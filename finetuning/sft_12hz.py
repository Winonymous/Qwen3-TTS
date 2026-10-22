# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from huggingface_hub import snapshot_download
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"],
                        help="Weight dtype. bfloat16 trains in pure bf16; float32 keeps fp32 master weights "
                             "(bf16 autocast for compute) and needs more memory.")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Linear learning-rate warmup, in optimizer steps")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop after this many optimizer steps and save a checkpoint (0 = train all epochs)")
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Also save a checkpoint every N optimizer steps (0 = off)")
    parser.add_argument("--save_epochs", type=int, default=1,
                        help="Save a checkpoint every N epochs (0 = only the final one)")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")

    MODEL_PATH = args.init_model_path
    if not os.path.isdir(MODEL_PATH):
        # A Hub repo id: resolve it to the local snapshot dir (uses the HF cache / offline mode), since
        # checkpoints are built by copying this directory and reading its config.json.
        MODEL_PATH = snapshot_download(MODEL_PATH)

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)
    # bf16 here means pure-bf16 training (no fp32 master weights); fp32 means autocast mixed precision.
    accelerator.print(f"Model weights dtype: {next(qwen3tts.model.parameters()).dtype}")

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    def save_checkpoint(name):
        if not accelerator.is_main_process:
            return
        output_dir = os.path.join(args.output_model_path, name)
        shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

        input_config_file = os.path.join(MODEL_PATH, "config.json")
        output_config_file = os.path.join(output_dir, "config.json")
        with open(input_config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {
            args.speaker_name.lower(): 3000
        }
        talker_config["spk_is_dialect"] = {
            args.speaker_name.lower(): False
        }
        config_dict["talker_config"] = talker_config

        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        unwrapped_model = accelerator.unwrap_model(model)
        # checkpoints are stored in bf16 like the Base model, whatever dtype was used for training
        state_dict = {
            k: (v.detach().to("cpu", torch.bfloat16) if v.is_floating_point() else v.detach().to("cpu"))
            for k, v in unwrapped_model.state_dict().items()
        }

        drop_prefix = "speaker_encoder"
        keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
        for k in keys_to_drop:
            del state_dict[k]

        weight = state_dict['talker.model.codec_embedding.weight']
        state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)
        accelerator.print(f"Saved {output_dir}")

    num_epochs = args.num_epochs
    opt_step = 0  # completed optimizer steps
    done = False
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            if args.warmup_steps > 0:
                lr = args.lr * min(1.0, (opt_step + 1) / args.warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr

            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # Inference always runs text embeddings through `text_projection` (text_hidden -> hidden MLP), so
                # training must too; the mask goes after it because the projection has a bias.
                input_text_embedding = model.talker.text_projection(
                    model.talker.model.text_embedding(input_text_ids)
                ) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                # the speaker slot sits right before the final codec_pad of the header; its index depends on whether
                # the sample has a language token
                speaker_pos = batch['speaker_pos'].to(input_ids.device)
                input_codec_embedding[torch.arange(input_ids.shape[0], device=input_ids.device), speaker_pos] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                opt_step += 1
                if args.save_steps > 0 and opt_step % args.save_steps == 0:
                    save_checkpoint(f"checkpoint-step-{opt_step}")
                if args.max_steps > 0 and opt_step >= args.max_steps:
                    done = True

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Opt step {opt_step} | Loss: {loss.item():.4f}")

            if done:
                break

        if done:
            save_checkpoint(f"checkpoint-step-{opt_step}")
            break
        if (args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0) or epoch == num_epochs - 1:
            save_checkpoint(f"checkpoint-epoch-{epoch}")

if __name__ == "__main__":
    train()
