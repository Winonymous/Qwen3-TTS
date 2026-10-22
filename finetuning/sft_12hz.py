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
import torch.nn.functional as F
from accelerate import Accelerator
from dataset import TTSDataset
from huggingface_hub import snapshot_download
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None


def compute_losses(model, batch):
    """Training / validation loss for one batch.

    Returns (loss, talker_loss, sub_talker_loss, speaker_embedding).
    """
    input_ids = batch['input_ids']
    codec_ids = batch['codec_ids']
    ref_mels = batch['ref_mels']
    text_embedding_mask = batch['text_embedding_mask']
    codec_embedding_mask = batch['codec_embedding_mask']
    attention_mask = batch['attention_mask']
    codec_0_labels = batch['codec_0_labels']
    codec_mask = batch['codec_mask']

    speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()

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

    # No `labels=` here: the model's built-in loss is Hugging Face's causal-LM loss, which shifts the
    # labels once more, so passing the already-shifted labels[:, 1:] trained the talker to predict two
    # frames ahead (check_loss_alignment.py: base CE 1.2 at the correct offset vs 7.8 at the shifted one).
    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        output_hidden_states=True
    )
    talker_loss = F.cross_entropy(
        outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
        codec_0_labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    # The sub-talker for a frame is conditioned on the hidden state that predicted that frame's first
    # code, i.e. the one at the position *before* the frame's own input, as in generation.
    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]

    sub_talker_logits, _ = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    # its built-in loss shifts the labels too; logits[:, j] predicts code group j + 1
    sub_talker_loss = F.cross_entropy(
        sub_talker_logits.float().reshape(-1, sub_talker_logits.shape[-1]),
        talker_codec_ids[:, 1:].reshape(-1),
    )

    loss = talker_loss + 0.3 * sub_talker_loss
    return loss, talker_loss, sub_talker_loss, speaker_embedding


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
    parser.add_argument("--val_jsonl", type=str, default=None,
                        help="Held-out JSONL (e.g. the test split from prepare_data.py). When given, the validation "
                             "loss is computed before training, after every saved checkpoint and at the end of every "
                             "epoch, and the best checkpoint is kept as checkpoint-best.")
    parser.add_argument("--val_batch_size", type=int, default=0,
                        help="Validation batch size (0 = same as --batch_size)")
    parser.add_argument("--no_initial_eval", action="store_true",
                        help="Skip the validation pass before training starts")
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

    val_dataloader = None
    if args.val_jsonl:
        val_data = [json.loads(line) for line in open(args.val_jsonl).readlines()]
        val_dataset = TTSDataset(val_data, qwen3tts.processor, config)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.val_batch_size or args.batch_size, shuffle=False,
            collate_fn=val_dataset.collate_fn,
        )
        accelerator.print(f"Validation: {len(val_data)} samples from {args.val_jsonl}")

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    if val_dataloader is not None:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            qwen3tts.model, optimizer, train_dataloader, val_dataloader
        )
    else:
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

    best_val = float("inf")

    def evaluate():
        """Mean (loss, talker, sub-talker) over the validation set."""
        model.eval()
        sums = torch.zeros(4, device=accelerator.device)
        with torch.no_grad():
            for val_batch in val_dataloader:
                loss, talker_loss, sub_talker_loss, _ = compute_losses(model, val_batch)
                n = val_batch['input_ids'].shape[0]
                sums += torch.tensor(
                    [loss.item() * n, talker_loss.item() * n, sub_talker_loss.item() * n, n],
                    device=accelerator.device,
                )
        sums = accelerator.reduce(sums, reduction="sum")
        model.train()
        return (sums[:3] / sums[3]).tolist()

    def run_eval(tag, keep_best=True):
        nonlocal best_val
        if val_dataloader is None:
            return
        val_loss, val_talker, val_sub = evaluate()
        accelerator.print(
            f"Validation @ {tag}: loss {val_loss:.4f} (talker {val_talker:.4f}, sub-talker {val_sub:.4f})"
        )
        if keep_best and val_loss < best_val:
            best_val = val_loss
            save_checkpoint("checkpoint-best")
            accelerator.print(f"New best validation loss {val_loss:.4f} -> checkpoint-best")

    num_epochs = args.num_epochs
    opt_step = 0  # completed optimizer steps
    done = False
    if not args.no_initial_eval:
        run_eval("start (untrained)", keep_best=False)
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            if args.warmup_steps > 0:
                lr = args.lr * min(1.0, (opt_step + 1) / args.warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr

            with accelerator.accumulate(model):

                loss, talker_loss, sub_talker_loss, speaker_embedding = compute_losses(model, batch)
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                opt_step += 1
                if args.save_steps > 0 and opt_step % args.save_steps == 0:
                    save_checkpoint(f"checkpoint-step-{opt_step}")
                    run_eval(f"opt step {opt_step}")
                if args.max_steps > 0 and opt_step >= args.max_steps:
                    done = True

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Opt step {opt_step} | Loss: {loss.item():.4f} "
                    f"(talker {talker_loss.item():.4f}, sub-talker {sub_talker_loss.item():.4f})"
                )

            if done:
                break

        if done:
            save_checkpoint(f"checkpoint-step-{opt_step}")
            run_eval(f"opt step {opt_step} (final)")
            break
        if (args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0) or epoch == num_epochs - 1:
            save_checkpoint(f"checkpoint-epoch-{epoch}")
        run_eval(f"end of epoch {epoch}")

if __name__ == "__main__":
    train()
