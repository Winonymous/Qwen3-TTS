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
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

LANGUAGE_ID_RANGE = (2048, 2148)  # free slots among the codec special tokens (suppressed at generation)
SPEAKER_ID_START = 3000


def _used_codec_ids(talker_config):
    used = set()
    for k, v in talker_config.to_dict().items():
        if isinstance(v, int) and k.endswith("_id"):
            used.add(v)
    for name in ("spk_id", "codec_language_id"):
        used.update(v for v in (getattr(talker_config, name, None) or {}).values() if isinstance(v, int))
    return used


def register_speakers_and_languages(talker_config, speakers, languages):
    """Assign codec-embedding ids to speakers / languages that the checkpoint does not know yet.

    Names already in the config keep their id (so an existing speaker/language is simply fine-tuned further).
    Returns (new_speaker_ids, new_language_ids) as {name: id}.
    """
    if talker_config.codec_language_id is None:
        talker_config.codec_language_id = {}
    if talker_config.spk_id is None:
        talker_config.spk_id = {}
    if talker_config.spk_is_dialect is None:
        talker_config.spk_is_dialect = {}
    used = _used_codec_ids(talker_config)

    def alloc(candidates):
        for i in candidates:
            if i not in used:
                used.add(i)
                return i
        raise RuntimeError("Out of free codec embedding ids; enlarge talker_config.vocab_size to add more.")

    new_languages = {}
    for lang in languages:
        if lang == "auto" or lang in talker_config.codec_language_id:
            continue
        if "dialect" in lang:
            raise ValueError(f"Language name {lang!r} must not contain 'dialect' (reserved by get_supported_languages).")
        lang_id = alloc(list(range(*LANGUAGE_ID_RANGE)) + list(range(SPEAKER_ID_START)))
        talker_config.codec_language_id[lang] = lang_id
        new_languages[lang] = lang_id

    new_speakers = {}
    for spk in speakers:
        if spk in talker_config.spk_id:
            continue
        spk_id = alloc(range(SPEAKER_ID_START, talker_config.vocab_size))
        talker_config.spk_id[spk] = spk_id
        talker_config.spk_is_dialect[spk] = False
        new_speakers[spk] = spk_id
    return new_speakers, new_languages


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test",
                        help="Speaker used for rows without a `speaker` field.")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl, encoding="utf-8").readlines()
    train_data = [json.loads(line) for line in train_data]

    # Speakers / languages come from the `speaker` / `language` fields of each row.
    speakers = sorted({(d.get("speaker") or args.speaker_name).lower() for d in train_data})
    languages = sorted({(d.get("language") or "auto").lower() for d in train_data})
    talker_config = config.talker_config
    new_speakers, new_languages = register_speakers_and_languages(talker_config, speakers, languages)
    accelerator.print(f"Speakers: {speakers} (new: {new_speakers})")
    accelerator.print(f"Languages: {languages} (new: {new_languages})")

    # New language embeddings start from the mean of the known ones plus noise at the same scale,
    # so they begin inside the existing language-embedding distribution and are then trained.
    codec_weight = qwen3tts.model.talker.model.codec_embedding.weight
    with torch.no_grad():
        known = [i for name, i in talker_config.codec_language_id.items() if name not in new_languages]
        if known:
            known_rows = codec_weight[known].float()
            mean, std = known_rows.mean(0), known_rows.std()
        else:
            mean, std = torch.zeros_like(codec_weight[0]).float(), codec_weight.float().std()
        for lang_id in new_languages.values():
            codec_weight[lang_id] = (mean + torch.randn_like(mean) * std).to(codec_weight.dtype)

    dataset = TTSDataset(train_data, qwen3tts.processor, config, default_speaker=args.speaker_name)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    speaker_sum, speaker_count = {}, {}
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_pos = batch['speaker_pos']
                speaker_idx = batch['speaker_idx']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()

                # Running mean of each speaker's encoder embedding; written into that speaker's codec row at save time.
                all_emb = accelerator.gather(speaker_embedding).float()
                all_idx = accelerator.gather(speaker_idx)
                for emb, idx in zip(all_emb, all_idx.tolist()):
                    name = dataset.speakers[idx]
                    speaker_sum[name] = speaker_sum.get(name, 0) + emb
                    speaker_count[name] = speaker_count.get(name, 0) + 1

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
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

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            config_dict["talker_config"] = {
                **config_dict.get("talker_config", {}),
                "spk_id": dict(talker_config.spk_id),
                "spk_is_dialect": dict(talker_config.spk_is_dialect),
                "codec_language_id": dict(talker_config.codec_language_id),
            }

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            for name, total in speaker_sum.items():
                mean_embedding = total / speaker_count[name]
                weight[talker_config.spk_id[name]] = mean_embedding.detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

if __name__ == "__main__":
    train()
