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
import gc
import json
import os
import random
import shutil
import traceback
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import timedelta

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from dataset import TTSDataset
from huggingface_hub import snapshot_download
from logging_utils import setup_logging
from text_utils import normalize_text
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoConfig

SPEAKER_ID_START = 3000  # first codec-embedding id used for fine-tuned speakers
LANGUAGE_ID_RANGE = (2048, 2148)  # free slots among the codec special tokens (never produced as audio codes)


def used_codec_ids(talker_config):
    """Every codec-embedding id the config already uses (special tokens, speakers, languages)."""
    used = set()
    for key, value in talker_config.to_dict().items():
        if isinstance(value, int) and key.endswith("_id"):
            used.add(value)
    for name in ("spk_id", "codec_language_id"):
        used.update(v for v in (getattr(talker_config, name, None) or {}).values() if isinstance(v, int))
    return used


def register_languages(talker_config, languages):
    """Give every language the checkpoint does not know a free codec-embedding id.

    Updates `talker_config.codec_language_id` in place and returns {name: id} for the new languages.
    "auto" needs no id (it means: no language token).
    """
    if talker_config.codec_language_id is None:
        talker_config.codec_language_id = {}
    used = used_codec_ids(talker_config)
    candidates = list(range(*LANGUAGE_ID_RANGE)) + list(range(LANGUAGE_ID_RANGE[0]))
    new_languages = {}
    for lang in languages:
        if lang == "auto" or lang in talker_config.codec_language_id:
            continue
        if "dialect" in lang:
            raise ValueError(f"Language name {lang!r} must not contain 'dialect' (reserved by get_supported_languages)")
        free = next((i for i in candidates if i not in used), None)
        if free is None:
            raise RuntimeError("No free codec-embedding id left for a new language")
        used.add(free)
        talker_config.codec_language_id[lang] = free
        new_languages[lang] = free
    return new_languages


def compute_losses(model, batch, per_sample=False):
    """Training / validation loss for one batch.

    Returns (loss, talker_loss, sub_talker_loss, speaker_embedding). With `per_sample=True` a fifth element
    (talker_loss_per_sample, sub_talker_loss_per_sample), each of shape (batch,), is returned as well; it is used to
    report validation loss per language and speaker.
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
    input_codec_embedding[torch.arange(input_ids.shape[0], device=input_ids.device), speaker_pos] = \
        speaker_embedding.to(input_codec_embedding.dtype)

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
    if not per_sample:
        return loss, talker_loss, sub_talker_loss, speaker_embedding

    batch_size = input_ids.shape[0]
    labels = codec_0_labels[:, 1:]
    token_loss = F.cross_entropy(
        outputs.logits.float().reshape(-1, outputs.logits.shape[-1]), labels.reshape(-1),
        ignore_index=-100, reduction="none",
    ).view(batch_size, -1)
    talker_ps = token_loss.sum(1) / (labels != -100).sum(1).clamp(min=1)

    frames = codec_mask.sum(1)  # codec frames per sample; talker_codec_ids is ordered sample by sample
    frame_loss = F.cross_entropy(
        sub_talker_logits.float().reshape(-1, sub_talker_logits.shape[-1]),
        talker_codec_ids[:, 1:].reshape(-1), reduction="none",
    ).view(talker_codec_ids.shape[0], -1).mean(1)
    owner = torch.arange(batch_size, device=frames.device).repeat_interleave(frames)
    sub_ps = torch.zeros(batch_size, device=frame_loss.device).index_add_(0, owner, frame_loss)
    sub_ps = sub_ps / frames.clamp(min=1)
    return loss, talker_loss, sub_talker_loss, speaker_embedding, (talker_ps, sub_ps)


class LossModule(nn.Module):
    """The model together with its loss, so that one training step is one `forward` call.

    With several GPUs accelerate wraps the model in DistributedDataParallel, which only synchronizes gradients when
    the computation goes through its `forward`. compute_losses uses the sub-modules (talker, speaker_encoder)
    directly, so it is called from here instead: `model(batch)` runs it inside DDP's forward. The wrapped Qwen3-TTS
    model is `self.inner`; its state dict (and so the checkpoint) is unchanged.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, batch, per_sample=False):
        return compute_losses(self.inner, batch, per_sample=per_sample)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--speaker_encoder_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Where the speaker encoder comes from when --init_model_path is a checkpoint of this "
                             "script (those have none)")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test",
                        help="Speaker for rows without a `speaker` field. Rows can carry their own `speaker` name to "
                             "train several speakers at once.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"],
                        help="Weight dtype. bfloat16 trains in pure bf16; float32 keeps fp32 master weights "
                             "(bf16 autocast for compute) and needs more memory.")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Linear learning-rate warmup, in optimizer steps")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Batches whose gradients are added up before each optimizer step. The effective batch "
                             "size is batch_size x number of GPUs x grad_accum.")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop after this many optimizer steps and save a checkpoint (0 = train all epochs)")
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Also save a checkpoint every N optimizer steps (0 = off)")
    parser.add_argument("--save_epochs", type=int, default=1,
                        help="Save a checkpoint every N epochs (0 = only the final one)")
    parser.add_argument("--language_init_noise", type=float, default=0.1,
                        help="New languages (not in the checkpoint) start from the mean embedding of the known "
                             "languages plus Gaussian noise of this many standard deviations")
    parser.add_argument("--val_jsonl", type=str, default=None,
                        help="Held-out JSONL (e.g. the test split from prepare_data.py). When given, the validation "
                             "loss is computed before training, after every saved checkpoint and at the end of every "
                             "epoch, and the best checkpoint is kept as checkpoint-best.")
    parser.add_argument("--val_batch_size", type=int, default=0,
                        help="Validation batch size (0 = same as --batch_size)")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Smoke test: train on a random subset of this many rows (0 = all). The validation "
                             "set is cut to --num_val_samples (default: a fifth of this, at least 1).")
    parser.add_argument("--num_val_samples", type=int, default=0,
                        help="Use a random subset of this many validation rows (0 = all, or a fifth of "
                             "--num_samples when that is set). A large validation set makes every evaluation slow.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes. Each sample loads and resamples its reference audio, "
                             "which is slow in the main process.")
    parser.add_argument("--sample_seed", type=int, default=42, help="Seed for --num_samples")
    parser.add_argument("--max_codec_frames", type=int, default=0,
                        help="Drop train/val rows whose audio has more codec frames than this (0 = no limit). "
                             "The codec runs at 12.5 frames per second, so 375 frames = 30 s.")
    parser.add_argument("--balance_by", type=str, default="none",
                        choices=["none", "language", "speaker", "language_speaker"],
                        help="Sample training rows so that small groups are seen more often than their share of "
                             "the data. none = plain shuffling. Rows are drawn with replacement; an epoch is "
                             "still len(train) draws.")
    parser.add_argument("--balance_power", type=float, default=0.5,
                        help="With --balance_by: a row's sampling weight is group_size ** -power. 0 = no "
                             "balancing, 1 = every group equally likely, 0.5 = square-root balancing (a milder "
                             "compromise that repeats small groups less).")
    parser.add_argument("--best_by", type=str, default="pooled", choices=["pooled", "macro"],
                        help="What picks checkpoint-best: pooled = validation loss over all rows (large groups "
                             "dominate); macro = mean of the per-(language, speaker) losses (every group counts "
                             "equally; recommended for several languages).")
    parser.add_argument("--normalize_text", action="store_true",
                        help="Normalize the text of every train/val row (Unicode NFC, invisible characters removed, "
                             "whitespace collapsed) while loading. Use it on JSONL files prepared without "
                             "normalization; no re-encoding needed. Also applies to --sample_file sentences.")
    parser.add_argument("--max_text_chars", type=int, default=0,
                        help="Drop train/val rows whose text has more characters than this (0 = no limit). "
                             "Batches are padded to their longest row, so one runaway transcript (for example a "
                             "repeated ASR hallucination) can use all the GPU memory.")
    parser.add_argument("--sample_epochs", type=int, default=0,
                        help="Every N epochs (and at the end if --max_steps stops the run), load the saved "
                             "checkpoint like at inference and synthesize a few sentences into "
                             "<output_model_path>/samples/epoch-N/ (also logged to wandb). 0 = off.")
    parser.add_argument("--sample_steps", type=int, default=0,
                        help="The same, every N optimizer steps instead of every N epochs, into "
                             "<output_model_path>/samples/step-N/. The checkpoint is the one saved by --save_steps, "
                             "so this must be a multiple of --save_steps. 0 = off.")
    parser.add_argument("--sample_file", type=str, default=None,
                        help="Text file with one sentence per line to synthesize for every speaker. Default: the "
                             "first --num_sample_texts validation rows of each (speaker, language) group.")
    parser.add_argument("--sample_language", type=str, default=None,
                        help="Language for --sample_file sentences (default: the training language)")
    parser.add_argument("--num_sample_texts", type=int, default=2,
                        help="Sentences per (speaker, language) group when --sample_file is not given")
    parser.add_argument("--sample_batch_size", type=int, default=8,
                        help="Sentences synthesized per generate call while sampling (they run in parallel in "
                             "the batch; one call can mix speakers and languages)")
    parser.add_argument("--sample_background", action="store_true",
                        help="Synthesize the samples in a separate process (infer_12hz.py) so training goes on "
                             "while they are generated. The audio is logged to wandb when the job has finished.")
    parser.add_argument("--sample_gpu", type=str, default=None,
                        help="With --sample_background: GPU id for the sampling process (default: the training "
                             "GPU; a different GPU does not slow training down)")
    parser.add_argument("--sample_max_new_tokens", type=int, default=300,
                        help="Generation limit in codec frames (12.5 per second, so 300 = 24 s)")
    parser.add_argument("--no_initial_eval", action="store_true",
                        help="Skip the validation pass before training starts")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Log losses, learning rate and validation to this Weights & Biases project "
                             "(default: wandb is off). Use WANDB_MODE=offline on a node without internet.")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None,
                        help="Everything printed (losses, warnings, errors) is also written here "
                             "(default: <output_model_path>/train.log)")
    args = parser.parse_args()
    if args.sample_steps > 0 and (args.save_steps <= 0 or args.sample_steps % args.save_steps):
        parser.error("--sample_steps needs --save_steps and must be a multiple of it (samples are made from the "
                     "checkpoint that --save_steps saved)")
    setup_logging(args.log_file or os.path.join(args.output_model_path, "train.log"), args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16",
        # some parameters get no gradient (the speaker encoder's output is detached), which DDP must be told about
        # a long timeout: the other processes wait while the main one saves checkpoints and generates samples
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True),
                         InitProcessGroupKwargs(timeout=timedelta(hours=3))],
    )
    accelerator.print(f"Processes: {accelerator.num_processes}, batch size per GPU {args.batch_size}, "
                      f"grad_accum {args.grad_accum} -> effective batch "
                      f"{args.batch_size * accelerator.num_processes * args.grad_accum}")

    wandb_run = None
    if args.wandb_project and accelerator.is_main_process:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name,
            config=vars(args), dir=args.output_model_path,
        )

    def wandb_log(data, step):
        if wandb_run is not None:
            try:
                wandb_run.log(data, step=step)
            except Exception:  # a logging problem must not kill a long training run
                traceback.print_exc()

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
    if getattr(qwen3tts.model, "speaker_encoder", None) is None:
        # Checkpoints written by this script have no speaker encoder (it is only needed to build speaker embeddings
        # from reference clips), so when continuing from one it is taken from the Base model. It is frozen anyway.
        base_path = args.speaker_encoder_path
        if not os.path.isdir(base_path):
            base_path = snapshot_download(base_path)
        accelerator.print(f"{MODEL_PATH} has no speaker encoder; loading it from {base_path}")
        base = Qwen3TTSModel.from_pretrained(base_path, dtype=qwen3tts.model.dtype, attn_implementation="sdpa")
        if getattr(base.model, "speaker_encoder", None) is None:
            raise SystemExit(f"{base_path} has no speaker encoder either; pass a Base model as --speaker_encoder_path")
        qwen3tts.model.speaker_encoder = base.model.speaker_encoder
        del base
    config = AutoConfig.from_pretrained(MODEL_PATH)
    # bf16 here means pure-bf16 training (no fp32 master weights); fp32 means autocast mixed precision.
    accelerator.print(f"Model weights dtype: {next(qwen3tts.model.parameters()).dtype}")

    def read_jsonl(path):
        # Line by line: readlines() would hold the whole file as text next to the parsed rows, in every process.
        # The codes become int16 arrays: as Python lists of ints they take about 7 times more memory than the JSON
        # text, in each of the processes (hundreds of thousands of rows x 16 codes per frame).
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["audio_codes"] = np.asarray(row["audio_codes"], dtype=np.int16)
                    rows.append(row)
        return rows

    train_data = read_jsonl(args.train_jsonl)
    val_data = read_jsonl(args.val_jsonl) if args.val_jsonl else []
    if args.normalize_text:
        changed = 0
        for d in train_data + val_data:
            clean = normalize_text(d["text"])
            changed += clean != d["text"]
            d["text"] = clean
        accelerator.print(f"--normalize_text changed {changed} of {len(train_data) + len(val_data)} row(s)")

    for name, rows in (("train", train_data), ("val", val_data)):
        if rows:
            texts = sorted(len(d["text"]) for d in rows)
            frames = sorted(len(d["audio_codes"]) for d in rows)
            accelerator.print(f"Length stats ({name}): text chars median {texts[len(texts) // 2]}, "
                              f"longest {texts[-3:]}; codec frames median {frames[len(frames) // 2]}, "
                              f"longest {frames[-3:]}")
    if args.max_text_chars > 0:
        n_train, n_val = len(train_data), len(val_data)
        train_data = [d for d in train_data if len(d["text"]) <= args.max_text_chars]
        val_data = [d for d in val_data if len(d["text"]) <= args.max_text_chars]
        accelerator.print(f"--max_text_chars {args.max_text_chars}: train {len(train_data)}/{n_train}, "
                          f"val {len(val_data)}/{n_val} rows kept")
        if not train_data:
            raise SystemExit("No training rows left after --max_text_chars")
    if args.max_codec_frames > 0:
        n_train, n_val = len(train_data), len(val_data)
        train_data = [d for d in train_data if len(d["audio_codes"]) <= args.max_codec_frames]
        val_data = [d for d in val_data if len(d["audio_codes"]) <= args.max_codec_frames]
        accelerator.print(f"--max_codec_frames {args.max_codec_frames}: train {len(train_data)}/{n_train}, "
                          f"val {len(val_data)}/{n_val} rows kept")
        if not train_data:
            raise SystemExit("No training rows left after --max_codec_frames")
    if args.num_samples > 0 or args.num_val_samples > 0:
        rng = random.Random(args.sample_seed)
        n_train_all, n_val_all = len(train_data), len(val_data)
        if args.num_samples > 0 and len(train_data) > args.num_samples:
            train_data = rng.sample(train_data, args.num_samples)
        n_val = args.num_val_samples or max(1, args.num_samples // 5)
        if len(val_data) > n_val:
            val_data = rng.sample(val_data, n_val)
        accelerator.print(f"Subsampling (--num_samples/--num_val_samples): train {len(train_data)}/{n_train_all}, "
                          f"val {len(val_data)}/{n_val_all} rows")

    # Speakers come from each row's `speaker` field (rows without one use --speaker_name). Names are lowercased
    # because inference looks them up with `speaker.lower()`. Each speaker gets its own row in the codec embedding table.
    speakers = sorted({(d.get("speaker") or args.speaker_name).lower() for d in train_data + val_data})
    speaker2idx = {name: i for i, name in enumerate(speakers)}
    existing_ids = dict(config.talker_config.spk_id or {})
    used_ids = set(existing_ids.values())
    speaker_ids = {}
    next_id = SPEAKER_ID_START
    for name in speakers:
        if name in existing_ids:
            speaker_ids[name] = existing_ids[name]
            continue
        while next_id in used_ids:
            next_id += 1
        if next_id >= config.talker_config.vocab_size:
            raise RuntimeError(f"No free codec embedding id left for speaker {name!r}")
        speaker_ids[name] = next_id
        used_ids.add(next_id)
    accelerator.print(f"Speakers: {speaker_ids}")

    # Languages come from each row's `language` field (missing = "auto", i.e. no language token). A language the
    # checkpoint does not know gets a new id and a fresh embedding that is trained like the rest of the model.
    languages = sorted({(d.get("language") or "auto").lower() for d in train_data + val_data})
    new_languages = register_languages(config.talker_config, languages)
    accelerator.print(f"Languages: {languages} (new: {new_languages})")
    if new_languages:
        codec_weight = qwen3tts.model.talker.model.codec_embedding.weight
        known_ids = [i for n, i in config.talker_config.codec_language_id.items() if n not in new_languages]
        with torch.no_grad():
            if known_ids:
                known_rows = codec_weight[known_ids].float()
                mean, std = known_rows.mean(0), known_rows.std()
            else:
                mean, std = torch.zeros_like(codec_weight[0]).float(), codec_weight.float().std()
            for lang_id in new_languages.values():
                init = mean + torch.randn_like(mean) * std * args.language_init_noise
                codec_weight[lang_id] = init.to(codec_weight.dtype)

    language2idx = {name: i for i, name in enumerate(languages)}
    dataset = TTSDataset(train_data, qwen3tts.processor, config, speaker2idx=speaker2idx,
                         default_speaker=args.speaker_name, language2idx=language2idx)

    sampler = None
    if args.balance_by != "none":
        def group_key(d):
            lang = (d.get("language") or "auto").lower()
            spk = (d.get("speaker") or args.speaker_name).lower()
            return {"language": lang, "speaker": spk, "language_speaker": f"{lang}/{spk}"}[args.balance_by]

        keys = [group_key(d) for d in train_data]
        counts = Counter(keys)
        weights = [counts[k] ** -args.balance_power for k in keys]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_data), replacement=True)
        total_weight = sum(weights)
        for group in sorted(counts):
            share = counts[group] ** (1 - args.balance_power) / total_weight
            accelerator.print(f"Balanced sampling [{group}]: {counts[group]} rows = "
                              f"{counts[group] / len(train_data):.1%} of the data, drawn {share:.1%} of the time")
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                                  collate_fn=dataset.collate_fn, num_workers=args.num_workers,
                                  persistent_workers=args.num_workers > 0)

    val_dataloader = None
    if args.val_jsonl:
        val_dataset = TTSDataset(val_data, qwen3tts.processor, config, speaker2idx=speaker2idx,
                                 default_speaker=args.speaker_name, language2idx=language2idx)
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.val_batch_size or args.batch_size, shuffle=False,
            collate_fn=val_dataset.collate_fn, num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        accelerator.print(f"Validation: {len(val_data)} samples from {args.val_jsonl}")

    loss_model = LossModule(qwen3tts.model)
    optimizer = AdamW(loss_model.parameters(), lr=args.lr, weight_decay=0.01)

    if val_dataloader is not None:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            loss_model, optimizer, train_dataloader, val_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            loss_model, optimizer, train_dataloader
        )

    # Running mean of each speaker's encoder embedding over all training batches; it becomes that speaker's row.
    speaker_sum, speaker_count = {}, {}

    def save_checkpoint(name):
        # Every process runs this: each has only seen its own share of the batches, so the speaker embedding sums
        # are added up over all of them. Only the main process writes the files.
        dim = accelerator.unwrap_model(model).inner.talker.model.codec_embedding.weight.shape[1]
        sums = torch.zeros(len(speakers), dim, device=accelerator.device)
        counts = torch.zeros(len(speakers), device=accelerator.device)
        for k, speaker in enumerate(speakers):
            if speaker_count.get(speaker, 0) > 0:
                sums[k] = speaker_sum[speaker]
                counts[k] = speaker_count[speaker]
        sums = accelerator.reduce(sums, reduction="sum")
        counts = accelerator.reduce(counts, reduction="sum")
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
        # only speakers that have been seen in training so far have a meaningful embedding
        trained = [speaker for k, speaker in enumerate(speakers) if counts[k] > 0]
        # speakers of the checkpoint this run started from stay (their embedding rows are in the weights)
        old_ids = dict(existing_ids)
        talker_config["spk_id"] = {**old_ids, **{name: speaker_ids[name] for name in trained}}
        talker_config["spk_is_dialect"] = {name: False for name in talker_config["spk_id"]}
        # keeps the checkpoint's own languages and adds the new ones
        talker_config["codec_language_id"] = dict(config.talker_config.codec_language_id)
        config_dict["talker_config"] = talker_config

        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        unwrapped_model = accelerator.unwrap_model(model).inner
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
        for speaker in trained:
            k = speakers.index(speaker)
            mean_embedding = sums[k] / counts[k]
            weight[speaker_ids[speaker]] = mean_embedding.detach().to(weight.device).to(weight.dtype)
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)
        accelerator.print(f"Saved {output_dir}")

    # Sentences synthesized after each epoch: (speaker, language, text)
    sample_items = []
    if args.sample_epochs > 0 or args.sample_steps > 0:
        if args.sample_file:
            with open(args.sample_file, encoding="utf-8") as f:
                sample_texts = [line.strip() for line in f if line.strip()]
            if args.normalize_text:
                sample_texts = [normalize_text(t) for t in sample_texts]
            real_languages = [lang for lang in languages if lang != "auto"]
            sample_language = (args.sample_language or (real_languages[0] if real_languages else "auto")).lower()
            sample_items = [(sp, sample_language, t) for sp in speakers for t in sample_texts]
        else:
            per_group = {}
            for d in (val_data or train_data):
                key = ((d.get("speaker") or args.speaker_name).lower(), (d.get("language") or "auto").lower())
                if len(per_group.setdefault(key, [])) < args.num_sample_texts:
                    per_group[key].append(d["text"])
            sample_items = [(sp, lang, t) for (sp, lang), texts in sorted(per_group.items()) for t in texts]
        accelerator.print(f"Sampling every {args.sample_epochs} epoch(s) / {args.sample_steps} step(s): "
                          f"{len(sample_items)} sentence(s)")

    def _generate_samples(ckpt_name, tag):
        import soundfile as sf

        out_dir = os.path.join(args.output_model_path, "samples", tag)
        os.makedirs(out_dir, exist_ok=True)
        tts = None
        try:
            # Load the saved checkpoint exactly as inference does, so this also checks the checkpoint itself.
            tts = Qwen3TTSModel.from_pretrained(
                os.path.join(args.output_model_path, ckpt_name), device_map=str(accelerator.device),
                dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            )
            audio_logs = {}
            with open(os.path.join(out_dir, "samples.tsv"), "w", encoding="utf-8") as manifest:
                # --sample_batch_size sentences per generate call (one call can mix speakers and languages)
                for first in range(0, len(sample_items), args.sample_batch_size):
                    batch = sample_items[first:first + args.sample_batch_size]
                    wavs, sr = tts.generate_custom_voice(
                        text=[t for _, _, t in batch], language=[lang for _, lang, _ in batch],
                        speaker=[spk for spk, _, _ in batch],
                        max_new_tokens=args.sample_max_new_tokens, do_sample=False,
                    )
                    for j, ((speaker, language, text), wav) in enumerate(zip(batch, wavs)):
                        name = f"{first + j:02d}_{speaker}_{language}"
                        sf.write(os.path.join(out_dir, name + ".wav"), wav, sr)
                        manifest.write(f"{name}\t{text}\n")
                        if wandb_run is not None:
                            audio_logs[f"samples/{tag}/{name}"] = wandb.Audio(wav, sample_rate=sr, caption=text)
            if audio_logs:
                wandb_log(audio_logs, step=opt_step)
                accelerator.print(f"Logged {len(audio_logs)} audio clip(s) of {tag} to wandb at step {opt_step}")
            accelerator.print(f"Wrote {len(sample_items)} sample(s) to {out_dir}")
        except Exception:  # a sampling problem must not kill a long training run
            traceback.print_exc()
            accelerator.print(f"Sampling failed for {ckpt_name}; training continues")
        finally:
            del tts
            gc.collect()
            torch.cuda.empty_cache()

    pending_samples = []  # (process, tag, out_dir, log file) of background sampling jobs

    def _launch_background_samples(ckpt_name, tag):
        """Start infer_12hz.py on the saved checkpoint as a separate process; training does not wait for it."""
        import subprocess
        import sys

        out_dir = os.path.join(args.output_model_path, "samples", tag)
        os.makedirs(out_dir, exist_ok=True)
        items_path = os.path.join(args.output_model_path, "samples", "items.tsv")
        with open(items_path, "w", encoding="utf-8") as f:
            f.write("id\tspeaker\tlanguage\ttext\n")
            for k, (speaker, language, text) in enumerate(sample_items):
                f.write(f"{k:02d}\t{speaker}\t{language}\t{' '.join(text.split())}\n")
        gpu = args.sample_gpu if args.sample_gpu is not None else str(accelerator.device.index or 0)
        cmd = [
            sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer_12hz.py"),
            "--checkpoint", os.path.join(args.output_model_path, ckpt_name), "--input", items_path,
            "--output_dir", out_dir, "--text_column", "text", "--id_column", "id",
            "--speaker_column", "speaker", "--language_column", "language",
            "--batch_size", str(args.sample_batch_size), "--max_new_tokens", str(args.sample_max_new_tokens),
            "--gpus", gpu,
        ]
        log = open(os.path.join(out_dir, "subprocess.out"), "w")
        pending_samples.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), tag, out_dir, log))
        accelerator.print(f"Sampling for {tag} started in the background on GPU {gpu}; wavs go to {out_dir}")

    def collect_samples(wait=False):
        """Report finished background sampling jobs and log their audio to wandb (wait=True: wait for all)."""
        for entry in list(pending_samples):
            proc, tag, out_dir, log = entry
            if wait:
                proc.wait()
            if proc.poll() is None:
                continue
            pending_samples.remove(entry)
            log.close()
            if proc.returncode != 0:
                accelerator.print(f"Background sampling for {tag} failed (exit code {proc.returncode}); "
                                  f"see {os.path.join(out_dir, 'subprocess.out')}")
                continue
            accelerator.print(f"Background sampling for {tag} finished: {out_dir}")
            if wandb_run is None:
                continue
            try:
                import soundfile as sf

                audio_logs = {}
                with open(os.path.join(out_dir, "manifest.tsv"), encoding="utf-8") as f:
                    next(f, None)  # header: id, speaker, language, status, seconds, path, text
                    for line in f:
                        fields = line.rstrip("\n").split("\t")
                        if len(fields) >= 7 and fields[3] == "ok":
                            wav, sr = sf.read(fields[5])
                            audio_logs[f"samples/{tag}/{fields[0]}_{fields[1]}_{fields[2]}"] = wandb.Audio(
                                wav, sample_rate=sr, caption=f"{tag}: {fields[6]}")
                if audio_logs:
                    wandb_log(audio_logs, step=opt_step)  # logged at the step when the job finished
                    accelerator.print(f"Logged {len(audio_logs)} audio clip(s) of {tag} to wandb at step {opt_step}")
                else:
                    accelerator.print(f"No audio clip of {tag} to log: nothing with status ok in "
                                      f"{os.path.join(out_dir, 'manifest.tsv')}")
            except Exception:
                traceback.print_exc()

    def generate_samples(ckpt_name, tag):
        if not sample_items:
            return
        collect_samples()  # log background jobs that finished while training went on
        if args.sample_background:
            if accelerator.is_main_process:
                _launch_background_samples(ckpt_name, tag)
            return
        if accelerator.is_main_process:
            _generate_samples(ckpt_name, tag)
        accelerator.wait_for_everyone()

    def autocast_ctx():
        # With fp32 master weights the forward must run under bf16 autocast: Flash Attention 2 only accepts
        # fp16/bf16. compute_losses calls model.talker / model.speaker_encoder directly, which bypasses the
        # autocast that accelerate puts around the top-level forward, so it is applied here. With bf16 weights
        # nothing changes (disabled).
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.dtype == "float32")

    best_val = float("inf")

    n_groups = len(languages) * len(speakers)

    def evaluate():
        """Pooled mean (loss, talker, sub-talker) over the validation set, and the same per (language, speaker).

        The pooled numbers are batch means weighted by batch size (as in earlier runs). The per-group numbers
        average each sample's own loss, so they differ slightly from the pooled ones.
        """
        model.eval()
        sums = torch.zeros(4, device=accelerator.device)
        group_sums = torch.zeros(n_groups, 4, device=accelerator.device)  # loss, talker, sub-talker, count
        with torch.no_grad():
            for i, val_batch in enumerate(val_dataloader):
                if i % 50 == 0:
                    accelerator.print(f"  validation batch {i}/{len(val_dataloader)}")
                with autocast_ctx():
                    loss, talker_loss, sub_talker_loss, _, (talker_ps, sub_ps) = model(
                        val_batch, per_sample=True)
                n = val_batch['input_ids'].shape[0]
                sums += torch.tensor(
                    [loss.item() * n, talker_loss.item() * n, sub_talker_loss.item() * n, n],
                    device=accelerator.device,
                )
                group_id = val_batch['language_idx'] * len(speakers) + val_batch['speaker_idx']
                per_sample = torch.stack(
                    [talker_ps + 0.3 * sub_ps, talker_ps, sub_ps, torch.ones_like(talker_ps)], dim=1)
                group_sums.index_add_(0, group_id.to(accelerator.device), per_sample.float())
        sums = accelerator.reduce(sums, reduction="sum")
        group_sums = accelerator.reduce(group_sums, reduction="sum")
        model.train()

        groups = {}
        for g in range(n_groups):
            count = group_sums[g, 3].item()
            if count > 0:
                key = (languages[g // len(speakers)], speakers[g % len(speakers)])
                groups[key] = (*(group_sums[g, :3] / count).tolist(), int(count))
        return (sums[:3] / sums[3]).tolist(), groups

    def run_eval(tag, keep_best=True):
        nonlocal best_val
        if val_dataloader is None:
            return
        (val_loss, val_talker, val_sub), groups = evaluate()
        macro = sum(g[0] for g in groups.values()) / len(groups) if groups else val_loss
        log = {"val/loss": val_loss, "val/talker_loss": val_talker, "val/sub_talker_loss": val_sub,
               "val/macro_loss": macro}
        for (lang, spk), (g_loss, g_talker, g_sub, _) in groups.items():
            log[f"val_group/{lang}_{spk}/loss"] = g_loss
            log[f"val_group/{lang}_{spk}/talker_loss"] = g_talker
            log[f"val_group/{lang}_{spk}/sub_talker_loss"] = g_sub
        wandb_log(log, step=opt_step)
        accelerator.print(
            f"Validation @ {tag}: loss {val_loss:.4f} (talker {val_talker:.4f}, sub-talker {val_sub:.4f}), "
            f"macro {macro:.4f} over {len(groups)} group(s)"
        )
        if len(groups) > 1:
            for (lang, spk), (g_loss, g_talker, g_sub, count) in sorted(groups.items()):
                accelerator.print(f"    {lang}/{spk}: loss {g_loss:.4f} (talker {g_talker:.4f}, "
                                  f"sub-talker {g_sub:.4f}), {count} rows")
        score = macro if args.best_by == "macro" else val_loss
        if keep_best and score < best_val:
            best_val = score
            save_checkpoint("checkpoint-best")
            accelerator.print(f"New best validation {args.best_by} loss {score:.4f} -> checkpoint-best")

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

                with autocast_ctx():
                    loss, talker_loss, sub_talker_loss, speaker_embedding = model(batch)
                for emb, idx in zip(speaker_embedding.float(), batch['speaker_idx'].tolist()):
                    name = speakers[idx]
                    speaker_sum[name] = speaker_sum.get(name, 0) + emb
                    speaker_count[name] = speaker_count.get(name, 0) + 1

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                opt_step += 1
                wandb_log({
                    "train/loss": loss.item(), "train/talker_loss": talker_loss.item(),
                    "train/sub_talker_loss": sub_talker_loss.item(), "train/grad_norm": float(grad_norm),
                    "train/lr": optimizer.param_groups[0]["lr"], "train/epoch": epoch,
                }, step=opt_step)
                if args.save_steps > 0 and opt_step % args.save_steps == 0:
                    save_checkpoint(f"checkpoint-step-{opt_step}")
                    run_eval(f"opt step {opt_step}")
                    if (args.sample_steps > 0 and opt_step % args.sample_steps == 0
                            and not (args.max_steps > 0 and opt_step >= args.max_steps)):  # the last step samples below
                        generate_samples(f"checkpoint-step-{opt_step}", f"step-{opt_step}")
                if args.max_steps > 0 and opt_step >= args.max_steps:
                    done = True

            if step % 10 == 0:
                if pending_samples and accelerator.is_main_process:
                    collect_samples()  # log background samples as soon as their job has finished
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Opt step {opt_step} | Loss: {loss.item():.4f} "
                    f"(talker {talker_loss.item():.4f}, sub-talker {sub_talker_loss.item():.4f})"
                )

            if done:
                break

        if done:
            save_checkpoint(f"checkpoint-step-{opt_step}")
            run_eval(f"opt step {opt_step} (final)")
            generate_samples(f"checkpoint-step-{opt_step}", f"step-{opt_step}")
            break
        sample_due = args.sample_epochs > 0 and (epoch + 1) % args.sample_epochs == 0
        if (args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0) or epoch == num_epochs - 1 or sample_due:
            save_checkpoint(f"checkpoint-epoch-{epoch}")
        run_eval(f"end of epoch {epoch}")
        if sample_due:
            generate_samples(f"checkpoint-epoch-{epoch}", f"epoch-{epoch}")

    collect_samples(wait=True)  # background sampling of the last checkpoint may still be running
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train()
