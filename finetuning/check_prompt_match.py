# coding=utf-8
"""Check that the fine-tuning input equals the prompt the Base model builds at inference.

For one row of the training JSONL it
  1. builds the training input embeddings exactly as `sft_12hz.py` does, and
  2. captures the input embeddings the Base model feeds its talker in `generate_voice_clone`
     (x_vector_only_mode=True, non_streaming_mode=True, the layout training imitates),
then compares them position by position up to and including `codec_bos`.

If they match, the training prompt is not the problem. If a segment differs, that segment is the bug.

Usage (from the finetuning/ directory):
    python check_prompt_match.py --train_jsonl train_with_codes.jsonl
    python check_prompt_match.py --train_jsonl train_with_codes.jsonl --skip_projection   # old behaviour
"""
import argparse
import json

import torch
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from transformers import AutoConfig


class _Captured(Exception):
    def __init__(self, inputs_embeds):
        self.inputs_embeds = inputs_embeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--index", type=int, default=0, help="Which row of the JSONL to use")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn", type=str, default="sdpa")
    parser.add_argument("--skip_projection", action="store_true",
                        help="Feed raw text embeddings (no text_projection), as the original training loop did")
    args = parser.parse_args()

    tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path, device_map=args.device, dtype=torch.bfloat16, attn_implementation=args.attn,
    )
    model = tts.model
    config = AutoConfig.from_pretrained(args.init_model_path)

    with open(args.train_jsonl, encoding="utf-8") as f:
        row = [json.loads(line) for line in f][args.index]
    language = row.get("language") or "Auto"
    print(f"row {args.index}: language={language!r} text={row['text']!r}")

    # ---- 1) training-side embeddings (same math as sft_12hz.py) ----
    dataset = TTSDataset([row], tts.processor, config)
    batch = dataset.collate_fn([dataset[0]])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    speaker_pos = int(batch["speaker_pos"][0]) if "speaker_pos" in batch else 6

    with torch.no_grad():
        speaker_embedding = model.speaker_encoder(batch["ref_mels"].to(model.dtype))
        text_embedding = model.talker.model.text_embedding(input_ids[:, :, 0])
        if not args.skip_projection:
            text_embedding = model.talker.text_projection(text_embedding)
        text_embedding = text_embedding * batch["text_embedding_mask"]
        codec_embedding = model.talker.model.codec_embedding(input_ids[:, :, 1]) * batch["codec_embedding_mask"]
        codec_embedding[:, speaker_pos, :] = speaker_embedding
        train_embeds = (text_embedding + codec_embedding)[0]

    # ---- 2) inference-side embeddings, captured at the talker.generate call ----
    original_generate = model.talker.generate

    def capture(*a, **kw):
        raise _Captured(kw["inputs_embeds"])

    model.talker.generate = capture
    try:
        tts.generate_voice_clone(
            text=row["text"],
            language=language,
            ref_audio=row["ref_audio"] if isinstance(row["ref_audio"], str) else row["ref_audio"][0],
            x_vector_only_mode=True,
            non_streaming_mode=True,
        )
        raise SystemExit("talker.generate was never called; cannot capture the inference prompt")
    except _Captured as c:
        infer_embeds = c.inputs_embeds[0]
    finally:
        model.talker.generate = original_generate

    # ---- 3) compare up to and including codec_bos ----
    n = infer_embeds.shape[0]
    print(f"inference prompt length: {n}; training prefix compared: positions 0..{n - 1}")
    train_prefix = train_embeds[:n].float()
    infer_prefix = infer_embeds.float().to(train_prefix.device)
    if train_prefix.shape != infer_prefix.shape:
        raise SystemExit(f"shape mismatch: training {tuple(train_prefix.shape)} vs inference {tuple(infer_prefix.shape)}")

    diff = (train_prefix - infer_prefix).abs().max(dim=-1).values  # per-position max abs difference
    scale = infer_prefix.abs().max().item()
    header_end = speaker_pos + 2
    segments = {
        "role tokens": (0, 3),
        "codec header (language prefix, speaker slot, pad)": (3, header_end),
        "text + eos": (header_end, n - 1),
        "final codec_bos": (n - 1, n),
    }
    print(f"(largest embedding value is about {scale:.2f}; bf16 noise is around 1e-2)")
    for name, (a, b) in segments.items():
        if b > a:
            print(f"  {name:52s} positions {a:3d}..{b - 1:3d}  max diff {diff[a:b].max().item():.4f}")
    bad = (diff > 0.05).nonzero().flatten().tolist()
    print("MATCH" if not bad else f"MISMATCH at positions {bad[:20]}{' ...' if len(bad) > 20 else ''}")


if __name__ == "__main__":
    main()
