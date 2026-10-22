# coding=utf-8
"""Check which targets the Base model's own predictions line up with.

At lr=0 the fine-tuning loss of the Base model sat near 11, worse than a uniform guess (about 10.3). The prompt itself
matches inference (see check_prompt_match.py), so the suspect is the targets. This script measures, on real
training rows and without touching any weights:

  A) Talker: cross-entropy and top-1 accuracy of the logits at position p against the codec-0 token that sits at
     position p+k, for k = 0..3. A correct next-token setup should be best at k = 1.
     (The loss in sft_12hz.py passes labels[:, 1:] to a Hugging Face loss that shifts once more, which would
     make it behave like k = 2.)
  B) Sub-talker: for each choice of which hidden state is paired with a frame, the cross-entropy of group i+1
     against the logits at index i ("aligned") and against the next group ("shifted").

Usage (from finetuning/):
    python check_loss_alignment.py --train_jsonl train_with_codes.jsonl --num_rows 8
"""
import argparse
import json

import torch
import torch.nn.functional as F
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from transformers import AutoConfig


def ce_acc(logits, targets):
    valid = targets != -100
    if valid.sum() == 0:
        return float("nan"), float("nan"), 0
    lg = logits[valid].float()
    tg = targets[valid]
    return F.cross_entropy(lg, tg).item(), (lg.argmax(-1) == tg).float().mean().item(), int(valid.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--num_rows", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn", type=str, default="sdpa")
    args = parser.parse_args()

    tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path, device_map=args.device, dtype=torch.bfloat16, attn_implementation=args.attn,
    )
    model = tts.model
    config = AutoConfig.from_pretrained(args.init_model_path)

    with open(args.train_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][: args.num_rows]
    dataset = TTSDataset(rows, tts.processor, config)
    batch = dataset.collate_fn([dataset[i] for i in range(len(rows))])
    batch = {k: v.to(model.device) for k, v in batch.items()}

    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    codec_mask = batch["codec_mask"]
    labels = batch["codec_0_labels"]  # label[q] == the codec-0 token that is the input at position q
    speaker_pos = batch["speaker_pos"] if "speaker_pos" in batch else torch.full((input_ids.shape[0],), 6, device=model.device)
    B, T = labels.shape

    with torch.no_grad():
        spk = model.speaker_encoder(batch["ref_mels"].to(model.dtype))
        text = model.talker.text_projection(model.talker.model.text_embedding(input_ids[:, :, 0])) * batch["text_embedding_mask"]
        codec = model.talker.model.codec_embedding(input_ids[:, :, 1]) * batch["codec_embedding_mask"]
        codec[torch.arange(B, device=model.device), speaker_pos] = spk
        embeds = text + codec
        for i in range(1, 16):
            emb_i = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            embeds = embeds + emb_i * codec_mask.unsqueeze(-1)

        out = model.talker(inputs_embeds=embeds, attention_mask=batch["attention_mask"], output_hidden_states=True)
        logits = out.logits            # (B, T, V): logits[:, p] is computed from the input at position p
        hidden = out.hidden_states[0][-1]

        print(f"uniform guess over the talker vocab: ln({logits.shape[-1]}) = {torch.log(torch.tensor(float(logits.shape[-1]))):.2f}")
        print("\nA) talker: logits at position p vs the codec-0 token at position p+k")
        print("   k   CE     top-1 acc   n")
        best = None
        for k in range(0, 4):
            ce, acc, n = ce_acc(logits[:, : T - k].reshape(-1, logits.shape[-1]), labels[:, k:].reshape(-1))
            print(f"   {k}   {ce:6.3f}  {acc:8.3f}   {n}")
            if best is None or ce < best[1]:
                best = (k, ce)
        print(f"   -> best alignment: k = {best[0]}  (k = 1 is the correct next-token setup;"
              f" sft_12hz.py's labels[:, 1:] plus the HF shift behaves like k = 2)")

        print("\nB) sub-talker: which hidden state belongs to a frame")
        frames = codec_ids[codec_mask]  # (N, 16) frames in order
        for name, sel in (("codec_mask[:, 1:]  (hidden of the position BEFORE the frame)", codec_mask[:, 1:]),
                          ("codec_mask[:, :-1] (hidden of the frame's own position)    ", codec_mask[:, :-1])):
            hs = hidden[:, :-1][sel]
            if hs.shape[0] != frames.shape[0]:
                print(f"   {name}: count mismatch ({hs.shape[0]} vs {frames.shape[0]}), skipped")
                continue
            sub_logits, sub_loss = model.talker.forward_sub_talker_finetune(frames, hs)  # (N, 15, V)
            aligned, acc_a, _ = ce_acc(sub_logits.reshape(-1, sub_logits.shape[-1]), frames[:, 1:].reshape(-1))
            shifted, acc_s, _ = ce_acc(sub_logits[:, :-1].reshape(-1, sub_logits.shape[-1]), frames[:, 2:].reshape(-1))
            print(f"   {name}: aligned CE {aligned:6.3f} (acc {acc_a:.3f}) | shifted CE {shifted:6.3f} | "
                  f"loss reported by the model {sub_loss.item():6.3f}")
        print("   -> the hidden-state choice with the lowest 'aligned' CE is the right pairing")


if __name__ == "__main__":
    main()
