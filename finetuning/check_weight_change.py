# coding=utf-8
"""How much did fine-tuning actually change the weights?

Compares a checkpoint with the base model, tensor by tensor, and prints the fraction of elements that are not
identical. In pure bf16 training a small learning rate rounds most updates away, so a low fraction means most of
the model did not train.

Example:
    python check_weight_change.py --base Qwen/Qwen3-TTS-12Hz-1.7B-Base --checkpoint output/checkpoint-epoch-2
"""
import argparse
import glob
import os
from collections import defaultdict

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def tensor_index(model_dir):
    index = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                index[key] = path
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    base_dir = args.base if os.path.isdir(args.base) else snapshot_download(args.base)
    base, ckpt = tensor_index(base_dir), tensor_index(args.checkpoint)

    changed, total = defaultdict(int), defaultdict(int)
    max_abs_diff = defaultdict(float)
    for key, ckpt_file in ckpt.items():
        if key not in base:
            continue
        with safe_open(base[key], framework="pt") as fb, safe_open(ckpt_file, framework="pt") as fc:
            a, b = fb.get_tensor(key), fc.get_tensor(key)
        if a.shape != b.shape or not a.is_floating_point():
            continue
        a, b = a.float(), b.float()
        group = ".".join(key.split(".")[:3])  # e.g. talker.model.layers, talker.code_predictor.model
        changed[group] += int((a != b).sum())
        total[group] += a.numel()
        max_abs_diff[group] = max(max_abs_diff[group], float((a - b).abs().max()))

    print(f"{'module':45s} {'changed':>9s} {'max |diff|':>11s}")
    for group in sorted(total):
        print(f"{group:45s} {changed[group] / total[group]:9.2%} {max_abs_diff[group]:11.2e}")
    print(f"{'ALL':45s} {sum(changed.values()) / sum(total.values()):9.2%}")


if __name__ == "__main__":
    main()
