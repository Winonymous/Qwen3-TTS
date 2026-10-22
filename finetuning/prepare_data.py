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
import hashlib
import json
import os
import random
from collections import defaultdict

import librosa
import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32
SAMPLE_RATE = 24000  # the training code only accepts 24 kHz reference audio


def resample_ref_audio(path, out_dir, cache):
    """Return a path to a 24 kHz mono wav of `path`.

    Files that are already 24 kHz mono are used as they are; others are resampled once into `out_dir`.
    """
    if path in cache:
        return cache[path]
    info = sf.info(path)
    if info.samplerate == SAMPLE_RATE and info.channels == 1:
        cache[path] = path
        return path
    os.makedirs(out_dir, exist_ok=True)
    digest = hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:8]
    out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(path))[0]}_{digest}.wav")
    if not os.path.exists(out_path):
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        sf.write(out_path, wav, SAMPLE_RATE)
    cache[path] = out_path
    return out_path


def audio_is_readable(path):
    """True if the file can be opened and has samples (checked with soundfile, then librosa for other formats)."""
    try:
        return sf.info(path).frames > 0
    except Exception:
        pass
    try:
        y, _ = librosa.load(path, sr=None, mono=True, duration=0.5)
        return len(y) > 0
    except Exception:
        return False


def split_rows(rows, test_ratio, seed):
    """Split rows into (train, test), keeping the original order inside each split.

    Rows are grouped by (speaker, language), and each group is split separately so that every speaker and
    language is represented in the test set. A group with a single row stays in train.
    """
    groups = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[(row.get('speaker'), row.get('language'))].append(idx)

    rng = random.Random(seed)
    test_idx = set()
    for key in sorted(groups, key=str):
        idxs = groups[key]
        if len(idxs) < 2:
            continue
        n_test = min(max(1, round(len(idxs) * test_ratio)), len(idxs) - 1)
        test_idx.update(rng.sample(idxs, n_test))

    train = [r for i, r in enumerate(rows) if i not in test_idx]
    test = [r for i, r in enumerate(rows) if i in test_idx]
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--resampled_dir", type=str, default=None,
                        help="Where 24 kHz copies of reference audio are written "
                             "(default: <output_jsonl dir>/ref_24k)")
    parser.add_argument("--test_ratio", type=float, default=0.05,
                        help="Fraction of rows held out as a test (validation) set; 0 writes everything to "
                             "--output_jsonl. Rows are split per (speaker, language) group.")
    parser.add_argument("--test_jsonl", type=str, default=None,
                        help="Where the test split is written (default: <output_jsonl>_test.jsonl)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the train/test split")
    args = parser.parse_args()
    if not 0.0 <= args.test_ratio < 1.0:
        parser.error("--test_ratio must be in [0, 1)")
    if args.test_jsonl is None:
        stem, ext = os.path.splitext(args.output_jsonl)
        args.test_jsonl = f"{stem}_test{ext or '.jsonl'}"
    if args.resampled_dir is None:
        args.resampled_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_jsonl)), "ref_24k")

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    total_lines = open(args.input_jsonl).readlines()
    total_lines = [json.loads(line.strip()) for line in total_lines]

    # Drop rows whose audio cannot be read: one corrupt file would otherwise abort the whole encoding run.
    readable = [audio_is_readable(line['audio']) for line in total_lines]
    bad_audio = [line['audio'] for line, ok in zip(total_lines, readable) if not ok]
    if bad_audio:
        bad_path = args.output_jsonl + ".bad_audio.txt"
        with open(bad_path, 'w') as f:
            f.write("\n".join(bad_audio) + "\n")
        print(f"Skipping {len(bad_audio)} row(s) with unreadable audio (listed in {bad_path}); "
              f"first: {bad_audio[0]}")
        total_lines = [line for line, ok in zip(total_lines, readable) if ok]

    # Reference audio must be 24 kHz for the mel extractor used in training. The target `audio` needs no
    # resampling here: the tokenizer resamples it to its own rate when encoding.
    ref_cache = {}
    for line in total_lines:
        ref = line['ref_audio']
        refs = [ref] if isinstance(ref, str) else ref
        refs = [resample_ref_audio(r, args.resampled_dir, ref_cache) for r in refs]
        line['ref_audio'] = refs[0] if isinstance(ref, str) else refs
    changed = sum(1 for k, v in ref_cache.items() if k != v)
    print(f"Reference audio: {len(ref_cache)} unique file(s), {changed} resampled to 24 kHz mono")

    def encode_batch(batch_lines):
        """Encode a batch; if it fails, retry file by file and skip only the files that fail."""
        try:
            enc_res = tokenizer_12hz.encode([line['audio'] for line in batch_lines])
            results = list(zip(enc_res.audio_codes, batch_lines))
        except Exception as e:
            print(f"Batch encode failed ({type(e).__name__}); retrying file by file")
            results = []
            for line in batch_lines:
                try:
                    results.append((tokenizer_12hz.encode([line['audio']]).audio_codes[0], line))
                except Exception as e2:
                    print(f"Skipping {line['audio']}: {type(e2).__name__}: {e2}")
        encoded = []
        for code, line in results:
            line['audio_codes'] = code.cpu().tolist()
            encoded.append(line)
        return encoded

    def encode_rows(rows):
        encoded = []
        batch_lines = []
        for line in rows:
            batch_lines.append(line)
            if len(batch_lines) >= BATCH_INFER_NUM:
                encoded.extend(encode_batch(batch_lines))
                batch_lines = []
        if batch_lines:
            encoded.extend(encode_batch(batch_lines))
        return encoded

    def write_jsonl(rows, path):
        with open(path, 'w') as f:
            for line in rows:
                f.writelines(json.dumps(line, ensure_ascii=False) + '\n')

    if args.test_ratio > 0:
        train_rows, test_rows = split_rows(total_lines, args.test_ratio, args.seed)
    else:
        train_rows, test_rows = total_lines, []
    print(f"Split: {len(train_rows)} train rows, {len(test_rows)} test rows")

    write_jsonl(encode_rows(train_rows), args.output_jsonl)
    print(f"Wrote {args.output_jsonl}")
    if test_rows:
        write_jsonl(encode_rows(test_rows), args.test_jsonl)
        print(f"Wrote {args.test_jsonl}")

if __name__ == "__main__":
    main()
