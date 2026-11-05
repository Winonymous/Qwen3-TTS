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
import base64
import io
import json
import multiprocessing as mp
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import librosa
import soundfile as sf
from audio_utils import normalize_loudness
from logging_utils import setup_logging
from qwen_tts import Qwen3TTSTokenizer
from text_utils import normalize_text

BATCH_INFER_NUM = 32


def short(audio):
    """Printable form of an audio field: a path as it is, a base64 data URL cut short."""
    return audio if len(audio) <= 200 else audio[:60] + "...(truncated)"


def audio_is_readable(path):
    """True if the file can be opened and has samples (checked with soundfile, then librosa for other formats).

    `path` may also be a `data:audio;base64,...` URL (see hf_to_jsonl.py).
    """
    if path.startswith("data:audio"):
        try:
            return sf.info(io.BytesIO(base64.b64decode(path.split(",", 1)[1]))).frames > 0
        except Exception:
            return False
    try:
        return sf.info(path).frames > 0
    except Exception:
        pass
    try:
        y, _ = librosa.load(path, sr=None, mono=True, duration=0.5)
        return len(y) > 0
    except Exception:
        return False


def load_trimmed(path, top_db, keep_ms, sr=24000):
    """Load `path` (or a data: URL) as mono float32 at `sr` and cut the leading/trailing silence.

    Everything quieter than `top_db` below the peak counts as silence; `keep_ms` of it is kept on each side so the
    speech does not start or stop abruptly. Audio that is silent throughout is returned untouched.
    """
    if path.startswith("data:audio"):
        path = io.BytesIO(base64.b64decode(path.split(",", 1)[1]))
    y, _ = librosa.load(path, sr=sr, mono=True)
    _, (start, end) = librosa.effects.trim(y, top_db=top_db)
    if end <= start:
        return y
    keep = int(sr * keep_ms / 1000)
    return y[max(0, start - keep):min(len(y), end + keep)]


def load_audio(path, args, sr=24000):
    """Target audio as a mono waveform at `sr`, with the optional silence trimming and loudness normalization."""
    if args.trim_silence:
        y = load_trimmed(path, args.trim_top_db, args.keep_silence_ms, sr)
    else:
        if path.startswith("data:audio"):
            path = io.BytesIO(base64.b64decode(path.split(",", 1)[1]))
        y, _ = librosa.load(path, sr=sr, mono=True)
    if args.normalize_loudness:
        y = normalize_loudness(y, sr, args.target_db)  # after trimming, so the pauses do not count
    return y


def audio_duration(path):
    """Duration in seconds, or None if it cannot be determined cheaply (`path` may be a data: URL)."""
    try:
        if path.startswith("data:audio"):
            info = sf.info(io.BytesIO(base64.b64decode(path.split(",", 1)[1])))
        else:
            info = sf.info(path)
        return info.frames / info.samplerate
    except Exception:
        return None


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


def encode_audios(tokenizer, audios, args):
    if args.trim_silence or args.normalize_loudness:
        waves = [load_audio(a, args) for a in audios]
        return tokenizer.encode(waves, sr=24000)
    return tokenizer.encode(audios)


def encode_batch(tokenizer, batch_lines, args, tag):
    """Encode a batch; if it fails, retry file by file and skip only the files that fail."""
    try:
        enc_res = encode_audios(tokenizer, [line['audio'] for line in batch_lines], args)
        results = list(zip(enc_res.audio_codes, batch_lines))
    except Exception as e:
        print(f"{tag} Batch encode failed ({type(e).__name__}); retrying file by file", flush=True)
        results = []
        for line in batch_lines:
            try:
                results.append((encode_audios(tokenizer, [line['audio']], args).audio_codes[0], line))
            except Exception as e2:
                print(f"{tag} Skipping {short(line['audio'])}: {type(e2).__name__}: {e2}", flush=True)
    encoded = []
    for code, line in results:
        line['audio_codes'] = code.cpu().tolist()
        del line['audio']  # training only needs the codes; keeps the output small
        encoded.append(line)
    return encoded


def encode_worker(rank, device, rows, part_path, args):
    """Load the tokenizer on `device`, encode `rows` (a list, or the path of a shard JSONL) and write `part_path`."""
    tag = f"[worker {rank} / {device}]"
    if isinstance(rows, str):
        with open(rows, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
    tokenizer = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_model_path, device_map=device)
    print(f"{tag} tokenizer loaded, {len(rows)} rows to encode", flush=True)
    start = time.time()
    with open(part_path, 'w', encoding='utf-8') as out:
        for number, first in enumerate(range(0, len(rows), args.batch_size)):
            for line in encode_batch(tokenizer, rows[first:first + args.batch_size], args, tag):
                out.write(json.dumps(line, ensure_ascii=False) + '\n')
            out.flush()
            done = min(first + args.batch_size, len(rows))
            if number % 10 == 0 or done == len(rows):
                print(f"{tag} {done}/{len(rows)} rows ({time.time() - start:.0f}s)", flush=True)


def write_jsonl(rows, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in rows:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None,
                        help="Encode on this single device (e.g. cuda:1 or cpu). Default: use --gpus")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids to encode on, one worker per GPU (default: all visible GPUs)")
    parser.add_argument("--procs_per_gpu", type=int, default=1,
                        help="Workers (tokenizer copies) per GPU; 2-3 can raise throughput when batches are small "
                             "or the CPU-side loading (--trim_silence / --normalize_loudness) is the bottleneck")
    parser.add_argument("--batch_size", type=int, default=BATCH_INFER_NUM, help="Clips per encode call, per worker")
    parser.add_argument("--num_threads", type=int, default=min(8, os.cpu_count() or 1),
                        help="Threads for the readability and duration checks that run before encoding")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--test_ratio", type=float, default=0.05,
                        help="Fraction of rows held out as a test (validation) set; 0 writes everything to "
                             "--output_jsonl. Rows are split per (speaker, language) group.")
    parser.add_argument("--test_jsonl", type=str, default=None,
                        help="Where the test split is written (default: <output_jsonl>_test.jsonl)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the train/test split")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Smoke test: encode only a random subset of this many rows (0 = all). It is taken "
                             "after the filters and before the train/test split, so --test_ratio applies to the "
                             "subset. Skips the slow encoding of the rest.")
    parser.add_argument("--no_normalize_text", dest="normalize_text", action="store_false",
                        help="Keep the text exactly as it is. By default it is normalized (Unicode NFC, invisible "
                             "characters removed, whitespace collapsed) so that the same word is always spelled "
                             "the same way. Apply the same normalization at inference "
                             "(text_utils.normalize_text).")
    parser.add_argument("--trim_silence", action="store_true",
                        help="Cut the leading and trailing silence of every target clip before encoding. The model "
                             "learns the silence it is trained on, so clips with long pauses at the edges give "
                             "generated speech with long pauses before and after.")
    parser.add_argument("--trim_top_db", type=float, default=35.0,
                        help="With --trim_silence: sound this many dB below the clip's peak counts as silence "
                             "(lower = trims more)")
    parser.add_argument("--keep_silence_ms", type=int, default=100,
                        help="With --trim_silence: silence kept before and after the speech, in milliseconds")
    parser.add_argument("--normalize_loudness", action="store_true",
                        help="Bring every target clip to the same speech level (--target_db) before encoding. "
                             "The model learns the loudness it is trained on, so quiet and loud clips in the "
                             "data give quiet and loud generated speech.")
    parser.add_argument("--target_db", type=float, default=-23.0,
                        help="With --normalize_loudness: speech level in dBFS (RMS of the active speech). "
                             "-23 is a comfortable level; the gain is limited so that clips do not clip")
    parser.add_argument("--max_text_chars", type=int, default=0,
                        help="Drop rows whose text has more characters than this (0 = no limit). Catches runaway "
                             "transcripts, e.g. a repeated ASR hallucination, which can use all the GPU memory.")
    parser.add_argument("--max_duration", type=float, default=0.0,
                        help="Drop rows whose target audio is longer than this many seconds (0 = no limit). "
                             "Very long clips use a lot of memory in training and tend to hurt it.")
    parser.add_argument("--min_duration", type=float, default=0.0,
                        help="Drop rows whose target audio is shorter than this many seconds (0 = no limit)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Everything printed is also written here (default: <output_jsonl>.log). Worker "
                             "progress lines go to the console only.")
    args = parser.parse_args()
    setup_logging(args.log_file or args.output_jsonl + ".log", args)
    if not 0.0 <= args.test_ratio < 1.0:
        parser.error("--test_ratio must be in [0, 1)")
    if args.device and args.gpus:
        parser.error("use either --device or --gpus, not both")
    if args.test_jsonl is None:
        stem, ext = os.path.splitext(args.output_jsonl)
        args.test_jsonl = f"{stem}_test{ext or '.jsonl'}"

    if args.device:
        devices = [args.device]
    else:
        import torch

        gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))
        if not gpus:
            raise SystemExit("No GPU available; pass --device cpu to encode on the CPU")
        devices = [f"cuda:{g}" for g in gpus for _ in range(args.procs_per_gpu)]
    print(f"Encoding workers: {devices}")

    with open(args.input_jsonl, encoding='utf-8') as f:
        total_lines = [json.loads(line) for line in f if line.strip()]

    # Drop rows whose audio cannot be read: one corrupt file would otherwise abort the whole encoding run.
    with ThreadPoolExecutor(args.num_threads) as pool:
        readable = list(pool.map(audio_is_readable, [line['audio'] for line in total_lines]))
    bad_audio = [short(line['audio']) for line, ok in zip(total_lines, readable) if not ok]
    if bad_audio:
        bad_path = args.output_jsonl + ".bad_audio.txt"
        with open(bad_path, 'w') as f:
            f.write("\n".join(bad_audio) + "\n")
        print(f"Skipping {len(bad_audio)} row(s) with unreadable audio (listed in {bad_path}); "
              f"first: {bad_audio[0]}")
        total_lines = [line for line, ok in zip(total_lines, readable) if ok]

    if args.normalize_text:
        changed = 0
        for line in total_lines:
            clean = normalize_text(line['text'])
            changed += clean != line['text']
            line['text'] = clean
        print(f"Text normalization changed {changed} of {len(total_lines)} row(s)")

    if args.max_text_chars > 0:
        long_rows =[line for line in total_lines if len(line['text']) > args.max_text_chars]
        if long_rows:
            long_path = args.output_jsonl + ".long_text.txt"
            with open(long_path, 'w', encoding='utf-8') as f:
                for line in long_rows:
                    f.write(f"{len(line['text'])}\t{short(line['audio'])}\t{line['text'][:100]!r}\n")
            print(f"Dropping {len(long_rows)} row(s) with more than {args.max_text_chars} characters of text "
                  f"(listed in {long_path}); longest: {max(len(r['text']) for r in long_rows)} chars")
            total_lines = [line for line in total_lines if len(line['text']) <= args.max_text_chars]

    if args.max_duration > 0 or args.min_duration > 0:
        with ThreadPoolExecutor(args.num_threads) as pool:
            durations = list(pool.map(audio_duration, [line['audio'] for line in total_lines]))
        kept = []
        for line, d in zip(total_lines, durations):
            too_long = args.max_duration > 0 and d is not None and d > args.max_duration
            too_short = args.min_duration > 0 and d is not None and d < args.min_duration
            if not (too_long or too_short):
                kept.append(line)
        print(f"Duration filter (min {args.min_duration}s, max {args.max_duration}s): kept {len(kept)} of "
              f"{len(total_lines)} rows, dropped {len(total_lines) - len(kept)}")
        total_lines = kept

    if args.num_samples > 0 and len(total_lines) > args.num_samples:
        total_lines = random.Random(args.seed).sample(total_lines, args.num_samples)
        print(f"--num_samples: using a random subset of {len(total_lines)} rows")

    # No audio is resampled or written here: the tokenizer resamples `audio` in memory when encoding, and
    # dataset.py resamples `ref_audio` to 24 kHz in memory when it is loaded for training.

    if args.test_ratio > 0:
        train_rows, test_rows = split_rows(total_lines, args.test_ratio, args.seed)
    else:
        train_rows, test_rows = total_lines, []
    print(f"Split: {len(train_rows)} train rows, {len(test_rows)} test rows")

    # Train and test rows are encoded together (the tokenizer is loaded once per worker) and told apart again by
    # `_split`; `_i` is the row's position, which restores the original order after the workers' parts are merged.
    for name, rows in (("train", train_rows), ("test", test_rows)):
        for i, row in enumerate(rows):
            row['_split'], row['_i'] = name, i
    todo = train_rows + test_rows
    if not todo:
        raise SystemExit("No rows left to encode after the filters")

    n_workers = min(len(devices), len(todo))
    shards = [todo[k::n_workers] for k in range(n_workers)]  # interleaved: every worker gets the same mix of clips
    part_paths = [f"{args.output_jsonl}.part{rank}.jsonl" for rank in range(n_workers)]
    shard_paths = []
    crashed = []
    if n_workers == 1:
        encode_worker(0, devices[0], shards[0], part_paths[0], args)
    else:
        for rank, shard in enumerate(shards):  # each worker reads its own file instead of receiving a huge pickle
            path = f"{args.output_jsonl}.shard{rank}.jsonl"
            write_jsonl(shard, path)
            shard_paths.append(path)
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=encode_worker, args=(rank, devices[rank], shard_paths[rank], part_paths[rank], args))
                 for rank in range(n_workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        crashed = [rank for rank, p in enumerate(procs) if p.exitcode != 0]
        if crashed:
            print(f"Worker(s) {crashed} exited with an error; the rows they had not finished are missing. "
                  f"Their part files are kept: {[part_paths[r] for r in crashed]}")

    encoded = {"train": [], "test": []}
    for path in part_paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                encoded[row.pop('_split')].append(row)
    for name, path, expected in (("train", args.output_jsonl, len(train_rows)),
                                 ("test", args.test_jsonl, len(test_rows))):
        rows = sorted(encoded[name], key=lambda r: r['_i'])
        for row in rows:
            del row['_i']
        if not rows and not expected:
            continue
        write_jsonl(rows, path)
        print(f"Wrote {path}: {len(rows)} of {expected} rows"
              + ("" if len(rows) == expected else f" ({expected - len(rows)} failed to encode)"))
    for path in shard_paths + [p for rank, p in enumerate(part_paths) if rank not in crashed]:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
