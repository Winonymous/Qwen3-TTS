# coding=utf-8
"""Convert a Hugging Face audio dataset into the raw JSONL that `prepare_data.py` expects, without writing any wav files.

Each output line has: {"audio": ..., "text": ..., "ref_audio": ..., "speaker": ..., "language": ...}

The audio is never decoded or re-encoded here (`Audio(decode=False)`). For every clip, `audio` / `ref_audio` is:
  - the path of the file the dataset already has on disk (the HF cache or a local audio folder), when there is one;
  - otherwise (e.g. audio stored as bytes inside parquet) a `data:audio;base64,...` URL of the original bytes.
`prepare_data.py` and `dataset.py` read both forms and resample in memory, so no new audio files are created.
Base64 URLs make the JSONL large, so prefer a dataset whose files live on disk when you have the choice.

Example:
    python hf_to_jsonl.py --dataset my-org/yoruba-tts --split train \
        --audio_column audio --text_column sentence --speaker_column speaker --language yoruba \
        --output_jsonl train_raw.jsonl
    python prepare_data.py --input_jsonl train_raw.jsonl --output_jsonl train_with_codes.jsonl
"""
import argparse
import base64
import io
import json
import os
import random
from collections import Counter, defaultdict

import soundfile as sf
from datasets import Audio, load_dataset

from logging_utils import setup_logging


def clip_source(audio):
    """Return a path (if the file exists on disk) or a base64 data URL for a `decode=False` audio cell."""
    path, data = audio.get("path"), audio.get("bytes")
    if path and os.path.isfile(path):
        return path
    if data:
        return "data:audio;base64," + base64.b64encode(data).decode("ascii")
    return None


def clip_duration(audio):
    """(duration in seconds, None) from the file header only (nothing is decoded), or (None, why) if unreadable."""
    if not isinstance(audio, dict):
        return None, f"audio cell is {type(audio).__name__}, not a dict (None = missing audio)"
    path, data = audio.get("path"), audio.get("bytes")
    if not (path and os.path.isfile(path)) and not data:
        return None, f"no file on disk (path={path!r}) and no bytes"
    try:
        info = sf.info(path if path and os.path.isfile(path) else io.BytesIO(data))
        return info.frames / info.samplerate, None
    except Exception as e:
        return None, f"soundfile cannot read it: {type(e).__name__}: {e} (path={path!r}, bytes={len(data) if data else 0})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Hub dataset id or a local dataset path")
    parser.add_argument("--config", type=str, default=None, help="Dataset config / subset name")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--audio_column", type=str, default="audio")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--speaker_column", type=str, default=None,
                        help="Column with the speaker name for each row")
    parser.add_argument("--speaker", type=str, default=None,
                        help="Speaker name for every row, or only for rows with an empty --speaker_column cell")
    parser.add_argument("--language_column", type=str, default=None,
                        help="Column with the language name for each row")
    parser.add_argument("--language", type=str, default=None,
                        help="Language for every row, or only for rows with an empty --language_column cell "
                             "(default: none = Auto). Use the same value as `language=` at inference.")
    parser.add_argument("--min_seconds", type=float, default=None, help="Skip clips shorter than this")
    parser.add_argument("--max_seconds", type=float, default=None,
                        help="Skip clips longer than this (very long clips are what runs a small GPU out of memory)")
    parser.add_argument("--max_rows", type=int, default=None, help="Only use the first N rows (for a quick test)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Random subset of this many usable rows (rows that are skipped are replaced by other "
                             "random rows until the amount is reached)")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default=None,
                        help="Everything printed is also written here (default: <output_jsonl>.log)")
    args = parser.parse_args()
    setup_logging(args.log_file or args.output_jsonl + ".log", args)

    ds = load_dataset(args.dataset, args.config, split=args.split, cache_dir=args.cache_dir)
    print(f"Loaded {args.dataset!r} config={args.config!r} split={args.split!r}: {len(ds)} rows")
    print(f"Columns: {ds.column_names}")
    print(f"Features: {ds.features}")
    missing = [c for c in (args.audio_column, args.text_column, args.speaker_column, args.language_column)
               if c is not None and c not in ds.column_names]
    if missing:
        parser.error(f"column(s) {missing} not found (columns: {ds.column_names})")
    ds = ds.cast_column(args.audio_column, Audio(decode=False))  # keep the original files / bytes untouched
    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    if len(ds):
        first = ds[0]
        audio = first[args.audio_column]
        audio_info = ({"path": audio.get("path"), "bytes": len(audio["bytes"]) if audio.get("bytes") else None}
                      if isinstance(audio, dict) else audio)
        print(f"First row: audio={audio_info}, text={first[args.text_column]!r}")

    # Pass 1: header-only scan -> which rows are usable, and each speaker's clip durations.
    rows, skipped = [], Counter()
    skip_examples = defaultdict(list)  # reason -> first few (row index, detail)

    def skip(reason, i, detail=""):
        skipped[reason] += 1
        if len(skip_examples[reason]) < 5:
            skip_examples[reason].append((i, detail))

    # With --num_samples the rows are visited in random order and the scan stops as soon as that many usable rows
    # are found, so skipped rows are replaced by other random ones instead of shrinking the sample.
    order = list(range(len(ds)))
    if args.num_samples:
        random.Random(args.sample_seed).shuffle(order)
    durations, scanned = [], 0
    for i in order:
        if args.num_samples and len(rows) >= args.num_samples:
            break
        scanned += 1
        row = ds[i]
        text = row[args.text_column]
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            skip("empty text", i, f"cell={row[args.text_column]!r}")
            continue
        duration, why = clip_duration(row[args.audio_column])
        if duration is None:
            skip("unreadable audio", i, why)
            continue
        durations.append(duration)
        if args.min_seconds and duration < args.min_seconds:
            skip(f"shorter than --min_seconds {args.min_seconds}", i, f"{duration:.2f}s")
            continue
        if args.max_seconds and duration > args.max_seconds:
            skip(f"longer than --max_seconds {args.max_seconds}", i, f"{duration:.2f}s")
            continue
        speaker = (str(row[args.speaker_column] or "").strip() if args.speaker_column else "") or args.speaker
        language = (str(row[args.language_column] or "").strip() if args.language_column else "") or args.language
        rows.append({"idx": i, "text": text, "speaker": speaker, "language": language, "duration": duration})
        if scanned % 5000 == 0:
            print(f"  scanned {scanned}/{len(ds)} rows: {len(rows)} usable, {sum(skipped.values())} skipped", flush=True)

    print(f"Scan: {len(rows)} usable, {sum(skipped.values())} skipped of {scanned} rows looked at ({len(ds)} in the dataset)")
    if args.num_samples and len(rows) < args.num_samples:
        print(f"Warning: only {len(rows)} usable rows exist, fewer than --num_samples {args.num_samples}")
    if durations:
        durations.sort()
        print(f"Readable clip durations: min {durations[0]:.2f}s, median {durations[len(durations) // 2]:.2f}s, "
              f"max {durations[-1]:.2f}s ({len(durations)} clips)")
    for reason, count in skipped.most_common():
        print(f"  skipped {count} rows: {reason}")
        for i, detail in skip_examples[reason]:
            print(f"      e.g. row {i}: {detail}")
    if not rows:
        raise SystemExit("No usable rows. See the skip reasons above (wrong --text_column / --audio_column, audio "
                         "files not found on this machine, or --min_seconds / --max_seconds too strict?).")

    # One fixed reference clip per speaker: the median-length one (a typical clip, not an outlier).
    by_speaker = defaultdict(list)
    for r in rows:
        by_speaker[r["speaker"] or ""].append(r)
    ref_idx = {}
    for spk, group in by_speaker.items():
        group.sort(key=lambda r: r["duration"])
        ref = group[(len(group) - 1) // 2]
        ref_idx[spk] = ref["idx"]
        print(f"Reference for speaker {spk or '(none)'!r}: row {ref['idx']} ({ref['duration']:.2f}s, "
              f"median of {len(group)} clips)")

    # Pass 2: write the rows; each reference is read once and reused.
    keep = {r["idx"]: r for r in rows}
    ref_rows = set(ref_idx.values())
    ref_src = {i: clip_source(ds[i][args.audio_column]) for i in sorted(ref_rows)}
    n, no_source = 0, 0
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for i in sorted(keep):
            r = keep[i]
            audio = clip_source(ds[i][args.audio_column])
            if audio is None:
                no_source += 1
                continue
            item = {"audio": audio, "text": r["text"], "ref_audio": ref_src[ref_idx[r["speaker"] or ""]]}
            if r["speaker"]:
                item["speaker"] = r["speaker"]
            if r["language"]:
                item["language"] = r["language"]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {args.output_jsonl} (skipped {sum(skipped.values()) + no_source})")


if __name__ == "__main__":
    main()
