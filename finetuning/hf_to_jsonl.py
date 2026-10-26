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
        --audio_column audio --text_column sentence --speaker_column voice_over --language yoruba \
        --output_jsonl train_raw.jsonl
    python prepare_data.py --input_jsonl train_raw.jsonl --output_jsonl train_with_codes.jsonl
"""
import argparse
import base64
import io
import json
import os
from collections import defaultdict

import soundfile as sf
from datasets import Audio, load_dataset


def clip_source(audio):
    """Return a path (if the file exists on disk) or a base64 data URL for a `decode=False` audio cell."""
    path, data = audio.get("path"), audio.get("bytes")
    if path and os.path.isfile(path):
        return path
    if data:
        return "data:audio;base64," + base64.b64encode(data).decode("ascii")
    return None


def clip_duration(audio):
    """Duration in seconds from the file header only (nothing is decoded), or None if unreadable."""
    path, data = audio.get("path"), audio.get("bytes")
    try:
        info = sf.info(path if path and os.path.isfile(path) else io.BytesIO(data))
        return info.frames / info.samplerate
    except Exception:
        return None


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
    args = parser.parse_args()

    ds = load_dataset(args.dataset, args.config, split=args.split, cache_dir=args.cache_dir)
    missing = [c for c in (args.audio_column, args.text_column, args.speaker_column, args.language_column)
               if c is not None and c not in ds.column_names]
    if missing:
        parser.error(f"column(s) {missing} not found (columns: {ds.column_names})")
    ds = ds.cast_column(args.audio_column, Audio(decode=False))  # keep the original files / bytes untouched
    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    # Pass 1: header-only scan -> which rows are usable, and each speaker's clip durations.
    rows, skipped = [], 0
    for i, row in enumerate(ds):
        text = (row[args.text_column] or "").strip()
        duration = clip_duration(row[args.audio_column]) if text else None
        if (duration is None
                or (args.min_seconds and duration < args.min_seconds)
                or (args.max_seconds and duration > args.max_seconds)):
            skipped += 1
            continue
        speaker = (str(row[args.speaker_column] or "").strip() if args.speaker_column else "") or args.speaker
        language = (str(row[args.language_column] or "").strip() if args.language_column else "") or args.language
        rows.append({"idx": i, "text": text, "speaker": speaker, "language": language, "duration": duration})

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
    ref_src = {}
    for i, row in enumerate(ds):
        if i in ref_rows:
            ref_src[i] = clip_source(row[args.audio_column])
    n = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            r = keep.get(i)
            audio = clip_source(row[args.audio_column]) if r else None
            if r is None or audio is None:
                continue
            item = {"audio": audio, "text": r["text"], "ref_audio": ref_src[ref_idx[r["speaker"] or ""]]}
            if r["speaker"]:
                item["speaker"] = r["speaker"]
            if r["language"]:
                item["language"] = r["language"]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {args.output_jsonl} (skipped {skipped})")


if __name__ == "__main__":
    main()
