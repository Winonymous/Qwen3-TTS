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
    parser.add_argument("--duration_column", type=str, default=None,
                        help="Column with each clip's duration in seconds. The --min_seconds / --max_seconds filter "
                             "is then applied to it first (fast, no audio is opened) and --num_samples draws only "
                             "from the rows that pass")
    parser.add_argument("--min_seconds", type=float, default=None, help="Skip clips shorter than this")
    parser.add_argument("--max_seconds", type=float, default=None,
                        help="Skip clips longer than this (very long clips are what runs a small GPU out of memory)")
    parser.add_argument("--max_rows", type=int, default=None, help="Only use the first N rows that pass the filter (for a quick test; "
                             "the first rows of a dataset are not random, prefer --num_samples)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Random subset of this many usable rows (rows that are skipped are replaced by other "
                             "random rows until the amount is reached)")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=None,
                        help="Processes for the filter (and the header scan) (default: 1)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Everything printed is also written here (default: <output_jsonl>.log)")
    args = parser.parse_args()
    setup_logging(args.log_file or args.output_jsonl + ".log", args)

    ds = load_dataset(args.dataset, args.config, split=args.split, cache_dir=args.cache_dir)
    print(f"Loaded {args.dataset!r} config={args.config!r} split={args.split!r}: {len(ds)} rows")
    print(f"Columns: {ds.column_names}")
    print(f"Features: {ds.features}")
    missing = [c for c in (args.audio_column, args.text_column, args.speaker_column, args.language_column,
                           args.duration_column)
               if c is not None and c not in ds.column_names]
    if missing:
        parser.error(f"column(s) {missing} not found (columns: {ds.column_names})")
    ds = ds.cast_column(args.audio_column, Audio(decode=False))  # keep the original files / bytes untouched
    if len(ds):  # text only: reading the audio cell would load the whole first clip into memory
        first_text = ds[args.text_column][0]
        print(f"First row text ({len(first_text or '')} characters): {(first_text or '')[:200]!r}")

    skipped = Counter()
    skip_examples = defaultdict(list)  # reason -> first few (row index, detail)

    # Duration of every clip: the column if there is one, otherwise read from each file's header (into a small
    # extra column; the audio itself is never copied).
    if args.duration_column:
        duration_col = args.duration_column
    else:
        duration_col = "_duration"
        header_ds = ds.select_columns([args.audio_column]).map(
            lambda audio: {duration_col: clip_duration(audio)[0]}, input_columns=[args.audio_column],
            remove_columns=[args.audio_column], num_proc=args.num_proc, desc="Reading durations from the audio headers")
        ds = ds.add_column(duration_col, header_ds[duration_col])

    def duration_reason(duration):
        """Why a clip is dropped on its duration alone, or None if it passes."""
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = float("nan")
        if duration != duration or duration <= 0:
            return "duration column is not a number" if args.duration_column else "unreadable audio"
        if args.min_seconds and duration < args.min_seconds:
            return f"shorter than --min_seconds {args.min_seconds}"
        if args.max_seconds and duration > args.max_seconds:
            return f"longer than --max_seconds {args.max_seconds}"
        return None

    def has_text(text):
        return isinstance(text, str) and bool(text.strip())

    # Stage 1: FILTER every row with Dataset.filter, cheapest column first. The duration filter reads only the small
    # duration column; the text column (transcripts can be thousands of characters) is only read for the rows that
    # survive it, and the audio is never loaded. Nothing is sampled yet.
    print(f"Filtering {len(ds)} rows on {duration_col!r} ...", flush=True)
    all_durations = ds[duration_col]
    for i, d in enumerate(all_durations):  # skip summary: which rule drops how many rows
        why = duration_reason(d)
        if why:
            skipped[why] += 1
            if len(skip_examples[why]) < 5:
                skip_examples[why].append((i, f"duration={d!r}"))
    short_enough = ds.filter(lambda d: duration_reason(d) is None, input_columns=[duration_col],
                             num_proc=args.num_proc, desc="Filtering on duration")
    print(f"Duration filter: {len(short_enough)} of {len(ds)} rows pass; checking the text of those ...", flush=True)
    filtered = short_enough.filter(has_text, input_columns=[args.text_column], num_proc=args.num_proc,
                                   desc="Filtering empty text")
    if len(short_enough) != len(filtered):
        skipped["empty text"] += len(short_enough) - len(filtered)
    print(f"Filter: {len(filtered)} of {len(ds)} rows pass, {sum(skipped.values())} skipped")
    good = sorted(d for d in all_durations if isinstance(d, (int, float)) and d == d and d > 0)
    if good:
        print(f"Clip durations (all rows): min {good[0]:.2f}s, median {good[len(good) // 2]:.2f}s, "
              f"max {good[-1]:.2f}s ({len(good)} clips)")
    for why, count in skipped.most_common():
        print(f"  skipped {count} rows: {why}")
        for i, detail in skip_examples[why]:
            print(f"      e.g. row {i}: {detail}")
    if not len(filtered):
        raise SystemExit("No usable rows (wrong --text_column / --audio_column / --duration_column, audio files not "
                         "found on this machine, or --min_seconds / --max_seconds too strict?).")
    if args.max_rows:  # applied to the rows that passed the filter, not to the raw dataset
        filtered = filtered.select(range(min(args.max_rows, len(filtered))))

    # Stage 2: SAMPLE from the rows that passed, with Dataset.shuffle(seed). A drawn row whose audio turns out to be
    # missing is replaced by the next one in the shuffled order, so the sample still reaches --num_samples.
    if args.num_samples:
        filtered = filtered.shuffle(seed=args.sample_seed)
    audio_ds = filtered.select_columns([args.audio_column])

    def get_audio(pos):
        return audio_ds[pos][args.audio_column]

    def has_audio(audio):
        return isinstance(audio, dict) and bool((audio.get("path") and os.path.isfile(audio["path"]))
                                                or audio.get("bytes"))

    wanted = args.num_samples or len(filtered)
    positions = []
    if args.num_samples and args.duration_column:  # without a duration column the header scan proved the audio
        for pos in range(len(filtered)):
            if len(positions) >= wanted:
                break
            if has_audio(get_audio(pos)):
                positions.append(pos)
            else:
                skipped["no audio file or bytes"] += 1
    else:
        positions = list(range(min(wanted, len(filtered))))
    if args.num_samples and len(positions) < args.num_samples:
        print(f"Warning: only {len(positions)} usable rows exist, fewer than --num_samples {args.num_samples}")
    print(f"Selected {len(positions)} rows" + (f" (random sample, seed {args.sample_seed})" if args.num_samples else ""))
    if not positions:
        raise SystemExit("No usable rows: none of the rows that passed the filter has audio.")

    texts = filtered[args.text_column]
    speaker_cells = filtered[args.speaker_column] if args.speaker_column else None
    language_cells = filtered[args.language_column] if args.language_column else None
    durations = filtered[duration_col]
    rows = []
    for pos in positions:
        rows.append({
            "pos": pos, "text": texts[pos].strip(), "duration": float(durations[pos]),
            "speaker": (str(speaker_cells[pos] or "").strip() if speaker_cells else "") or args.speaker,
            "language": (str(language_cells[pos] or "").strip() if language_cells else "") or args.language,
        })

    # One fixed reference clip per speaker: the median-length one (a typical clip, not an outlier).
    by_speaker = defaultdict(list)
    for r in rows:
        by_speaker[r["speaker"] or ""].append(r)
    ref_src = {}
    for spk, group in by_speaker.items():
        group.sort(key=lambda r: r["duration"])
        ref = group[(len(group) - 1) // 2]
        ref_src[spk] = clip_source(get_audio(ref["pos"]))
        print(f"Reference for speaker {spk or '(none)'!r}: {ref['duration']:.2f}s, median of {len(group)} clips")

    # Stage 3: write, grouped by speaker (then in the order they were selected) so each speaker's rows are contiguous.
    n, no_source = 0, 0
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: (r["speaker"] or "", r["pos"])):
            audio = clip_source(get_audio(r["pos"]))
            if audio is None:
                no_source += 1
                continue
            item = {"audio": audio, "text": r["text"], "ref_audio": ref_src[r["speaker"] or ""]}
            if r["speaker"]:
                item["speaker"] = r["speaker"]
            if r["language"]:
                item["language"] = r["language"]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {args.output_jsonl} (dropped by the filter: {sum(skipped.values())}, "
          f"no audio at write time: {no_source})")


if __name__ == "__main__":
    main()
