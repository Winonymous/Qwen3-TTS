# coding=utf-8
"""Convert a TSV into the raw JSONL that `prepare_data.py` expects.

Each output line has: {"audio": ..., "text": ..., "ref_audio": ...}

Example:
    python tsv_to_jsonl.py --input_tsv train.tsv --output_jsonl train_raw.jsonl --ref_path ./data/ref.wav
    python prepare_data.py --input_jsonl train_raw.jsonl --output_jsonl train_with_codes.jsonl
"""
import argparse
import csv
import json
import os


def audio_duration(path):
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate


def median_length_audio(paths):
    """Return the path whose duration is the median (lower middle for an even count)."""
    durations = []
    for p in paths:
        try:
            durations.append((audio_duration(p), p))
        except Exception as e:  # missing or unreadable file
            print(f"Skipping {p} for the median reference: {e}")
    if not durations:
        raise SystemExit("No readable audio files to pick a median-length reference from.")
    durations.sort()
    duration, path = durations[(len(durations) - 1) // 2]
    print(f"No reference audio given: using the median-length clip {path} ({duration:.2f}s) "
          f"out of {len(durations)} files")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tsv", type=str, required=True, help="TSV with a header row")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--audio_column", type=str, default="audio")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--ref_audio_column", type=str, default="ref_audio",
                        help="Column with the reference audio path (ignored if --ref_path is given)")
    parser.add_argument("--ref_path", type=str, default=None,
                        help="One reference audio used for every row (recommended: same ref for all rows). "
                             "If it is omitted and a row has no ref_audio, the median-length audio is used.")
    parser.add_argument("--check_audio", action="store_true",
                        help="Skip rows whose audio file does not exist")
    args = parser.parse_args()

    with open(args.input_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        columns = reader.fieldnames or []
        missing = [c for c in (args.audio_column, args.text_column) if c not in columns]
        if missing:
            parser.error(f"column(s) {missing} not found in {args.input_tsv} (columns: {columns})")
        has_ref_column = args.ref_audio_column in columns

        rows, skipped = [], 0
        for row in reader:
            audio = row[args.audio_column].strip()
            text = row[args.text_column].strip()
            ref_audio = args.ref_path or (row[args.ref_audio_column].strip() if has_ref_column else "")
            if not audio or not text or (args.check_audio and not os.path.isfile(audio)):
                skipped += 1
                continue
            rows.append({"audio": audio, "text": text, "ref_audio": ref_audio})

    # Rows without a reference share the median-length clip of the dataset.
    if any(not r["ref_audio"] for r in rows):
        fallback = median_length_audio([r["audio"] for r in rows])
        for r in rows:
            r["ref_audio"] = r["ref_audio"] or fallback

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output_jsonl} (skipped {skipped})")


if __name__ == "__main__":
    main()
