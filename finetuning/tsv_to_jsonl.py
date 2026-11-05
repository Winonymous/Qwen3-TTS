# coding=utf-8
"""Convert a TSV into the raw JSONL that `prepare_data.py` expects.

Each output line has: {"audio": ..., "text": ..., "ref_audio": ...}

Example:
    python tsv_to_jsonl.py --input_tsv train.tsv --output_jsonl train_raw.jsonl --ref_path ./data/ref.wav
    python prepare_data.py --input_jsonl train_raw.jsonl --output_jsonl train_with_codes.jsonl
"""
import argparse
import csv
import difflib
import json
import os
import random

from logging_utils import setup_logging
from text_utils import normalize_text


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
    parser.add_argument("--speaker_column", type=str, default=None,
                        help="Column with the speaker name for each row (several speakers in one file)")
    parser.add_argument("--speaker", type=str, default=None,
                        help="Speaker name for every row, or only for rows with an empty --speaker_column cell")
    parser.add_argument("--language_column", type=str, default=None,
                        help="Column with the language name for each row (several languages in one file)")
    parser.add_argument("--language", type=str, default=None,
                        help="Language for every row, or only for rows with an empty --language_column cell "
                             "(default: none = Auto). A language the base model does not know is added during "
                             "fine-tuning. Use the same value as `language=` at inference.")
    parser.add_argument("--no_normalize_text", dest="normalize_text", action="store_false",
                        help="Keep the text exactly as it is in the TSV. By default it is normalized (Unicode NFC, "
                             "invisible characters removed, whitespace collapsed) so that the same word is always "
                             "spelled the same way. Apply the same normalization at inference "
                             "(text_utils.normalize_text).")
    parser.add_argument("--max_text_chars", type=int, default=0,
                        help="Skip rows whose text has more characters than this (0 = no limit). Catches runaway "
                             "transcripts, e.g. a repeated ASR hallucination.")
    parser.add_argument("--check_audio", action="store_true",
                        help="Skip rows whose audio file does not exist")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Random subset of this many usable rows (skipped rows never count towards it)")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default=None,
                        help="Everything printed is also written here (default: <output_jsonl>.log)")
    args = parser.parse_args()
    setup_logging(args.log_file or args.output_jsonl + ".log", args)

    # utf-8-sig: a byte-order mark (written by Excel and by pandas with that encoding) would otherwise be glued to
    # the first column name, so that column looks "missing".
    with open(args.input_tsv, newline="", encoding="utf-8-sig") as f:
        header = f.readline()
        f.seek(0)
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames:
            reader.fieldnames = [c.strip() for c in reader.fieldnames]
        columns = reader.fieldnames or []
        print(f"Columns in {args.input_tsv}: {columns}")
        missing = [c for c in (args.audio_column, args.text_column, args.speaker_column, args.language_column)
                   if c is not None and c not in columns]
        if missing:
            hints = []
            if len(columns) == 1 and ("," in header or ";" in header):
                hints.append("the file has a single column, so it is probably comma- or semicolon-separated, not "
                             "tab-separated (this script reads tabs)")
            for c in missing:
                close = difflib.get_close_matches(c, columns, n=3, cutoff=0.5)
                if close:
                    hints.append(f"{c!r}: did you mean {close}?")
            parser.error(f"column(s) {missing} not found in {args.input_tsv}. Columns: {columns}. "
                         + " ".join(hints) + " Set the name with --audio_column / --text_column / "
                         "--speaker_column / --language_column.")
        has_ref_column = args.ref_audio_column in columns

        def cell(row, column):
            """A cell as stripped text; '' if the row is too short to have that column (the reader gives None)."""
            return (row.get(column) or "").strip()

        rows, skipped, too_long, normalized = [], 0, [], 0
        malformed = []  # (line number, fields found) for rows that do not have exactly one value per column
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                # Too few fields (a missing tab, or a text with a line break in it) or too many (a tab inside the
                # text). The text column is the usual victim, so the row is dropped rather than guessed at.
                found = sum(v is not None for k, v in row.items() if k is not None) + len(row.get(None) or [])
                malformed.append((reader.line_num, found))
                skipped += 1
                continue
            audio = cell(row, args.audio_column)
            text = cell(row, args.text_column)
            if args.normalize_text:
                clean = normalize_text(text)
                normalized += clean != text
                text = clean
            ref_audio = args.ref_path or (cell(row, args.ref_audio_column) if has_ref_column else "")
            if not audio or not text or (args.check_audio and not os.path.isfile(audio)):
                skipped += 1
                continue
            if args.max_text_chars > 0 and len(text) > args.max_text_chars:
                too_long.append((audio, len(text)))
                skipped += 1
                continue
            item = {"audio": audio, "text": text, "ref_audio": ref_audio}
            speaker = (cell(row, args.speaker_column) if args.speaker_column else "") or args.speaker
            if speaker:
                item["speaker"] = speaker
            language = (cell(row, args.language_column) if args.language_column else "") or args.language
            if language:
                item["language"] = language
            rows.append(item)

    # Sampling happens after the filters above, so skipped rows never count towards the amount.
    if args.num_samples and len(rows) > args.num_samples:
        print(f"Sampling {args.num_samples} of {len(rows)} usable rows")
        rows = random.Random(args.sample_seed).sample(rows, args.num_samples)
    elif args.num_samples:
        print(f"Warning: only {len(rows)} usable rows exist, fewer than --num_samples {args.num_samples}")

    # Rows without a reference share the median-length clip of their speaker (of the whole dataset if there are
    # no speakers).
    fallbacks = {}
    for r in rows:
        if r["ref_audio"]:
            continue
        key = r.get("speaker", "")
        if key not in fallbacks:
            fallbacks[key] = median_length_audio([x["audio"] for x in rows if x.get("speaker", "") == key])
        r["ref_audio"] = fallbacks[key]

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if malformed:
        print(f"Skipped {len(malformed)} row(s) that do not have {len(columns)} tab-separated fields; first few "
              f"(file line, fields found): {malformed[:5]}")
    if too_long:
        print(f"Skipped {len(too_long)} row(s) with more than {args.max_text_chars} characters of text; "
              f"longest: {max(n for _, n in too_long)} chars. First few: "
              + ", ".join(f"{a} ({n})" for a, n in too_long[:5]))
    if args.normalize_text:
        print(f"Text normalization changed {normalized} row(s)")
    print(f"Wrote {len(rows)} rows to {args.output_jsonl} (skipped {skipped})")


if __name__ == "__main__":
    main()
