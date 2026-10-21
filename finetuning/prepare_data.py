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
import csv
import io
import json
import os

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32
SAMPLE_RATE = 24000  # the training code only accepts 24 kHz reference audio


def load_hf_rows(args, parser):
    """Read rows from a Hugging Face dataset (Hub id or local path) into the same dicts the TSV path builds.

    Audio is decoded, resampled to 24 kHz mono and written to `--audio_out_dir` as wav files, because the
    training code reads `audio` / `ref_audio` from file paths.
    """
    import librosa
    import soundfile as sf
    from datasets import Audio, ClassLabel, load_dataset

    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split, cache_dir=args.hf_cache_dir)
    for col in (args.audio_column, args.text_column, args.speaker_column, args.language_column):
        if col is not None and col not in ds.column_names:
            parser.error(f"column '{col}' not found in {args.hf_dataset} (columns: {ds.column_names})")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    # decode=False: we decode ourselves, which works regardless of the installed `datasets` audio backend
    ds = ds.cast_column(args.audio_column, Audio(decode=False))

    def label(col, value):
        feature = ds.features[col]
        if isinstance(feature, ClassLabel) and isinstance(value, int):
            return feature.int2str(value)
        return str(value).strip() if value is not None else ''

    os.makedirs(args.audio_out_dir, exist_ok=True)
    rows, first_audio_of_speaker = [], {}
    for idx, example in enumerate(ds):
        wav_path = os.path.join(args.audio_out_dir, f"{idx:08d}.wav")
        if not os.path.exists(wav_path):
            audio = example[args.audio_column]
            source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
            wav, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True)
            sf.write(wav_path, wav, SAMPLE_RATE)

        item = {'audio': wav_path, 'text': example[args.text_column]}
        speaker = (label(args.speaker_column, example[args.speaker_column]) if args.speaker_column else '') or args.speaker
        language = (label(args.language_column, example[args.language_column]) if args.language_column else '') or args.language
        if speaker:
            item['speaker'] = speaker
        if language:
            item['language'] = language
        first_audio_of_speaker.setdefault(speaker or '', wav_path)
        rows.append(item)

    # Reference audio: one global file, else the first utterance of each speaker.
    for item in rows:
        item['ref_audio'] = args.ref_path or first_audio_of_speaker[item.get('speaker', '')]
    if args.ref_path is None:
        print(f"No --ref_path given: using the first utterance of each speaker as reference ({len(first_audio_of_speaker)} speaker(s)).")
    return rows


def load_tsv_rows(args, parser):
    # TSV with a header row. Each row is rebuilt with the keys the training code
    # expects (audio, text, ref_audio, and optionally speaker, language); other columns are dropped.
    with open(args.input_tsv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        columns = reader.fieldnames or []
        for col in (args.audio_column, args.text_column, args.speaker_column, args.language_column):
            if col is not None and col not in columns:
                parser.error(f"column '{col}' not found in {args.input_tsv} (columns: {columns})")
        if args.ref_path is None and 'ref_audio' not in columns:
            parser.error("no --ref_path given and the TSV has no 'ref_audio' column")

        rows = []
        for row in reader:
            item = {'audio': row[args.audio_column], 'text': row[args.text_column]}
            item['ref_audio'] = args.ref_path if args.ref_path is not None else row['ref_audio']
            # per-row column value wins; the global value fills in when there is no column or the cell is empty
            speaker = (row[args.speaker_column].strip() if args.speaker_column else '') or args.speaker
            language = (row[args.language_column].strip() if args.language_column else '') or args.language
            if speaker:
                item['speaker'] = speaker
            if language:
                item['language'] = language
            rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_tsv", type=str, default=None,
                        help="Local TSV input (use this or --hf_dataset)")
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="Hugging Face dataset id or local path (use this or --input_tsv). "
                             "The --*_column options then name dataset columns.")
    parser.add_argument("--hf_config", type=str, default=None, help="Dataset config / subset name")
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Only use the first N rows of the Hugging Face dataset (for quick tests)")
    parser.add_argument("--audio_out_dir", type=str, default=None,
                        help="Where decoded Hugging Face audio is written as 24 kHz wav "
                             "(default: <output_jsonl dir>/hf_audio)")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=BATCH_INFER_NUM,
                        help="Audios encoded per forward pass; lower it if you run out of GPU memory")
    parser.add_argument("--audio_column", type=str, default="audio",
                        help="Column holding each utterance's audio (a path in a TSV, an Audio feature in a HF dataset)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column holding the transcript")
    parser.add_argument("--ref_path", type=str, default=None,
                        help="Reference speaker audio used for every row. If omitted, a TSV's `ref_audio` "
                             "column is used; for a HF dataset, each speaker's first utterance is used.")
    parser.add_argument("--speaker_column", type=str, default=None,
                        help="Column holding the speaker name for each row")
    parser.add_argument("--speaker", type=str, default=None,
                        help="Speaker name for every row (or only for rows with an empty "
                             "--speaker_column cell). If neither is set, training falls back "
                             "to its own --speaker_name.")
    parser.add_argument("--language_column", type=str, default=None,
                        help="Column holding the language name for each row")
    parser.add_argument("--language", type=str, default=None,
                        help="Language name for every row (or only for rows with an empty "
                             "--language_column cell). If neither is set, training uses `Auto`.")
    args = parser.parse_args()
    if (args.input_tsv is None) == (args.hf_dataset is None):
        parser.error("pass exactly one of --input_tsv or --hf_dataset")
    if args.audio_out_dir is None:
        args.audio_out_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_jsonl)), "hf_audio")

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    total_lines = load_hf_rows(args, parser) if args.hf_dataset else load_tsv_rows(args, parser)

    final_lines = []
    batch_lines = []
    batch_audios = []
    for line in total_lines:

        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= args.batch_size:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line['audio_codes'] = code.cpu().tolist()
                final_lines.append(line)
            batch_lines.clear()
            batch_audios.clear()

    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, line in zip(enc_res.audio_codes, batch_lines):
            line['audio_codes'] = code.cpu().tolist()
            final_lines.append(line)
        batch_lines.clear()
        batch_audios.clear()

    final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]

    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.writelines(line + '\n')

if __name__ == "__main__":
    main()
