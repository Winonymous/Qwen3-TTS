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
import json

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_tsv", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--audio_column", type=str, default="audio",
                        help="TSV column holding the path to each utterance's audio")
    parser.add_argument("--text_column", type=str, default="text",
                        help="TSV column holding the transcript")
    parser.add_argument("--ref_path", type=str, default=None,
                        help="Reference speaker audio used for every row. "
                             "If omitted, a `ref_audio` column in the TSV is used.")
    parser.add_argument("--speaker_column", type=str, default=None,
                        help="TSV column holding the speaker name for each row")
    parser.add_argument("--speaker", type=str, default=None,
                        help="Speaker name for every row (or only for rows with an empty "
                             "--speaker_column cell). If neither is set, training falls back "
                             "to its own --speaker_name.")
    parser.add_argument("--language_column", type=str, default=None,
                        help="TSV column holding the language name for each row")
    parser.add_argument("--language", type=str, default=None,
                        help="Language name for every row (or only for rows with an empty "
                             "--language_column cell). If neither is set, training uses `Auto`.")
    args = parser.parse_args()

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

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

        total_lines = []
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
            total_lines.append(item)

    final_lines = []
    batch_lines = []
    batch_audios = []
    for line in total_lines:

        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= BATCH_INFER_NUM:
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
