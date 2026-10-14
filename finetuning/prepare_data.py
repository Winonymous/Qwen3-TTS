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
    args = parser.parse_args()

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    # TSV with a header row. Rows are normalized to the keys the training code
    # expects (audio, text, ref_audio); any other columns (speaker, language, ...) pass through.
    with open(args.input_tsv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        columns = reader.fieldnames or []
        for col in (args.audio_column, args.text_column):
            if col not in columns:
                parser.error(f"column '{col}' not found in {args.input_tsv} (columns: {columns})")
        if args.ref_path is None and 'ref_audio' not in columns:
            parser.error("no --ref_path given and the TSV has no 'ref_audio' column")

        total_lines = []
        for row in reader:
            item = {k: v for k, v in row.items() if k not in (args.audio_column, args.text_column)}
            item['audio'] = row[args.audio_column]
            item['text'] = row[args.text_column]
            if args.ref_path is not None:
                item['ref_audio'] = args.ref_path
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
