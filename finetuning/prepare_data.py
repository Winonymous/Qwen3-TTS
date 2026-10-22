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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--resampled_dir", type=str, default=None,
                        help="Where 24 kHz copies of reference audio are written "
                             "(default: <output_jsonl dir>/ref_24k)")
    args = parser.parse_args()
    if args.resampled_dir is None:
        args.resampled_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_jsonl)), "ref_24k")

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    total_lines = open(args.input_jsonl).readlines()
    total_lines = [json.loads(line.strip()) for line in total_lines]

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

    final_lines =    []
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

    with open(args.output_jsonl, 'w') as f:
        for line in final_lines:
            f.writelines(line + '\n')

if __name__ == "__main__":
    main()
