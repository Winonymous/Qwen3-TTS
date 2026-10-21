## Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base

The Qwen3-TTS-12Hz-1.7B/0.6B-Base model series currently supports single-speaker fine-tuning. Please run `pip install qwen-tts` first, then run the command below:

```
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

Then follow the steps below to complete the entire fine-tuning workflow. Multi-speaker fine-tuning and other advanced fine-tuning features will be supported in future releases.

### 1) Input JSONL format

Prepare your training file as a JSONL (one JSON object per line). Each line must contain:

- `audio`: path to the target training audio (wav)
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav)

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

Optional columns (also carried through `prepare_data.py`):

- `speaker`: speaker name for this row (falls back to `--speaker_name`). Several speakers can be trained in one run; each gets its own id in the codec embedding table, and its embedding is the mean speaker-encoder embedding of its `ref_audio`s. Names are lowercased.
- `language`: language name (default `Auto`). Languages the checkpoint does not know (e.g. `yoruba`) are registered automatically: they get a new language id, an embedding initialised from the mean of the existing language embeddings plus noise, and are trained. Names must not contain `dialect`.

The saved `config.json` merges the new `spk_id` / `codec_language_id` with the existing ones, so at inference use `generate_custom_voice(text, speaker="<name>", language="<name>")`.

`ref_audio` recommendation:
- Strongly recommended: use the same `ref_audio` for all samples.
- Keeping `ref_audio` identical across the dataset usually improves speaker consistency and stability during generation.


### 1b) Or use a Hugging Face dataset

Instead of a TSV, `prepare_data.py` can read a Hub dataset (or a local `load_dataset` path). The `--audio_column`, `--text_column`, `--speaker_column` and `--language_column` options then name dataset columns (`audio` must be an `Audio` feature; `ClassLabel` columns are converted to their names). Audio is decoded, resampled to 24 kHz mono and written as wav files to `--audio_out_dir` (default: `hf_audio/` next to the output JSONL), since training reads audio from file paths. Already-written files are reused on re-runs.

```bash
python prepare_data.py   --hf_dataset your-org/your-dataset --hf_split train   --audio_column audio --text_column sentence   --speaker_column speaker_id --language yoruba   --output_jsonl train_with_codes.jsonl
```

If you don't pass `--ref_path`, each speaker's first utterance is used as its reference audio (pick a clean one with `--ref_path` if that matters). Use `--max_samples 100` for a quick test; for private datasets run `huggingface-cli login` first. `run_sft_multigpu.sh` accepts the same through `HF_DATASET`, `HF_CONFIG`, `HF_SPLIT` and `MAX_SAMPLES`.

### 2) Prepare data (extract `audio_codes`)

`prepare_data.py` reads a TSV with a header row and writes a training JSONL that includes `audio_codes`. Use `--audio_column` / `--text_column` to name the TSV columns (defaults: `audio`, `text`) and `--ref_path` to use one reference audio for every row (otherwise a `ref_audio` column is required). Speaker and language can be set per row with `--speaker_column` / `--language_column`, globally with `--speaker` / `--language`, or both (the column value wins; the global value fills empty cells). Columns not mapped by an argument are dropped from the output.

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_tsv train_raw.tsv \
  --audio_column full_path \
  --text_column sentence \
  --speaker_column voice_over \
  --language_column language \
  --ref_path ./data/ref.wav \
  --output_jsonl train_with_codes.jsonl
```


### 3) Fine-tune

Run SFT using the prepared JSONL:

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_test
```

Checkpoints will be written to:
- `output/checkpoint-epoch-0`
- `output/checkpoint-epoch-1`
- `output/checkpoint-epoch-2`
- ...


### 4) Quick inference test

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="speaker_test",
)
sf.write("output.wav", wavs[0], sr)
```

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

RAW_TSV="train_raw.tsv"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="speaker_1"

python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_tsv ${RAW_TSV} \
  --output_jsonl ${TRAIN_JSONL}

python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
```