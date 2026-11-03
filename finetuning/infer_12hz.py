# coding=utf-8
"""Synthesize many sentences with a fine-tuned custom-voice checkpoint, in parallel.

Three levels of parallelism:
  * batching: `--batch_size` sentences go through the model in one `generate_custom_voice` call;
  * several GPUs: `--gpus 0,1,2,3` (default: all visible GPUs) runs one worker per GPU, each with its own copy of
    the model and its own share of the sentences;
  * several workers per GPU: `--procs_per_gpu 2` keeps a GPU busy when batches are small (the model is only ~4 GB).

Input is a CSV, TSV or JSONL file with one sentence per row. Every sentence is synthesized for every speaker in the
checkpoint (or for `--speakers`, or for the row's own `--speaker_column`).

Example:
    python infer_12hz.py --checkpoint output/checkpoint-best --input test_sentences.tsv \
        --text_column sentence --language hausa --output_dir wavs --batch_size 8 --gpus 0,1
"""
import argparse
import csv
import json
import os
import re
import time

from logging_utils import setup_logging
from text_utils import normalize_text


def read_rows(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    tab = ext in (".tsv", ".txt")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t" if tab else ",",
                                quoting=csv.QUOTE_NONE if tab else csv.QUOTE_MINIMAL)
        return list(reader)


def safe_name(value):
    return re.sub(r"[^\w.-]+", "_", str(value))


def one_line(text):
    return re.sub(r"\s+", " ", text)


def build_items(args, rows, checkpoint_speakers, checkpoint_languages):
    """One item per (sentence, speaker): {id, speaker, language, text, path}."""
    items = []
    for index, row in enumerate(rows):
        text = (row.get(args.text_column) or "").strip()
        if args.normalize_text:
            text = normalize_text(text)
        if not text:
            continue
        row_id = row.get(args.id_column) if args.id_column else index
        language = (row.get(args.language_column) if args.language_column else None) or args.language
        language = language.strip().lower()
        if language != "auto" and language not in checkpoint_languages:
            raise SystemExit(f"Language {language!r} (row {index}) is not in the checkpoint: "
                             f"{sorted(checkpoint_languages)}")
        if args.speaker_column:
            speakers = [row[args.speaker_column].strip().lower()]
        else:
            speakers = args.speakers or checkpoint_speakers
        for speaker in speakers:
            if speaker not in checkpoint_speakers:
                raise SystemExit(f"Speaker {speaker!r} (row {index}) is not in the checkpoint: "
                                 f"{sorted(checkpoint_speakers)}")
            path = os.path.join(args.output_dir, f"{safe_name(row_id)}_{safe_name(speaker)}_{safe_name(language)}.wav")
            items.append({"id": str(row_id), "speaker": speaker, "language": language, "text": text, "path": path})
    return items


def run_worker(rank, gpu, items, args):
    """Load the checkpoint on `gpu` and synthesize `items` in batches. Writes manifest.part<rank>.tsv."""
    import soundfile as sf
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    tag = f"[worker {rank} / cuda:{gpu}]"
    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint, device_map=f"cuda:{gpu}", dtype=torch.bfloat16, attn_implementation=args.attn,
    )
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": args.do_sample}
    if args.do_sample:
        gen_kwargs.update(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

    results = []  # (item, status, seconds)

    def synthesize(batch):
        wavs, sr = tts.generate_custom_voice(
            text=[i["text"] for i in batch], language=[i["language"] for i in batch],
            speaker=[i["speaker"] for i in batch], **gen_kwargs,
        )
        done = []
        for item, wav in zip(batch, wavs):
            sf.write(item["path"], wav, sr)
            done.append((item, "ok", len(wav) / sr))
        return done

    start = time.time()
    for first in range(0, len(items), args.batch_size):
        batch = items[first:first + args.batch_size]
        try:
            results.extend(synthesize(batch))
        except Exception as e:  # e.g. out of memory: retry the batch one sentence at a time
            print(f"{tag} batch failed ({type(e).__name__}: {e}); retrying one by one", flush=True)
            torch.cuda.empty_cache()
            for item in batch:
                try:
                    results.extend(synthesize([item]))
                except Exception as e2:
                    results.append((item, f"failed: {type(e2).__name__}: {e2}", 0.0))
        print(f"{tag} {min(first + args.batch_size, len(items))}/{len(items)} done "
              f"({time.time() - start:.0f}s)", flush=True)

    with open(os.path.join(args.output_dir, f"manifest.part{rank}.tsv"), "w", encoding="utf-8") as f:
        for item, status, seconds in results:
            f.write("\t".join([item["id"], item["speaker"], item["language"], one_line(status), f"{seconds:.2f}",
                               item["path"], one_line(item["text"])]) + "\n")


def merge_manifests(args, n_workers):
    """manifest.tsv = rows from earlier runs (for --skip_existing) + this run's rows; this run wins on conflicts."""
    manifest_path = os.path.join(args.output_dir, "manifest.tsv")
    rows = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            next(f, None)  # header
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 7:
                    rows[fields[5]] = fields
    for rank in range(n_workers):
        part = os.path.join(args.output_dir, f"manifest.part{rank}.tsv")
        if not os.path.exists(part):
            continue
        with open(part, encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 7:
                    rows[fields[5]] = fields
        os.remove(part)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("id\tspeaker\tlanguage\tstatus\tseconds\tpath\ttext\n")
        for fields in sorted(rows.values(), key=lambda r: (r[0], r[1])):
            f.write("\t".join(fields) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint directory")
    parser.add_argument("--input", type=str, required=True, help="CSV, TSV or JSONL with one sentence per row")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--id_column", type=str, default=None,
                        help="Column used in the output file names (default: the row number)")
    parser.add_argument("--language", type=str, default="auto",
                        help="Language for every row (or rows with an empty --language_column cell); use the "
                             "name the model was trained with, e.g. hausa")
    parser.add_argument("--language_column", type=str, default=None)
    parser.add_argument("--speakers", type=str, nargs="*", default=None,
                        help="Speakers to synthesize every sentence for (default: all speakers in the checkpoint)")
    parser.add_argument("--speaker_column", type=str, default=None,
                        help="Use the speaker named in this column for each row instead of crossing all speakers")
    parser.add_argument("--normalize_text", action="store_true",
                        help="Apply text_utils.normalize_text; use it if the model was trained with normalized text")
    parser.add_argument("--batch_size", type=int, default=8, help="Sentences per generate call, per worker")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids (default: all visible GPUs)")
    parser.add_argument("--procs_per_gpu", type=int, default=1,
                        help="Workers (model copies) per GPU; 2-3 can raise throughput when batches are small")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Generation limit in codec frames (12.5 per second, so 300 = 24 s)")
    parser.add_argument("--do_sample", action="store_true", help="Sample instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--attn", type=str, default="flash_attention_2",
                        help="Attention implementation (flash_attention_2, sdpa, eager)")
    parser.add_argument("--skip_existing", action="store_true", help="Do not redo sentences whose wav exists")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Everything printed is also written here (default: <output_dir>/infer.log)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.log_file or os.path.join(args.output_dir, "infer.log"), args)

    with open(os.path.join(args.checkpoint, "config.json"), encoding="utf-8") as f:
        talker_config = json.load(f).get("talker_config", {})
    checkpoint_speakers = sorted((talker_config.get("spk_id") or {}).keys())
    checkpoint_languages = set((talker_config.get("codec_language_id") or {}).keys())
    if not checkpoint_speakers:
        raise SystemExit(f"{args.checkpoint} has no speakers; use a checkpoint saved by sft_12hz.py")
    args.speakers = [s.lower() for s in args.speakers] if args.speakers else None

    rows = read_rows(args.input)
    items = build_items(args, rows, checkpoint_speakers, checkpoint_languages)
    total = len(items)
    if args.skip_existing:
        items = [i for i in items if not os.path.exists(i["path"])]
    print(f"{total} sentence/speaker pairs, {total - len(items)} already done, {len(items)} to synthesize "
          f"(speakers: {args.speakers or checkpoint_speakers})")
    if not items:
        return

    import torch

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))
    if not gpus:
        raise SystemExit("No GPU available")
    workers = [g for g in gpus for _ in range(args.procs_per_gpu)]

    # Longest sentences first and dealt out round-robin: every worker gets the same mix of lengths, and the
    # sentences inside a batch have similar lengths (less padding).
    items.sort(key=lambda i: len(i["text"]), reverse=True)
    shards = [items[k::len(workers)] for k in range(len(workers))]

    start = time.time()
    if len(workers) == 1:
        run_worker(0, workers[0], shards[0], args)
    else:
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=run_worker, args=(rank, gpu, shard, args))
                 for rank, (gpu, shard) in enumerate(zip(workers, shards))]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        crashed = [rank for rank, p in enumerate(procs) if p.exitcode != 0]
        if crashed:
            print(f"Worker(s) {crashed} exited with an error; their sentences are missing. "
                  f"Rerun with --skip_existing to retry only those.")
    elapsed = time.time() - start

    rows_done = merge_manifests(args, len(workers))
    ok = [r for r in rows_done.values() if r[3] == "ok"]
    failed = [r for r in rows_done.values() if r[3] != "ok"]
    audio_seconds = sum(float(r[4]) for r in ok)
    print(f"Done: {len(ok)} ok, {len(failed)} failed in {elapsed:.0f}s; {audio_seconds:.0f}s of audio "
          f"({audio_seconds / max(elapsed, 1e-9):.1f}x real time). Manifest: "
          f"{os.path.join(args.output_dir, 'manifest.tsv')}")
    for r in failed[:5]:
        print(f"  failed: {r[0]}/{r[1]}: {r[3]}")


if __name__ == "__main__":
    main()
