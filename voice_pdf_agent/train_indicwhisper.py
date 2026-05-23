"""Fine-tune IndicWhisper on bhajan audio-transcript pairs.

Prepare a CSV with columns for audio file paths and transcripts:
  path,transcript
  ./audio/bhajan1.wav,राधे राधे जपो...
  ./audio/bhajan2.wav,हरे कृष्ण हरे राम...

Usage:
  python train_indicwhisper.py \
    --train_csv bhajan_train.csv \
    --output_dir outputs/indicwhisper_bhajan \
    --base_model ai4bharat/indicwhisper \
    --language hi \
    --num_train_epochs 5

After training, use the fine-tuned model in live_search.py:
  python live_search.py --pdf lyrics.pdf --use_indicwhisper \
    --indicwhisper_model outputs/indicwhisper_bhajan \
    --use_index
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune IndicWhisper on bhajan data")
    p.add_argument("--train_csv", required=True, help="CSV with audio paths and transcripts")
    p.add_argument("--eval_csv", default=None, help="Optional CSV for evaluation")
    p.add_argument("--audio_col", default="path", help="Column name for audio file paths")
    p.add_argument("--text_col", default="transcript", help="Column name for transcripts")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_model", default="ai4bharat/indicwhisper",
                   help="HuggingFace model ID to start from")
    p.add_argument("--language", default="hi", help="BCP-47 language code (hi=Hindi, sa=Sanskrit)")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2,
                   help="Effective batch = batch_size * grad_accum")
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", default=None)
    return p.parse_args()


wer_metric = evaluate.load("wer")


@dataclass
class DataCollatorSeq2Seq:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # mask padding with -100 so it's ignored in the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # strip leading BOS token that the tokenizer may prepend
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def make_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}
    return compute_metrics


def main():
    args = parse_args()

    print(f"Loading processor and model from: {args.base_model}")
    processor = WhisperProcessor.from_pretrained(
        args.base_model, language=args.language, task="transcribe"
    )

    use_cuda = torch.cuda.is_available()
    # only use half-precision when explicitly requested via --fp16
    use_bf16 = use_cuda and args.fp16 and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and args.fp16 and not use_bf16
    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # load directly in target dtype to avoid a temporary fp32 copy in VRAM
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=load_dtype
    )
    if use_cuda:
        torch.cuda.empty_cache()
        model = model.to("cuda")

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # --- dataset ---
    data_files = {"train": args.train_csv}
    if args.eval_csv:
        data_files["eval"] = args.eval_csv
    ds = load_dataset("csv", data_files=data_files)

    audio_col = args.audio_col
    text_col = args.text_col

    def prepare_batch(batch):
        import soundfile as sf
        import numpy as np
        # load audio directly from path — avoids cast_column compatibility issues
        audio, sr = sf.read(batch[audio_col])
        inputs = processor(audio, sampling_rate=sr, return_tensors="np")
        batch["input_features"] = inputs.input_features[0]
        # tokenize transcript
        labels = processor.tokenizer(batch[text_col]).input_ids
        batch["labels"] = labels
        return batch

    remove_cols = ds["train"].column_names
    ds_proc = ds.map(prepare_batch, remove_columns=remove_cols, num_proc=1)

    # --- training arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="epoch" if args.eval_csv else "no",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=225,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        remove_unused_columns=False,
        load_best_model_at_end=bool(args.eval_csv),
        metric_for_best_model="wer" if args.eval_csv else None,
        greater_is_better=False,
        dataloader_num_workers=0,
    )
    print(f"Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc.get("eval"),
        data_collator=DataCollatorSeq2Seq(processor=processor),
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )

    print("Starting fine-tuning...")
    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nFine-tuned model saved to: {args.output_dir}")
    print("Use it with:")
    print(f"  python live_search.py --pdf lyrics.pdf --use_indicwhisper \\")
    print(f"    --indicwhisper_model {args.output_dir} --use_index")


if __name__ == "__main__":
    main()
