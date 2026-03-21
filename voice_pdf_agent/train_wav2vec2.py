"""Train Wav2Vec2 ASR (CTC) on your audio-text pairs.

Prepare a CSV with columns pointing to audio file paths and transcripts, for example:
  path,transcript
  ./audio/utt1.wav,This is the first sentence.
  ./audio/utt2.wav,Another example.

Usage (example):
  python train_wav2vec2.py \
    --train_csv train.csv \
    --audio_col path --text_col transcript \
    --output_dir outputs/wav2vec2 \
    --model_name facebook/wav2vec2-base-960h \
    --per_device_train_batch_size 8 --num_train_epochs 5

Notes:
- Requires a GPU for practical training speed.
- Install requirements: `pip install -r requirements.txt`
"""
from __future__ import annotations
import argparse
import os
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from dataclasses import dataclass
import evaluate
import jiwer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True, help="CSV file with audio paths and transcripts")
    p.add_argument("--audio_col", default="path", help="CSV column name for audio file paths")
    p.add_argument("--text_col", default="transcript", help="CSV column name for transcripts")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="facebook/wav2vec2-base-960h")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    return p.parse_args()


wer_metric = evaluate.load("wer")


def main():
    args = parse_args()
    ds = load_dataset("csv", data_files={"train": args.train_csv})
    audio_col = args.audio_col
    text_col = args.text_col

    # cast audio column to Audio so datasets loads it as arrays
    ds = ds.cast_column(audio_col, Audio(sampling_rate=16_000))

    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)

    def prepare_batch(batch):
        # batch[audio_col] will be a dict with 'array' and 'sampling_rate'
        audio = batch[audio_col]["array"]
        inputs = processor(audio, sampling_rate=16_000, return_tensors="np", padding=True)
        batch["input_values"] = inputs.input_values[0]
        with processor.as_target_processor():
            labels = processor(batch[text_col]).input_ids
        batch["labels"] = labels
        return batch

    ds_proc = ds["train"].map(prepare_batch, remove_columns=ds["train"].column_names)

    @dataclass
    class DataCollatorCTC:
        processor: Wav2Vec2Processor

        def __call__(self, features):
            input_values = [f["input_values"] for f in features]
            labels = [f["labels"] for f in features]
            batch = self.processor.pad({"input_values": input_values}, return_tensors="pt")
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad({"input_ids": labels}, return_tensors="pt")
            # replace padding with -100 to ignore in loss
            labels = labels_batch["input_ids"].masked_fill(labels_batch["input_ids"] == processor.tokenizer.pad_token_id, -100)
            batch["labels"] = labels
            return batch

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        # decode labels
        label_ids = pred.label_ids
        # replace -100
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    data_collator = DataCollatorCTC(processor=processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        evaluation_strategy="no",
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        learning_rate=args.learning_rate,
        save_total_limit=2,
        save_steps=500,
        logging_steps=100,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds_proc,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
