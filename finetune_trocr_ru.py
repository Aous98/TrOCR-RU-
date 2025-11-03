import os, argparse
import pandas as pd
from PIL import Image
import torch
from datasets import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    TrainerCallback,
)

# ---- environment niceties (can also export in shell) ----
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:true")

# ---------- config ----------
DEF_MODEL = "microsoft/trocr-base-stage1"   # byte-level decoder
MAX_TXT_LEN = 384                           # ↓ from 512 to save VRAM
IMG_FP16 = True
FREEZE_ENCODER_EPOCHS = 1                   # warm-up for 1 epoch
# ----------------------------

def read_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    assert {"image_path","text"} <= set(df.columns), "TSV must have columns: image_path, text"
    return df

def build_datasets(train_tsv, val_tsv, processor):
    def _preprocess(examples):
        images = [Image.open(p).convert("RGB") for p in examples["image_path"]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        labels = processor.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=MAX_TXT_LEN,
            truncation=True,
        ).input_ids
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

    train_df = read_tsv(train_tsv)
    val_df   = read_tsv(val_tsv)
    train_ds = Dataset.from_pandas(train_df)
    val_ds   = Dataset.from_pandas(val_df)

    train_ds = train_ds.map(_preprocess, batched=True, remove_columns=train_ds.column_names)
    val_ds   = val_ds.map(_preprocess,   batched=True, remove_columns=val_ds.column_names)
    return train_ds, val_ds

class UnfreezeEncoderAtEpoch(TrainerCallback):
    def __init__(self, encoder_params, trigger_epoch: int):
        self.encoder_params = list(encoder_params)
        self.trigger_epoch = trigger_epoch
        self.done = False

    def on_epoch_end(self, args, state, control, **kwargs):
        if (not self.done) and state.epoch is not None and state.epoch >= self.trigger_epoch:
            for p in self.encoder_params:
                p.requires_grad = True
            print(f"🧊 Encoder unfrozen at epoch {state.epoch:.2f}")
            self.done = True
        return control

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)

    # essential config
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = MAX_TXT_LEN
    model.config.use_cache = False                     # ↓ save VRAM during training

    train_ds, val_ds = build_datasets(args.train_tsv, args.val_tsv, processor)

    # memory-friendly hyperparams for ~8GB VRAM
    per_dev_bs = 1
    grad_accum = 12           # effective batch 12
    lr = 3e-5
    num_epochs = args.epochs
    eval_steps = 1000
    save_steps = 1000
    logging_steps = 50

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=per_dev_bs,
        per_device_eval_batch_size=per_dev_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        # Use Trainer's tqdm progress bar:
        disable_tqdm=False,
        # keep eval light (no text generation during eval to save VRAM)
        predict_with_generate=False,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=2,
        fp16=(IMG_FP16 and device=="cuda"),
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        eval_accumulation_steps=4,
    )

    # warm-up: freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        tokenizer=processor.feature_extractor,  # image side
    )

    # callback to unfreeze after epoch 1
    if FREEZE_ENCODER_EPOCHS > 0:
        trainer.add_callback(UnfreezeEncoderAtEpoch(model.encoder.parameters(), FREEZE_ENCODER_EPOCHS))

    print(f"Starting fine-tuning for {num_epochs} epochs…")
    # Single call — Trainer handles epochs, tqdm, eval & saves
    trainer.train(resume_from_checkpoint=True if os.path.isdir(args.outdir) and any(n.startswith("checkpoint-") for n in os.listdir(args.outdir)) else None)

    # save final
    print("Saving fine-tuned model…")
    model.save_pretrained(args.outdir)
    processor.save_pretrained(args.outdir)
    print(f"✅ Saved to: {args.outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--val_tsv", required=True)
    ap.add_argument("--outdir", default="trocr_ru_finetuned")
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()
    main(args)

