"""
Modal Script for Fine-tuning Laya (AI Slop Detector)
======================================================
This runs on a free T4 GPU on modal.com.

Usage:
  modal run finetune_laya.py
"""

import modal
from pathlib import Path
import os

# Create the Modal app
app = modal.App("laya-slop-finetuner")

# Define the image with all dependencies
laya_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", 
        "transformers>=4.48.0", 
        "datasets", 
        "pandas", 
        "scikit-learn", 
        "accelerate",
        "evaluate",
        "tiktoken",
        "sentencepiece",
        "tokenizers>=0.21.0"
    )
    .add_local_file("laya_slop_dataset.csv", remote_path="/data/laya_slop_dataset.csv")
)

# Create a volume to save the model permanently
model_volume = modal.Volume.from_name("laya-models-vol", create_if_missing=True)

# The function that runs on the GPU
@app.function(
    image=laya_image, 
    gpu="T4",           # Use a free T4 GPU
    timeout=7200,       # 2 hour timeout
    volumes={"/model_cache": model_volume}
)
def train():
    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification, 
        TrainingArguments, 
        Trainer
    )
    from sklearn.model_selection import train_test_split

    print("🚀 Starting Laya fine-tuning on Modal GPU...")

    # 1. Load the dataset
    print("Loading dataset...")
    df = pd.read_csv("/data/laya_slop_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    
    # 2. Split train/eval
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

    # 3. Load Laya (ModernBERT base)
    print("Loading answerdotai/ModernBERT-base model & tokenizer...")
    model_id = "answerdotai/ModernBERT-base"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except ValueError:
        print("Fast tokenizer failed, falling back to slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    # Laya is a sequence classifier, we map our binary labels (0=human, 1=slop)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=2, 
        ignore_mismatched_sizes=True  # In case original laya has different num_labels
    )

    # 4. Tokenize
    print("Tokenizing data...")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=512
        )
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

    # 5. Training Arguments
    args = TrainingArguments(
        output_dir="/model_cache/laya-slop-detector-checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
    )

    # 6. Compute Metrics
    import evaluate
    import numpy as np
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # 7. Train
    print("Starting Training...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    # 8. Save the best model
    print("Saving fine-tuned model to permanent Volume...")
    save_path = "/model_cache/laya-slop-final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Commit the volume changes so they persist!
    model_volume.commit()
    
    print(f"✅ Training complete! Model permanently saved inside Modal Volume at {save_path}")


@app.local_entrypoint()
def main():
    print("Deploying fine-tuning job to Modal...")
    train.remote()
    print("Job finished!")
