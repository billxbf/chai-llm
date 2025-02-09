from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
import wandb
from transformers import TrainingArguments, Trainer, EvalPrediction
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, log_loss
from accelerate import PartialState
from trl import SFTTrainer, SFTConfig
import torch.nn.functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_STRING = PartialState().process_index


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)


def set_seeds(seed):
    """Set seeds for reproducibility """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_latest_checkpoint(directory):
    checkpoints = [d for d in os.listdir(
        directory) if d.startswith('checkpoint-')]
    if not checkpoints:
        return directory
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    return os.path.join(directory, checkpoints[-1])


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc}


def print_trainable_layers(model):
    print("\n=== Trainable Layers ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("=======================\n")


def train(args):

    wandb.init(project="chai", name="nemo-sft")

    # check global batch size
    if args.bs < args.bs_per_device * torch.cuda.device_count():
        print("Warning: Global batch size must be greater or equal to bs_per_device * num_device. Upscaling global batch size to ",
              args.bs_per_device * torch.cuda.device_count())
        args.bs = args.bs_per_device * torch.cuda.device_count()

    print(f'Loading dataset from {args.dataset_path} ...')
    dataset = load_from_disk(args.dataset_path)
    trainset = dataset[args.train_split]
    valset = dataset[args.val_split] if args.val_split else None

    print(f'Loading model and tokenizer ...')
    if os.path.exists(args.tokenizer_path):
        args.tokenizer_path = get_latest_checkpoint(args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if os.path.exists(args.model_path):
        args.model_path = get_latest_checkpoint(args.model_path)

    if args.use_lora:
        print("Using LoRA!")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        # Configure LoRA
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
        if args.lora_weights:
            if os.path.exists(args.lora_weights):
                args.lora_weights = get_latest_checkpoint(args.lora_weights)
            model = PeftModel.from_pretrained(model, args.lora_weights)
        else:
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print_trainable_layers(model)
    else:
        print("Full Finetune!")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f'Setting up trainer ...')
    training_args = SFTConfig(
        # optimizing
        optim="adamw_8bit" if not args.use_lora else "adamw_torch",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.bs_per_device,
        gradient_accumulation_steps=args.bs // (
            args.bs_per_device * torch.cuda.device_count()),
        weight_decay=args.wd,
        # logging
        save_strategy="epoch",
        # eval_strategy="steps",
        # eval_steps=0.1,
        do_eval=False,
        logging_steps=0.01,
        output_dir=args.model_save_path,
        # misc
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        save_only_model=args.save_only_model,
        remove_unused_columns=False,
        report_to="wandb",
    )

    def format_func(example):
        return example["text"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=valset,
        formatting_func=format_func,
    )

    print(f'Start training ...')
    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_save_path", required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--val_split", required=False, type=str, default=None)
    parser.add_argument("--epoch", required=False, type=int, default=1)
    parser.add_argument("--lr", required=False, type=float, default=5e-6)
    parser.add_argument("--bs", required=False, type=int, default=64)
    parser.add_argument("--bs_per_device", required=False, type=int, default=4)
    parser.add_argument("--wd", required=False, type=float, default=0.0)
    parser.add_argument("--save_only_model", required=False,
                        type=bool, default=False)
    parser.add_argument("--use_lora", action="store_true",
                        help="Whether to use LoRA for training")
    parser.add_argument('--lora_weights', type=str, default=None)
    parser.add_argument("--seed", required=False, type=int, default=666)
    args = parser.parse_args()

    set_seeds(args.seed)
    train(args)


if __name__ == "__main__":
    main()
