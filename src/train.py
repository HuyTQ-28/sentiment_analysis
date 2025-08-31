import os
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback

from .config import get_config
from .model import create_lora_model
from .data_loader import SentimentDataset
from .utils import compute_metrics

def train():
    config = get_config()
    device = config["device"]

    # Tải tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_name"])

    # Tải và chuẩn bị dữ liệu
    train_df = pd.read_csv(config["train_file"])
    val_df = pd.read_csv(config["val_file"])

    train_dataset = SentimentDataset(train_df, tokenizer, config["max_seq_length"])
    val_dataset = SentimentDataset(val_df, tokenizer, config["max_seq_length"])

    # Tải model
    model = create_lora_model(config).to(device)

    # Chuẩn bị training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        label_smoothing_factor=config["label_smoothing_factor"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        eval_strategy=config["eval_strategy"],
        eval_steps=config["eval_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        logging_dir=config["logging_dir"],
        logging_steps=config["logging_steps"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        fp16=config["fp16"],
        # gradient_accumulation_steps=config["gradient_accumulation_steps"],
    )

    print("Đang khởi tạo Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    print("Đang merge và lưu model hoàn chỉnh...")
    merged_model = model.merge_and_unload()

    final_model_path = os.path.join(config["output_dir"], "final_model")
    
    merged_model.save_pretrained(final_model_path)

    tokenizer.save_pretrained(final_model_path)

    print(f"Model hoàn chỉnh đã được lưu tại {final_model_path}")


if __name__ == "__main__":
    train()
