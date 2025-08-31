import torch
from pathlib import Path

def get_config(base_path="."):
    root_path = Path(base_path)
    if base_path == "/data":
        data_root = root_path
        save_root = root_path
    else:
        data_root = root_path / "data"
        save_root = root_path

    config = {
        "train_file": str(data_root / "processed" / "train.csv"),
        "val_file": str(data_root / "processed" / "val.csv"),
        "test_file": str(data_root / "processed" / "test.csv"),
        
        # Tham số mô hình
        "pretrained_model_name": "bert-base-uncased",
        "num_labels": 6,
        "max_seq_length": 160,

        # Tham số LoRA
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "lora_target_modules": ["query", "key", "value"],

        # Ánh xạ nhãn
        "id2label": {
            0: 'sadness',
            1: 'joy',
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        },

        # Tham số huấn luyện
        "num_epochs": 15,
        "learning_rate": 2e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "label_smoothing_factor": 0.1,
        "per_device_train_batch_size": 128,
        "per_device_eval_batch_size": 256,
        "gradient_accumulation_steps": 1,

        "eval_strategy": "steps",
        "eval_steps": 200,
        "save_strategy": "steps",
        "save_steps": 200,
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "greater_is_better": True,
        "fp16": True,
        
        "output_dir": str(save_root / "checkpoints"),
        "logging_dir": str(save_root / "checkpoints" / "logs"),
        
        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    return config