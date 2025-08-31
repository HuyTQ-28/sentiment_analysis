from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

def create_lora_model(config):
    base_model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=config["pretrained_model_name"],
    num_labels=config["num_labels"],
    id2label=config["id2label"],
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
        bias="none",
        modules_to_save=["classifier"],
    )

    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    return lora_model