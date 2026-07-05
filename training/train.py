import yaml
import json
import wandb
import argparse
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ── Prompt template ───────────────────────────────────────
# Three fields matching your dataset structure exactly
PROMPT_TEMPLATE = """Below is an instruction that describes 
a security surveillance task. Analyze the detection log and 
provide a triage decision.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = None  # set after tokenizer loads

def format_examples(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for inst, inp, out in zip(instructions, inputs, outputs):
        text = PROMPT_TEMPLATE.format(inst, inp, out) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# ── Load config ───────────────────────────────────────────
def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

# ── Load data ─────────────────────────────────────────────
def load_data(train_path, val_path):
    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    return train_dataset, val_dataset

# ── Main training function ────────────────────────────────
def train(config_path):
    config = load_config(config_path)

    # Initialize W&B
    wandb.init(
        project=config["wandb_project"],
        name=config["wandb_run_name"],
        config=config,
    )

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=config["load_in_4bit"],
    )

    # Set EOS token globally
    global EOS_TOKEN
    EOS_TOKEN = tokenizer.eos_token

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )

    # Load and format data
    train_dataset, val_dataset = load_data(
        config["train_data_path"],
        config["val_data_path"]
    )
    train_dataset = train_dataset.map(
        format_examples, batched=True
    )
    val_dataset = val_dataset.map(
        format_examples, batched=True
    )

    # Detect GPU capabilities
    fp16_enabled = not torch.cuda.is_bf16_supported()
    bf16_enabled = torch.cuda.is_bf16_supported()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config[
            "per_device_train_batch_size"
        ],
        gradient_accumulation_steps=config[
            "gradient_accumulation_steps"
        ],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        logging_steps=10,
        seed=config["seed"],
        report_to="wandb",
        run_name=config["wandb_run_name"],
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=training_args,
    )

    # Train
    trainer_stats = trainer.train()

    # Log final stats to W&B
    wandb.log({
        "final_train_loss": trainer_stats.training_loss,
        "total_steps": trainer_stats.global_step,
        "total_time_seconds": trainer_stats.metrics[
            "train_runtime"
        ],
    })

    # Save adapter only
    model.save_pretrained(config["adapter_save_path"])
    tokenizer.save_pretrained(config["adapter_save_path"])

    print(f"\nAdapter saved to {config['adapter_save_path']}")
    print(f"Training loss: {trainer_stats.training_loss:.4f}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)