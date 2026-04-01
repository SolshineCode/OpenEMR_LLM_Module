"""
Medical LLM Fine-Tuning Script using Unsloth

This script fine-tunes a base LLM on medical data using Unsloth for
efficient training with LoRA/QLoRA.

Unsloth provides:
- 2x faster training than standard HuggingFace
- 80% less memory usage
- No accuracy degradation

Usage:
    python train_medical_llm.py --base_model unsloth/Llama-3.2-1B-Instruct \\
                                --dataset medical_qa \\
                                --output_dir ./models/medical_adapter

For Google Colab (free GPU), use the notebook version:
    fine_tuning/train_medical_llm.ipynb
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments

# Unsloth imports (will fail if not installed)
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Unsloth not installed. Install with:")
    print('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')

from trl import SFTTrainer


# Medical instruction prompt template
MEDICAL_PROMPT_TEMPLATE = """Below is a medical question. Provide an accurate, evidence-based response.

### Question:
{question}

### Response:
{answer}"""

# Chat template for instruction-tuned models
CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful medical assistant. Provide accurate, evidence-based information. Always recommend consulting healthcare providers for medical decisions.<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|>"""


def load_medical_dataset(dataset_name: str, split: str = "train") -> Dataset:
    """
    Load a medical QA dataset.

    Supported datasets:
    - medmcqa: Medical Multiple Choice QA
    - pubmedqa: PubMed QA dataset
    - medqa: Medical Question Answering
    - custom: Load from local JSON file
    """
    if dataset_name == "medmcqa":
        ds = load_dataset("openlifescienceai/medmcqa", split=split)
        # Format for training
        def format_medmcqa(example):
            question = example["question"]
            # Get the correct answer
            options = [example["opa"], example["opb"], example["opc"], example["opd"]]
            answer_idx = example["cop"]  # 0-3
            answer = options[answer_idx] if 0 <= answer_idx < 4 else options[0]
            explanation = example.get("exp", "")
            full_answer = f"{answer}\n\nExplanation: {explanation}" if explanation else answer
            return {"question": question, "answer": full_answer}

        return ds.map(format_medmcqa)

    elif dataset_name == "pubmedqa":
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
        def format_pubmedqa(example):
            return {
                "question": example["question"],
                "answer": f"{example['final_decision']}\n\n{example['long_answer']}"
            }
        return ds.map(format_pubmedqa)

    elif dataset_name == "medqa":
        ds = load_dataset("bigbio/med_qa", split=split)
        def format_medqa(example):
            return {
                "question": example["question"],
                "answer": example["answer"]
            }
        return ds.map(format_medqa)

    elif dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        # Load from local file
        with open(dataset_name) as f:
            if dataset_name.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        return Dataset.from_list(data)

    else:
        # Try loading from HuggingFace Hub
        return load_dataset(dataset_name, split=split)


def format_prompts(examples, template: str = "instruction"):
    """Format examples into training prompts."""
    if template == "chat":
        template_str = CHAT_TEMPLATE
    else:
        template_str = MEDICAL_PROMPT_TEMPLATE

    texts = []
    for question, answer in zip(examples["question"], examples["answer"]):
        text = template_str.format(question=question, answer=answer)
        texts.append(text)

    return {"text": texts}


def train(
    base_model: str = "unsloth/Llama-3.2-1B-Instruct",
    dataset_name: str = "medmcqa",
    output_dir: str = "./models/medical_adapter",
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 5,
    max_steps: int = -1,  # -1 means use num_train_epochs
    save_steps: int = 100,
    logging_steps: int = 10,
    seed: int = 42,
    use_chat_template: bool = True,
    export_gguf: bool = True,
    quantization_method: str = "q4_k_m",
):
    """
    Fine-tune a medical LLM using Unsloth.

    Args:
        base_model: Base model to fine-tune (HuggingFace model ID)
        dataset_name: Dataset to use for training
        output_dir: Directory to save the fine-tuned model
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        max_steps: Max training steps (-1 for full epochs)
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        seed: Random seed
        use_chat_template: Use chat template vs instruction template
        export_gguf: Export to GGUF format for llama.cpp
        quantization_method: GGUF quantization method
    """
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth is not installed")

    print(f"Loading base model: {base_model}")

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Optimized checkpointing
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"Loading dataset: {dataset_name}")
    dataset = load_medical_dataset(dataset_name)

    # Format dataset
    template = "chat" if use_chat_template else "instruction"
    dataset = dataset.map(
        lambda x: format_prompts(x, template),
        batched=True,
    )

    print(f"Dataset size: {len(dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=seed,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Train
    print("Starting training...")
    trainer_stats = trainer.train()

    print(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds")

    # Save the LoRA adapter
    print(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Export to GGUF if requested
    if export_gguf:
        print(f"Exporting to GGUF format ({quantization_method})...")
        gguf_path = f"{output_dir}/medical-llm-{quantization_method}.gguf"

        # Merge LoRA and export
        model.save_pretrained_gguf(
            output_dir,
            tokenizer,
            quantization_method=quantization_method,
        )
        print(f"GGUF model saved to {gguf_path}")

    print("Done!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a medical LLM with Unsloth")

    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="medmcqa",
        help="Dataset name or path to JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/medical_adapter",
        help="Output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for full epochs)",
    )
    parser.add_argument(
        "--no_gguf",
        action="store_true",
        help="Skip GGUF export",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="GGUF quantization method",
    )

    args = parser.parse_args()

    train(
        base_model=args.base_model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        export_gguf=not args.no_gguf,
        quantization_method=args.quantization,
    )


if __name__ == "__main__":
    main()
