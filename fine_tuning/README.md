# Medical LLM Fine-Tuning with Unsloth

This module provides tools for fine-tuning LLMs on medical data using [Unsloth](https://github.com/unslothai/unsloth) for efficient training.

## Why Unsloth?

- **2x faster** training than standard HuggingFace
- **80% less memory** usage with 4-bit quantization
- **No accuracy loss** compared to full precision
- Easy export to **GGUF format** for llama.cpp deployment

## Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment
python -m venv venv-finetune
source venv-finetune/bin/activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# Install other dependencies
pip install -r requirements-unsloth.txt
```

### 2. Prepare Your Data

Create a JSON file with question-answer pairs:

```json
[
  {
    "question": "What are the symptoms of diabetes?",
    "answer": "Common symptoms include increased thirst, frequent urination..."
  }
]
```

See `sample_training_data.json` for examples.

### 3. Run Training

```bash
# Using a public medical dataset
python train_medical_llm.py \
    --base_model unsloth/Llama-3.2-1B-Instruct \
    --dataset medmcqa \
    --output_dir ./models/medical_adapter \
    --epochs 1

# Using your own data
python train_medical_llm.py \
    --base_model unsloth/Llama-3.2-1B-Instruct \
    --dataset path/to/your/data.json \
    --output_dir ./models/medical_adapter
```

### 4. Use the Fine-Tuned Model

#### Option A: With llama.cpp (Recommended)

The training script exports a GGUF file that can be used with llama.cpp:

```bash
# Start llama.cpp server with the fine-tuned model
llama-server -m ./models/medical_adapter/medical-llm-q4_k_m.gguf --port 8080
```

Then configure the OpenEMR LLM module to use llama.cpp backend.

#### Option B: With the LoRA Adapter

Set in `.env`:
```env
LLM_BACKEND=huggingface
HUGGINGFACE_MODEL=unsloth/Llama-3.2-1B-Instruct
USE_FINETUNED_MODEL=true
FINETUNED_ADAPTER_PATH=./models/medical_adapter
```

## Available Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| `medmcqa` | Medical multiple choice QA | ~180K |
| `pubmedqa` | PubMed question answering | ~1K labeled |
| `medqa` | Medical licensing exam questions | ~10K |

## Recommended Base Models

| Model | Parameters | Notes |
|-------|------------|-------|
| `unsloth/Llama-3.2-1B-Instruct` | 1B | Fast, good for testing |
| `unsloth/Llama-3.2-3B-Instruct` | 3B | Good balance |
| `unsloth/Mistral-7B-Instruct-v0.2` | 7B | High quality |
| `unsloth/Phi-3-mini-4k-instruct` | 3.8B | Microsoft's efficient model |

## Training on Google Colab (Free GPU)

1. Open Google Colab
2. Upload the `train_medical_llm.py` script
3. Run:

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# Run training
!python train_medical_llm.py --max_steps 100  # Quick test
```

## Tips for Better Results

1. **Data Quality**: Ensure your training data is accurate and well-formatted
2. **Data Diversity**: Include various medical topics and question types
3. **Epochs**: More epochs = better memorization, but watch for overfitting
4. **Learning Rate**: Start with 2e-4, reduce if loss is unstable
5. **LoRA Rank**: Higher rank (32, 64) for more complex adaptations

## Export Options

| Format | Use Case |
|--------|----------|
| `q4_k_m` | Best balance of size/quality for llama.cpp |
| `q5_k_m` | Higher quality, larger file |
| `q8_0` | Near full precision |
| `f16` | Full precision, largest file |

## Directory Structure

```
fine_tuning/
├── __init__.py
├── README.md
├── requirements-unsloth.txt
├── train_medical_llm.py      # Main training script
└── sample_training_data.json # Example data format
```

## Disclaimer

Fine-tuned medical models should be thoroughly validated before any clinical use. These tools are intended for research and educational purposes. Always consult healthcare professionals for medical decisions.
