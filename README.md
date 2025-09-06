# Robust Navigation Instruction Tuning

This repository contains code for fine-tuning large language models on navigation instructions with robustness to malicious inputs.

## Setup

1. Create a conda environment:
```bash
conda create -n finetune python=3.10
conda activate finetune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a cache directory for models:
```bash
mkdir -p model_cache
```

## Files

- `robust_instruct.py`: Main training script
- `model_config.py`: Model configurations and factory
- `prompt_templates.py`: Prompt templates for training and inference
- `finetuning_dataset_corrected.json`: Training dataset
- `vln_data_expert.json`: Expert data for VLN
- `vln_data_expert_inferred.json`: Inferred data (generated during training)

## Usage

1. List available models:
```bash
python robust_instruct.py --list-models
```

2. Train a model:
```bash
python robust_instruct.py --model qwen2.5-7b --output-dir ./ft-r2r-qwen2.5-7b
```

3. Run evaluation only:
```bash
python robust_instruct.py --model qwen2.5-7b --skip-training --checkpoint-dir ./ft-r2r-qwen2.5-7b
```

## Model Options

Currently supported models:
- qwen2.5-14b (Qwen2.5-14B-Instruct)
- qwen2.5-7b (Qwen2.5-7B-Instruct)
- llama3.1-8b (Llama-3.1-8B-Instruct)
- llama3.2-3b (Llama-3.2-3B-Instruct)
- mistral-7b (Mistral-7B-Instruct-v0.3)

## Output

The training script will:
1. Fine-tune the model using LoRA
2. Save the trained model to the specified output directory
3. Generate evaluation results and inferred VLN data
4. Save detailed metrics and predictions
