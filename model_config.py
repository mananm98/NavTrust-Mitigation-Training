"""
Model configuration and factory for easy model swapping in the robust instruction tuning pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig


@dataclass
class ModelConfig:
    """Configuration class for model settings"""
    
    # Model identification
    name: str
    model_id: str
    
    # Model loading parameters
    torch_dtype: str = "auto"
    attn_implementation: Optional[str] = "flash_attention_2"
    device_map: str = "auto"
    cache_dir: Optional[str] = None
    
    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    use_rslora: bool = True
    
    # Training configuration
    num_train_epochs: int = 8
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.15
    max_length: int = 1024
    
    # Training optimization
    bf16: bool = True
    gradient_checkpointing: bool = True
    weight_decay: float = 0.01
    optim: str = "adamw_torch_fused"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    max_grad_norm: float = 1.0
    
    # Evaluation settings
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Inference settings
    max_new_tokens: int = 256
    do_sample: bool = False
    inference_batch_size: int = 4
    
    def to_lora_config(self) -> LoraConfig:
        """Convert to LoRA configuration object"""
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            target_modules=self.lora_target_modules,
            use_rslora=self.use_rslora,
            task_type="CAUSAL_LM",
        )
    
    def get_training_args_dict(self, output_dir: str) -> Dict[str, Any]:
        """Get training arguments as dictionary"""
        return {
            "output_dir": output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "logging_steps": 10,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "bf16": self.bf16,
            "dataset_text_field": "text",
            "max_length": self.max_length,
            "packing": True,
            "dataloader_num_workers": 4,
            "gradient_checkpointing": self.gradient_checkpointing,
            "weight_decay": self.weight_decay,
            "optim": self.optim,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "max_grad_norm": self.max_grad_norm,
            "dataloader_drop_last": True,
        }
    
    @classmethod
    def from_json(cls, json_path: str) -> "ModelConfig":
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class ModelFactory:
    """Factory class for creating models and tokenizers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
    
    def get_tokenizer(self):
        """Get or create tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                use_fast=True,
                cache_dir=self.config.cache_dir
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer
    
    def get_model(self):
        """Get or create model"""
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=self.config.torch_dtype,
                attn_implementation=self.config.attn_implementation,
                device_map=self.config.device_map,
                cache_dir=self.config.cache_dir
            )
        return self._model
    
    def create_trainer_components(self, train_dataset, eval_dataset, output_dir: str):
        """Create all components needed for SFT training"""
        from trl import SFTTrainer, SFTConfig
        
        model = self.get_model()
        tokenizer = self.get_tokenizer()
        peft_config = self.config.to_lora_config()
        training_args = SFTConfig(**self.config.get_training_args_dict(output_dir))
        
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=training_args,
        )
        
        return trainer, model, tokenizer
    
    def inference_batch(self, inputs: List[str], prompt_template_fn=None) -> List[str]:
        """Batch inference with the model"""
        import torch
        
        model = self.get_model()
        tokenizer = self.get_tokenizer()
        model.eval()
        
        results = []
        batch_size = self.config.inference_batch_size
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            
            # Apply prompt template if provided
            if prompt_template_fn:
                prompts = [prompt_template_fn(x) for x in batch_inputs]
            else:
                prompts = batch_inputs
            
            # Tokenize batch
            encoded = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True,
                padding_side="left", 
                truncation=True, 
                max_length=2048
            )
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            
            # Generate batch
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode batch
            for j, output in enumerate(outputs):
                # Remove input tokens to get only generated part
                input_len = encoded['input_ids'][j].shape[0]
                generated = output[input_len:]
                decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
                results.append(decoded)
            
            print(f"Processed {min(i+batch_size, len(inputs))}/{len(inputs)} samples")
        
        return results


# Predefined model configurations with optimized hyperparameters
PREDEFINED_CONFIGS = {
    "qwen2.5-14b": ModelConfig(
        name="Qwen2.5-14B-Instruct",
        model_id="Qwen/Qwen2.5-14B-Instruct",
        # Optimized for large model: higher rank for more capacity
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        per_device_train_batch_size=1,  # Reduced for memory
        gradient_accumulation_steps=16,  # Increased to maintain effective batch size
        learning_rate=2e-5,  # Lower LR for large models
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=6,  # Fewer epochs for large models
        gradient_checkpointing=True,
        weight_decay=0.01,
        max_grad_norm=1.0,
        inference_batch_size=2,
    ),
    
    "qwen2.5-7b": ModelConfig(
        name="Qwen2.5-7B-Instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        # Balanced configuration for 7B model
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        num_train_epochs=8,
        gradient_checkpointing=True,
        weight_decay=0.01,
        max_grad_norm=1.0,
        inference_batch_size=4,
    ),
    
    "llama3.1-8b": ModelConfig(
        name="Llama-3.1-8B-Instruct",
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        # Optimized for Llama architecture
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,  # Lower LR for Llama stability
        lr_scheduler_type="cosine",
        warmup_ratio=0.2,  # More warmup for Llama
        num_train_epochs=8,
        gradient_checkpointing=True,
        weight_decay=0.01,
        max_grad_norm=0.5,  # Lower grad norm for stability
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_batch_size=4,
    ),
    
    "llama3.2-3b": ModelConfig(
        name="Llama-3.2-3B-Instruct",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        # Optimized for smaller 3B model - can use higher LR and more aggressive training
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,  # Higher LR for smaller model
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=12,  # More epochs for smaller model
        gradient_checkpointing=False,  # Not needed for 3B
        weight_decay=0.01,
        max_grad_norm=1.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_batch_size=8,  # Higher batch size for faster inference
    ),
    
    "mistral-7b": ModelConfig(
        name="Mistral-7B-Instruct-v0.3",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        # Optimized for Mistral architecture - known to train fast
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,  # Lower dropout for Mistral
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=4e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=10,  # Optimal for Mistral
        gradient_checkpointing=False,  # Mistral is efficient
        weight_decay=0.01,
        max_grad_norm=1.0,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_batch_size=6,
    ),
}


def get_model_config(model_name: str, cache_dir: Optional[str] = None) -> ModelConfig:
    """Get model configuration by name"""
    if model_name in PREDEFINED_CONFIGS:
        config = PREDEFINED_CONFIGS[model_name]
        if cache_dir:
            config.cache_dir = cache_dir
        return config
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(PREDEFINED_CONFIGS.keys())}")


def list_available_models() -> List[str]:
    """List all available predefined models"""
    return list(PREDEFINED_CONFIGS.keys())
