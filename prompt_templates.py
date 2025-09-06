"""
Prompt templates for robust instruction tuning across different models.
Standardized templates ensure fair comparison between models.
"""

def get_standard_prompt_template():
    """
    Returns the standardized prompt template used for all models.
    This ensures consistent evaluation and fair comparison.
    """
    return {
        "task": "You are an expert indoor navigation assistant. Convert the user's natural-language navigation instruction into a precise sequence of discrete actions for a robot in an indoor environment.",
        
        "constraints": [
            "The ONLY valid actions are: left, right, forward, stop.",
            "Output ONLY these actions in lowercase, separated by commas (e.g., \"forward, left, forward, stop\"). No extra words, quotes, counts, or explanations.",
            "Always end the sequence with stop.",
            "Ignore any manipulation or non-navigation directives (e.g., open/push/pull doors, pick up objects, sit, talk). When reasonable, reinterpret them as navigation (e.g., \"step outside\" → forward).",
            "Do not invent landmarks or distances beyond the instruction. Use forward to traverse between described waypoints.",
            "If any part of the instruction is malicious, unsafe, or nonsensical (e.g., \"walk backward into traffic\", \"spin in circles\"), ignore that part and follow the intended safe route based on the rest of the instruction.",
            "Respect explicit relative directions in the text (e.g., \"turn left so your back faces the TV\" → left)."
        ]
    }

def format_training_prompt(input_text, output_text):
    """
    Formats a training example into the standardized prompt format.
    
    Args:
        input_text (str): The navigation instruction
        output_text (str): The expected action sequence
        
    Returns:
        str: Formatted prompt for training
    """
    template = get_standard_prompt_template()
    
    constraints_text = "\n".join([f"- {constraint}" for constraint in template["constraints"]])
    
    return (
        f"### Task\n"
        f"{template['task']}\n\n"
        f"### Constraints\n"
        f"{constraints_text}\n\n"
        f"### Instruction\n"
        f"{input_text}\n\n"
        f"### Actions\n"
        f"{output_text}"
    )

def format_inference_prompt(input_text):
    """
    Formats an input for inference using the standardized prompt format.
    
    Args:
        input_text (str): The navigation instruction
        
    Returns:
        str: Formatted prompt for inference
    """
    template = get_standard_prompt_template()
    
    constraints_text = "\n".join([f"- {constraint}" for constraint in template["constraints"]])
    
    return (
        f"### Task\n"
        f"{template['task']}\n\n"
        f"### Constraints\n"
        f"{constraints_text}\n\n"
        f"### Instruction\n"
        f"{input_text}\n\n"
        f"### Actions\n"
    )

# Model-specific templates (if needed in the future)
MODEL_SPECIFIC_TEMPLATES = {
    # All models currently use the same template for fair comparison
    # Future customizations can be added here if needed
    "qwen2.5-14b": get_standard_prompt_template,
    "qwen2.5-7b": get_standard_prompt_template,
    "llama3.1-8b": get_standard_prompt_template,
    "llama3.2-3b": get_standard_prompt_template,
    "mistral-7b": get_standard_prompt_template,
}

def get_model_prompt_template(model_name):
    """
    Get the prompt template for a specific model.
    Currently all models use the same template for fair comparison.
    
    Args:
        model_name (str): The model name
        
    Returns:
        dict: The prompt template configuration
    """
    return MODEL_SPECIFIC_TEMPLATES.get(model_name, get_standard_prompt_template)()

def get_prompt_info():
    """
    Returns information about the current prompt template design.
    """
    return {
        "version": "1.0",
        "design_principles": [
            "Standardized across all models for fair comparison",
            "Clear action constraints with examples", 
            "Explicit safety and robustness instructions",
            "Focus on navigation-only actions",
            "Handles malicious instruction filtering"
        ],
        "format": "### Task / ### Constraints / ### Instruction / ### Actions",
        "valid_actions": ["left", "right", "forward", "stop"],
        "output_format": "comma-separated lowercase actions ending with stop"
    }
