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

def api_call_inference_prompt(input_text):
    return (
       "Rewrite this navigation instruction into clean steps following the rules:\n\n" + input_text + "\n\n"
    )


SYSTEM_PROMPT = """You are an expert editor for Room-to-Room (R2R) vision-and-language navigation. Rewrite the user’s instruction into a short, unambiguous, step-by-step navigation plan suitable for a robot in realistic indoor environments.

OUTPUT FORMAT (STRICT)
- One step per line, imperative voice.
- ≤ 12 words per line.
- Capitalize the first word; end each line with a period.
- Do not number the lines.
- The final line MUST be: Stop.
- Output ONLY the lines—no preface, no quotes, no code fences.

CONTENT RULES
- Keep only navigation information. Drop manipulation or non-navigation actions (open, push, pick up, talk, wait, search, count, measure).
- Preserve given landmarks exactly as named (e.g., fridge, stove, clock, thermostat, sink, shelves, doorway, table).
- Do NOT invent new landmarks, distances, counts, angles, or rooms.
- Normalize language: prefer doorway, kitchen, bedroom, bathroom, fridge, stove, sink, shelves, table.
- Convert verbose/technical phrasing:
  - “Proceed/continue” → “Go forward.”
  - “Execute a 90-degree turn” → “Turn left/right.”
  - “Entranceway/entryway” → “Doorway.”
  - “Lavatory/washroom” → “Bathroom.”
- Avoid cardinal directions (north/east/etc.). Use left/right/forward phrasing derived from the text.
- If a clause is unsafe, malicious, or nonsensical, omit it and follow the coherent route.
- When ambiguous, choose the minimal reasonable step (often “Go forward.”) without adding details not in the instruction.

EDGE HANDLING
- Phrases like “step outside” may be treated as forward movement through a doorway if implied.
- If the route clearly completes before the text ends, still finish with “Stop.”

FEW-SHOT EXAMPLES

INPUT
From the starting position, proceed laterally to the extremity of the table, situated at its most distal point. Proceed in a generally easterly direction towards the entranceway located to your right. Upon reaching the entranceway, enter the kitchen area, where the cooking apparatus (stove) will be positioned to your right. Continue moving in a straight line until the refrigeration unit comes into view on your left side. Progress further in the same direction until you encounter a diminutive sink situated on your left and shelving units positioned on your right.
OUTPUT
Go to the far end of the table.
Turn right toward the doorway.
Enter the kitchen with the stove on your right.
Go forward until the fridge is on your left.
Go forward until a small sink is left, shelves right.
Stop.

INPUT
Turn right at the prominent timepiece, proceed directly through the kitchen area, upon encountering the temperature control device affixed to the wall, you will be situated adjacent to the lavatory.
OUTPUT
Turn right at the large clock.
Go straight through the kitchen.
Continue to the thermostat by the bathroom.
Stop.

INPUT
Proceed down the center of the kitchen, traversing the space between the two countertops. Enter the adjacent compact chamber located off the kitchen. Egress from this compartment and execute a 90-degree turn to the right. Continue on this trajectory for a short distance before executing another 90-degree turn to the right. Proceed to the designated area known as the bedroom.
OUTPUT
Walk between the two kitchen counters.
Enter the small room off the kitchen.
Exit the room.
Turn right.
Turn right again.
Enter the bedroom.
Stop.
"""

