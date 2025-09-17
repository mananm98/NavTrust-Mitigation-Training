#!/usr/bin/env python3
"""
Create VLN data with OpenAI API inferred instructions.
Replaces expert instructions with OpenAI-generated outputs.
"""

import json
import os
import openai
from prompt_templates import format_inference_prompt, SYSTEM_PROMPT, api_call_inference_prompt
import time

def load_expert_data():
    """Load expert records from finetuning dataset"""
    with open("vln_data_expert.json", "r") as f:
        data = json.load(f)

    expert_records = data["episodes"]
    print(f"Loaded {len(expert_records)} expert records")
    return expert_records

def setup_openai_client():
    """Setup OpenAI client with API key"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    openai.api_key = api_key
    return True

def call_openai_api(client, input_text, model="gpt-4o", max_retries=5):
    """Call OpenAI API with navigation instruction and retry with exponential backoff"""

    prompt = api_call_inference_prompt(input_text)

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200,
            )

            return response.choices[0].message.content.strip()

        except openai.error.RateLimitError as e:
            wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 5, 9, 17 seconds
            print(f"  Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:  # Last attempt
                return None
            wait_time = 2  # Short wait for other errors
            time.sleep(wait_time)

    print(f"  Failed after {max_retries} attempts")
    return None

def generate_api_responses(expert_records, model="gpt-4o"):
    """Generate API responses for all expert inputs"""
    setup_openai_client()
    input_to_output = {}

    print(f"Generating responses using {model}...")

    for i, record in enumerate(expert_records):
        input_text = record["instruction"]["instruction_text"]

        print(f"Processing {i+1}/{len(expert_records)}: {input_text[:60]}...")

        # Call OpenAI API
        api_output = call_openai_api(None, input_text, model)

        if api_output:
            input_to_output[input_text] = api_output
            print(f"  -> Generated: {api_output[:60]}...")
        else:
            print(f"  -> Failed to generate response")

        # Rate limiting to stay under 10k TPM limit
        time.sleep(3)

    print(f"Generated {len(input_to_output)} API responses")
    return input_to_output

def create_vln_api_inferred(input_to_output):
    """Create VLN data with API-inferred instructions"""
    # Load original VLN expert data
    with open("vln_data_expert.json", "r") as f:
        vln_data = json.load(f)

    # Create new data structure
    vln_api_inferred = {
        "episodes": []
    }

    episodes_with_api_predictions = 0

    for episode in vln_data["episodes"]:
        original_text = episode["instruction"]["instruction_text"]

        # Create new episode with dict unpacking (no .copy() needed)
        new_episode = {
            **episode,
            "instruction": {
                **episode["instruction"],
                "instruction_text": input_to_output.get(original_text.strip(), original_text)
            }
        }

        if original_text.strip() in input_to_output:
            episodes_with_api_predictions += 1

        vln_api_inferred["episodes"].append(new_episode)

    # Save the new file
    with open("vln_data_expert_API_call_inferred.json", "w") as f:
        json.dump(vln_api_inferred, f, indent=2)

    print(f"Created vln_data_expert_API_call_inferred.json")
    print(f"Total episodes: {len(vln_api_inferred['episodes'])}")
    print(f"Episodes with API predictions: {episodes_with_api_predictions}")

    return vln_api_inferred

def main():
    """Main pipeline"""
    print("=== Creating VLN Data with OpenAI API Inference ===")

    # Load expert data
    expert_records = load_expert_data()

    # Generate API responses
    input_to_output = generate_api_responses(expert_records)

    # Create VLN file with API inferences
    vln_api_inferred = create_vln_api_inferred(input_to_output)

    print("\n=== Pipeline Complete ===")
    print("Files created:")
    print("  - vln_data_expert_API_call_inferred.json")

if __name__ == "__main__":
    main()