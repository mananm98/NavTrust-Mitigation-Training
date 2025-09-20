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

def load_data():
    """Load records from finetuning dataset by type, excluding malicious types"""
    with open("finetuning_dataset_corrected.json", "r") as f:
        data = json.load(f)

    # Separate data by type, excluding malicious types
    data_by_type = {}
    for record in data["data"]:
        record_type = record["type"]
        if not record_type.startswith("malicious"):
            if record_type not in data_by_type:
                data_by_type[record_type] = []
            data_by_type[record_type].append(record)

    # Print summary
    total_records = sum(len(records) for records in data_by_type.values())
    print(f"Loaded {total_records} records across {len(data_by_type)} types:")
    for record_type, records in data_by_type.items():
        print(f"  {record_type}: {len(records)} records")

    return data_by_type

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

def generate_api_responses(data_records, model="gpt-4o"):
    """Generate API responses for all expert inputs"""
    setup_openai_client()
    api_outputs_by_type = {}

    print(f"Generating responses using {model}...")

    for data_type, records in data_records.items():
        print(f"⭐️⭐️⭐️ Processing {data_type} Records ......")
        api_outputs_by_type[data_type] = {}
        
        for i, record in enumerate(records):
            instruction_id = record["instruction_id"]
            input_text = record["input"]

            print(f"Processing {i+1}/{len(records)}: {input_text[:60]}...")

            # Call OpenAI API
            api_output = call_openai_api(None, input_text, model)

            if api_output:
                print(f"  -> Generated: {api_output[:60]}...")
                api_outputs_by_type[data_type][instruction_id] = (input_text, api_output)
            else:
                print(f"  -> Failed to generate response")

        # Rate limiting to stay under 10k TPM limit
        time.sleep(3)

    return api_outputs_by_type

def create_vln_api_inferred(api_outputs_by_type):
    """Create VLN data with API-inferred instructions"""

    # Load original VLN expert data
    with open("vln_data_expert.json", "r") as f:
        vln_data = json.load(f)

    for data_type, input_to_output in api_outputs_by_type.items():
        print(f"⭐️⭐️⭐️ Creating vln file for the data_type {data_type}")
        # Create new data structure
        vln_api_inferred = {
            "episodes": []
        }

        for i, episode in enumerate(vln_data["episodes"]):
            instruction_id = i + 1

            # Create new episode with dict unpacking (no .copy() needed)
            new_episode = {
                **episode,
                "instruction": {
                    **episode["instruction"],
                    "instruction_text": input_to_output.get(instruction_id)[1]
                }
            }

            vln_api_inferred["episodes"].append(new_episode)

        # Save the new file
        with open(f"vln_data_{data_type}_API_call_inferred.json", "w") as f:
            json.dump(vln_api_inferred, f, indent=2)

        print(f"Created vln_data_{data_type}_API_call_inferred.json")
        print(f"Total episodes: {len(vln_api_inferred['episodes'])}")
        # print(f"Episodes with API predictions: {episodes_with_api_predictions}")

    return vln_api_inferred

def main():
    """Main pipeline"""
    print("=== Creating VLN Data with OpenAI API Inference ===")

    # Load expert data
    data_records = load_data()

    # Generate API responses
    api_outputs_by_type = generate_api_responses(data_records)

    # Create VLN file with API inferences
    vln_api_inferred = create_vln_api_inferred(api_outputs_by_type)

    print("\n=== Pipeline Complete ===")
if __name__ == "__main__":
    main()