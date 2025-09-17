#!/usr/bin/env python3
"""
Test the call_openai_api function with your updated system prompt.
"""

import json
import os
from create_vln_api_inferred import call_openai_api, setup_openai_client

def test_api_call():
    """Test the OpenAI API call with a sample expert input"""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Set it with:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    # Load a sample expert record
    with open("finetuning_dataset_corrected.json", "r") as f:
        data = json.load(f)

    expert_records = [r for r in data["data"] if r["type"] == "expert"]

    # Test with first expert record
    sample_record = expert_records[0]
    input_text = sample_record["input"]
    expected_output = sample_record["output"]

    print("=== Testing OpenAI API Call ===")
    print(f"\nInput: {input_text}")
    print(f"\nExpected output: {expected_output}")

    try:
        setup_openai_client()
        api_output = call_openai_api(None, input_text)

        if api_output:
            print(f"\nAPI Output:\n{api_output}")
        else:
            print("\nAPI call failed - no output returned")

    except Exception as e:
        print(f"\nError during API call: {e}")

if __name__ == "__main__":
    test_api_call()