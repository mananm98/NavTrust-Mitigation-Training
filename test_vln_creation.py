#!/usr/bin/env python3
"""
Test the create_vln_api_inferred function with mock data.
"""

import json
from create_vln_api_inferred import create_vln_api_inferred

def test_vln_creation():
    """Test VLN creation with mock input_to_output mapping"""

    # Load a few VLN episodes to see what we're working with
    with open("vln_data_expert.json", "r") as f:
        vln_data = json.load(f)

    print("=== VLN Data Structure ===")
    print(f"Total episodes: {len(vln_data['episodes'])}")

    # Show first episode structure
    first_episode = vln_data["episodes"][0]
    print(f"\nFirst episode keys: {list(first_episode.keys())}")
    print(f"Instruction keys: {list(first_episode['instruction'].keys())}")
    print(f"Original instruction: {first_episode['instruction']['instruction_text'][:100]}...")

    # Create mock input_to_output mapping for testing
    # Let's say we have API responses for 2-3 instructions
    mock_input_to_output = {}

    # Take first few episodes and create mock mappings
    for i in range(min(3, len(vln_data["episodes"]))):
        episode = vln_data["episodes"][i]
        original_text = episode["instruction"]["instruction_text"]

        # Create a mock API response (shortened version)
        mock_api_response = f"Go to location {i+1}.\nTurn right.\nStop."
        mock_input_to_output[original_text.strip()] = mock_api_response

    print(f"\n=== Mock API Responses ===")
    print(f"Created {len(mock_input_to_output)} mock mappings")
    for i, (original, mock_response) in enumerate(mock_input_to_output.items()):
        print(f"\nMapping {i+1}:")
        print(f"  Original: {original[:80]}...")
        print(f"  Mock API: {mock_response}")

    # Test the function
    print(f"\n=== Testing create_vln_api_inferred ===")
    try:
        result = create_vln_api_inferred(mock_input_to_output)

        print(f"\n✅ Success! Created file with {len(result['episodes'])} episodes")

        # Show some examples of what changed
        print(f"\n=== Verification ===")
        for i in range(min(3, len(result["episodes"]))):
            episode = result["episodes"][i]
            original_episode = vln_data["episodes"][i]

            original_text = original_episode["instruction"]["instruction_text"]
            new_text = episode["instruction"]["instruction_text"]

            if original_text != new_text:
                print(f"\nEpisode {i+1} - CHANGED:")
                print(f"  Before: {original_text[:60]}...")
                print(f"  After:  {new_text}")
            else:
                print(f"\nEpisode {i+1} - UNCHANGED (no API mapping)")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_vln_creation()