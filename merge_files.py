import json
import sys

def merge_jsonl_files(input_files, output_file):
    with open(output_file, 'w') as outfile:
        for input_file in input_files:
            with open(input_file, 'r') as infile:
                for line in infile:
                    # Skip empty lines
                    if line.strip():
                        # Verify it's valid JSON
                        try:
                            json.loads(line)
                            outfile.write(line)
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line in {input_file}")

if __name__ == "__main__":
    input_files = ["results/2024-11-20.jsonl", "results/2024-11-21.jsonl", "results/2024-11-21b.jsonl"]
    output_file = "results/merged.jsonl"
    merge_jsonl_files(input_files, output_file)