#!/usr/bin/env python3
"""
Extract unique scenario IDs from INCORRECT_MedQAExt (1).txt file.
Outputs space-separated list ready for use with --scenario_ids parameter.
"""

import re
import sys

def extract_scenario_ids(filename):
    """Extract unique scenario IDs from the incorrect scenarios file."""
    scenario_ids = []
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
            # Find all "Scene X" patterns
            matches = re.findall(r'Scene (\d+)', content)
            scenario_ids = [int(m) for m in matches]
            unique_ids = sorted(set(scenario_ids))
            return unique_ids
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    filename = "INCORRECT_MedQAExt (1).txt"
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    unique_ids = extract_scenario_ids(filename)
    
    # Output space-separated for CLI use
    print(" ".join(map(str, unique_ids)))
    
    # Also print summary
    print(f"\n# Summary: {len(unique_ids)} unique scenario IDs extracted", file=sys.stderr)
