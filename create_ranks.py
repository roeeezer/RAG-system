#!/usr/bin/env python3
import json
import sys

def create_ranks_file(json_path: str, ranks_path: str):
    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # The queries are under the "queries" key
    queries = data["queries"]

    # Open the output file
    with open(ranks_path, 'w', encoding='utf-8') as out_file:
        for query in queries:
            query_id = query["number"]
            rank_1_based = query.get("rank")  # Might be None if not found
            
            # Convert to 0-based rank
            if rank_1_based is None:
                # Gold doc not found in top K => treat as beyond 1000
                rank_0_based = 1000  
            else:
                rank_0_based = rank_1_based - 1  # Convert from 1-based to 0-based

            # If the 0-based rank is larger than 1000, cap it at 1000
            if rank_0_based > 1000:
                rank_0_based = 1000

            # Write to file as: QUERY_ID RANK
            out_file.write(f"{query_id} {rank_0_based}\n")


if __name__ == "__main__":
    """
    Usage:
        python create_ranks.py path/to/results.json path/to/ranks.txt
    """
    if len(sys.argv) != 3:
        print("Usage: python create_ranks.py <path_to_json> <path_to_ranks_txt>")
        sys.exit(1)

    json_path = sys.argv[1]
    ranks_path = sys.argv[2]

    create_ranks_file(json_path, ranks_path)
    print(f"ranks.txt created at: {ranks_path}")
