import argparse
import time
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import os

os.environ["HF_DATASETS_NO_TQDM"] = "1"
from datasets.utils.logging import set_verbosity_error

import re

def count_occurrences_multi(batch, entities_lower):
    """
    Count case-insensitive occurrences for each entity in entities_lower in every example.
    Counts only whole words using regular expressions.
    
    Parameters:
        batch (dict): A batch of examples with a "text" field.
        entities_lower (list of str): List of entities in lowercase.
        
    Returns:
        dict: A dictionary where each key is an entity and the value is a list of counts
              (one count per example in the batch).
    """
    # Initialize a dictionary for storing counts for each entity.
    results = {entity: [] for entity in entities_lower}
    
    # Precompile regex patterns for each entity to count whole word occurrences.
    patterns = {entity: re.compile(rf'\b{re.escape(entity)}\b') for entity in entities_lower}
    
    # Process each example in the batch.
    for text in batch["text"]:
        lower_text = text.lower()  # Lowercase the text once per example.
        # For each entity, count its occurrences as whole words.
        for entity in entities_lower:
            count = len(patterns[entity].findall(lower_text))
            results[entity].append(count)
    
    return results


def count_occurrences_in_dataset(dataset, entities, num_proc):
    """
    Count occurrences of multiple entities in the dataset by processing it only once.
    
    Parameters:
        dataset (Dataset or dict): The dataset or dictionary of splits.
        entities (list): List of entities (strings) to search for.
        num_proc (int): Number of processes for parallel processing.
        
    Returns:
        dict: A mapping of each entity (in lowercase) to its total count.
    """
    # Convert entities to lowercase once.
    # Convert to lowercase and remove duplicates.
    entities_lower = list(set(e.lower() for e in entities))

    
    # Process the dataset using .map() only one time.
    set_verbosity_error()
    processed_dataset = dataset.map(
        lambda batch: count_occurrences_multi(batch, entities_lower),
        batched=True,
        batch_size=100,
        num_proc=num_proc
    )
    
    # Now, for each entity, sum the counts across all examples.
    total_counts = {}
    for entity in entities_lower:
        total_counts[entity] = sum(processed_dataset[entity])
    
    return total_counts

def main(csv_path, num_proc, split):
    # Load the dataset once.
    print("Loading dataset...")
    if split:
        dataset = load_dataset("bookcorpus", "plain_text", split=split)
    else:
        dataset = load_dataset("bookcorpus", "plain_text")
    
    # Load the CSV containing entities.
    df = pd.read_csv(csv_path)
    df["Entity"] = df["Entity"].str.strip('"')
    entities = df["Entity"].tolist()
    
    # Process the dataset once and time the operation.
    print("Counting occurrences for all entities in one pass:")
    start_time = time.time()
    total_counts = count_occurrences_in_dataset(dataset, entities, num_proc)
    end_time = time.time()
    
    total_time = end_time - start_time
    num_examples = len(entities)
    avg_time_per_example = total_time / num_examples
    
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per example: {avg_time_per_example:.6f} seconds")
    
    # Map the counts back to the DataFrame.
    df["count"] = [total_counts[entity.lower()] for entity in entities]
    
    # Build output filename: output_{inputfilename}
    base_filename = os.path.basename(csv_path)
    output_csv = f"output_{base_filename}"
    
    df.to_csv(output_csv, index=False)
    print(f"\nOutput written to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Iterate over a CSV of entities and count occurrences in BookCorpus using parallel processing."
    )
    parser.add_argument("csv_path", help="Path to CSV file containing an 'Entity' column.")
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Number of processes to use for parallel processing (default: 4).")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process (e.g., 'train').")
    args = parser.parse_args()
    
    main(args.csv_path, args.num_proc, args.split)
