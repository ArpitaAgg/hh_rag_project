import sys
import io
from datasets import load_dataset_builder, load_dataset

# Ensure UTF-8 output handling for Windows consoles (to support Indic scripts)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def format_sample(sample, index):
    """Format a single dataset record into a readable string."""
    lines = [f"--- Sample Record #{index} ---"]
    lines.append(f"Source Language  : {sample.get('source_lang')}")
    lines.append(f"Target Language  : {sample.get('target_lang')}")
    lines.append(f"Query ID         : {sample.get('query_id')}")
    lines.append(f"Query Type       : {sample.get('query_type')}")
    lines.append(f"Original Query (English)   : {sample.get('Eng_Query')}")
    lines.append(f"Translated Query (Indic)   : {sample.get('query')}")
    lines.append(f"Original Answer (English)  : {sample.get('Eng_Answer')}")
    lines.append(f"Translated Answer (Indic)  : {sample.get('Answer')}")
    
    passages = sample.get('passages', {})
    eng_passages = passages.get('English_passages', [])
    trans_passages = passages.get('Translated_passages', [])
    is_selected = passages.get('is_selected', [])
    
    lines.append(f"Number of Candidate Passages: {len(eng_passages)}")
    if eng_passages:
        # Show first passage preview
        selected_flag = is_selected[0] if is_selected else 0
        lines.append(f"  Passage [0] (Selected: {selected_flag}):")
        lines.append(f"    English: {eng_passages[0][:150]}...")
        if trans_passages:
            lines.append(f"    Indic  : {trans_passages[0][:150]}...")
    
    lines.append("")
    return "\n".join(lines)

def inspect_dataset():
    dataset_name = "ai4bharat/MSMARCO-XI"
    print(f"Loading metadata for dataset: {dataset_name} ...")
    
    # 1. Load Metadata (Builder pattern avoids downloading the ~129GB dataset files)
    builder = load_dataset_builder(dataset_name)
    info = builder.info
    
    output_lines = []
    output_lines.append("==========================================")
    output_lines.append("    DATASET INSPECTION REPORT             ")
    output_lines.append("==========================================")
    output_lines.append(f"Dataset Name: {dataset_name}\n")
    
    # 2. Available Splits & Counts
    output_lines.append("--- Available Splits & Example Counts ---")
    splits = info.splits if info.splits else {}
    for split_name, split_info in splits.items():
        output_lines.append(f"  - Split '{split_name}': {split_info.num_examples:,} examples ({split_info.num_bytes / (1024**3):.2f} GB)")
    output_lines.append("")
    
    # 3. Column Names & Schema
    output_lines.append("--- Column Names & Features ---")
    features = info.features if info.features else {}
    for col_name, feature_type in features.items():
        output_lines.append(f"  - {col_name}: {feature_type}")
    output_lines.append("")
    
    # 4. Stream 2-3 Sample Records (streaming=True loads records on-the-fly without full download)
    output_lines.append("--- Sample Records (Fetched via Streaming) ---")
    try:
        stream_ds = load_dataset(dataset_name, split="validation", streaming=True)
        samples = list(stream_ds.take(2))
        for i, sample in enumerate(samples, start=1):
            output_lines.append(format_sample(sample, i))
    except Exception as e:
        output_lines.append(f"Could not fetch streaming samples: {e}\n")
    
    output_text = "\n".join(output_lines)
    
    # Print to Console
    print(output_text)
    
    # 5. Save to dataset_info.txt (with UTF-8 encoding for Indic characters)
    output_file = "dataset_info.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    
    print(f"\nDataset inspection report saved to '{output_file}'.")

if __name__ == "__main__":
    inspect_dataset()
