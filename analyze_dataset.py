import os
import sys
import json
from collections import Counter
from datasets import load_dataset

# Ensure UTF-8 output handling for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def analyze_dataset(sample_count=500, num_samples_to_save=10):
    dataset_name = "ai4bharat/MSMARCO-XI"
    print(f"Streaming first {sample_count} examples from '{dataset_name}' (train split)...")
    
    # 1. Load dataset using HF streaming mode
    ds = load_dataset(dataset_name, split="train", streaming=True)
    
    # Take sample_count records
    raw_samples = list(ds.take(sample_count))
    total_read = len(raw_samples)
    print(f"Successfully read {total_read} examples.\n")
    
    # Data structures for analysis
    source_langs = Counter()
    target_langs = Counter()
    query_types = Counter()
    
    query_lengths_char = []
    query_lengths_word = []
    answer_lengths_char = []
    answer_lengths_word = []
    
    passages_per_example = []
    total_selected_passages = 0
    examples_with_selected_passage = 0
    
    clean_samples_for_json = []
    
    for i, sample in enumerate(raw_samples):
        src_lang = sample.get("source_lang", "unknown")
        tgt_lang = sample.get("target_lang", "unknown")
        q_type = sample.get("query_type", "unknown")
        
        source_langs[src_lang] += 1
        target_langs[tgt_lang] += 1
        query_types[q_type] += 1
        
        query_text = sample.get("query") or ""
        answer_text = sample.get("Answer") or ""
        
        query_lengths_char.append(len(query_text))
        query_lengths_word.append(len(query_text.split()))
        
        answer_lengths_char.append(len(answer_text))
        answer_lengths_word.append(len(answer_text.split()))
        
        # Passage inspection
        passages = sample.get("passages", {})
        eng_passages = passages.get("English_passages", [])
        trans_passages = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])
        
        num_passages = max(len(eng_passages), len(trans_passages), len(is_selected))
        passages_per_example.append(num_passages)
        
        selected_count = sum(1 for val in is_selected if val == 1 or val is True)
        total_selected_passages += selected_count
        if selected_count > 0:
            examples_with_selected_passage += 1
            
        # Collect top 10 representative samples for JSON export
        if i < num_samples_to_save:
            clean_samples_for_json.append({
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "query": query_text,
                "Answer": answer_text,
                "passages": passages,
                "query_type": q_type
            })

    # Calculations
    avg_query_char = sum(query_lengths_char) / total_read if total_read else 0
    avg_query_word = sum(query_lengths_word) / total_read if total_read else 0
    
    avg_answer_char = sum(answer_lengths_char) / total_read if total_read else 0
    avg_answer_word = sum(answer_lengths_word) / total_read if total_read else 0
    
    avg_passages_per_ex = sum(passages_per_example) / total_read if total_read else 0
    pct_examples_with_selected = (examples_with_selected_passage / total_read * 100) if total_read else 0
    
    # 2. Build analysis report
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("   MSMARCO-XI DATASET ANALYSIS (500 SAMPLES)      ")
    report_lines.append("==================================================")
    report_lines.append(f"Examples Analyzed: {total_read}\n")
    
    report_lines.append("--- 1. Languages Discovered ---")
    report_lines.append(f"Source Languages: {dict(source_langs)}")
    report_lines.append(f"Target Languages: {dict(target_langs)}\n")
    
    report_lines.append("--- 2. Query Type Distribution ---")
    for qtype, count in query_types.most_common():
        pct = (count / total_read) * 100
        report_lines.append(f"  - {qtype}: {count} ({pct:.1f}%)")
    report_lines.append("")
    
    report_lines.append("--- 3. Text Length Statistics ---")
    report_lines.append(f"Average Query Length : {avg_query_char:.1f} characters ({avg_query_word:.1f} words)")
    report_lines.append(f"Average Answer Length: {avg_answer_char:.1f} characters ({avg_answer_word:.1f} words)\n")
    
    report_lines.append("--- 4. Passages & Relevance Statistics ---")
    report_lines.append(f"Average Passages per Example   : {avg_passages_per_ex:.1f}")
    report_lines.append(f"Total Selected (Relevant) Passages: {total_selected_passages}")
    report_lines.append(f"Examples with >=1 Selected Passage: {examples_with_selected_passage} of {total_read} ({pct_examples_with_selected:.1f}%)\n")
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # 3. Create 'data' directory if not exists
    os.makedirs("data", exist_ok=True)
    
    # 4. Save JSON sample records (ensure_ascii=False preserves non-ASCII Indic scripts)
    json_path = os.path.join("data", "sample_records.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_samples_for_json, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(clean_samples_for_json)} sample records to '{json_path}'.")
    
    # 5. Save Analysis Report text file
    txt_path = os.path.join("data", "dataset_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved analysis report to '{txt_path}'.")

if __name__ == "__main__":
    analyze_dataset(sample_count=500, num_samples_to_save=10)
