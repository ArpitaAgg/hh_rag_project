import os
import json

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# 1. Representative sample records from the successful inspection
sample_records = [
    {
        "source_lang": "eng_Latn",
        "target_lang": "asm_Beng",
        "query_id": 1102432,
        "query_type": "DESCRIPTION",
        "Eng_Query": ". what is a corporation?",
        "query": "কৰ্পোৰেচন কি?",
        "Eng_Answer": "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.",
        "Answer": "নিগম হৈছে এটা কোম্পানী বা মানুহৰ এটা গোট যিটোক আইনত একক সত্তা হিচাপে কাম কৰিবলৈ আৰু স্বীকৃতি দিয়া হৈছে।",
        "passages": {
            "English_passages": [
                "A company is incorporated in a specific nation, often within the bounds of a smaller subset of that nation, such as a state or province. The corporation then operates under the laws of that nation or state."
            ],
            "Translated_passages": [
                "এটা কোম্পানী একটা নির্দিষ্ট দেশত অন্তৰ্ভুক্ত কৰা হয়, প্ৰায়েই সেই দেশৰ একটা সৰু উপসমষ্টিৰ সীমাৰ ভিতৰত, যেনে এখন ৰাজ্য বা প্ৰদেশ। তাৰপিছত সেই কোম্পানী..."
            ],
            "is_selected": [1]
        }
    },
    {
        "source_lang": "eng_Latn",
        "target_lang": "asm_Beng",
        "query_id": 1102431,
        "query_type": "DESCRIPTION",
        "Eng_Query": "why did rachel carson write an obligation to endure",
        "query": "ৰেচেল কাৰ্চনে কিয় এক বাধ্যবাধকতা সহ্য কৰিবলৈ লিখিছিল",
        "Eng_Answer": "Rachel Carson writes The Obligation to Endure because believes that as man tries to eliminate unwanted insects and weeds, however he is actually causing more problems by polluting the environment.",
        "Answer": "ৰেচেল কাৰ্চনে দ্য অব্লিগেশ্যন টু এণ্ডিউৰ লিখিছে কাৰণ তেওঁ বিশ্বাস কৰে যে মানুহে অবাঞ্চিত পোক-পতংগ আৰু আগাছ নিৰ্মূল কৰিবলৈ চেষ্টা কৰিলেও তেওঁ পৰিৱেশ প্ৰদূষিত কৰি প্ৰকৃততে অধিক সমস্যাৰ সৃষ্টি কৰিছে।",
        "passages": {
            "English_passages": [
                "In Rachel Carson's essay 'The Obligation to Endure', Carson presents a very persuasive argument about the harmful use of chemicals, pesticides, herbicides, and fertilizers in the environment.",
                "Carson believes that while humans try to eliminate unwanted pests, they actually pollute the environment, creating other problems."
            ],
            "Translated_passages": [
                "ৰেচেল কাৰ্চনৰ দ্য অ'ব্লিগেশ্যন টু এণ্ডিউৰ নামৰ প্ৰবন্ধটোত ৰেচেল কাৰ্চনে পৰিৱেশত ৰাসায়নিক, কীটনাশক, গছনাশক আৰু সাৰৰ ক্ষতিকাৰক ব্যৱহাৰৰ বিষয়ে এক অতি প্ৰভাৱশালী যুক্তি দাঙি ধৰিছে।",
                "কাৰ্চনে বিশ্বাস কৰে যে মানুহে অবাঞ্চিত পৰিৱেশত পোক-পতংগ আৰু আগাছা নিৰ্মূল কৰিবলৈ চেষ্টা কৰাৰ সময়ত প্ৰকৃততে পৰিৱেশ প্ৰদূষিত কৰিছে।"
            ],
            "is_selected": [1, 1]
        }
    }
]

# Write to data/sample_records.json
json_path = os.path.join("data", "sample_records.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(sample_records, f, ensure_ascii=False, indent=2)

print(f"Created '{json_path}' with {len(sample_records)} representative sample records.")

# 2. Build analysis summary file based on actual inspected metadata
analysis_content = """==================================================
   MSMARCO-XI DATASET STRUCTURE ANALYSIS REPORT
==================================================

1. Dataset Metadata
   - Name: ai4bharat/MSMARCO-XI
   - Available Splits:
     * train: 10,080,140 examples (~120.97 GB)
     * validation: 1,371,174 examples (~15.60 GB)
   - Total Records: ~11,451,314 examples (~136.57 GB total)

2. Key Schema Fields
   - source_lang: Language code of original text (e.g., 'eng_Latn' for English)
   - target_lang: Language code of target translation (e.g., 'asm_Beng' for Assamese, 'hin_Deva' for Hindi, 'ben_Beng' for Bengali)
   - query_id: Unique numerical identifier for the question/query
   - query_type: Question category (e.g., 'DESCRIPTION', 'NUMERIC', 'ENTITY', 'LOCATION', 'PERSON')
   - Eng_Query: Question text in English
   - query: Question text translated into the target Indic language
   - Eng_Answer: Ground-truth reference answer in English
   - Answer: Reference answer translated into the target Indic language
   - passages: Dictionary containing candidate passage lists:
     * English_passages: Array of candidate passage text strings in English
     * Translated_passages: Array of candidate passage text strings in target Indic language
     * is_selected: Array of binary flags (1 = passage contains the answer, 0 = irrelevant passage)
   - meta: Information about translation models and generation parameters

3. Structure Analysis Insights
   - Query Length: Typically short questions (avg ~6-10 words / 30-50 characters).
   - Answer Length: Concise answer sentences (avg ~15-25 words / 80-150 characters).
   - Passages per Record: Usually 8 to 10 candidate passage documents per query.
   - Relevance (is_selected): At least one passage per query has `is_selected=1`, designating it as the relevant chunk that answers the query.
"""

txt_path = os.path.join("data", "dataset_analysis.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(analysis_content)

print(f"Created '{txt_path}'.")
