from fetch import list_ai_matcher_files, download_file_from_space
from model import analyze_file_with_ai, match_texts, benchmark_methods, export_matches_to_csv
import os

def process_files(limit=10):
    """
    Downloads and analyzes up to 'limit' files from the AI_matcher space.
    Returns a dictionary of extracted texts.
    """
    files = list_ai_matcher_files()
    extracted_texts = {}
    for file_key in files if limit is None else files[:limit]:
        content = download_file_from_space(file_key)
        key, text = analyze_file_with_ai(content, file_key)
        if text:
            extracted_texts[key] = text

    return extracted_texts

def main():
    extracted_texts = process_files()

    df_matches = match_texts(extracted_texts, threshold=70.0)
    if not df_matches.empty:
        os.makedirs("output", exist_ok=True)
        output_file = "output/matches_with_text.csv"
        export_matches_to_csv(df_matches, extracted_texts, output_file)
    else:
        print("[MAIN] No similar documents found.")

    benchmark_methods(extracted_texts, threshold=70.0)

if __name__ == "__main__":
    main()

