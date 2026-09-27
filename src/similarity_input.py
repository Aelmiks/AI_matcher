from model import extract_text, find_similar_to_input
from fetch import list_ai_matcher_files, download_file_from_space
import os

def load_existing_texts(limit=None):
    files = list_ai_matcher_files()
    extracted = {}
    for file_key in files if limit is None else files[:limit]:
        if not file_key.lower().endswith((".docx", ".pdf", ".jpg", ".jpeg", ".png")):
            continue  # ignore unsupported types like .txt
        content = download_file_from_space(file_key)
        text = extract_text(content, file_key)
        extracted[file_key] = text
    return extracted


def load_input_image(path: str) -> str:
    with open(path, "rb") as f:
        return extract_text(f.read(), path)

if __name__ == "__main__":
    existing_texts = load_existing_texts()

    input_path = "AI_matcher/tests/DNI_ancienne.jpg"
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        exit()

    input_text = load_input_image(input_path)
    df = find_similar_to_input(input_text, existing_texts, top_k=5)

    print("\nTop 5 similar documents:\n")
    print(df)