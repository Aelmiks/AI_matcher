import os
import json
import pickle
import argparse
from typing import List, Tuple
from dotenv import load_dotenv, find_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import vstack
from tqdm import tqdm

from src.fetch import list_ai_matcher_files, download_file_from_space
from src.model import extract_text

# --- Load environment variables ---
load_dotenv(find_dotenv())

# --- Global configuration ---
VALID_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".docx")
MIN_TEXT_LENGTH = 20
MAX_FEATURES = 20000
OUTPUT_PATH = "output/text_vectors.pkl"
CACHE_PATH = "output/.vectorized_files.json"

def load_vectorized_files(cache_path: str) -> List[str]:
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump([], f)
        return []
    with open(cache_path, "r") as f:
        return json.load(f)

def save_vectorized_files(keys: List[str], cache_path: str = "output/.vectorized_files.json"):
    """
    Saves the list of already processed files to the specified JSON cache file.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(keys, f)

def get_keys_to_process() -> List[str]:
    all_keys = list_ai_matcher_files()
    done_keys = set(load_vectorized_files(CACHE_PATH))
    return [k for k in all_keys if k not in done_keys and k.lower().endswith(VALID_EXTENSIONS)]

def load_existing_index() -> Tuple[List[str], any, TfidfVectorizer]:
    if not os.path.exists(OUTPUT_PATH):
        return [], None, None
    with open(OUTPUT_PATH, "rb") as f:
        data = pickle.load(f)
        return data["keys"], data["vectors"], data["vectorizer"]

def extract_valid_texts(keys: List[str], mode: str) -> Tuple[List[str], List[str]]:
    texts, valid_keys = [], []
    for key in tqdm(keys, desc=f"{mode} TF-IDF: Reading & OCR"):
        try:
            content = download_file_from_space(key)
            text = extract_text(content, key)
            if text and len(text.strip()) >= MIN_TEXT_LENGTH:
                texts.append(text)
                valid_keys.append(key)
        except Exception as e:
            print(f"[ERROR] {key}: {type(e).__name__} - {e}")
    return texts, valid_keys

def save_index(keys: List[str], matrix, vectorizer: TfidfVectorizer) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"keys": keys, "vectors": matrix, "vectorizer": vectorizer}, f)
    save_vectorized_files(keys)

def update_tfidf_index(limit: int = None):
    keys_to_process = get_keys_to_process()
    if limit:
        keys_to_process = keys_to_process[:limit]
    if not keys_to_process:
        print("[TF-IDF] No new files to process.")
        return

    old_keys, old_matrix, vectorizer = load_existing_index()

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
        texts, new_keys = extract_valid_texts(keys_to_process, "Initial")
        if not texts:
            print("[TF-IDF] No valid documents found.")
            return
        matrix = normalize(vectorizer.fit_transform(texts))
        final_keys = new_keys
    else:
        texts, new_keys = extract_valid_texts(keys_to_process, "Incremental")
        if not texts:
            print("[TF-IDF] No valid new documents to vectorize.")
            return
        new_matrix = normalize(vectorizer.transform(texts))
        matrix = vstack([old_matrix, new_matrix])
        final_keys = old_keys + new_keys

    save_index(final_keys, matrix, vectorizer)
    print(f"[TF-IDF] Updated index saved: {len(final_keys)} documents total.")
    print(f"[TF-IDF] Matrix shape: {matrix.shape}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    update_tfidf_index(limit=args.limit)