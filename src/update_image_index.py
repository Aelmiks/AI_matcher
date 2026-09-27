import os
import json
import pickle
from typing import List, Tuple
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.preprocessing import normalize
from src.fetch import list_ai_matcher_files, download_file_from_space
from src.image_features_mobilenet import extract_visual_features
from src.update_tfidf_index import load_vectorized_files, save_vectorized_files, parse_args

# --- Load environment variables ---
load_dotenv(find_dotenv())

# --- Global configuration ---
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
OUTPUT_PATH = "output/image_vectors_mobilenet.pkl"
CACHE_PATH = "output/.vectorized_images_mobilenet.json"

def get_image_keys_to_process() -> List[str]:
    all_keys = list_ai_matcher_files()
    done_keys = set(load_vectorized_files(CACHE_PATH))
    return [k for k in all_keys if k not in done_keys and k.lower().endswith(VALID_IMAGE_EXTENSIONS)]

def load_existing_image_index() -> Tuple[List[str], np.ndarray]:
    if not os.path.exists(OUTPUT_PATH):
        return [], None
    with open(OUTPUT_PATH, "rb") as f:
        data = pickle.load(f)
        return data["keys"], data["vectors"]

def save_image_index(keys: List[str], vectors: np.ndarray):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"keys": keys, "vectors": vectors}, f)
    save_vectorized_files(keys, CACHE_PATH)

def update_image_index(limit: int = None):
    keys_to_process = get_image_keys_to_process()
    if limit:
        keys_to_process = keys_to_process[:limit]
    if not keys_to_process:
        print("[IMAGE] No new image files to process.")
        return

    old_keys, old_vectors = load_existing_image_index()

    new_vectors, new_keys = [], []
    for key in tqdm(keys_to_process, desc="Image Feature Extraction"):
        try:
            content = download_file_from_space(key)
            vector = extract_visual_features(content)
            if vector is not None:
                new_vectors.append(vector)
                new_keys.append(key)
                # Save progress after each image
                if old_vectors is not None:
                    tmp_matrix = np.vstack([old_vectors, normalize(np.vstack(new_vectors))])
                    tmp_keys = old_keys + new_keys
                else:
                    tmp_matrix = normalize(np.vstack(new_vectors))
                    tmp_keys = new_keys
                save_image_index(tmp_keys, tmp_matrix)
        except Exception as e:
            print(f"[ERROR] {key}: {type(e).__name__} - {e}")

    print(f"[IMAGE] Updated index saved: {len(old_keys) + len(new_keys)} images total.")
    if new_keys:
        print(f"[IMAGE] Matrix shape: {tmp_matrix.shape}")

if __name__ == "__main__":
    args = parse_args()
    update_image_index(limit=args.limit)