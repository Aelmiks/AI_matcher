from fetch import list_ai_matcher_files, download_file_from_space
from image_features_mobilenet import extract_visual_features
import pickle
import os

output_path = "output/image_vectors.pkl"
os.makedirs("output", exist_ok=True)

def is_image_file(key: str):
    return key.lower().endswith((".jpg", ".jpeg", ".png"))

def build_index_from_digitalocean():
    files = list_ai_matcher_files()
    image_files = [f for f in files if is_image_file(f)]
    
    features = {}

    print(f"[INFO] Found {len(image_files)} image files.")

    for i, file_key in enumerate(image_files):
        try:
            content = download_file_from_space(file_key)
            vec = extract_visual_features(content)
            features[file_key] = vec
            print(f"[{i+1}/{len(image_files)}] {file_key}")
        except Exception as e:
            print(f"[ERROR] Failed to process {file_key} — {e}")

    # Save vectors
    with open(output_path, "wb") as f:
        pickle.dump(features, f)
    
    print(f"\n[SAVED] {len(features)} image vectors saved to {output_path}")

if __name__ == "__main__":
    build_index_from_digitalocean()
