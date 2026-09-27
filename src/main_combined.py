from fetch import list_ai_matcher_files, download_file_from_space
from model import analyze_file_with_ai, find_similar_to_input
from image_features_mobilenet import extract_visual_features
from visual_matcher import build_index, search_similar_images

from sklearn.preprocessing import normalize
import os
import pickle
import pandas as pd
import numpy as np

# === Configuration: weight for text similarity in the combined score ===
ALPHA = 0.5  # 0 = only image similarity, 1 = only text similarity

# === Load precomputed image embeddings ===
with open("output/image_vectors.pkl", "rb") as f:
    visual_db = pickle.load(f)

keys_visual, vectors_visual = list(visual_db.keys()), list(visual_db.values())
vectors_visual = normalize(np.stack(vectors_visual))
index_visual, _, _ = build_index(visual_db, n_neighbors=5)

# === Step 1: Download and analyze documents ===
def process_files(limit=100):
    """
    Fetch files from DigitalOcean, extract text and visual features.
    Returns:
        texts (dict): {file_key: extracted_text}
        images (dict): {file_key: image_vector}
    """
    files = list_ai_matcher_files()
    texts, images = {}, {}

    for file_key in files[:limit]:
        content = download_file_from_space(file_key)

        # Extract text and keep file_key as identifier
        _, extracted_text = analyze_file_with_ai(content, file_key)
        if extracted_text:
            texts[file_key] = extracted_text

        # Extract image vector if applicable
        if file_key.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                vec = extract_visual_features(content)
                if vec is not None and vec.size > 0:
                    images[file_key] = vec
            except Exception:
                pass

    return texts, images

# === Step 2: Combine similarity results ===
def main():
    texts, images = process_files(limit=100)
    combined_results = []

    for query_key, query_text in texts.items():
        # Textual similarity
        df_text_sim = find_similar_to_input(query_text, texts, top_k=5)
        text_scores = dict(df_text_sim.values)

        # Visual similarity
        img_vec = images.get(query_key, None)
        if img_vec is not None and img_vec.size > 0:
            res_vis = search_similar_images(img_vec, index=index_visual, keys=keys_visual, vectors=vectors_visual, top_k=5)
            vis_scores = dict(res_vis)
        else:
            vis_scores = {}

        # Merge both scores
        all_matches = set(text_scores.keys()) | set(vis_scores.keys())
        for match in all_matches:
            score_text = text_scores.get(match)
            score_vis = vis_scores.get(match)

            if score_text is not None and score_vis is not None:
                score_combined = round(ALPHA * score_text + (1 - ALPHA) * score_vis, 2)
                result_type = "combined"
            elif score_text is not None:
                score_combined = round(score_text, 2)
                result_type = "text"
            elif score_vis is not None:
                score_combined = round(score_vis, 2)
                result_type = "image"
            else:
                continue

            combined_results.append({
                "query": query_key,
                "match": match,
                "score": score_combined,
                "type": result_type
            })

    # Compile results into CSV
    df = pd.DataFrame(combined_results)
    df = df[df["query"] != df["match"]]  # Exclude self-matches
    df = df.sort_values(by="score", ascending=False)

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/matches_combined.csv", index=False)
    print("Combined results saved to output/matches_combined.csv")

if __name__ == "__main__":
    main()