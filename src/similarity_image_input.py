import pickle
import numpy as np
from image_features import extract_visual_features
from visual_matcher import build_index, search_similar_images

def load_index() -> tuple[list[str], list[np.ndarray]]:
    """
    Loads the image vector dictionary from a pickle file.
    Returns keys (file names) and vectors.
    """
    with open("output/image_vectors.pkl", "rb") as f:
        data = pickle.load(f)
    return list(data.keys()), list(data.values())

def load_query_image(path: str) -> np.ndarray:
    """
    Loads an image and extracts its visual feature vector.
    """
    with open(path, "rb") as f:
        return extract_visual_features(f.read())

if __name__ == "__main__":
    # Load query image
    query_path = "AI_matcher/tests/DNI_ancienne.jpg"
    query_vector = load_query_image(query_path)

    # Load indexed vectors
    keys, vectors = load_index()

    # Building the index with scikit-learn
    index, _, _ = build_index(dict(zip(keys, vectors)), n_neighbors=5)

    # Perform the search
    results = search_similar_images(query_vector, index=index, keys=keys, vectors=vectors, top_k=5)

    # Show results
    print("\nSimilar search results for the image query :")
    for key, score in results:
        print(f"{key} — approximate similarity : {score:.2f}%")