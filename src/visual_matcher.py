import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def build_index(embeddings_dict: dict, n_neighbors: int = 5):
    """
    Builds a NearestNeighbors index from a dictionary {filename: vector}.
    Normalizes all vectors before fitting the index.
    Returns the model and list of keys (filenames).
    """
    keys = list(embeddings_dict.keys())
    vectors = np.stack([embeddings_dict[k] for k in keys])
    vectors = normalize(vectors)  # Normalize to unit norm

    index = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    index.fit(vectors)

    return index, keys, vectors

def search_similar_images(query_vec: np.ndarray, index, keys: list, vectors: np.ndarray, top_k: int = 5):
    """
    Finds the top_k nearest images in the NearestNeighbors index.
    """
    distances, indices = index.kneighbors(query_vec.reshape(1, -1), n_neighbors=top_k)
    results = [(keys[i], round(100 - distances[0][j], 2)) for j, i in enumerate(indices[0])]
    return results