from image_features import extract_visual_features
from visual_matcher import build_index, search_similar_images

image_path = "AI_matcher/tests/DNI_ancienne.jpg"
with open(image_path, "rb") as f:
    vec = extract_visual_features(f.read())

print("Feature vector shape:", vec.shape)
print("First 10 values:", vec[:10])

dummy_db = {"DNI ancienne.jpg": vec}
index, keys, vectors = build_index(dummy_db, n_neighbors=1)
results = search_similar_images(vec, index, keys, vectors, top_k=1)

print("\nSearch results (scikit-learn NN) :")
for filename, score in results:
    print(f"{filename} — approximate similarity : {score}%")
