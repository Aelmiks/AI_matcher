import io
from PIL import Image
import pytesseract
import fitz
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
import csv
import os
import pickle
from src.fetch import list_ai_matcher_files, download_file_from_space
from tqdm import tqdm

def convert_pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[Image.Image]:
    """
    Converts each page of a PDF file into a PIL Image (RGB mode).

    Args:
        pdf_bytes (bytes): Raw bytes of the PDF file.
        dpi (int): Resolution for rendering pages. Defaults to 150.

    Returns:
        list[Image.Image]: A list of images, one per page.
    """
    images = []
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in pdf:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
    except Exception as e:
        print(f"[PDF->IMG ERROR] Failed to convert PDF to image: {e}")
    return images

def extract_text(file_bytes: bytes, file_key: str) -> str:
    """
    Extracts raw text from DOCX, PDF, or image files. If no text is found in DOCX, applies OCR on embedded images.
    
    Args:
        file_bytes (bytes): The raw content of the file.
        file_key (str): The filename or key, used to detect file type.

    Returns:
        str: Extracted text.
    """
    file_key_lower = file_key.lower()

    if file_key_lower.endswith(".docx"):
        text = []
        doc = Document(io.BytesIO(file_bytes))

        # Extract visible text
        text.extend(p.text for p in doc.paragraphs if p.text.strip())

        # If no text, apply OCR on embedded images
        if not text:
            for rel in doc.part._rels:
                rel_obj = doc.part._rels[rel]
                if "image" in rel_obj.target_ref:
                    try:
                        image_data = rel_obj.target_part.blob
                        img = Image.open(io.BytesIO(image_data)).convert("RGB")
                        ocr_text = pytesseract.image_to_string(img, lang="eng+fra+ara+spa")
                        if ocr_text.strip():
                            text.append(ocr_text)
                    except Exception as e:
                        print(f"[DOCX OCR ERROR] Failed to OCR image in {file_key}: {e}")

        return "\n".join(text)

    elif file_key_lower.endswith(".pdf"):
        text = ""
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        for page in pdf:
            text += page.get_text()
        
        # If no usable text found, fallback to image-based OCR
        if not text.strip() or len(text.strip()) < 20:
            print(f"[PDF->OCR] No text found in {file_key}, trying image-based fallback.")
            images = convert_pdf_to_images(file_bytes)
            for img in images:
                try:
                    ocr_text = pytesseract.image_to_string(img, lang="eng+fra+ara+spa")
                    if ocr_text.strip():
                        text += "\n" + ocr_text
                except Exception as e:
                    print(f"[PDF OCR ERROR] Page OCR failed: {e}")

        return text

    elif file_key_lower.endswith((".jpg", ".jpeg", ".png")):
        try:
            # Open image and convert to grayscale + binarize
            image = Image.open(io.BytesIO(file_bytes)).convert("L")  # Grayscale
            image = image.point(lambda x: 0 if x < 160 else 255, '1')  # Binarize

            return pytesseract.image_to_string(image, lang="eng+fra+ara+spa")
        except Exception as e:
            print(f"[OCR ERROR] Cannot process image {file_key}: {e}")
            return ""

    else:
        raise ValueError(f"Unsupported file type : {file_key}")

def analyze_file_with_ai(file_content: bytes, file_key: str) -> tuple[str, str]:
    """
    Analyze a file and return (file_key, extracted_text).
    """
    try:
        text = extract_text(file_content, file_key)
        print(f"[AI MODEL] {file_key} — extracted text ({len(text)} characters).")
        return file_key, text
    except Exception as e:
        print(f"[AI MODEL] Error for {file_key} : {e}")
        return file_key, ""

def vectorize_texts(texts: list[str]) -> np.ndarray:
    """
    Transforms a list of raw text documents into a normalized TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(max_features=20000)
    matrix = vectorizer.fit_transform(texts)
    return normalize(matrix)

def match_texts(extracted_texts: dict, threshold: float = 70.0) -> pd.DataFrame:
    """
    Compute cosine similarities between extracted texts passed as a dictionary.
    """
    names = list(extracted_texts.keys())
    texts = list(extracted_texts.values())

    if not texts:
        print("[AI MODEL] No texts to process.")
        return pd.DataFrame()

    results = find_matches_naive(names, texts, threshold)

    df = pd.DataFrame(results, columns=["file_A", "file_B", "similarity_%"])
    df = df.sort_values(by="similarity_%", ascending=False)

    print(f"[AI MODEL] Found {len(df)} matching document pairs with similarity ≥ {threshold}%")
    return df

def find_matches_naive(names, texts, threshold):
    """
    Computes pairwise cosine similarities between all documents using an O(n²) approach.
    Returns a list of document pairs with similarity above the given threshold.
    """
    matrix = vectorize_texts(texts)
    sim = (matrix @ matrix.T).toarray()
    np.fill_diagonal(sim, 0)
    results = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            score_pct = sim[i, j] * 100
            if score_pct >= threshold:
                results.append((names[i], names[j], round(score_pct, 2)))
    return results

def find_matches_nn(names, texts, threshold, k_neighbors):
    """
    Uses scikit-learn's NearestNeighbors to find the k most similar documents for each entry.
    Returns pairs with similarity above the given threshold.
    """
    matrix = vectorize_texts(texts)
    nn = NearestNeighbors(metric='cosine', n_neighbors=min(k_neighbors + 1, len(names)), algorithm='auto')
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)
    results = []
    for i, neighbors in enumerate(indices):
        for j, neighbor_idx in enumerate(neighbors[1:]):
            similarity_pct = (1 - distances[i][j]) * 100
            if similarity_pct >= threshold:
                results.append((names[i], names[neighbor_idx], round(similarity_pct, 2)))
    return results

def benchmark_methods(extracted_texts: dict, threshold: float = 70.0, k_neighbors: int = 10):
    """
    Benchmarks the current O(n²) matching vs a NearestNeighbors approach.
    Measures and prints execution times for both methods.
    """
    names = list(extracted_texts.keys())
    texts = list(extracted_texts.values())

    if not texts:
        print("[AI MODEL] No texts to process for benchmarking.")
        return

    print(f"\n[Benchmark] Testing on {len(names)} documents.\n")

    # O(n²) method
    start = time.time()
    results_naive = find_matches_naive(names, texts, threshold)
    end = time.time()
    print(f"[Benchmark] Naive O(n²) method took {end - start:.2f} seconds and found {len(results_naive)} pairs.")

    # Nearest Neighbors method
    start = time.time()
    results_nn = find_matches_nn(names, texts, threshold, k_neighbors)
    end = time.time()
    print(f"[Benchmark] NearestNeighbors method took {end - start:.2f} seconds and found {len(results_nn)} pairs.")

    print("\n[Benchmark] Comparison completed.\n")

def clean_text(text: str) -> str:
    """
    Trims and cleans extracted text to a max of 300 characters, removing line breaks and leading/trailing spaces.
    """
    return text[:300].replace("\n", " ").strip()

def export_matches_to_csv(matches: pd.DataFrame, extracted_texts: dict, output_path: str):
    """
    Exports similar pairs with text snippets to a CSV file.
    """
    if matches.empty:
        print("[EXPORT] No matches to export.")
        return

    rows = []
    for _, row in matches.iterrows():
        a, b, score = row["file_A"], row["file_B"], row["similarity_%"]
        text_a = clean_text(extracted_texts.get(a, ""))
        text_b = clean_text(extracted_texts.get(b, ""))
        rows.append([a, b, score, text_a, text_b])

    header = ["file_A", "file_B", "similarity (%)", "excerpt_A", "excerpt_B"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[EXPORT] Results exported in {output_path}")

def find_similar_to_input(input_text: str, extracted_texts: dict, top_k: int = 5) -> pd.DataFrame:
    """
    Compares a new input text against already extracted texts.
    Returns a DataFrame with the top_k most similar documents.
    """
    if not extracted_texts:
        print("[AI MODEL] No documents available for comparison.")
        return pd.DataFrame()

    names = list(extracted_texts.keys())
    texts = list(extracted_texts.values())

    all_texts = [input_text] + texts
    matrix = vectorize_texts(all_texts)

    input_vec = matrix[0]
    doc_vecs = matrix[1:]

    sims = (input_vec @ doc_vecs.T).toarray().flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = [(names[i], round(sims[i] * 100, 2)) for i in top_indices]
    df = pd.DataFrame(results, columns=["file", "similarity_%"])
    return df

def build_tfidf_index(output_path="output/text_vectors.pkl", limit=None):
    """
    Extracts text from DigitalOcean documents and builds a TF-IDF matrix.
    Saves matrix, file names, and vectorizer to a pickle file.

    Args:
        output_path (str): Path to save the TF-IDF index.
        limit (int or None): Maximum number of documents to process. If None, process all.
    """
    files = list_ai_matcher_files()
    print(f"[DEBUG] {len(files)} files found in DigitalOcean")
    print(f"[DEBUG] Processing up to {limit if limit is not None else 'all'} files")

    texts, keys = [], []

    for file_key in tqdm(files if limit is None else files[:limit], desc="Extracting text"):
        if not file_key.lower().endswith((".docx", ".pdf", ".jpg", ".jpeg", ".png")):
            continue
        try:
            content = download_file_from_space(file_key)
            text = extract_text(content, file_key)
            if not text.strip() or len(text.strip()) < 20:
                continue  # skip empty or unusable OCR
            texts.append(text)
            keys.append(file_key)
        except Exception as e:
            print(f"[OCR ERROR] Cannot process {file_key}: {e}")

    if not texts:
        print("[TF-IDF] No usable text extracted.")
        return

    vectorizer = TfidfVectorizer(max_features=20000)
    matrix = vectorizer.fit_transform(texts)
    matrix = normalize(matrix)

    os.makedirs("output", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "keys": keys,
            "vectors": matrix,
            "vectorizer": vectorizer
        }, f)

    print(f"TF-IDF index saved to {output_path} — {len(keys)} documents retained.")