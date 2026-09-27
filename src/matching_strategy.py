from PIL import Image
import fitz  # PyMuPDF

TEXT_EXTENSIONS = (".docx", ".pdf", ".txt")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

def choose_matching_strategy(key: str, file_bytes: bytes) -> str:
    """
    Choose between text-based or image-based matching depending on the content.

    Returns:
        "text" or "image"
    """
    key_lower = key.lower()

    if key_lower.endswith(IMAGE_EXTENSIONS):
        return "image"

    if key_lower.endswith(".docx"):
        return "text"

    if key_lower.endswith(".pdf"):
        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            text = "".join(page.get_text().strip() for page in pdf)
            if len(text) >= 20:
                return "text"
            # Otherwise fallback: check if it can be rendered visually
            page = pdf[0]
            _ = page.get_pixmap(dpi=150)
            return "image"
        except Exception:
            return "image"

    return "text"  # Fallback for unknown types