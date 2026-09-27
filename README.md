# AI Translation Matcher

**Find reusable translation references through document content and visual similarity.**

Python project combining OCR, text retrieval and computer vision to help translators identify documents resembling previously processed material. The repository includes indexing pipelines, exploratory notebooks and a Streamlit interface for searching and previewing results.

## Why this project

Translation workflows often receive recurring forms, scanned documents and files with similar layouts. This project explores two complementary signals: what a document says, and what it looks like. Similarity scores rank possible references; they do not certify translation quality or semantic equivalence across languages.

## Technical approach

| Component | Implementation |
| --- | --- |
| Document ingestion | S3-compatible storage through boto3; optional WebDAV synchronization |
| Text extraction | PyMuPDF, python-docx and Tesseract OCR, including scanned PDF fallbacks |
| Text retrieval | TF-IDF, normalization, cosine similarity and nearest-neighbor search |
| Visual retrieval | Pretrained MobileNetV2 embeddings; separate ResNet50 extractor for exploration |
| Search strategies | Text-only, image-only, and sequential or weighted combinations |
| Interface | Streamlit upload, ranked results, document previews and CSV export |
| Incremental processing | Saved indexes and manifests avoid processing unchanged files again |

```text
Documents → text extraction / OCR → TF-IDF index ───────┐
          → image feature extraction → visual index ──┤→ ranked references → preview
```

## Repository guide

- `src/app_streamlit.py`: interactive text and image search interface.
- `src/model.py`: extraction, TF-IDF indexing and similarity functions.
- `src/update_tfidf_index.py`, `src/update_image_index.py`: incremental index builders.
- `src/image_features_mobilenet.py`, `src/image_features.py`: visual feature extractors.
- `src/Notebooks/`: four experiments comparing text, image and sequential retrieval.
- `src/webdav_sync/`: optional infrastructure integration.
- `.env.example`: configuration names without credentials.

## Set up

Use Python 3.9–3.11 for the original pinned PyTorch stack. Dependency installation has not been revalidated on every platform.

```bash
git clone https://github.com/Aelmiks/AI_matcher.git
cd AI_matcher
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Install Tesseract separately, with the English, French, Arabic and Spanish language packs used by the extraction pipeline. Set the `DO_SPACE_*` variables in `.env` for a document collection you are authorized to use. The storage loader expects objects under `AI_matcher/` in the configured bucket.

From the repository root, build the indexes and start the interface:

```bash
python -m src.update_tfidf_index --limit 20
python -m src.update_image_index --limit 20
python -m streamlit run src/app_streamlit.py
```

Start with a small collection: OCR, image embedding downloads and index generation can consume significant time, memory and disk space. Use the same feature extractor when building and querying an image index. The interface expects `output/text_vectors.pkl` and `output/image_vectors_mobilenet.pkl`; older exploratory scripts use different index names.

To explore the notebooks:

```bash
cd src/Notebooks
jupyter notebook
```

Notebook input documents belong under `AI_matcher/tests/` at the repository root and are deliberately excluded from Git. Some exploratory scripts retain assumptions about locally generated indexes; adjust these to your own collection.

## Scope and limitations

This is a portfolio export of a development project, not a packaged production service. TF-IDF uses lexical overlap and is not a multilingual semantic embedding model. Visual matching depends on image quality, document layout and the selected encoder. The UI's PDF text extraction is simpler than the OCR fallback in `src/model.py`.

There is no public benchmark corpus or independently reproduced quality score in this export. Notebook outputs were cleared to avoid disclosing source documents. Full indexing and remote integrations were not run during publication.

## Data and repository scope

Private documents, credentials, generated indexes, model artifacts and environments are excluded. `src/test.py` from the local working folder was a literal data listing and is intentionally omitted. No license file is included in this repository.
