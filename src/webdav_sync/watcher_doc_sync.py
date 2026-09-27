import os
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Set, List
from webdav3.client import Client
import boto3
from dotenv import load_dotenv, find_dotenv
from src.update_image_index import update_image_index
from src.update_tfidf_index import update_tfidf_index

# --- Load environment variables ---
load_dotenv(find_dotenv())

# --- Configuration ---
DEFAULT_PREFIX = "AI_matcher/"
DEFAULT_TMP_DIR = "/tmp"
REMOTE_PATH = "/production/subprojects"
POLL_INTERVAL = 15
IGNORED_CACHE_PATH = ".ignored_files.json"
SYNCED_CACHE_PATH = ".synced_files.json"
MAX_IGNORED_ENTRIES = 50000
MAX_SYNCED_ENTRIES = 1_000_000
ALLOWED_EXTENSIONS = (".pdf", ".jpeg", ".jpg", ".png", ".docx")

# --- Logging configuration ---
logger = logging.getLogger("WebDAVSync")
logger.setLevel(logging.INFO)

# Remove any existing handlers
for h in logger.handlers[:]:
    logger.removeHandler(h)

# Console handler (stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- Initialize DigitalOcean Spaces ---
s3 = boto3.client(
    's3',
    region_name=os.getenv("DO_SPACE_REGION"),
    endpoint_url=os.getenv("DO_SPACE_ENDPOINT"),
    aws_access_key_id=os.getenv("DO_SPACE_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("DO_SPACE_SECRET_ACCESS_KEY"),
)
BUCKET_NAME = os.getenv("DO_SPACE_NAME")
if not BUCKET_NAME:
    raise EnvironmentError("DO_SPACE_NAME is not set in environment variables")

# --- Initialize WebDAV client ---
client = Client({
    'webdav_hostname': os.getenv("WEBDAV_HOSTNAME"),
    'webdav_login': os.getenv("WEBDAV_LOGIN"),
    'webdav_password': os.getenv("WEBDAV_PASSWORD"),
})

def list_existing_keys(prefix: str = DEFAULT_PREFIX) -> Set[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = set()
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        contents = page.get("Contents")
        if contents:
            keys.update(obj["Key"] for obj in contents if "Key" in obj)
    return keys

def load_cache(path: str) -> Set[str]:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return set(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load cache {path}: {e}")
    return set()

def save_cache(path: str, data: Set[str], max_len: int) -> None:
    try:
        if len(data) > max_len:
            data = set(list(data)[-max_len:])
        with open(path, "w") as f:
            json.dump(list(data), f)
    except Exception as e:
        logger.warning(f"Failed to save cache {path}: {e}")

def download_and_upload(file_name: str, prefix: str, tmp_dir: str) -> bool:
    destination_key = f"{prefix}{file_name}"
    local_path = os.path.join(tmp_dir, file_name)
    try:
        client.download_sync(
            remote_path=f"{REMOTE_PATH}/{file_name}",
            local_path=local_path
        )
        with open(local_path, "rb") as file_obj:
            s3.upload_fileobj(file_obj, BUCKET_NAME, destination_key)
        logger.info(f"Uploaded {file_name} to {destination_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to process {file_name}: {e}")
        return False
    finally:
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
                logger.debug(f"Removed temporary file {local_path}")
            except Exception as cleanup_err:
                logger.warning(f"Cleanup failed for {local_path}: {cleanup_err}")

def watch_and_sync_files(prefix: str = DEFAULT_PREFIX, tmp_dir: str = DEFAULT_TMP_DIR, interval: int = POLL_INTERVAL) -> None:
    logger.info(f"Starting WebDAV watcher on: {REMOTE_PATH}, interval: {interval}s")
    synced_keys = load_cache(SYNCED_CACHE_PATH)
    ignored_files = load_cache(IGNORED_CACHE_PATH)

    while True:
        logger.info("⏳ Starting new sync cycle...")
        try:
            remote_files: List[str] = client.list(REMOTE_PATH)
            total_files = len(remote_files)
            logger.info(f"📂 Found {total_files} files on WebDAV")

            for index, file_name in enumerate(remote_files, start=1):
                percent = (index / total_files) * 100 if total_files else 0
                tag = f"[{index}/{total_files} | {percent:.1f}%]"

                if file_name in ignored_files:
                    logger.debug(f"{tag} Skipped (in ignored cache): {file_name}")
                    continue
                if file_name.endswith("/"):
                    logger.debug(f"{tag} Skipped (directory): {file_name}")
                    continue
                if not file_name.lower().endswith(ALLOWED_EXTENSIONS):
                    logger.info(f"{tag} Ignored (unsupported extension): {file_name}")
                    ignored_files.add(file_name)
                    save_cache(IGNORED_CACHE_PATH, ignored_files, MAX_IGNORED_ENTRIES)
                    continue

                full_key = f"{prefix}{file_name}"
                if full_key in synced_keys:
                    logger.debug(f"{tag} Skipped (already synced): {file_name}")
                    continue

                logger.info(f"{tag} New file detected: {file_name}")
                if download_and_upload(file_name, prefix, tmp_dir):
                    synced_keys.add(full_key)
                    save_cache(SYNCED_CACHE_PATH, synced_keys, MAX_SYNCED_ENTRIES)

        except Exception as loop_err:
            logger.error(f"❌ Watcher failed during sync: {loop_err}")

        logger.info("✅ Sync cycle completed")
        logger.info("🔄 Triggering index update...")

        update_image_index()
        update_tfidf_index()

        logger.info(f"🕒 Sleeping for {interval} seconds...\n{'-'*60}")
        time.sleep(interval)

if __name__ == "__main__":
   watch_and_sync_files()
