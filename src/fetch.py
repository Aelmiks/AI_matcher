import boto3
from botocore.exceptions import ClientError
import requests
import io
import os
from dotenv import load_dotenv, find_dotenv

# --- Load environment variables ---
load_dotenv(find_dotenv())

session = boto3.session.Session()
s3 = session.client(
    's3',
    region_name=os.getenv("DO_SPACE_REGION"),
    endpoint_url=os.getenv("DO_SPACE_ENDPOINT"),
    aws_access_key_id=os.getenv("DO_SPACE_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("DO_SPACE_SECRET_ACCESS_KEY"),
)

def list_ai_matcher_files():
    """
    List all file keys from the 'AI_matcher/' folder in the configured DO Space.
    Uses pagination to retrieve more than 1000 files.
    """
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=os.getenv("DO_SPACE_NAME"),
        Prefix="AI_matcher/"
    )

    all_keys = []
    for page in page_iterator:
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith("/"):  # Ignore folders
                all_keys.append(key)
    return all_keys

def download_private_file(bucket_name: str, file_key: str, expiration: int = 3600) -> bytes:
    """
    Download a file from a private DigitalOcean Space using a signed URL.
    """
    try:
        signed_url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': bucket_name, 'Key': file_key},
            ExpiresIn=expiration
        )
        response = requests.get(signed_url)
        response.raise_for_status()
        return response.content

    except ClientError as e:
        print(f"[S3 CLIENT ERROR] {file_key} — {e.response['Error']['Message']}")
    except Exception as e:
        print(f"[SIGNED URL ERROR] {file_key} — {type(e).__name__}: {e}")
    return None

def download_file_from_space(key: str, expiration: int = 3600):
    """
    Download a file from DigitalOcean Space using a signed URL (private access).
    """
    return download_private_file(os.getenv("DO_SPACE_NAME"), key, expiration)