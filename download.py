#!/usr/bin/env python3

import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
import os
import zipfile
import time
import sys


def download(s3_bucket, s3_object_key):
    "Function that downloads from s3 bucket and tracks progress"

    meta_data = s3.head_object(Bucket=s3_bucket, Key=s3_object_key)
    total_length = int(meta_data.get('ContentLength', 0))
    downloaded = 0

    def progress(chunk):
        nonlocal downloaded
        downloaded += chunk
        done = int(50 * downloaded / total_length)
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
        sys.stdout.flush()

    print(f'Downloading {s3_object_key}')
    with open(os.path.join(os.path.expanduser("~"), os.path.basename(s3_object_key)), 'wb') as f:
        s3.download_fileobj(s3_bucket, s3_object_key, f, Callback=progress)

BUCKET_NAME = 'pcfun-download'
PATH = 'pcfun.zip'
home_dir = os.path.expanduser("~")

print(f'Should be installing required {PATH} data into your home directory then unzipping the hidden folder.')

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
print(f"Downloading {PATH} from public S3 bucket.")

time_start_download = time.time()
download(s3_bucket=BUCKET_NAME, s3_object_key=PATH)
time_end_download = time.time()
print(f'Time taken to download {PATH}: {(time_end_download-time_start_download)/60:0.2f} min')

# try:
#     print("Downloading .pcfun.zip from public S3 bucket.")
#     s3.Bucket(BUCKET_NAME).download_file(PATH, os.path.join(home_dir,PATH))
# except botocore.exceptions.ClientError as e:
#     if e.response['Error']['Code'] == "404":
#         print("The object does not exist.")
#     if e.response['Error']['Code'] == "403":
#         print("Access to the file you've queried is forbidden for some reason. Perhaps a permissions issue.")
#     else:
#         raise

print('Extracting zipped folder and storing it in home directory.')
time_start_unzip = time.time()
with zipfile.ZipFile(os.path.join(home_dir,PATH), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(home_dir))
time_end_unzip = time.time()
print(f'Time taken to unzip {PATH}: {(time_end_unzip-time_start_unzip)/60:0.2f} min')