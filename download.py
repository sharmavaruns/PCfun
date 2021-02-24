#!/usr/bin/env python3

import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
import os
import zipfile
import time
import sys
import argparse


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

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--input_bucket_filename', type=str,
                    help='Name of file to be pulled from the Bucket. Should be "pcfun.zip".',
                    default='pcfun.zip',required=False)

kwargs = vars(parser.parse_args())
print(kwargs)

BUCKET_NAME = 'pcfun-download'
PATH = kwargs['input_bucket_filename']
home_dir = os.path.expanduser("~")

print(f'Should be installing required {PATH} data into your home directory then unzipping the hidden folder.')

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
print(f"Downloading {PATH} from public S3 bucket.")

time_start_download = time.time()
download(s3_bucket=BUCKET_NAME, s3_object_key=PATH)
time_end_download = time.time()
print(f'\nTime taken to download {PATH}: {(time_end_download-time_start_download)/60:0.2f} min')

if '.zip' in PATH:
    print('Extracting zipped folder and storing it in home directory.')
    time_start_unzip = time.time()
    with zipfile.ZipFile(os.path.join(home_dir,PATH), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(home_dir))
    time_end_unzip = time.time()
    print(f'Time taken to unzip {PATH}: {(time_end_unzip-time_start_unzip)/60:0.2f} min')