import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
import os
import zipfile

BUCKET_NAME = 'pcfun-download'
PATH = '.pcfun.zip'
home_dir = os.path.expanduser("~")

s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))

try:
    s3.Bucket(BUCKET_NAME).download_file(PATH, os.path.join(home_dir,PATH))
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    if e.response['Error']['Code'] == "403":
        print("Access to the file you've queried is forbidden for some reason. Perhaps a permissions issue.")
    else:
        raise

with zipfile.ZipFile(os.path.join(home_dir,PATH), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(home_dir))