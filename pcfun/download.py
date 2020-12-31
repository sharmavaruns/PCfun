import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
import os
import zipfile

print('Should be installing required pcfun data into your home directory then unzipping the hidden folder.')

BUCKET_NAME = 'pcfun-download'
PATH = '.pcfun.zip'
home_dir = os.path.expanduser("~")

s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))

try:
    print("Downloading .pcfun.zip from public S3 bucket.")
    s3.Bucket(BUCKET_NAME).download_file(PATH, os.path.join(home_dir,PATH))
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    if e.response['Error']['Code'] == "403":
        print("Access to the file you've queried is forbidden for some reason. Perhaps a permissions issue.")
    else:
        raise

print('Extracting zipped folder and storing it in home directory.')
with zipfile.ZipFile(os.path.join(home_dir,PATH), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(home_dir))