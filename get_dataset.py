import os
from typing import List
import boto3


def fetch(s3_dataset_root_url: str, relative_filepaths: List[str], download_directory='.'):
    s3 = boto3.resource('s3')
    _, bucket, dataset_path = s3_dataset_root_url.split('amazonaws.com')[1].split('/', 2)
    for filepath in relative_filepaths:
        source = os.path.join(dataset_path, filepath)
        destination = os.path.join(download_directory, filepath)
        s3.Bucket(bucket).download_file(source, destination)