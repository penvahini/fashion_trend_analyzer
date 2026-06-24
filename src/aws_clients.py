# src/aws_clients.py
"""
Centralized boto3 client factory.

Points at LocalStack when AWS_ENDPOINT_URL is set (local dev), or real AWS
otherwise (set AWS_ENDPOINT_URL unset + real credentials/region to go live).
Bucket/queue names and the endpoint are all environment-driven so the same
code path works against LocalStack or production AWS.
"""
import os
import boto3

S3_BUCKET = os.getenv("S3_BUCKET", "fashion-trend-images")
SQS_QUEUE_NAME = os.getenv("SQS_QUEUE_NAME", "fashion-trend-ingestion")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")  # e.g. http://localhost:4566 for LocalStack
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

def get_s3_client():
    return boto3.client("s3", endpoint_url=AWS_ENDPOINT_URL, region_name=AWS_REGION)

def get_sqs_client():
    return boto3.client("sqs", endpoint_url=AWS_ENDPOINT_URL, region_name=AWS_REGION)

def get_queue_url(sqs_client=None):
    sqs_client = sqs_client or get_sqs_client()
    return sqs_client.get_queue_url(QueueName=SQS_QUEUE_NAME)["QueueUrl"]
