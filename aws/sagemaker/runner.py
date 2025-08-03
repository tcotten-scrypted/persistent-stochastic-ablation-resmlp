#!/usr/bin/env python3
"""
AWS SageMaker Runner for Persistent Stochastic Ablation Experiments

This script reads configurations from current_batch_configurations.txt and launches
SageMaker training jobs for each architecture with all ablation modes.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
"""

import sagemaker
from sagemaker.pytorch import PyTorch
from dotenv import load_dotenv
from pathlib import Path
import os
import boto3
import botocore
import time
import argparse

# --- Robustly load environment variables from .env file ---
script_dir = Path(__file__).resolve().parent
dotenv_path = script_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- Main Configuration ---
SAGEMAKER_ROLE_ARN = os.getenv("AWS_SAGEMAKER_ROLE")
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
AWS_REGION = os.getenv("AWS_REGION")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "scrypted-ai-training")
S3_PREFIX = os.getenv("S3_PREFIX", "psa-experiment")
S3_OUTPUT_PATH = f"s3://{S3_BUCKET_NAME}/{S3_PREFIX}"

INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.g4dn.xlarge")
USE_SPOT_INSTANCES = os.getenv("USE_SPOT_INSTANCES", "false").lower() == "true"
MAX_RUN_SECONDS = int(os.getenv("MAX_RUN_SECONDS", "172800"))
MAX_WAIT_SECONDS = MAX_RUN_SECONDS

ENTRY_SCRIPT = "train.py"
CONFIG_FILE_PATH = "current_batch_configurations.txt"
NUM_RUNS_PER_JOB = 10


def get_latest_job_status(sagemaker_client, base_job_name):
    """
    Checks SageMaker for the most recent job with a given base name and returns its status.
    Returns None if no job is found.
    """
    try:
        response = sagemaker_client.list_training_jobs(
            NameContains=base_job_name,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        if response['TrainingJobSummaries']:
            return response['TrainingJobSummaries'][0]['TrainingJobStatus']
        return None
    except botocore.exceptions.ClientError:
        return "CLIENT_ERROR"


def main(args):
    """
    Reads configurations and launches SageMaker jobs only if they are not already
    completed or currently in progress.
    """
    if not all([SAGEMAKER_ROLE_ARN, AWS_ACCOUNT_ID, AWS_REGION]):
        print("🔴 FATAL ERROR: Required variables not found in .env file.")
        print(f"   Expected: AWS_SAGEMAKER_ROLE, AWS_ACCOUNT_ID, AWS_REGION")
        print(f"   Found in {dotenv_path}: {list(os.environ.keys()) if os.path.exists(dotenv_path) else 'File not found'}")
        return

    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    sagemaker_client = sagemaker_session.sagemaker_client

    # Check if config file exists
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"🔴 FATAL ERROR: Configuration file {CONFIG_FILE_PATH} not found.")
        print(f"   Please create this file with one architecture per line (e.g., '7*3', '9*1')")
        return

    with open(CONFIG_FILE_PATH, 'r') as f:
        architectures = [line.strip() for line in f if line.strip()]

    if not architectures:
        print(f"🔴 FATAL ERROR: No configurations found in {CONFIG_FILE_PATH}")
        return

    ablation_modes = ["none", "full", "hidden", "output"]
    
    print("✅ Starting State-Aware Batch Execution.")
    print(f"   Configurations: {architectures}")
    print(f"   Ablation modes: {ablation_modes}")
    print(f"   Runs per job: {NUM_RUNS_PER_JOB}")
    if args.force_rerun:
        print("   --force-rerun flag is active. All jobs will be launched regardless of status.")

    jobs_launched = 0
    jobs_skipped = 0

    for arch in architectures:
        for mode in ablation_modes:
            arch_safe_name = arch.replace('*', 'x')
            base_job_name = f"psa-{arch_safe_name}-{mode}"

            if not args.force_rerun:
                latest_status = get_latest_job_status(sagemaker_client, base_job_name)
                if latest_status == "Completed":
                    print(f"✅ SKIPPING: {base_job_name} (Already Completed)")
                    jobs_skipped += 1
                    continue
                if latest_status in ["InProgress", "Stopping"]:
                    print(f"⏳ SKIPPING: {base_job_name} (Currently {latest_status})")
                    jobs_skipped += 1
                    continue
                if latest_status in ["Failed", "Stopped"]:
                    print(f"🟡 RETRYING: {base_job_name} (Last status was {latest_status})")

            hyperparameters = {
                "arch": f"[{arch}]",
                "ablation-mode": mode,
                "model-dir": "/opt/ml/model",
                "meta-loops": 100,
                "num-runs": NUM_RUNS_PER_JOB,
            }

            pytorch_estimator = PyTorch(
                entry_point=ENTRY_SCRIPT,
                source_dir=str(script_dir),  # <--- THE CRITICAL FIX
                role=SAGEMAKER_ROLE_ARN,
                instance_count=1,
                instance_type=INSTANCE_TYPE,
                framework_version="2.0.0",
                py_version="py310",
                hyperparameters=hyperparameters,
                output_path=S3_OUTPUT_PATH,
                sagemaker_session=sagemaker_session,
                base_job_name=base_job_name,
                use_spot_instances=USE_SPOT_INSTANCES,
                max_wait=MAX_WAIT_SECONDS if USE_SPOT_INSTANCES else None,
                max_run=MAX_RUN_SECONDS
            )

            try:
                pytorch_estimator.fit(wait=False)
                print(f"🚀 LAUNCHED: {pytorch_estimator.latest_training_job.name}")
                jobs_launched += 1
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == 'ResourceLimitExceeded':
                    print(f"🔴 DENIED: {base_job_name} (Account resource limit exceeded)")
                    print(f"   SageMaker will not queue this job. Retry when resources are available.")
                else:
                    print(f"🔴 ERROR launching {base_job_name}: {e}")
            
            time.sleep(1)

    print(f"\n✅ Batch execution summary:")
    print(f"   Jobs launched: {jobs_launched}")
    print(f"   Jobs skipped: {jobs_skipped}")
    print(f"   Total configurations: {len(architectures) * len(ablation_modes)}")
    print("\n   Monitor progress in the AWS SageMaker console.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SageMaker training jobs for PSA experiments")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Launch all jobs even if they have already completed successfully."
    )
    args = parser.parse_args()
    main(args) 