import boto3
from dotenv import load_dotenv
from pathlib import Path
import os
import tarfile
import numpy as np
from rich.console import Console
from rich.progress import track

# --- Robustly load environment variables from .env file ---
script_dir = Path(__file__).resolve().parent
dotenv_path = script_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- Main Configuration ---
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = "scrypted-ai-training"
S3_PREFIX = "psa-experiment"
CONFIG_FILE = "reproduction/configurations.txt"
SUMMARY_FILE = "results/psa_simplemlp_summary.md"
TRIALS_FILE = "results/psa_simplemlp_trials.md"
CONSOLE = Console()

NUM_INPUTS = 784
NUM_OUTPUTS = 10

params = {}

def count_parameters(input_size, hidden_layers, output_size):
    total_params = 0

    if not hidden_layers:
        # No hidden layers, direct input to output
        total_params = output_size * (input_size + 1)
    else:
        # Input to first hidden
        total_params += hidden_layers[0] * (input_size + 1)
        # Hidden to hidden
        for i in range(len(hidden_layers) - 1):
            total_params += hidden_layers[i + 1] * (hidden_layers[i] + 1)
        # Last hidden to output
        total_params += output_size * (hidden_layers[-1] + 1)

    return total_params

def get_job_results(s3_client, arch, mode):
    """
    Finds the latest job for an arch/mode and returns its results.
    Returns a dictionary with both stats and raw scores, or an error string.
    """
    arch_safe_name = arch.replace('*', 'x')
    job_prefix = f"{S3_PREFIX}/psa-{arch_safe_name}-{mode}"
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=job_prefix, Delimiter='/')
    
    all_job_prefixes = []
    for page in pages:
        if 'CommonPrefixes' in page:
            all_job_prefixes.extend([p['Prefix'] for p in page.get('CommonPrefixes', [])])
    
    if not all_job_prefixes:
        return "NO_JOB_FOUND"
    
    latest_job_prefix = sorted(all_job_prefixes)[-1]
    s3_key_results = f"{latest_job_prefix}output/model.tar.gz"

    try:
        with open("model.tar.gz", "wb") as f:
            s3_client.download_fileobj(S3_BUCKET_NAME, s3_key_results, f)

        with tarfile.open("model.tar.gz", "r:gz") as tar:
            results_file_member = next((m for m in tar.getmembers() if os.path.basename(m.name) == "results.txt"), None)
            
            if results_file_member:
                content = tar.extractfile(results_file_member).read().decode('utf-8')
                scores = [float(line.strip()) for line in content.strip().split('\n') if line]
                
                if not scores: return "EMPTY_RESULTS"
                
                # Return both the calculated stats and the raw scores list
                stats = {
                    "mean": np.mean(scores), "std": np.std(scores),
                    "min": np.min(scores), "max": np.max(scores), "count": len(scores)
                }
                return {"stats": stats, "scores": scores}
            else:
                return "RESULTS_NOT_FOUND"

    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return "FAILED_OR_INCOMPLETE"
        return f"ERROR: {str(e)}"
    except Exception as e:
        return f"CLIENT_ERROR: {str(e)}"
    finally:
        if os.path.exists("model.tar.gz"):
            os.remove("model.tar.gz")


def main():
    """Iterates through all configs, fetches results, and builds markdown files."""
    if not AWS_REGION:
        CONSOLE.print(f"🔴 FATAL ERROR: AWS_REGION not found in your .env file at {dotenv_path}.")
        return

    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    boto_session = boto3.Session(region_name=AWS_REGION)
    s3_client = boto_session.client('s3')

    with open(CONFIG_FILE, 'r') as f:
        configs = [line.strip() for line in f if line.strip()]
        
    ablation_modes = ["none", "full", "hidden", "output"]
    all_results = {}

    CONSOLE.print(f"🔍 Starting results collection from S3 Bucket: [bold cyan]{S3_BUCKET_NAME}/{S3_PREFIX}[/bold cyan]")
    
    for config in track(configs, description="Processing architectures..."):
        all_results[config] = {}

        L, H = map(int, config.split('*'))
        hidden_layers = [H] * L

        params[config] = count_parameters(NUM_INPUTS, hidden_layers, NUM_OUTPUTS)

        for mode in ablation_modes:
            result = get_job_results(s3_client, config, mode)
            all_results[config][mode] = result

    CONSOLE.print(f"\n✅ Collection complete. Writing results to [bold green]{SUMMARY_FILE}[/bold green] and [bold green]{TRIALS_FILE}[/bold green]")
    
    # --- PART 1: WRITE THE SUMMARY FILE ---
    with open(SUMMARY_FILE, 'w') as f:
        f.write("# Persistent Stochastic Ablation - SimpleMLP Results Summary\n\n")
        f.write("## Statistical Summary\n\n")
        for config, modes in all_results.items():
            p_count = params[config]

            f.write(f"Results Shape {{{config}}} Parameters {{{p_count}}}\n")
            
            for mode, result in modes.items():
                mode_name = mode.capitalize()
                if isinstance(result, dict):
                    stats = result['stats']
                    f.write(f"* {mode_name}: Mean={stats['mean']:.2f}% | Std={stats['std']:.2f}% | Min={stats['min']:.2f}% | Max={stats['max']:.2f}% (n={stats['count']})\n")
                else:
                    f.write(f"* {mode_name}: {result}\n")
            f.write("\n")

    # --- PART 2: WRITE THE TRIALS FILE ---
    with open(TRIALS_FILE, 'w') as f:
        f.write("# Persistent Stochastic Ablation - SimpleMLP Raw Trial Data\n\n")
        f.write("## Raw Trial Data\n\n")
        for config, modes in all_results.items():
            f.write(f"### Architecture: {config}\n\n")

            # Determine the maximum number of trials for this architecture to set table rows
            max_trials = 0
            for result in modes.values():
                if isinstance(result, dict):
                    max_trials = max(max_trials, result['stats']['count'])
            
            if max_trials == 0:
                f.write("No successful trials found for this architecture.\n\n")
                continue

            # Write table header
            f.write("| Trial | None | Full | Hidden | Output |\n")
            f.write("|:-----:|:----:|:----:|:------:|:------:|\n")
            
            # Write table rows
            for i in range(max_trials):
                row = [f"| {i+1} "]
                for mode in ablation_modes:
                    result = modes[mode]
                    if isinstance(result, dict) and i < len(result['scores']):
                        score = f"{result['scores'][i]:.2f}%"
                        row.append(f"| {score} ")
                    else:
                        row.append("| N/A ")
                row.append("|\n")
                f.write("".join(row))
            f.write("\n")

    CONSOLE.print("✨ Done.")


if __name__ == "__main__":
    main()