# Persistent Stochastic Ablation (PSA) - SimpleMLP

Source code for the mini-paper "Beyond Pruning and Dropout: Evolving Robust Networks via Persistent Stochastic Ablation"

## Prerequisites

### Poetry Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. If you don't have Poetry installed, you can install it using one of the following methods:

#### Option 1: Official Installer (Recommended)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Option 2: pip
```bash
pip install poetry
```

#### Option 3: Homebrew (macOS)
```bash
brew install poetry
```

After installation, ensure Poetry is in your PATH. You may need to restart your terminal or run:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Project Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:tcotten-scrypted/persistent-stochastic-ablation-mlp.git
   cd persistent-stochastic-ablation-mlp
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```
   
   This will create a virtual environment in `.venv/` within the project directory.

3. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```
   
   Or run commands directly with Poetry:
   ```bash
   poetry run train
   ```

## Project Structure

```
├── src/           # source code
├── scripts/       # utility and execution scripts
├── aws/           # AWS-related configurations and scripts
├── models/        # user-generated model files
├── dataset/       # downloaded datasets
├── results/       # experimental results
└── pyproject.toml # Poetry configuration and dependencies     
```

## Usage

### Basic Training

Train with default settings:
```bash
poetry run train
```

### Training with Custom Parameters

Train with custom architecture and ablation mode:
```bash
poetry run train -- --arch "[4*4, 2*8]" --ablation-mode hidden --lr 1e-3
```

Train with custom batch size and meta-loops:
```bash
poetry run train -- --batch-size 512 --meta-loops 200 --ablation-mode full
```

### Available Parameters

- `--arch`: Define architecture with string, e.g., `"[4*4, 2*8, 1*16]"`
- `--ablation-mode`: Set ablation mode (`none`, `full`, `hidden`, `output`)
- `--lr`: Learning rate (default: 1e-4)
- `--batch-size`: Batch size for training (default: 256)
- `--meta-loops`: Total meta-loops to run (default: 100)
- `--model-dir`: Path to store/retrieve models (default: "models/")
- `--debug`: Enable verbose debug logging
- `--num-workers`: DataLoader workers (default: 4)
- `--device`: Override device detection (`cpu`, `cuda`, `mps`)

### Device Detection

The script automatically detects the best available device:
- **CUDA** (highest priority): NVIDIA GPUs
- **Metal** (MPS): Apple Silicon (M1/M2/M3) chips
- **CPU** (fallback): When no GPU acceleration is available

You can override the device detection with `--device`:
```bash
poetry run train -- --device cpu      # Force CPU
poetry run train -- --device cuda     # Force CUDA
poetry run train -- --device mps      # Force Metal (Apple Silicon)
```

### Examples

**Control experiment (no ablation):**
```bash
poetry run train -- --ablation-mode none
```

**Hidden layer ablation:**
```bash
poetry run train -- --ablation-mode hidden --arch "[2*512, 1*256]"
```

**Output layer ablation:**
```bash
poetry run train -- --ablation-mode output --lr 5e-4
```

**Full ablation with custom settings:**
```bash
poetry run train -- --ablation-mode full --batch-size 128 --meta-loops 150 --debug
```

### Resuming Training

The training script automatically resumes from the last known good (LKG) checkpoint if one exists. This allows you to:

**Continue training on the same architecture:**
```bash
# First run
poetry run train -- --arch "[1*1024]" --ablation-mode hidden

# Continue training (resumes from checkpoint)
poetry run train -- --arch "[1*1024]" --ablation-mode hidden

# Switch ablation mode (resumes from same checkpoint)
poetry run train -- --arch "[1*1024]" --ablation-mode output
```

**Important Notes:**
- **Same Architecture**: You can resume training and even switch ablation modes on the same architecture
- **Different Architecture**: If you change `--arch` or `--hidden-layers`, you must clean the checkpoint first:
  ```bash
  poetry run clean  # Remove old checkpoint
  poetry run train -- --arch "[2*512]" --ablation-mode hidden  # Start fresh
  ```
- **Checkpoint Location**: Models are saved as `models/mnist_lkg.safetensors`
- **Resume Behavior**: Training automatically loads the LKG checkpoint and continues from where it left off

### Cleaning Model Files

Remove the default checkpoint:
```bash
poetry run clean
```

Remove all files in the models directory:
```bash
poetry run clean --all
```

Remove specific files:
```bash
poetry run clean file1.safetensors file2.safetensors
poetry run clean models/file1.safetensors  # With models/ prefix
```

Dry run (show what would be removed):
```bash
poetry run clean --dry-run
poetry run clean --all --dry-run
```

Alias for clean-all:
```bash
poetry run clean-all  # Same as poetry run clean --all
```

### Reproduction Tools

Generate commands for reproducing all experimental configurations:
```bash
poetry run generate-reproduction-tests
```

This generates:
- **Individual Commands**: All 59 architectures × 4 ablation modes
- **Batch Script**: Executable `run_reproduction.sh` for automated execution
- **Organized Output**: Commands grouped by architecture with clear headers

### AWS SageMaker Integration

Parse and analyze SageMaker training results:
```bash
poetry run sagemaker-results-parser
```

**Requirements:**
- AWS credentials configured
- `.env` file with AWS configuration in `aws/sagemaker/` (copy from `.env.example`)
- S3 bucket access to configured bucket/prefix

**Output Files:**
- `results/psa_simplemlp_summary.md` - Statistical summary of all experiments
- `results/psa_simplemlp_trials.md` - Raw trial data in markdown tables

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2025 Tim Cotten <tcotten@scrypted.ai> <tcotten2@gmu.edu>
