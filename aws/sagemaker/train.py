# train.py - SageMaker Version (ResMLP)
#
# Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu> 
#
# Description:
# A comprehensive training harness for testing architectural variations and six
# distinct live ablation strategies on the MNIST dataset using ResMLP architecture.
# Adapted for AWS SageMaker execution with environment-specific modifications.
#
# Core Concepts:
# 1. Six Ablation Modes:
#    - 'none': Control group.
#    - 'decay': Traditional weight decay regularization
#    - 'dropout': Traditional dropout regularization
#    - 'full': Partially ablates a neuron in ANY linear layer (hidden or output).
#    - 'hidden': Fully ablates a neuron in a HIDDEN layer only.
#    - 'output': Partially ablates a neuron in the OUTPUT layer only.
#
# 2. Each meta-loop starts with a fresh copy of the Last Known Good (LKG) model.
# This copy is ablated once (for ablative modes) to create the active model state
# for the current loop. The LKG state itself is only updated when the validation
# accuracy improves, preserving the best model found so far. A "bounty" tracks
# the highest validation accuracy achieved across all meta-loops.
#
# 3. Ablation strategy is always ONE neuron randomly selected from the
# LKG model at the start of each meta-loop. For the 'full' mode this means randomly
# selecting a layer first (including the output layer), then a random neuron from
# within the layer. For the 'hidden' mode this means randomly selecting from
# a list of all available neurons from the hidden layers only. Each meta-loop
# starts fresh with a single ablation - ablations do not accumulate across loops.
#
# 4. Dynamic Architecture: Use --arch "[4*4, 2*8]" to define complex models.
# 5. Frustration Engine: The training orchestration driving the experiments.
# 6. ResMLP Architecture: Residual connections to solve vanishing gradient problems.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import argparse
import logging
from pathlib import Path
import random
from dataclasses import dataclass
import os
import copy
import re
import platform

# --- Dependency Imports ---
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.progress import (
    Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn,
)
from rich.table import Table
from safetensors.torch import save_file, load_file

# --- 1. Configuration & Argument Parsing ---

def detect_best_device() -> str:
    """Detect the best available device, prioritizing CUDA > Metal > CPU."""
    # Check for CUDA first (highest priority)
    if torch.cuda.is_available():
        return "cuda"
    
    # Check for Metal (Apple Silicon)
    if platform.system() == "Darwin":  # macOS
        try:
            # Check if MPS (Metal Performance Shaders) is available
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except (AttributeError, RuntimeError):
            pass
    
    # Fallback to CPU
    return "cpu"

@dataclass
class Config:
    """Configuration class for all hyperparameters and settings."""
    MODEL_DIR: str = "models/"
    CHECKPOINT_NAME: str = "mnist_lkg.safetensors"
    INPUT_SIZE: int = 28 * 28
    HIDDEN_LAYERS: list[int] = None
    ARCH_STRING: str = ""
    OUTPUT_SIZE: int = 10
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 256
    NUM_META_LOOPS: int = 100
    DEVICE: str = detect_best_device()
    ABLATION_MODE: str = "none"
    LOG_INTERVAL: int = 20
    DEBUG: bool = False
    NUM_WORKERS: int = 4
    NUM_RUNS: int = 1  # New field for number of experiment runs
    WEIGHT_DECAY: float = 1e-4
    DROPOUT_RATE: float = 0.1
    GLOBAL_META_LOOPS: int = 0  # Total meta-loops trained across all sessions
    BOUNTY_META_LOOP: int = 0   # Global meta-loop where bounty was last improved

    def getCheckpointPath(self) -> Path:
        return Path(self.MODEL_DIR) / self.CHECKPOINT_NAME

def parse_arch_string(arch_str: str) -> list[int]:
    """Parse architecture string '[4*4, 2*8]' into list [4, 4, 8, 8]."""
    if not re.match(r'^\[[\d\s,\*]+\]$', arch_str):
        raise ValueError(f"Invalid architecture string format: {arch_str}")

    content = arch_str.strip()[1:-1] # Remove brackets
    if not content:
        return []

    hidden_dims = []
    parts = [part.strip() for part in content.split(',')]
    for part in parts:
        try:
            count, size = [int(p.strip()) for p in part.split('*')]
            hidden_dims.extend([size] * count)
        except ValueError:
            raise ValueError(f"Malformed segment in architecture string: '{part}'")
    return hidden_dims

def get_config() -> Config:
    """Parse command-line arguments and return a Config dataclass instance."""
    parser = argparse.ArgumentParser(description="Persistent Stochastic Ablation Trainer for MNIST")
    parser.add_argument("--model-dir", type=str, default="models/", help="Path to store/retrieve models.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--meta-loops", type=int, default=100, help="Total meta-loops to run.")
    parser.add_argument(
        "--arch", type=str, default=None,
        help="Define architecture with a string, e.g., '[4*4, 2*8, 1*16]'."
    )
    parser.add_argument(
        "--hidden-layers", type=int, nargs='+', default=[1024],
        help="A list of hidden layer sizes (used if --arch is not provided)."
    )
    parser.add_argument(
        "--ablation-mode", type=str, default="none", choices=["none", "decay", "dropout", "full", "hidden", "output"],
        help="Set the ablation mode."
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of times to run the experiment for statistical analysis.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], 
                         help="Override device detection (cpu, cuda, mps)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay rate (only used with --ablation-mode decay)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (only used with --ablation-mode dropout)")
    args = parser.parse_args()

    if args.arch:
        hidden_layers = parse_arch_string(args.arch)
        arch_string_for_display = args.arch
    else:
        hidden_layers = args.hidden_layers
        arch_string_for_display = f"[{'*'.join(map(str, [len(hidden_layers), hidden_layers[0]])) if len(set(hidden_layers)) == 1 else 'Custom'}]"


    # SageMaker sets /opt/ml/model as the model directory
    # Use this if we're in a SageMaker environment, otherwise use the provided path
    if os.path.exists("/opt/ml/model"):
        model_dir = Path("/opt/ml/model")
    else:
        model_dir = Path(args.model_dir)
    
    model_dir.mkdir(parents=True, exist_ok=True)

    # Handle device override
    device = args.device if args.device else detect_best_device()

    config = Config(
        MODEL_DIR=str(model_dir),
        LEARNING_RATE=args.lr,
        BATCH_SIZE=args.batch_size,
        NUM_META_LOOPS=args.meta_loops,
        HIDDEN_LAYERS=hidden_layers,
        ARCH_STRING=arch_string_for_display,
        ABLATION_MODE=args.ablation_mode,
        DEBUG=args.debug,
        NUM_WORKERS=args.num_workers,
        NUM_RUNS=args.num_runs,
        WEIGHT_DECAY=args.weight_decay,
        DROPOUT_RATE=args.dropout,
    )
    # Override device if specified
    if args.device:
        config.DEVICE = device
    return config

def setup_logging(is_debug: bool, console: Console) -> logging.Logger:
    """Configure logging with rich formatting."""
    log_level = "DEBUG" if is_debug else "INFO"
    logging.basicConfig(level=log_level, format="%(message)s", datefmt="[%X]",
                        handlers=[RichHandler(rich_tracebacks=True, show_path=is_debug, console=console)])
    return logging.getLogger("rich")

# --- 2. Core Model and Ablator ---

class ResidualBlock(nn.Module):
    """A single residual block: Linear -> ReLU -> Linear, with a skip connection."""
    def __init__(self, width: int, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(width, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.layers(x))

class ResNetStack(nn.Module):
    """A stack of multiple ResidualBlocks of the same width."""
    def __init__(self, depth: int, width: int, dropout_rate: float = 0.1):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout_rate) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

class ResMLP(nn.Module):
    """A flexible ResNet-based MLP built from heterogeneous stacks of residual blocks."""
    def __init__(self, config: Config):
        super().__init__()
        self.flatten = nn.Flatten()
        self.config = config
        self.layer_stack = nn.ModuleList()

        if not config.HIDDEN_LAYERS:
            # Handle edge case of no hidden layers
            self.layer_stack.append(nn.Linear(config.INPUT_SIZE, config.OUTPUT_SIZE))
            return

        # Parse architecture into (depth, width) tuples
        arch_defs = []
        if config.HIDDEN_LAYERS:
            i = 0
            while i < len(config.HIDDEN_LAYERS):
                size = config.HIDDEN_LAYERS[i]
                count = 1
                while i + count < len(config.HIDDEN_LAYERS) and config.HIDDEN_LAYERS[i + count] == size:
                    count += 1
                arch_defs.append({'depth': count, 'width': size})
                i += count
        
        # Build the dynamic architecture
        current_width = config.INPUT_SIZE
        for i, definition in enumerate(arch_defs):
            block_width = definition['width']
            block_depth = definition['depth']

            # Add a transition layer if widths differ
            if current_width != block_width:
                self.layer_stack.append(nn.Linear(current_width, block_width))
                self.layer_stack.append(nn.ReLU())
                # Add dropout for transition layers
                self.layer_stack.append(nn.Dropout(config.DROPOUT_RATE))
            
            self.layer_stack.append(ResNetStack(block_depth, block_width, config.DROPOUT_RATE))
            current_width = block_width

        # Add the final projection layer
        self.layer_stack.append(nn.Linear(current_width, config.OUTPUT_SIZE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        
        # --- Start of Fix ---
        dropout_layers = []
        original_modes = []

        # If the mode is NOT dropout, we need to find and disable all dropout layers
        if self.config.ABLATION_MODE != 'dropout':
            # Recursively find ALL nn.Dropout modules in the entire model
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    dropout_layers.append(module)
                    original_modes.append(module.training) # Store its original state
                    module.eval() # Set to eval mode to disable it
        # --- End of Fix ---

        # The forward pass itself remains the same
        for layer in self.layer_stack:
            x = layer(x)
        
        # --- Start of Fix ---
        # Restore the original training state of the dropout layers
        if self.config.ABLATION_MODE != 'dropout':
            for layer, original_mode in zip(dropout_layers, original_modes):
                layer.train(original_mode)
        # --- End of Fix ---
                
        return x


class Ablator:
    """Handles different modes of neuron ablation for the ResMLP."""
    def __init__(self, model: ResMLP, mode: str, log: logging.Logger):
        self.log = log
        self.mode = mode
        self.ablatable_targets = []
        # Only set up ablation targets for actual ablation modes
        if self.mode == "hidden": 
            self._index_hidden_neurons(model)
        elif self.mode == "full": 
            self._index_full_layers(model)
        elif self.mode == "output": 
            self._index_output_layer(model)
        # baseline modes (none, decay, dropout) don't need ablation targets

    def _get_all_linear_layers_in_order(self, model: ResMLP) -> list[nn.Linear]:
        """Traverses the model and returns a flat list of linear layers in compute order."""
        linear_layers = []
        for module in model.layer_stack:
            if isinstance(module, nn.Linear):
                linear_layers.append(module)
            elif isinstance(module, ResNetStack):
                for block in module.blocks:
                    # Layers inside a ResidualBlock are in a Sequential
                    linear_layers.append(block.layers[0])
                    linear_layers.append(block.layers[3])  # Second linear layer (index 3 due to dropout)
        return linear_layers

    def _index_full_layers(self, model: ResMLP):
        """Indexes all linear layers for the 'full' ablation mode."""
        all_linear_layers = self._get_all_linear_layers_in_order(model)
        module_to_name = {v: k for k, v in model.named_modules()}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear): 
                self.ablatable_targets.append({'name': name, 'module': module})
        self.log.info(f"Ablator (full mode) indexed {len(self.ablatable_targets)} total linear layers.")

    def _index_hidden_neurons(self, model: ResMLP):
        """Indexes hidden neurons for the 'hidden' ablation mode."""
        all_linear_layers = self._get_all_linear_layers_in_order(model)
        if len(all_linear_layers) <= 1:
            self.log.warning("Ablator (hidden mode) found no hidden layers to index.")
            return
        
        # Hidden layers are all linear layers EXCEPT the first and last
        hidden_layers_with_successors = list(zip(all_linear_layers[1:-1], all_linear_layers[2:]))
        module_to_name = {v: k for k, v in model.named_modules()}
        
        self.log.info(f"Ablator (hidden mode) identified {len(hidden_layers_with_successors)} hidden layers for ablation.")

        for current_layer, successor_layer in hidden_layers_with_successors:
            layer_name = module_to_name[current_layer]
            next_layer_name = module_to_name[successor_layer]
            for neuron_idx in range(current_layer.out_features):
                self.ablatable_targets.append({
                    "layer_name": layer_name, "neuron_idx": neuron_idx,
                    "incoming_weight_key": f"{layer_name}.weight",
                    "incoming_bias_key": f"{layer_name}.bias",
                    "outgoing_weight_key": f"{next_layer_name}.weight"
                })
        if self.ablatable_targets: 
            self.log.info(f"Ablator (hidden mode) indexed {len(self.ablatable_targets)} hidden neurons.")

    def _index_output_layer(self, model: ResMLP):
        """Indexes only the final linear layer for the 'output' ablation mode."""
        all_linear_layers = self._get_all_linear_layers_in_order(model)
        if not all_linear_layers:
            self.log.warning("Ablator (output mode) found no linear layers.")
            return
        output_layer_module = all_linear_layers[-1]
        module_to_name = {v: k for k, v in model.named_modules()}
        output_layer_name = module_to_name[output_layer_module]
        self.ablatable_targets.append({'name': output_layer_name, 'module': output_layer_module})
        self.log.info(f"Ablator (output mode) indexed the final output layer: '{output_layer_name}'.")

    def ablate(self, model: ResMLP) -> dict:
        if self.mode == "none" or not self.ablatable_targets: 
            return model.state_dict()
        state_dict = copy.deepcopy(model.state_dict())

        if self.mode == "output":
            target = self.ablatable_targets[0] 
            name, module = target['name'], target['module']
            idx = random.randint(0, module.out_features - 1)
            self.log.info(f"🧠 (Output Mode) Partially ablating neuron {idx} in output layer '{name}'.")
            state_dict[f"{name}.weight"][idx, :] = 0.0
            if module.bias is not None: 
                state_dict[f"{name}.bias"][idx] = 0.0
        
        elif self.mode == "full":
            target = random.choice(self.ablatable_targets)
            name, module = target['name'], target['module']
            idx = random.randint(0, module.out_features - 1)
            self.log.info(f"🧠 (Full Mode) Partially ablating neuron {idx} in layer '{name}'.")
            state_dict[f"{name}.weight"][idx, :] = 0.0
            if module.bias is not None: 
                state_dict[f"{name}.bias"][idx] = 0.0
        
        elif self.mode == "hidden":
            target = random.choice(self.ablatable_targets)
            name, idx = target['layer_name'], target['neuron_idx']
            self.log.info(f"🧠 (Hidden Mode) Fully ablating neuron {idx} in hidden layer '{name}'.")
            state_dict[target['incoming_weight_key']][idx, :] = 0.0
            state_dict[target['incoming_bias_key']][idx] = 0.0
            state_dict[target['outgoing_weight_key']][:, idx] = 0.0
        
        return state_dict

# --- 3. UI and Helper Functions ---

def display_architecture_summary(model: ResMLP, config: Config, console: Console):
    """Creates and prints a summary table of the ResMLP model's architecture."""
    table = Table(title=f"ResMLP Architecture Summary: {config.ARCH_STRING}", show_header=True, header_style="bold magenta")
    
    headers, shapes, params = ["Metric"], ["Shape"], ["Parameters"]
    
    current_width = config.INPUT_SIZE
    shapes.append(f"{current_width}")
    params.append("-")

    for i, module in enumerate(model.layer_stack):
        if isinstance(module, nn.Linear):
            # Infer if it's initial, transition, or final
            if i == 0 and len(model.layer_stack) > 1: 
                header = "Initial Proj."
            elif i == len(model.layer_stack) - 1: 
                header = "Final Proj."
            else: 
                header = "Transition"
            
            shapes.append(f"{module.in_features} → {module.out_features}")
            p = sum(p.numel() for p in module.parameters())
            params.append(f"{p:,}")
        
        elif isinstance(module, nn.ReLU):
            header = f"ReLU_{i-1}"
            shapes.append("✓")
            params.append("-")
            continue # Skip adding a new column for ReLUs

        elif isinstance(module, nn.Dropout):
            header = f"Dropout_{i-1}"
            shapes.append("✓")
            params.append("-")
            continue # Skip adding a new column for Dropout

        elif isinstance(module, ResNetStack):
            block = module.blocks[0]
            depth = len(module.blocks)
            width = block.layers[0].in_features
            header = f"ResStack ({depth}x{width})"
            shapes.append(f"{width} → {width}")
            p = sum(p.numel() for p in module.parameters())
            params.append(f"{p:,}")
        
        headers.append(header)

    # Add columns and rows to table
    for h in headers: 
        table.add_column(h, justify="center")
    table.add_row(*shapes)
    table.add_row(*params)
    console.print(table)


def get_mnist_loaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    MODIFIED for high-performance SageMaker training with proper train/validation/test splits.
    Loads the entire MNIST dataset into memory to eliminate I/O bottlenecks.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_path = os.environ.get("SM_CHANNEL_TRAINING", "dataset")

    # Step 1: Load the full datasets from disk/S3 channel ONCE.
    full_train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    # Step 2: Split training data into train (50k) and validation (10k)
    train_size = 50000
    val_size = 10000
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_train_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(1337)  # Fixed seed for reproducibility
    )
    
    train_subset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(full_train_dataset, val_indices.indices)

    # Step 3: Iterate through the datasets and pull everything into RAM.
    train_images = torch.stack([img for idx in train_indices.indices for img, _ in [full_train_dataset[idx]]])
    train_labels = torch.tensor([label for idx in train_indices.indices for _, label in [full_train_dataset[idx]]])
    
    val_images = torch.stack([img for idx in val_indices.indices for img, _ in [full_train_dataset[idx]]])
    val_labels = torch.tensor([label for idx in val_indices.indices for _, label in [full_train_dataset[idx]]])
    
    test_images = torch.stack([img for img, _ in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])

    # Step 4: Create new, in-memory TensorDatasets.
    train_ds = TensorDataset(train_images, train_labels)
    val_ds = TensorDataset(val_images, val_labels)
    test_ds = TensorDataset(test_images, test_labels)

    # Step 5: Return DataLoaders that now wrap the in-memory datasets.
    # Use num_workers=0 for in-memory data to avoid overhead
    # Disable pin_memory on MPS (not supported) and use smaller batch for validation
    pin_memory = config.DEVICE == "cuda"  # Only use pin_memory for CUDA
    
    return DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory), \
           DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=pin_memory), \
           DataLoader(test_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=pin_memory)

# --- 4. Training & Evaluation Functions ---

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, config: Config, progress: Progress, task_id: TaskID):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(config.DEVICE), target.to(config.DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        progress.update(task_id, advance=1)
        if batch_idx % config.LOG_INTERVAL == 0:
            progress.update(task_id, description=f"Training... Loss: {loss.item():.4f}")

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, config: Config, progress: Progress) -> float:
    model.eval()
    correct, total = 0, 0
    eval_task = progress.add_task("[cyan]Evaluating...", total=len(loader))
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            progress.update(eval_task, advance=1)
    progress.remove_task(eval_task)
    return 100. * correct / total

# --- 5. Main Orchestration ---

def run_single_experiment(config: Config, console: Console, log: logging.Logger) -> float:
    """
    Contains the logic for ONE full experimental run (100 meta-loops).
    Returns the final best accuracy (bounty) for this single run.
    """
    model = ResMLP(config).to(config.DEVICE)
    ablator = Ablator(model, config.ABLATION_MODE, log)
    criterion = nn.CrossEntropyLoss()

    log.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
    display_architecture_summary(model, config, console)
    train_loader, val_loader, test_loader = get_mnist_loaders(config)

    lkg_score, bounty = -1.0, -1.0
    lkg_model_state = model.state_dict()
    active_model_state = copy.deepcopy(lkg_model_state)
    
    # Initialize global tracking for this run (SageMaker jobs are isolated)
    current_global_loops = 0

    try:
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(), console=console) as progress:
            meta_loop_task = progress.add_task("[bold]Meta-Loops[/bold]", total=config.NUM_META_LOOPS)
            for loop in range(config.NUM_META_LOOPS):
                current_global_loop = current_global_loops + loop + 1
                model.load_state_dict(active_model_state)
                
                # Reset optimizer each meta-loop for fair comparison
                # This ensures each meta-loop starts with a fresh optimizer state,
                # providing a fair comparison between ablation strategies without
                # momentum carryover from previous attempts.
                if config.ABLATION_MODE == "decay":
                    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
                    if loop == 0:  # Log once at the start
                        log.info(f"🔧 Using weight decay: {config.WEIGHT_DECAY}")
                elif config.ABLATION_MODE == "dropout":
                    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
                    if loop == 0:  # Log once at the start
                        log.info(f"🔧 Using dropout rate: {config.DROPOUT_RATE}")
                else:
                    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
                
                train_task = progress.add_task("Training...", total=len(train_loader))
                train_one_epoch(model, train_loader, optimizer, criterion, config, progress, train_task)
                progress.remove_task(train_task)

                new_score = evaluate(model, val_loader, criterion, config, progress)
                if new_score > lkg_score:
                    lkg_score, lkg_model_state = new_score, copy.deepcopy(model.state_dict())
                    if lkg_score > bounty: 
                        bounty = lkg_score
                        config.BOUNTY_META_LOOP = current_global_loop
                
                log.info(f"Loop {loop + 1}/{config.NUM_META_LOOPS} (Global: {current_global_loop}) | Current: {new_score:.2f}% | LKG: {lkg_score:.2f}% | Bounty: {bounty:.2f}%")

                temp_model = ResMLP(config)
                temp_model.load_state_dict(lkg_model_state)
                if config.ABLATION_MODE not in ['none', 'decay', 'dropout']:
                    active_model_state = ablator.ablate(temp_model)
                else:
                    active_model_state = copy.deepcopy(lkg_model_state)
                
                progress.update(meta_loop_task, advance=1)

    except KeyboardInterrupt:
        log.warning("\nTraining run interrupted by user.")
    finally:
        # Update global meta-loop count for this run
        current_global_loops += config.NUM_META_LOOPS
        
        # Final evaluation on test set
        if lkg_model_state:
            model.load_state_dict(lkg_model_state)
            with Progress(transient=True, console=console) as progress:
                final_test_accuracy = evaluate(model, test_loader, criterion, config, progress)
            log.info(f"🧪 Final Test Accuracy: {final_test_accuracy:.2f}%")
        
        log.info(f"🏆 Run Finished. Final Bounty (validation accuracy): {bounty:.2f}% @ {config.BOUNTY_META_LOOP}/{current_global_loops}")
        return bounty

def main():
    """Main orchestrator: sets up config and calls the experiment runner multiple times."""
    config = get_config()
    console = Console()
    log = setup_logging(config.DEBUG, console)
    all_bounties = []

    console.print(Panel(f"[bold yellow]Executing {config.NUM_RUNS} runs for ResMLP Arch: {config.ARCH_STRING}, Mode: {config.ABLATION_MODE}[/bold yellow]",
                        subtitle=f"Device: {config.DEVICE}"))

    for i in range(config.NUM_RUNS):
        console.print(Panel(f"[magenta]Starting Run {i + 1} of {config.NUM_RUNS}[/magenta]"))
        
        # IMPORTANT: Ensure a clean slate for each run by removing the old checkpoint
        checkpoint_path = config.getCheckpointPath()
        if checkpoint_path.exists():
            os.remove(checkpoint_path)
            log.warning(f"Removed previous checkpoint to ensure clean run: {checkpoint_path}")

        bounty = run_single_experiment(config, console, log)
        all_bounties.append(bounty)

    console.print(Panel.fit("[bold green]✅ All Runs for this Job Finished.[/bold green]"))
    log.info(f"Collected Bounties: {all_bounties}")

    # --- Save all results to a file that SageMaker will capture to S3 ---
    try:
        results_path = os.path.join(config.MODEL_DIR, "results.txt")
        with open(results_path, 'w') as f:
            for score in all_bounties:
                f.write(f"{score:.2f}\n")
        log.info(f"Saved all run results to {results_path} for parsing.")
    except Exception as e:
        log.error(f"Could not write final results for parsing: {e}")

if __name__ == "__main__":
    main()
