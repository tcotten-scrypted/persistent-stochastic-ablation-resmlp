#!/usr/bin/env python3
"""
Clean script for removing model files from the models directory.

Usage:
    poetry run clean                    # Remove default checkpoint
    poetry run clean --all             # Remove all files in models/
    poetry run clean file1.safetensors # Remove specific file
    poetry run clean --all file1.safetensors file2.safetensors  # Remove all + specific files
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import shutil

# Import the Config class from the trainer to get default values
sys.path.append(str(Path(__file__).parent.parent / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("trainer", Path(__file__).parent.parent / "src" / "train_psa_simplemlp.py")
trainer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer_module)
Config = trainer_module.Config

def get_default_checkpoint_path() -> Path:
    """Get the default checkpoint path from the trainer configuration."""
    config = Config()
    return config.getCheckpointPath()

def clean_files(files_to_remove: list[Path], console: Console, dry_run: bool = False) -> list[Path]:
    """Remove specified files and return list of successfully removed files."""
    removed_files = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        for file_path in files_to_remove:
            if file_path.exists():
                task = progress.add_task(f"Removing {file_path.name}...", total=None)
                if not dry_run:
                    file_path.unlink()
                removed_files.append(file_path)
                progress.update(task, description=f"Removed {file_path.name}")
            else:
                console.print(f"[yellow]⚠️  File not found: {file_path}[/yellow]")
    
    return removed_files

def main():
    parser = argparse.ArgumentParser(
        description="Clean model files from the models directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run clean                    # Remove default checkpoint
  poetry run clean --all             # Remove all files in models/
  poetry run clean file1.safetensors # Remove specific file
  poetry run clean --all file1.safetensors file2.safetensors  # Remove all + specific files
        """
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Remove all files in the models directory"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be removed without actually removing"
    )
    parser.add_argument(
        "files", 
        nargs="*", 
        help="Specific files to remove (can include models/ prefix or not)"
    )
    
    args = parser.parse_args()
    console = Console()
    
    # Get models directory from default config
    models_dir = Path(Config().MODEL_DIR)
    default_checkpoint = get_default_checkpoint_path()
    
    files_to_remove = []
    
    # Handle --all flag
    if args.all:
        if models_dir.exists():
            for file_path in models_dir.iterdir():
                if file_path.is_file():
                    files_to_remove.append(file_path)
        else:
            console.print(f"[yellow]⚠️  Models directory not found: {models_dir}[/yellow]")
    
    # Handle specific files
    for file_arg in args.files:
        file_path = Path(file_arg)
        
        # If it's already an absolute path or has models/ prefix, use as-is
        if file_path.is_absolute() or str(file_path).startswith("models/"):
            target_path = file_path
        else:
            # Otherwise, assume it's relative to models directory
            target_path = models_dir / file_path
        
        files_to_remove.append(target_path)
    
    # If no files specified and no --all, remove default checkpoint
    if not files_to_remove and not args.all:
        files_to_remove = [default_checkpoint]
    
    if not files_to_remove:
        console.print("[yellow]No files specified for removal.[/yellow]")
        return
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file_path in files_to_remove:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)
    
    # Display what will be removed
    table = Table(title="Files to Remove", show_header=True, header_style="bold red")
    table.add_column("File Path", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Status", justify="center")
    
    total_size = 0
    for file_path in unique_files:
        if file_path.exists():
            size = file_path.stat().st_size
            total_size += size
            size_str = f"{size:,} bytes"
            status = "[green]Exists[/green]"
        else:
            size_str = "N/A"
            status = "[yellow]Not Found[/yellow]"
        
        table.add_row(str(file_path), size_str, status)
    
    console.print(table)
    
    if total_size > 0:
        console.print(f"[bold]Total size to remove: {total_size:,} bytes[/bold]")
    
    # Confirm removal
    if args.dry_run:
        console.print("[blue]Dry run mode - no files were actually removed.[/blue]")
        return
    
    if not args.all and len(unique_files) > 1:
        response = console.input("[bold red]Remove these files? (y/N): [/bold red]")
        if response.lower() not in ['y', 'yes']:
            console.print("[yellow]Operation cancelled.[/yellow]")
            return
    
    # Perform removal
    removed_files = clean_files(unique_files, console, dry_run=args.dry_run)
    
    # Summary
    if removed_files:
        console.print(Panel(
            f"[bold green]✅ Successfully removed {len(removed_files)} file(s)[/bold green]",
            title="Clean Complete",
            border_style="green"
        ))
    else:
        console.print("[yellow]No files were removed.[/yellow]")

if __name__ == "__main__":
    main() 