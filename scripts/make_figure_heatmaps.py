#!/usr/bin/env python3
"""
Generates a suite of data-driven heat map visualizations for the SimpleMLP
design space, based on the results of Persistent Stochastic Ablation (PSA) trials.

This script produces five distinct plots to tell a comprehensive story:
1.  Baseline Performance: The raw performance of the control model.
2.  Ablation Effects: The quantitative benefit or harm of applying PSA.
3.  Instability/Chaos: The trial-to-trial variance of each architecture.
4.  Winning Strategy: The single best-performing mode for each architecture.
5.  Parameter Matching: Parameter counts across architectures to show design equivalences.

Author: Tim Cotten @cottenio <tcotten@scrypted.ai, tcotten2@gmu.edu>
Assisted by: Google's Gemini
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import re
from collections import defaultdict

# Color constants for consistent visualization
COLORS = {
    'bright_green': '#4CAF50',    # Beneficial effects
    'deep_green': '#1B5E20',      # Stable/Good performance
    'light_green': '#7CB342',     # Moderate stability
    'red': '#E53935',             # Harmful effects/Unstable
    'brown': '#A1887F',           # Insignificant/Neutral
    'black': '#212121',           # Untrainable
    'blue': '#1565C0',            # Lower performance
    'yellow': '#FFEB3B',          # Moderate instability
    'orange': '#FF8A65',          # High instability
    'purple': '#651FFF',          # Full ablation
    'pink': '#E040FB',            # Hidden ablation
    'cyan': '#00B0FF',            # Output ablation
}

def parse_configurations(config_file: Path) -> list[tuple[int, int]]:
    """
    Parses architecture configurations from a file.
    Expects format like "1*2048" on each line.
    
    Args:
        config_file: Path to the configuration file.
    
    Returns:
        A list of tuples, where each tuple is (depth, width).
    """
    architectures = []
    if not config_file.is_file():
        print(f"Error: Configuration file '{config_file}' not found.")
        return []
    
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '*' in line:
                    depth, width = map(int, line.split('*'))
                    architectures.append((depth, width))
    except ValueError as e:
        print(f"Error parsing configuration file: {e}")
        return []
        
    return architectures

def parse_trial_data(trials_file: Path) -> dict:
    """
    Parses raw trial data from the markdown results file.
    
    Args:
        trials_file: Path to the markdown file containing trial data.
        
    Returns:
        A dictionary mapping architecture keys (e.g., "1x2048") to a list
        of trial result dictionaries.
    """
    trial_results = defaultdict(list)
    if not trials_file.is_file():
        print(f"Error: Trials file '{trials_file}' not found.")
        return {}
        
    with open(trials_file, 'r') as f:
        content = f.read()

    # Regex to find each architecture block and its content
    arch_blocks = re.findall(r'### Architecture: ([\d\*]+)\n\n(.*?)(?=\n### Architecture:|$)', content, re.DOTALL)
    
    for arch_str, block in arch_blocks:
        arch_key = arch_str.replace('*', 'x')
        
        # Find the raw data table within the block
        table_match = re.search(r'\| Trial \|.*?\|(.*)', block, re.DOTALL)
        if not table_match:
            continue
        
        table_content = table_match.group(1)
        lines = table_content.strip().split('\n')
        
        for line in lines:
            if '|' not in line or '---' in line:
                continue

            parts = [p.strip().replace('%', '') for p in line.split('|')]
            if len(parts) >= 6:
                try:
                    trial_data = {
                        'none': float(parts[2]),
                        'full': float(parts[3]),
                        'hidden': float(parts[4]),
                        'output': float(parts[5])
                    }
                    trial_results[arch_key].append(trial_data)
                except (ValueError, IndexError):
                    continue
                    
    return dict(trial_results)

def calculate_summary_metrics(trial_results: dict) -> dict:
    """
    Calculates mean and standard deviation for each ablation mode across all trials.
    
    Args:
        trial_results: The dictionary of parsed trial data.
    
    Returns:
        A nested dictionary with summary statistics for each architecture.
    """
    metrics = {}
    for arch_key, trials in trial_results.items():
        if not trials:
            continue
        
        metrics[arch_key] = {}
        for mode in ['none', 'full', 'hidden', 'output']:
            accuracies = [t[mode] for t in trials]
            metrics[arch_key][mode] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies)
            }
    return metrics

def setup_plot(title: str) -> tuple[plt.Figure, plt.Axes]:
    """Creates a standard figure and axis with common styling."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.8, 4000)
    ax.set_ylim(0.8, 4000)
    ax.set_xlabel('Depth (# of layers)', fontsize=14)
    ax.set_ylabel('Width (neurons per layer)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    return fig, ax

def save_and_close(fig: plt.Figure, output_file: Path):
    """Saves a figure to a file and closes it to free memory."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Plot saved successfully to: {output_file}")

def plot_baseline_performance(architectures: list, metrics: dict, output_file: Path):
    """Plots the mean accuracy of the control 'none' mode."""
    title = 'SimpleMLP Design Space: Baseline Performance (Control)'
    fig, ax = setup_plot(title)
    
    depths = [arch[0] for arch in architectures]
    widths = [arch[1] for arch in architectures]
    accuracies = []
    untrainable_mask = []
    
    for d, w in zip(depths, widths):
        arch_key = f"{d}x{w}"
        if arch_key in metrics:
            m = metrics[arch_key]
            # Check if all modes achieve ≤11.35% (untrainable)
            all_untrainable = all(m[mode]['mean'] <= 11.35 for mode in ['none', 'full', 'hidden', 'output'])
            untrainable_mask.append(all_untrainable)
            
            if all_untrainable:
                accuracies.append(0)  # Placeholder value for untrainable
            else:
                accuracies.append(m['none']['mean'])
        else:
            accuracies.append(10)
            untrainable_mask.append(False)

    # Create custom color gradient: blue -> green
    from matplotlib.colors import LinearSegmentedColormap
    colors = [COLORS['blue'], COLORS['bright_green']]  # Blue -> green
    n_bins = 100
    custom_cmap = LinearSegmentedColormap.from_list('baseline_gradient', colors, N=n_bins)
    
    # Plot trainable architectures with color gradient
    trainable_depths = [d for i, d in enumerate(depths) if not untrainable_mask[i]]
    trainable_widths = [w for i, w in enumerate(widths) if not untrainable_mask[i]]
    trainable_accuracies = [acc for i, acc in enumerate(accuracies) if not untrainable_mask[i]]
    
    if trainable_accuracies:  # Only plot if there are trainable architectures
        sc = ax.scatter(trainable_depths, trainable_widths, c=trainable_accuracies, cmap=custom_cmap, s=120, edgecolors='black', linewidth=0.5, vmin=11.35, vmax=100)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Mean Accuracy of Control (%)', fontsize=12)
    
    # Plot untrainable architectures as black
    untrainable_depths = [d for i, d in enumerate(depths) if untrainable_mask[i]]
    untrainable_widths = [w for i, w in enumerate(widths) if untrainable_mask[i]]
    
    if untrainable_depths:  # Only plot if there are untrainable architectures
        ax.scatter(untrainable_depths, untrainable_widths, c=COLORS['black'], s=120, edgecolors='black', linewidth=0.5, label='Untrainable')
        ax.legend(loc='upper right', fontsize=10)
    
    save_and_close(fig, output_file)

def plot_ablation_effects(architectures: list, metrics: dict, trial_results: dict, output_file: Path):
    """Plots the performance difference between the best ablation mode and control with statistical significance and effect size."""
    title = 'SimpleMLP Design Space: Ablation Effects (Benefit/Harm)'
    fig, ax = setup_plot(title)

    from scipy import stats

    depths = [arch[0] for arch in architectures]
    widths = [arch[1] for arch in architectures]
    colors = []
    sizes = []
    untrainable_mask = []

    for d, w in zip(depths, widths):
        arch_key = f"{d}x{w}"
        if arch_key in trial_results and arch_key in metrics:
            trials = trial_results[arch_key]
            m = metrics[arch_key]
            # Check if untrainable
            all_untrainable = all(m[mode]['mean'] <= 11.35 for mode in ['none', 'full', 'hidden', 'output'])
            untrainable_mask.append(all_untrainable)
            if all_untrainable:
                colors.append(COLORS['black'])
                sizes.append(120)
            else:
                uplifts = [max(trial['full'], trial['hidden'], trial['output']) - trial['none'] for trial in trials]
                if len(uplifts) > 1 and np.std(uplifts) > 0:
                    t_stat, p_value = stats.ttest_1samp(uplifts, 0)
                    mean_uplift = np.mean(uplifts)
                    effect_size = abs(mean_uplift) / np.std(uplifts)
                    # Color
                    if p_value < 0.05:
                        if mean_uplift > 0:
                            colors.append(COLORS['bright_green'])  # Bright green
                        else:
                            colors.append(COLORS['red'])  # Red
                    else:
                        colors.append(COLORS['brown'])  # Brown
                    # Size (normalize effect size: 60-200)
                    normalized_size = 60 + (min(effect_size, 7) / 7) * 140
                    sizes.append(normalized_size)
                else:
                    colors.append(COLORS['brown'])  # Brown
                    sizes.append(60)
        else:
            colors.append(COLORS['brown'])  # Brown
            sizes.append(60)
            untrainable_mask.append(False)

    sc = ax.scatter(depths, widths, c=colors, s=sizes, edgecolors='black', linewidth=0.5)
    # Add legend for color
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['bright_green'], edgecolor='black', label='Beneficial'),
        Patch(facecolor=COLORS['red'], edgecolor='black', label='Harmful'),
        Patch(facecolor=COLORS['brown'], edgecolor='black', label='Insignificant'),
        Patch(facecolor=COLORS['black'], edgecolor='black', label='Untrainable'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    save_and_close(fig, output_file)

def plot_instability(architectures: list, metrics: dict, output_file: Path):
    """Plots the maximum standard deviation across all modes for each architecture."""
    title = 'SimpleMLP Design Space: Instability (Chaos) Map'
    fig, ax = setup_plot(title)

    depths = [arch[0] for arch in architectures]
    widths = [arch[1] for arch in architectures]
    max_stds = []
    untrainable_mask = []
    
    for d, w in zip(depths, widths):
        arch_key = f"{d}x{w}"
        if arch_key in metrics:
            m = metrics[arch_key]
            # Check if all modes achieve ≤11.35% (untrainable)
            all_untrainable = all(m[mode]['mean'] <= 11.35 for mode in ['none', 'full', 'hidden', 'output'])
            untrainable_mask.append(all_untrainable)
            
            if all_untrainable:
                max_stds.append(0)  # Placeholder value for untrainable
            else:
                max_stds.append(max(m['none']['std'], m['full']['std'], m['hidden']['std'], m['output']['std']))
        else:
            max_stds.append(0)
            untrainable_mask.append(False)

    # Create custom color gradient: green (stable) -> yellow -> orange -> red (unstable)
    from matplotlib.colors import LinearSegmentedColormap
    colors = [COLORS['deep_green'], COLORS['light_green'], COLORS['yellow'], COLORS['orange'], COLORS['red']]  # Deep green -> light green -> yellow -> orange -> bright red
    n_bins = 100
    custom_cmap = LinearSegmentedColormap.from_list('stability_gradient', colors, N=n_bins)
    
    # Plot trainable architectures with color gradient
    trainable_depths = [d for i, d in enumerate(depths) if not untrainable_mask[i]]
    trainable_widths = [w for i, w in enumerate(widths) if not untrainable_mask[i]]
    trainable_stds = [std for i, std in enumerate(max_stds) if not untrainable_mask[i]]
    
    if trainable_stds:  # Only plot if there are trainable architectures
        sc = ax.scatter(trainable_depths, trainable_widths, c=trainable_stds, cmap=custom_cmap, s=120, edgecolors='black', linewidth=0.5, vmin=0, vmax=30)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Maximum Standard Deviation (%)', fontsize=12)
    
    # Plot untrainable architectures as black
    untrainable_depths = [d for i, d in enumerate(depths) if untrainable_mask[i]]
    untrainable_widths = [w for i, w in enumerate(widths) if untrainable_mask[i]]
    
    if untrainable_depths:  # Only plot if there are untrainable architectures
        ax.scatter(untrainable_depths, untrainable_widths, c=COLORS['black'], s=120, edgecolors='black', linewidth=0.5, label='Untrainable')
        ax.legend(loc='upper right', fontsize=10)
    
    save_and_close(fig, output_file)

def plot_winning_strategy(architectures: list, metrics: dict, output_file: Path):
    """Plots a categorical map showing which mode had the highest mean accuracy."""
    title = 'SimpleMLP Design Space: Winning Strategy Map'
    fig, ax = setup_plot(title)
    
    depths = np.array([arch[0] for arch in architectures])
    widths = np.array([arch[1] for arch in architectures])
    
    # Updated color scheme with non-heat colors
    mode_colors = {
        'none': COLORS['brown'],      # Brown
        'full': COLORS['purple'],      # Purple
        'hidden': COLORS['pink'],    # Pink
        'output': COLORS['cyan'],    # Blue
        'untrainable': COLORS['black'] # Dark black
    }
    mode_labels = {
        'none': 'Control (None)', 
        'full': 'Full Ablation', 
        'hidden': 'Hidden Ablation', 
        'output': 'Output Ablation',
        'untrainable': 'Untrainable'
    }
    
    winners = []
    for d, w in zip(depths, widths):
        arch_key = f"{d}x{w}"
        winner = 'none' # Default
        if arch_key in metrics:
            m = metrics[arch_key]
            # Check if all modes achieve ≤11.35% (untrainable)
            all_untrainable = all(m[mode]['mean'] <= 11.35 for mode in ['none', 'full', 'hidden', 'output'])
            if all_untrainable:
                winner = 'untrainable'
            else:
                # Find the mode with the highest mean accuracy
                winner = max(m, key=lambda mode: m[mode]['mean'])
        winners.append(winner)
    winners = np.array(winners)

    for mode_name, label in mode_labels.items():
        mask = winners == mode_name
        if np.any(mask):
            ax.scatter(depths[mask], widths[mask], c=mode_colors[mode_name], s=120, edgecolors='black', linewidth=0.5, label=label)

    ax.legend(title='Winning Mode', loc='upper right', fontsize=10)
    
    save_and_close(fig, output_file)

def plot_parameter_matching(architectures: list, output_file: Path):
    """Plots parameter counts across architectures to show parameter matching between different designs."""
    title = 'SimpleMLP Design Space: Parameter Matching over Designs'
    fig, ax = setup_plot(title)

    depths = [arch[0] for arch in architectures]
    widths = [arch[1] for arch in architectures]
    
    # Calculate parameter counts for each architecture
    parameter_counts = []
    for d, w in zip(depths, widths):
        # Parameter count formula: input_size * hidden_layers + hidden_layers * output_size + biases
        input_size = 784  # MNIST flattened
        output_size = 10  # MNIST digits
        
        if d == 0:  # No hidden layers
            param_count = input_size * output_size + output_size  # weights + bias
        else:
            # Input to first hidden layer
            param_count = input_size * w + w  # weights + bias
            # Hidden to hidden layers
            for _ in range(d - 1):
                param_count += w * w + w  # weights + bias
            # Last hidden to output
            param_count += w * output_size + output_size  # weights + bias
        
        parameter_counts.append(param_count)
    
    # Find min and max for color scaling
    min_params = min(parameter_counts)
    max_params = max(parameter_counts)
    
    # Create logarithmic size scaling
    log_params = np.log10(parameter_counts)
    min_log = np.log10(min_params)
    max_log = np.log10(max_params)
    
    # Normalize sizes (7.5 to 469)
    sizes = 7.5 + ((log_params - min_log) / (max_log - min_log)) * 461.5
    
    # Create black-to-blue gradient based on parameter count
    from matplotlib.colors import LinearSegmentedColormap
    black_blue_colors = [COLORS['black'], COLORS['blue'], '#2196F3']  # Black to dark blue to light blue
    n_bins = 100
    black_blue_cmap = LinearSegmentedColormap.from_list('black_blue_gradient', black_blue_colors, N=n_bins)
    
    # Normalize parameter counts for color mapping (logarithmic to match exponential growth)
    log_params = np.log10(parameter_counts)
    min_log = np.log10(min_params)
    max_log = np.log10(max_params)
    normalized_params = (log_params - min_log) / (max_log - min_log)
    
    sc = ax.scatter(depths, widths, c=normalized_params, cmap=black_blue_cmap, s=sizes, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Parameter Count', fontsize=12)
    
    # Add legend for size
    from matplotlib.patches import Circle
    legend_elements = [
        Circle((0, 0), radius=8, facecolor=COLORS['black'], edgecolor='black', label=f'Small ({min_params:,} params)'),
        Circle((0, 0), radius=16, facecolor='#2196F3', edgecolor='black', label=f'Large ({max_params:,} params)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    save_and_close(fig, output_file)

def main():
    """Main function to parse arguments and generate all heat map visualizations."""
    parser = argparse.ArgumentParser(description="Generate a suite of heat map visualizations for SimpleMLP PSA results.")
    parser.add_argument('--config-file', type=Path, default=Path('reproduction/configurations.txt'),
                        help="Path to the architecture configurations file.")
    parser.add_argument('--trials-file', type=Path, default=Path('results/psa_simplemlp_trials.md'),
                        help="Path to the markdown file with raw trial data.")
    parser.add_argument('--output-dir', type=Path, default=Path('results/'),
                        help="Directory to save the output PNG files.")
    args = parser.parse_args()

    # --- Load and Process Data ---
    print("--- Starting Visualization Generation ---")
    architectures = parse_configurations(args.config_file)
    if not architectures:
        print("❌ Aborting: No architectures loaded. Check the configuration file path.")
        return

    trial_results = parse_trial_data(args.trials_file)
    if not trial_results:
        print("❌ Aborting: No trial data loaded. Check the trials file path and format.")
        return
        
    metrics = calculate_summary_metrics(trial_results)
    
    print(f"Loaded {len(architectures)} architectures.")
    print(f"Loaded trial data for {len(trial_results)} architectures.")
    print("--- Generating Plots ---")

    # --- Generate and Save Plots ---
    plot_baseline_performance(architectures, metrics, args.output_dir / 'SimpleMLP_Heatmap_Baseline_Performance.png')
    plot_ablation_effects(architectures, metrics, trial_results, args.output_dir / 'SimpleMLP_Heatmap_Ablation_Effects.png')
    plot_instability(architectures, metrics, args.output_dir / 'SimpleMLP_Heatmap_Instability.png')
    plot_winning_strategy(architectures, metrics, args.output_dir / 'SimpleMLP_Heatmap_Winning_Strategy.png')
    plot_parameter_matching(architectures, args.output_dir / 'SimpleMLP_Heatmap_Parameter_Matching.png')

    print("--- Visualization Generation Complete ---")

if __name__ == "__main__":
    main() 