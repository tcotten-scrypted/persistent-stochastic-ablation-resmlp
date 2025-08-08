# Utility Tools

This document describes the utility scripts available in the `scripts/` folder for various analysis and automation tasks.

## Architecture Analysis

### `make_table_architectures.py`

Generate LaTeX tables showing parameter counts for network architectures.

**Purpose**: Creates publication-ready LaTeX tables from the experimental configurations, categorizing architectures as "Matched" or "Asymmetric" based on the paper's classification.

**Usage**:
```bash
# Direct Python
python scripts/make_table_architectures.py

# Poetry command
poetry run make-architecture-table
```

**Output**: LaTeX table with columns:
- **Architecture**: The network configuration (e.g., "1*2048")
- **Parameters**: Total parameter count with comma formatting
- **Parameter Matching**: "Matched" or "Asymmetric" classification

**Input**: Reads from `reproduction/configurations.txt`

**Example Output**:
```latex
\begin{table}[ht]
\centering
\caption{Parameters for Network Architectures}
\label{tab:architecture_parameters}
\begin{tabular}{lcl}
\toprule
\textbf{Architecture} & \textbf{Parameters} & \textbf{Parameter Matching}\\
\midrule
1*2048      & 1,628,170       & Matched \\
2*939       & 1,629,175       & Matched \\
939*2       & 1,629,175       & Asymmetric \\
...
\bottomrule
\end{tabular}
\end{table}
```

**Features**:
- Automatic parameter counting using the same formula as the training script
- Categorization based on the paper's asymmetric configuration list
- Proper LaTeX formatting with booktabs-style tables
- Error handling for missing configuration files

## Results Analysis

### `make_table_trial_accuracy.py`

Generate LaTeX tables from trial accuracy results with statistical information.

**Purpose**: Creates publication-ready LaTeX tables from experimental results, showing mean accuracy with standard deviation for each architecture and ablation mode. Includes sophisticated winner detection with tolerance for statistical ties.

**Usage**:
```bash
# Direct Python
python scripts/make_table_trial_accuracy.py

# Poetry command
poetry run make-trial-accuracy-table

# With custom files and options
python scripts/make_table_trial_accuracy.py --datafile results/my_results.md --order reproduction/my_configs.txt --caption "Custom Caption" --label "custom_label"
```

**Output**: LaTeX table with columns:
- **Architecture**: The network configuration (e.g., "1*2048")
- **`none`**: Mean ± Std accuracy with no ablation
- **`full`**: Mean ± Std accuracy with full ablation
- **`hidden`**: Mean ± Std accuracy with hidden layer ablation
- **`output`**: Mean ± Std accuracy with output layer ablation
- **Winner**: The ablation mode(s) with highest accuracy (handles ties)

**Input**: 
- Reads from `results/psa_resmlp_summary.md` (default)
- Uses `reproduction/configurations.txt` for row ordering

**Example Output**:
```latex
\begin{table}[ht]
\centering
\caption{Mean Peak Accuracy (\%) with Standard Deviation over 10 Trials for ResMLP Architectures}
\label{tab:results_summary_stats}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l l l l l l}
\toprule
\textbf{Architecture} & {\textbf{`none`}} & {\textbf{`full`}} & {\textbf{`hidden`}} & {\textbf{`output`}} & \textbf{Winner} \\
 & {Mean $\pm$ Std (\%)} & {Mean $\pm$ Std (\%)} & {Mean $\pm$ Std (\%)} & {Mean $\pm$ Std (\%)} & \\
\midrule
1*2048 & \bfseries $98.33 \pm 0.04$ & $98.30 \pm 0.04$ & \bfseries $98.33 \pm 0.05$ & $98.17 \pm 0.07$ & Tie: \texttt{none}/\texttt{hidden} \\
1*1024 & \bfseries $98.26 \pm 0.05$ & $98.19 \pm 0.07$ & $98.25 \pm 0.07$ & $98.03 \pm 0.08$ & \texttt{none} \\
...
\bottomrule
\end{tabular}
}
\end{table}
```

**Features**:
- **Statistical Information**: Shows mean ± standard deviation for all results
- **Tie Detection**: Identifies statistical ties within 0.01% tolerance
- **Professional Formatting**: Uses `\resizebox` for table fitting and mathematical notation
- **Winner Highlighting**: Bold formatting for winning modes
- **Configurable Options**: Custom captions, labels, and file paths
- **Error Handling**: Graceful handling of missing files
- **Maintains Ordering**: Uses configuration file for consistent row ordering

## Visualization

### `make_figure_design_space.py`

Generate design space visualization for ResMLP architectures.

**Purpose**: Creates a logarithmic scatter plot showing the design space of all tested architectures, visualizing the relationship between network depth and width.

**Usage**:
```bash
# Direct Python
python scripts/make_figure_design_space.py

# Poetry command
poetry run make-design-space-figure

# With custom files
python scripts/make_figure_design_space.py --config-file reproduction/my_configs.txt --output-file results/my_plot.png
```

**Output**: High-resolution PNG file showing:
- **Logarithmic Scale**: Both depth and width axes use log scale for better visualization
- **Architecture Points**: Each tested architecture plotted as a rectangle
- **Labels**: Architecture notation (e.g., "1×2048") displayed for each point
- **Grid Lines**: Reference grid for easier reading of values
- **Professional Formatting**: Publication-ready figure with proper labels and title

**Input**: 
- Reads from `reproduction/configurations.txt` (default)
- Outputs to `results/SimpleMLP_Testing_Design_Space.png` (default)

**Features**:
- **Dynamic Configuration Loading**: Reads actual experimental configurations
- **High-Resolution Output**: 300 DPI PNG suitable for publications
- **Logarithmic Visualization**: Optimal for wide-ranging architectural parameters
- **Error Handling**: Graceful handling of missing files and parsing errors
- **Configurable Paths**: Custom input and output file locations
- **Memory Management**: Proper cleanup of matplotlib resources

### `make_figure_heatmaps.py`

Generate comprehensive heat map visualizations for ResMLP design space.

**Purpose**: Creates a suite of four data-driven heat map visualizations that tell a comprehensive story about the ResMLP design space under Persistent Stochastic Ablation (PSA).

**Usage**:
```bash
# Direct Python
python scripts/make_figure_heatmaps.py

# Poetry command
poetry run make-figure-heatmaps

# With custom files
python scripts/make_figure_heatmaps.py --config-file reproduction/my_configs.txt --trials-file results/my_trials.md --output-dir results/
```

**Output**: Four high-resolution PNG files:
1. **Baseline Performance**: Raw performance of control models (viridis colormap)
2. **Ablation Uplift**: Quantitative benefit/harm of PSA (coolwarm colormap)
3. **Instability Map**: Trial-to-trial variance/chaos (magma colormap)
4. **Winning Strategy**: Best-performing ablation mode per architecture (categorical)

**Input**: 
- Reads from `reproduction/configurations.txt` (default)
- Reads from `results/psa_resmlp_trials.md` (default)
- Outputs to `results/` directory (default)

**Features**:
- **Comprehensive Analysis**: Four complementary visualizations for complete understanding
- **Data-Driven**: Based on actual experimental trial data
- **Statistical Rigor**: Uses mean and standard deviation calculations
- **Professional Visualization**: High-resolution (300 DPI) publication-ready figures
- **Logarithmic Scale**: Optimal for wide-ranging architectural parameters
- **Memory Efficient**: Proper cleanup of matplotlib resources
- **Configurable**: Custom input files and output directory
- **Error Handling**: Graceful handling of missing files and parsing errors 