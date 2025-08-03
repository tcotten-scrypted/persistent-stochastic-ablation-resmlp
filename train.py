#!/usr/bin/env python3
"""
Training script wrapper for Persistent Stochastic Ablation MLP
"""

import sys
from pathlib import Path

# Add src to path and run the main training script
sys.path.insert(0, str(Path(__file__).parent / "src"))
from train_psa_simplemlp import main

if __name__ == "__main__":
    main() 