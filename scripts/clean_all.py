#!/usr/bin/env python3
"""
Clean-all script - alias for 'poetry run clean --all'
"""

import sys
from pathlib import Path

# Import and run the main clean script with --all flag
sys.path.append(str(Path(__file__).parent))
from clean import main

if __name__ == "__main__":
    # Add --all flag to sys.argv if it's not already there
    if "--all" not in sys.argv:
        sys.argv.insert(1, "--all")
    main() 