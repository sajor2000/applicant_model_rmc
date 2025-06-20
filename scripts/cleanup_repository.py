#!/usr/bin/env python3
"""
Clean up the repository structure and organize files.
"""

import os
import shutil
from pathlib import Path

def main():
    """Clean up the repository structure."""
    print("🧹 Cleaning up repository structure...")
    
    # Get the repository root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    
    # Create necessary directories
    directories = [
        "logs",
        "tmp",
        "data/processed",
        "data/raw",
        "models/trained",
        "models/checkpoints",
        "notebooks/exploratory",
        "notebooks/training",
        "docs/reports",
        "docs/images",
        "src/utils",
        "tests/unit",
        "tests/integration"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Add placeholder files for empty directories
    placeholder_content = "# This directory is intentionally left empty\n"
    
    empty_dirs = [
        "logs/.gitkeep",
        "tmp/.gitkeep", 
        "data/processed/.gitkeep",
        "data/raw/.gitkeep",
        "models/trained/.gitkeep",
        "models/checkpoints/.gitkeep",
        "notebooks/exploratory/.gitkeep",
        "notebooks/training/.gitkeep",
        "docs/reports/.gitkeep",
        "docs/images/.gitkeep",
        "src/utils/.gitkeep",
        "tests/unit/.gitkeep",
        "tests/integration/.gitkeep"
    ]
    
    for placeholder in empty_dirs:
        with open(placeholder, 'w') as f:
            f.write(placeholder_content)
        print(f"✓ Created placeholder: {placeholder}")
    
    # Create utils __init__.py
    with open("src/utils/__init__.py", 'w') as f:
        f.write('"""Utility functions."""\n')
    
    print("\n🎉 Repository cleanup completed!")
    print("\nRepository structure:")
    print("📁 Clean, organized, and ready for GitHub!")
    
    # Display final structure
    print("\n" + "="*50)
    print("FINAL REPOSITORY STRUCTURE")
    print("="*50)
    
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        level = root.replace(".", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Only show first few files to avoid clutter
        subindent = " " * 2 * (level + 1)
        for f in sorted(files[:5]):  # Show only first 5 files
            if not f.startswith('.') and not f.endswith('.pyc'):
                print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... ({len(files)-5} more files)")

if __name__ == "__main__":
    main()