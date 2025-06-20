"""Basic tests to verify the package structure and imports."""

import pytest
import sys
import os
from pathlib import Path

# Add the root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_package_structure():
    """Test that the basic package structure exists."""
    root_dir = Path(__file__).parent.parent
    
    # Check for main directories
    assert (root_dir / "github_repo").exists()
    assert (root_dir / "github_repo" / "src").exists()
    assert (root_dir / "output").exists()
    assert (root_dir / "scripts").exists()


def test_requirements_file():
    """Test that requirements.txt exists and is readable."""
    root_dir = Path(__file__).parent.parent
    requirements_file = root_dir / "requirements.txt"
    
    assert requirements_file.exists()
    
    with open(requirements_file, 'r') as f:
        content = f.read()
        assert 'pandas' in content
        assert 'numpy' in content
        assert 'scikit-learn' in content


def test_feature_engineer_import():
    """Test that we can import the feature engineer."""
    try:
        from src.feature_engineer import FeatureEngineer
        # If import succeeds, create an instance to verify it works
        fe = FeatureEngineer()
        assert fe is not None
    except ImportError:
        # If direct import fails, try with github_repo path
        sys.path.insert(0, str(Path(__file__).parent.parent / "github_repo"))
        from src.features.feature_engineer import FeatureEngineer
        fe = FeatureEngineer()
        assert fe is not None


def test_html_report_exists():
    """Test that the final HTML report exists."""
    root_dir = Path(__file__).parent.parent
    html_file = root_dir / "output" / "RMC_AI_Admissions_TRIPOD_Report_Final.html"
    
    assert html_file.exists()
    
    # Check that it contains key content
    with open(html_file, 'r') as f:
        content = f.read()
        assert 'Rush Medical College' in content
        assert 'Juan C. Rojas' in content
        assert '93.8%' in content
        assert 'GPT-4o' in content


def test_production_script_exists():
    """Test that the production script exists."""
    root_dir = Path(__file__).parent.parent
    script_file = root_dir / "scripts" / "process_2025_applications.py"
    
    assert script_file.exists()
    
    # Check that it's executable Python
    with open(script_file, 'r') as f:
        content = f.read()
        assert '#!/usr/bin/env python3' in content
        assert 'ProductionPipeline' in content


def test_documentation_files():
    """Test that documentation files exist."""
    root_dir = Path(__file__).parent.parent
    
    docs = [
        "README_PRODUCTION.md",
        "DEPLOYMENT_CHECKLIST_2025.md",
        "GITHUB_SETUP.md"
    ]
    
    for doc in docs:
        doc_file = root_dir / doc
        assert doc_file.exists(), f"Missing documentation file: {doc}"


if __name__ == "__main__":
    pytest.main([__file__])