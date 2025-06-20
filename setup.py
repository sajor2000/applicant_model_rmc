"""Setup script for Rush Medical College AI Admissions System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rush-ai-admissions",
    version="1.0.0",
    author="Rush University Medical Center",
    author_email="admissions-ai@rush.edu",
    description="AI-powered medical school admissions ranking system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RushUniversity/ai-admissions",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.4.1",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rush-admissions=rush_admissions.cli:main",
            "rush-admissions-web=rush_admissions.web:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rush_admissions": [
            "data/sample/*.csv",
            "models/README.md",
            "web/templates/*.html",
            "web/static/*.css",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/RushUniversity/ai-admissions/issues",
        "Documentation": "https://rush-ai-admissions.readthedocs.io/",
        "Source": "https://github.com/RushUniversity/ai-admissions",
    },
)