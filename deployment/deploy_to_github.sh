#!/bin/bash

# RMC AI Admissions Report - GitHub Pages Deployment Script
# This script automates the deployment of the TRIPOD-AI compliant report to GitHub Pages

echo "==================================================="
echo "RMC AI Admissions Report - GitHub Pages Deployment"
echo "==================================================="

# Configuration
REPO_PATH="$1"
SOURCE_DIR="/Users/JCR/Desktop/Windsurf IDE/rmc_admissions/deployment/rmc_ai_admissions_report"

# Check if repository path was provided
if [ -z "$REPO_PATH" ]; then
    echo "Error: Please provide the path to your GitHub repository"
    echo "Usage: ./deploy_to_github.sh /path/to/applicant_model_rmc"
    exit 1
fi

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found at $SOURCE_DIR"
    exit 1
fi

# Check if target repository exists
if [ ! -d "$REPO_PATH" ]; then
    echo "Error: Repository not found at $REPO_PATH"
    exit 1
fi

echo ""
echo "Source: $SOURCE_DIR"
echo "Target: $REPO_PATH"
echo ""

# Navigate to repository
cd "$REPO_PATH" || exit 1

# Check if it's a git repository
if [ ! -d ".git" ]; then
    echo "Error: $REPO_PATH is not a git repository"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Warning: You have uncommitted changes in your repository"
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 1
    fi
fi

# Copy the report folder
echo ""
echo "Copying report files..."
cp -r "$SOURCE_DIR" .

# Check if copy was successful
if [ ! -d "rmc_ai_admissions_report" ]; then
    echo "Error: Failed to copy report files"
    exit 1
fi

echo "Files copied successfully!"

# Add to git
echo ""
echo "Adding files to git..."
git add rmc_ai_admissions_report/

# Show what will be committed
echo ""
echo "Files to be committed:"
git status --porcelain rmc_ai_admissions_report/

# Commit
echo ""
read -p "Enter commit message (or press Enter for default): " COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Add TRIPOD-AI compliant AI admissions system report"
fi

git commit -m "$COMMIT_MSG"

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push origin "$CURRENT_BRANCH"

# Check if push was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================================="
    echo "Deployment Successful!"
    echo "==================================================="
    echo ""
    echo "Your report is now being deployed to GitHub Pages."
    echo "It may take a few minutes for the changes to appear."
    echo ""
    echo "Once deployed, your report will be available at:"
    echo "https://sajor2000.github.io/applicant_model_rmc/rmc_ai_admissions_report/"
    echo ""
    echo "Direct link to report:"
    echo "https://sajor2000.github.io/applicant_model_rmc/rmc_ai_admissions_report/index.html"
    echo ""
else
    echo ""
    echo "Error: Failed to push to GitHub"
    echo "Please check your GitHub credentials and try again"
    exit 1
fi

# Optional: Update main README
echo "==================================================="
echo "Optional: Update Main README"
echo "==================================================="
read -p "Would you like to add a link to the report in your main README? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    README_PATH="$REPO_PATH/README.md"
    if [ -f "$README_PATH" ]; then
        # Check if link already exists
        if grep -q "AI Model Performance Report" "$README_PATH"; then
            echo "Link already exists in README"
        else
            echo "" >> "$README_PATH"
            echo "## Documentation" >> "$README_PATH"
            echo "" >> "$README_PATH"
            echo "- [AI Model Performance Report](https://sajor2000.github.io/applicant_model_rmc/rmc_ai_admissions_report/) - TRIPOD-AI compliant report on model development and validation" >> "$README_PATH"
            echo "" >> "$README_PATH"
            
            git add README.md
            git commit -m "Add link to AI model performance report in README"
            git push origin "$CURRENT_BRANCH"
            
            echo "README updated successfully!"
        fi
    else
        echo "README.md not found in repository root"
    fi
fi

echo ""
echo "Deployment complete!"