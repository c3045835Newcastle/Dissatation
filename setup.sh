#!/bin/bash
# Quick setup script for Base Llama 3.1 8B Model

echo "=================================================="
echo "Base Llama 3.1 8B Model - Setup Script"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

# Test setup
echo ""
echo "Testing setup..."
python test_setup.py

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Request access to Llama 3.1 at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B"
echo "2. Create Hugging Face token at: https://huggingface.co/settings/tokens"
echo "3. Login with: huggingface-cli login"
echo "4. Run the model with: python inference.py"
echo ""
echo "For more information, see README_MODEL.md"
