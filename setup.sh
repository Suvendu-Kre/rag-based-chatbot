#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv venv
echo "Activating virtual environment..."
source venv/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Setup complete! Next steps:"
echo "  1. Copy .env.example to .env and fill in your credentials"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: python main.py"