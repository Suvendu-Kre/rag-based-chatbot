@echo off
echo Creating virtual environment...
python -m venv venv
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Setup complete! Next steps:
echo   1. Copy .env.example to .env and fill in your credentials
echo   2. Run: venv\Scripts\activate.bat
echo   3. Run: python main.py