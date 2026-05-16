@echo off
cd /d "%~dp0"
python -m streamlit run "%~dp0streamlit_app.py" --server.port 8501 --server.headless true
