@echo off
echo Starting Stock Forecast Dashboard...
cd /d "%~dp0"
call stock_forecast_venv\Scripts\activate
streamlit run app.py
pause
