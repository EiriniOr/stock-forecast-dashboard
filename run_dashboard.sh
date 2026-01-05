#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Stock Forecast Dashboard..."
source stock_forecast_venv/bin/activate 2>/dev/null || python3 -m venv stock_forecast_venv && source stock_forecast_venv/bin/activate
pip install -q -r requirements.txt
streamlit run app.py
