#!/bin/bash
# Always run inside the project venv
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app.py "$@"
