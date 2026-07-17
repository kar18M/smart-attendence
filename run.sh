#!/bin/bash
# Always run inside the project venv — never use system Python
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run app.py "$@"
