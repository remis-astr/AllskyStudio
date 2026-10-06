#!/bin/bash
# Lanceur AllskyStudio — active le venv puis démarre l'app
cd "$(dirname "$(readlink -f "$0")")"
source venv/bin/activate
exec python3 allsky_app.py
