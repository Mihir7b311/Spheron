#!/bin/bash

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install package in development mode
pip install -e .

# Run tests
pytest tests/ \
    --cov=src \
    --cov-report=html \
    --cov-report=term \
    -v \
    --durations=10