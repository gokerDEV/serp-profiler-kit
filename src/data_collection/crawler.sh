#!/bin/bash

# Get the URL from the command-line argument
url="$1"

# Check if a URL is provided
if [ -z "$url" ]; then
  echo "Error: Please provide a URL as an argument."
  exit 1
fi

# Shift the arguments to remove the URL
shift

# Run the Python crawler script and pass all arguments
python crawler.py "$url" "$@"

# (No need for temp files, spinner, or prompt)