#!/bin/bash

# Check if a CSV file path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 [CSV_FILE_PATH]"
  exit 1
fi

# Get the CSV file path from the command line argument
csv_file=$1

# Check if the CSV file exists
if [ ! -f "$csv_file" ]; then
  echo "Error: File $csv_file not found."
  exit 1
fi

# Run the Python program
python run1.py $csv_file
