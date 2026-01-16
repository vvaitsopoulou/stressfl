#!/bin/bash

# Script to download desired wesad dataset from the official source

# Include the URL of the dataset
WESAD_URL="https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"

#ensure the path the dataset will be written to will be the same no matter where bash script is executed from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIPPED_FILE="$SCRIPT_DIR/../data/raw/wesad/wesad.zip"
mkdir -p "$(dirname "$ZIPPED_FILE")"

# Download the WESAD dataset
echo "Downloading WESAD dataset..."

# troubleshooting
if command -v wget >/dev/null 2>&1; then
  wget -O "$ZIPPED_FILE" "$WESAD_URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 --retry-delay 2 -o "$ZIPPED_FILE" "$WESAD_URL"
else
  echo "Error: Neither wget nor curl is installed."
  exit 1
fi

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download successful. Unzipping the file..."

    # Unzip the file
    unzip -q "$ZIPPED_FILE" -d "$(dirname "$ZIPPED_FILE")"

    # Check if the unzip was successful
    if [ $? -eq 0 ]; then
        echo "Unzip successful. Deleting the zip file..."

        # Delete the zip file
        rm $ZIPPED_FILE

        if [ $? -eq 0 ]; then
            echo "Zip file deleted successfully."
        else
            echo "Error: Failed to delete the zip file."
        fi
    else
        echo "Error: Failed to unzip the file."
    fi
else
    echo "Error: Failed to download the file."
fi
