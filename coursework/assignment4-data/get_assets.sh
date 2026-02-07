#!/bin/bash

# Define the target directory
SOURCE_DIR="/data/classifiers"
ASSETS_DIR="$(pwd)/cs336_data/assets"

# Function to handle each file
handle_file() {
    local filename=$1
    local url=$2

    # Check if file exists in target directory
    if [ -e "$ASSETS_DIR/$filename" ]; then
        echo "File $filename already exists in $ASSETS_DIR, skipping..."
    # Check if file exists in source directory
    elif [ -f "$SOURCE_DIR/$filename" ]; then
        echo "File $filename exists in $SOURCE_DIR, creating softlink..."
        ln -s "$SOURCE_DIR/$filename" "$ASSETS_DIR/$filename"
    else
        echo "File $filename not found in $SOURCE_DIR, downloading..."
        wget "$url" -O "$ASSETS_DIR/$filename"
    fi
}

# Handle each file
handle_file "dolma_fasttext_nsfw_jigsaw_model.bin" "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-nsfw/resolve/main/model.bin"
handle_file "dolma_fasttext_hatespeech_jigsaw_model.bin" "https://huggingface.co/allenai/dolma-jigsaw-fasttext-bigrams-hatespeech/resolve/main/model.bin"