#!/usr/bin/env bash

# Script Name:  download_datasets.sh
# Description:  Downloads speech datasets if zip files are missing and extracts
#               them into separate, dedicated directories.
# Author:       Coding Assistance

# -----------------------------------------------------------------------------
# Safety Settings
# -----------------------------------------------------------------------------
set -euo pipefail

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log_info() {
    echo "[INFO] $1"
}

# Function: process_dataset
# Description: Checks for zip existence, downloads if missing, and unzips to target.
# Arguments:
#   $1 (str): The URL of the dataset.
#   $2 (str): The local filename for the zip archive.
#   $3 (str): The target directory name for extraction.
process_dataset() {
    local url="$1"
    local zip_file="$2"
    local target_dir="$3"

    echo "--------------------------------------------------------"
    log_info "Processing: ${target_dir}"

    # 1. Download: Check if zip exists
    if [[ -f "${zip_file}" ]]; then
        log_info "File '${zip_file}' already exists. Skipping download."
    else
        log_info "Downloading '${zip_file}'..."
        # -L: Follow redirects (crucial for Kaggle/HuggingFace links)
        # -o: Output to specific filename
        curl -L -o "${zip_file}" "${url}"
    fi

    # 2. Extract: Unzip to specific folder
    # We check if the directory exists to avoid re-unzipping if not needed,
    # though zip overwrite prompts might still occur if the dir exists but is empty.
    # To be safe and clean, we create the dir and unzip.
    if [[ -d "${target_dir}" ]]; then
        log_info "Target directory '${target_dir}' already exists. "
        log_info "Skipping extraction to prevent overwriting/duplication."
    else
        log_info "Extracting '${zip_file}' to '${target_dir}'..."
        mkdir -p "${target_dir}"
        
        # -q: Quiet mode (suppresses listing every file unzipped)
        # -d: Directory to extract into
        unzip -q "${zip_file}" -d "${target_dir}"
        
        log_info "Extraction complete."
    fi
}

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

main() {
    # Dataset 1: VCTK Corpus
    process_dataset \
        "https://www.kaggle.com/api/v1/datasets/download/pratt3000/vctk-corpus" \
        "vctk-corpus.zip" \
        "vctk_dataset"

    # Dataset 2: DARPA TIMIT
    process_dataset \
        "https://www.kaggle.com/api/v1/datasets/download/mfekadu/darpa-timit-acousticphonetic-continuous-speech" \
        "darpa-timit-acousticphonetic-continuous-speech.zip" \
        "darpa_timit_dataset"

    # Dataset 3: VoxCeleb (Dev Wav)
    process_dataset \
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip" \
        "vox1_dev_wav.zip" \
        "vox1_dev_dataset"
        
    echo "--------------------------------------------------------"
    log_info "All tasks finished successfully."
}

# Invoke main
main "$@"