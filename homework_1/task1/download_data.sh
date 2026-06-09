#!/bin/bash
# Author: Ladipo Ipadeola
# Date: 4-29-2026
# Description: This shell script is used to download and unpack the CARVANA dataset for homework 1. 
# It uses wget to download the dataset from a specified URL, unzips the downloaded file, 
# and then removes any unnecessary files such as .DS_Store and the original zip file to clean up the directory.

#TODO as a README : Ensure that the script is executable by setting the appropriate permissions chmod +x download_data.sh).


#Logger
LOG_FILE="script.log"

log(){
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Download and unpack data
if [ -d "./data" ]; then
    log "Data directory already exists. Skipping download."
else
    log "Downloading file"
    mkdir -p ./data && wget -P ./data https://www.dropbox.com/s/tc1qo73rrm3gt3m/CARVANA.zip
    log "Unzipping file"
    unzip -q ./data/CARVANA.zip -d ./data
    log "Removing unnecessary files"
    rm -rf ./data/train/.DS_Store ./data/train_masks/.DS_Store ./data/CARVANA.zip
    log "Script complete"
fi