#!/bin/bash
# Author: Ladipo Ipadeola
# Date: 4-29-2026
# Description: This shell script is used to download and unpack the CARVANA dataset for homework 1. 
# It uses wget to download the dataset from a specified URL, unzips the downloaded file, 
# and then removes any unnecessary files such as .DS_Store and the original zip file to clean up the directory.
# # Note: The URL provided in the script is a placeholder and should be replaced with the actual URL where the CARVANA dataset is hosted.

#TODO: Add error handling to check if the download and unzip processes were successful, and provide appropriate messages to the user.
#TODO: Consider adding a progress bar or some form of feedback during the download and unzip processes to enhance user experience.
#TODO: Ensure that the script is compatible with different operating systems, especially if there are differences in how wget and unzip work across platforms.
#TODO: Include a check to see if the dataset already exists before attempting to download and unpack it, to avoid unnecessary downloads and potential overwriting of existing data.
#TODO: Add comments to explain each step of the script for better readability and maintainability.
#TODO: Consider adding a cleanup function that can be called to remove the downloaded files and any temporary files created during the process, in case the user wants to free up space after using the dataset.
#TODO: Provide instructions on how to run the script, including any prerequisites such as installing wget and unzip, and any necessary permissions required to execute the script.
#TODO: Ensure that the script is executable by setting the appropriate permissions, and provide instructions on how to do this if necessary (e.g., using chmod +x download_data.sh).
#TODO: Test the script on different environments to ensure it works as expected and handles any potential issues that may arise during the download and unpacking process, such as network errors or insufficient disk space.
#TODO: Consider adding logging functionality to the script to keep a record of the download and unpacking process, which can be useful for debugging and tracking purposes.
#TODO: Update the script to handle any changes in the dataset URL or structure, and ensure that it remains functional even if there are updates to the dataset in the future.



# Download and unpack data
wget https://www.dropbox.com/s/tc1qo73rrm3gt3m/CARVANA.zip
unzip -q CARVANA.zip
rm -rf ./train/.DS_Store ./train_masks/.DS_Store CARVANA.zip
