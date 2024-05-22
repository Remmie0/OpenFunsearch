
# Funsearch Setup Guide

**NOTE:** We do not own Funsearch or any of the models used. Please refer to Google's DeepMind Funsearch for the original project. This project aims to help the community by providing easier access and expanding on top of the original Funsearch.

## Added Functionalities:
- Interchangeable LLMs (note that not all LLMs work; tested ones are listed in the files)
- Altered sandbox for easier access to input and output
- `eval.txt` that shows which prompts/response combinations lead to which results and if the result passed
- Data plotting functionality

## Overview
This document explains how to set up the Funsearch environment to run locally on a PC in easy-to-follow steps. This has been tested only on Ubuntu 22.04 and should work on Linux. Specific commands are mentioned to help the process. Notes and commands are enclosed in curly brackets '()'.

## Prerequisites
- Git for cloning the repository
- Decent hardware, as even smaller models are quite memory intensive (segmentation faults can happen)
- Anaconda
- Ensure your system is up to date:
  ```
  sudo apt update
  sudo apt upgrade -y
  ```
- Ensure the path to your folder does not contain spaces such as ' ' in the name, as this will crash the program. Solutions: change spaces to `_` or fix the path in the code.

## Steps

### Step 1: Gathering Repository and Installing Dependencies
1. Git clone the repository:
   ```
   git clone <Repository Link>
   ```
2. Open a terminal in the folder of the repo and create a conda environment:
   ```
   conda create -n myenv python=3.11
   ```
   *(you can rename `myenv`)*

3. Activate the conda environment:
   ```
   conda activate myenv
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Step 2: Installing Docker
1. Install packages to allow `apt` to use repositories over HTTPS:
   ```
   sudo apt install apt-transport-https ca-certificates curl software-properties-common -y
   ```

2. Get Docker's official GPG key:
   ```
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   ```

3. Set up the stable repository:
   ```
   sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
   ```

4. Install Docker Community Edition:
   ```
   sudo apt update
   sudo apt install docker-ce -y
   ```

**Optional:** 
- Verify if Docker is working correctly by running:
  ```
  sudo docker run hello-world
  ```

- Execute Docker without sudo:
  ```
  sudo usermod -aG docker ${USER}
  ```

### Step 3: Creating Sandbox Docker Image
1. Change the topline of the Dockerfile in the `open_funsearch` folder to your Python version.
2. Install the `funsearch_sandbox` Docker image by opening a terminal in the `open_funsearch` folder and running:
   ```
   sudo docker build -t funsearch_sandbox .
   ```

### Step 4: Choosing and Downloading LLM
**NOTE:** The specific results of LLMs differ widely and have a significant impact on the results of the method. For more information, please read the original paper by DeepMind. Within our testing capabilities, we noticed huge differences between some 8b and 13b models (we had too few resources for running any higher than that).

1. Choose and download an LLM from HuggingFace with the `.gguf` extension.
2. Place the downloaded LLM in the `open_funsearch` folder.

*Note: Tested LLMs are commented in the `run_funsearch.py`.*

### Step 5: Running Funsearch
1. Alter `run_funsearch.py` to match the exact model as downloaded and saved in the folder.
2. Open a terminal in the location and run:
   ```
   python3 run_funsearch.py
   ```

**Optional:**
- Specific settings can be altered through this file. You can also run Funsearch on other methods not specified in this setup guide. The goal of this guide is to get started with Funsearch in a more open and easily accessible way, not to be an in-depth guide on all endless possibilities.

### When Running Funsearch
The `last_eval.txt` file generates output:
- Runs that passed start with `RUNS_OK = True` and show the evaluated score as `TEST OUTPUT = Value`.
- The file will be saved into the backup folder with the datetime concatenation to analyze older runs. 

### Common Errors
- **Core dumped (Segmentation Fault):** This is most likely due to out-of-memory (on GPU). Solution: find a smaller model or reduce GPU layers in `main.py`. LLama models seem to get segmentation faults more often. Another fix for us was to update llama-cpp to the latest version after the release of Llama 3.
- **Sandbox not found:** There can be issues depending on the system with spaces ' ' in the absolute file path. Try to move the project to a location without spaces in the file path.
