#!/bin/bash

# Conda environment setup script for PianoVAM FingeringDetection

echo "Setting up conda environment for PianoVAM FingeringDetection..."

# Conda environment name
ENV_NAME="pianovam-fingering"

# Check if existing environment exists
if conda env list | grep -q $ENV_NAME; then
    echo "Environment '$ENV_NAME' already exists. Do you want to remove it and create a new one? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME
    else
        echo "Exiting script."
        exit 1
    fi
fi

# Create new conda environment with Python 3.9
echo "Creating new conda environment '$ENV_NAME' with Python 3.9..."
conda create -n $ENV_NAME python=3.9 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."

# Install packages specified in README.md
pip install miditok==3.0.5.post1
pip install shapely==2.0.7
pip install geopandas==1.0.1
pip install mido==1.3.3
pip install pretty_midi==0.2.10
pip install streamlit==1.37.1
pip install streamlit_image_coordinates==0.2.0
pip install numpy==1.24.1
pip install cupy-cuda12x
pip install torch torchvision torchaudio

# Install MediaPipe (required for video processing)
echo "Installing MediaPipe..."
pip install mediapipe

# Install additional useful packages
echo "Installing additional useful packages..."
pip install pandas matplotlib opencv-python stqdm psutil dill scipy symusic pillow

echo ""
echo "✅ Installation completed!"
echo ""
echo "To use the environment, run the following command:"
echo "    conda activate $ENV_NAME"
echo ""
echo "To run the application:"
echo "    streamlit run ./FingeringDetection/ASDF.py"
echo ""
echo "Installed packages:"
pip list

echo ""
echo "Environment setup completed!" 