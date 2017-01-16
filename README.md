# GPUstats
A Python library for getting the GPU status from NVIDA GPUs using nvidia-smi programmically in Python.

[Installation](#installation)
[Usage](#usage)

## Installation
1. Download or clone repository to your computer
2. Add folder to ~/.bashrc
  1. Open a new terminal (Press Ctrl+Alt+T)
  2. Open bashrc:

      ```
      gedit ~/.bashrc
      ```

  3. Added your GPUstats folder to PYTHONPATH (replace <path_to_gpustats> with your folder path):

      ```
      export PYTHONPATH="$PYTHONPATH:<path_to_gpustats>/gpustats"
      ```

  4. Save ~/.bashrc and close gedit
  5. Restart your terminal

## Usage
