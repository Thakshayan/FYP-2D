# FYP-2D

python -m venv packages
packages\Scripts\activate
git clone https://github.com/rcuocolo/PROSTATEx_masks

# for GPU
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# for CPU
pip install torch torchvision torchaudio