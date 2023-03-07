# FYP-2D

python -m venv packages <br />
packages\Scripts\activate <br />
git clone https://github.com/rcuocolo/PROSTATEx_masks <br />

# for GPU
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# for CPU
pip install torch torchvision torchaudio

# to run the web app
streamlit run app.py