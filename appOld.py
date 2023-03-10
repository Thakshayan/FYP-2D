import numpy as np
from PIL import Image
import streamlit as st
import argparse
from Models.ViTNet import ViTNet
import os
import torch
import nibabel as nib
import tempfile
import matplotlib.pyplot as plt
import SimpleITK as sitk
...

# Define a directory to save the uploaded NIfTI files
UPLOAD_DIRECTORY = "./images"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def args_parser():
    parser = argparse.ArgumentParser(
        description="Datature Open Source Prediction Script")
    ...
#     parser.add_argument(
#         "--model",
#         help="Path to tensorflow pb model",
#         required=True,
#   default='./saved_models/2D_Models/best_metric_model.pth'
#     )
#     parser.add_argument(
#         "--label",
#         help="Path to tensorflow label map",
#         required=True,
#   default="../label_map.pbtxt"
#     )
    parser.add_argument("--width",
                        help="Width of image to load into model",
                        default=640)
    parser.add_argument("--height",
                        help="Height of image to load into model",
                        default=640)
    parser.add_argument("--threshold",
                        help="Prediction confidence threshold",
                        default=0.7)

    return parser.parse_args()


#@st.cache(allow_output_mutation=True)

def load_model(model_path):
 model = ViTNet()
 if (os.path.exists(model_path)):
    model.load_state_dict(torch.load(
        os.path.join(model_path)))
 return

model_path = './saved_models/2D_Models/best_metric_model.pth'
st.title('Web App to detect Prostate Cancer')

#file = st.sidebar.file_uploader("Choose image to evaluate model", type=["jpg", "png"])
file = st.sidebar.file_uploader("Choose image to evaluate model",  type='nii.gz')

view = st.sidebar.button('View images')
detect = st.sidebar.button('Detect Prostate Cancer!')



args = args_parser()

model = load_model(model_path)


if  detect and file: 
   st.text('Running inference...')

if  view and file: 
    st.text('Running view')
    file_path = os.path.join(UPLOAD_DIRECTORY, file.name)
    with open(file_path, "wb") as file:
        file.write(file.getbuffer())
        st.write("Saved file:", file_path)
        
        # Load and display the NIfTI image
        nifti_image = nib.load("./images/ProstateX-0000_t2_tse_tra_4.nii.gz")
        image_data = nifti_image.get_fdata()
        st.text(image_data.shape)
    