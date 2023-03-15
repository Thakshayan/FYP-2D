import streamlit as st
import os
import nibabel as nib
from streamlit import session_state
import matplotlib.pyplot as plt
from utils.view import plot_slice,plot_image_label
import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from utils.transform import transformInput

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=( 64, 128, 256, 512), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    if (os.path.exists(model_path)):
        model.load_state_dict(torch.load(
        os.path.join(model_path),map_location=device))
    return model


model_path = './saved_models/3D_Models/best_metric_model.pth'

# Define a directory to save the uploaded NIfTI files
UPLOAD_DIRECTORY = "./images"

# Create the directory if it doesn't already exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

st.title('Web App to detect Prostate Cancer')

# Add a file uploader widget to the Streamlit app
nifti_file = st.sidebar.file_uploader("Choose image to check", type=["nii", "nii.gz"], key='file')

view = st.sidebar.button('View images')
detect = st.sidebar.button('Detect Prostate Cancer!')



# If the user has uploaded a file, save it to the directory and display the image
if nifti_file and view:
    session_state.clear()
    # Save the uploaded file to the upload directory
    file_path = os.path.join(UPLOAD_DIRECTORY, nifti_file.name)
    with open(file_path, "wb") as file:
        file.write(nifti_file.getbuffer())
        st.write("Saved file:", file_path)
    # Create a session state object
    session_state.file_path= file_path
    session_state.labeled = False


if 'file_path' in session_state and  not session_state.labeled:
    # Load and display the NIfTI image
    nifti_image = nib.load(session_state.file_path)
    image_data = nifti_image.get_fdata()
    st.text(image_data.shape)
    slider_value = st.slider('Select a value', min_value=0, max_value=image_data.shape[2] - 1,
                         value= image_data.shape[2] //2, key='1')
    
    plot_slice(image_data,slider_value)

if (detect and 'file_path' in session_state) or (not view and 'labeled' in session_state):
    
    model = load_model(model_path)
    nifti_image = nib.load(session_state.file_path)
    image_data = nifti_image.get_fdata()
    transformed_image = transformInput(image_data)
    
    label = model(transformed_image)
    
    st.text(label.shape)
    session_state.labeled = True
    plot_image_label(transformed_image, label)
