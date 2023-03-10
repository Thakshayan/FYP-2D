import streamlit as st
import os
import nibabel as nib
from streamlit import session_state
import matplotlib.pyplot as plt
from utils.view import plot_slice,plot_image_label
import torch
from Models.ViTNet import ViTNet
from utils.transform import transform_pipeline

def load_model(model_path):
    model = ViTNet()
    if (os.path.exists(model_path)):
        model.load_state_dict(torch.load(
        os.path.join(model_path)))
    return model

model_path = './saved_models/2D_Models/best_metric_model.pth'

# Define a directory to save the uploaded NIfTI files
UPLOAD_DIRECTORY = "./images"

# Create the directory if it doesn't already exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

st.title('Web App to detect Prostate Cancer')

# Add a file uploader widget to the Streamlit app
nifti_file = st.sidebar.file_uploader("Choose image to evaluate model", type=["nii", "nii.gz"], key='file')

view = st.sidebar.button('View images')
detect = st.sidebar.button('Detect Prostate Cancer!')
label = None


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
    


if 'file_path' in session_state and 'label_path' not in session_state:
    # Load and display the NIfTI image
    nifti_image = nib.load(session_state.file_path)
    image_data = nifti_image.get_fdata()
    st.text(image_data.shape)
    slider_value = st.slider('Select a value', min_value=0, max_value=image_data.shape[2] - 1,
                         value= image_data.shape[2] //2, key='1')
    slice_data = image_data[:,:,slider_value]
        
    plot_slice(slice_data)

if detect and 'file_path' in session_state:
    #session_state.label_path = "./images/ProstateX-0000.nii.gz"
    model = load_model(model_path)
    transformed_image = transform_pipeline(image_data)
    #label = model(image_data)
    #print(label)
    print(transformed_image)

if False:

    label_data = label.get_fdata()
    st.text(label_data.shape)
    image = nib.load(session_state.file_path)
    image_data = image.get_fdata()
    slider_value = st.slider('Select a value', min_value=0, max_value=image_data.shape[2] - 1,
                         value= image_data.shape[2] //2, key='2')
    slice_data = image_data[:,:,slider_value]
    slice_label = label_data[:,:,slider_value]
    plot_image_label(slice_data, slice_label)
