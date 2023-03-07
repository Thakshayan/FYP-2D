import streamlit as st
import os
import nibabel as nib
from check import *
from streamlit import session_state
import matplotlib.pyplot as plt



# Define a directory to save the uploaded NIfTI files
UPLOAD_DIRECTORY = "./images"

# Create the directory if it doesn't already exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

st.title('Web App to detect Prostate Cancer')

# Add a file uploader widget to the Streamlit app
nifti_file = st.sidebar.file_uploader("Choose image to evaluate model", type=["nii", "nii.gz"])

view = st.sidebar.button('View images')
detect = st.sidebar.button('Detect Prostate Cancer!')


# If the user has uploaded a file, save it to the directory and display the image
if nifti_file and view:
    # Save the uploaded file to the upload directory
    file_path = os.path.join(UPLOAD_DIRECTORY, nifti_file.name)
    with open(file_path, "wb") as file:
        file.write(nifti_file.getbuffer())
        st.write("Saved file:", file_path)
    # Create a session state object
    session_state.file_path= file_path

        

        



        

if 'file_path' in session_state:
    # Load and display the NIfTI image
    nifti_image = nib.load(session_state.file_path)
    image_data = nifti_image.get_fdata()
    st.text(image_data.shape)
    slider_value = st.slider('Select a value', min_value=0, max_value=image_data.shape[2] - 1,
                         value= image_data.shape[2] //2)
    slice_data = image_data[:,:,slider_value]
        
        # Create a figure and axes
    fig, ax = plt.subplots()

        # Display the 3D numpy array as an image
    ax.imshow(slice_data, cmap='gray')

        # Display the figure in Streamlit using st.pyplot()
    st.pyplot(fig)
