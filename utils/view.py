import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np



def dir_selector(folder_path='.'):
    dirnames = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    selected_folder = st.sidebar.selectbox('Select a folder', dirnames)
    if selected_folder is None:
        return None
    return os.path.join(folder_path, selected_folder)



@st.cache_data
def plot_slice(image, slice):
    
    slice_data = image[:,:,slice]
    # Create a figure and axes
    fig, ax = plt.subplots()

    plt.axis('off')

    # Display the 3D numpy array as an image
    ax.imshow(slice_data, cmap='gray')

    # Display the figure in Streamlit using st.pyplot()
    st.pyplot(fig)


def plot_image_label(image, label):
    label_data = label.get_array()
    slider_value = st.slider('Select a value', min_value=0, max_value=image.shape[4] - 1,
                         value= image.shape[4] //2, key='2')
    slice_image = image[0,0,:,:,slider_value]
    slice_label = label_data[0,0,:,:,slider_value]
    # session_state.labeled = True
    fig, ax = plt.subplots()

    # Display MRI image
    ax.imshow(slice_image, cmap='gray', alpha=1)

    # Set alpha channel of label image
    alpha = np.zeros_like(slice_label)
    alpha[slice_label > 0.5] = 0.4

    # Display label image
    ax.imshow(slice_label, cmap='Reds', alpha=alpha)

    # Add colorbar for label image
    #cbar = plt.colorbar(ax.imshow(label, cmap='jet', alpha=1))

    # Set title for plot
    ax.set_title('MRI Image with Label')

    # Display the figure in Streamlit using st.pyplot()
    st.pyplot(fig)