import numpy as np
from monai.transforms import Compose, AddChannel,Orientation,Resize,ToTensor, ScaleIntensity, ToTensord
import torch

def transformInput(numpy_image):
    # Example numpy array with shape (H, W)

    pixdim =(1.5, 1.5, 1.0)
    a_min=0
    a_max=500
    spatial_size= [128, 128,16] #[384, 384,18]
    central_slice_size = [128, 128,1]

    transformations = Compose([
        AddChannel(),
        ScaleIntensity(),
        Orientation(axcodes="RAS"),
        Resize(spatial_size=spatial_size),
        ToTensor()  # convert the numpy array to a PyTorch tensor
    ])

    # Apply the transformations to the numpy image
    transformed_tensor = transformations(numpy_image)
    #tensor_batch = transformed_tensor.unsqueeze(0)

    tensor = torch.unsqueeze(transformed_tensor, 0)
    # Print the shape of the transformed tensor

    print(tensor.shape)  # output: torch.Size([1, 256, 256])
    return tensor