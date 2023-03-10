from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)

from monai.data import DataLoader, Dataset
from monai.utils import first

pixdim =(1.5, 1.5, 1.0)
a_min=0
a_max=500
spatial_size= [128, 128,16] #[384, 384,18]
central_slice_size = [128, 128,1]

train_transforms = Compose(
      [
          LoadImaged(keys=["image", "label"]),
          AddChanneld(keys=["image", "label"]),
          Orientationd(keys=["image", "label"], axcodes="RAS"),
          ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
          CropForegroundd(keys=["image", "label"], source_key="image"),
          Resized(keys=["image", "label"], spatial_size=central_slice_size),
          ToTensord(keys=["image", "label"])

      ]
)

def transform_pipeline(image):
    DataLoader,
    ds = Dataset(data=[image], transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)
    return first(Dataset(data=[image], transform=train_transforms))

