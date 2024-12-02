import torch
from torch.utils.data import Dataset

# Custom Dataset class to handle images and text in batches
class ImageTextDataset(Dataset):
    def __init__(self, dataframe, image_column, text_column, image_preprocess):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe containing image paths and text data.
            image_column (str): The name of the column containing image paths.
            text_column (str): The name of the column containing text associated with the images.
            image_preprocess (function): Function that takes an image path and text and returns processed output.
        """
        self.dataframe = dataframe
        self.image_column = image_column
        self.text_column = text_column
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the image path and text from the DataFrame
        image_path = self.dataframe.iloc[idx][self.image_column]
        text = self.dataframe.iloc[idx][self.text_column]
        
        # Apply the image preprocess function to the image and text
        processed_data = self.image_preprocess(image_path, text_available = True, text= text)
        
        return processed_data