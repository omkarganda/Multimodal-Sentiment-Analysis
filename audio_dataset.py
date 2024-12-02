import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, dataframe, audio_array, sampling_rate, target_column, feature_extractor):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe containing file paths and targets.
            audio_column (str): Name of the column with audio file paths.
            target_column (str): Name of the column with target labels.
            feature_extractor (function): A function to extract audio features.
            sampling_rate (int): The target sampling rate for loading audio.
        """
        self.dataframe = dataframe
        self.audio_array = audio_array
        self.sampling_rate = sampling_rate
        self.target_column = target_column
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the file path and label
        audio_array = self.dataframe.iloc[idx][self.audio_array]
        sampling_rate = self.dataframe.iloc[idx][self.sampling_rate]
        label = self.dataframe.iloc[idx][self.target_column]


        # Extract features
        text, audio, image = self.feature_extractor(audio_array, sampling_rate)

        if text is None:
            text = torch.zeros((1, 512), dtype=torch.float32)
            audio = torch.zeros((1, 512))
            label = 2

    

        # Convert the label to tensor (assuming it's a single integer)
        label = torch.tensor(label, dtype=torch.long)

        return (text, audio, image), label