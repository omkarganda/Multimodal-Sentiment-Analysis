import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device='cpu'):
        super(MultimodalModel, self).__init__()

        self.device = device

        fused_representation_size = 512
        image_fused_representation_size = 512
        text_fused_representation_size = 512
        audio_fused_representation_size = 512
        num_attention_heads = 4

        self.num_classes = num_classes

        self.image_attention = nn.MultiheadAttention(
            embed_dim=image_fused_representation_size,
            num_heads=num_attention_heads,
            batch_first=True,
            device=self.device
        )
        self.image_attention_Linear = nn.Linear(
            image_fused_representation_size, fused_representation_size, device=self.device)

        self.text_attention = nn.MultiheadAttention(
            embed_dim=text_fused_representation_size,
            num_heads=num_attention_heads,
            batch_first=True,
            device=self.device
        )
        self.text_attention_Linear = nn.Linear(
            text_fused_representation_size, fused_representation_size, device=self.device)

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=audio_fused_representation_size,
            num_heads=num_attention_heads,
            batch_first=True,
            device=self.device
        )
        self.audio_attention_Linear = nn.Linear(
            audio_fused_representation_size, fused_representation_size, device=self.device)

        self.fusion_layer = nn.Linear(
            image_fused_representation_size + text_fused_representation_size + audio_fused_representation_size,
            fused_representation_size,
            device=self.device
        )

        self.nonlinear_layer_1 = nn.Linear(
            fused_representation_size, fused_representation_size, device=self.device)
        self.nonlinear_layer_2 = nn.Linear(
            fused_representation_size, fused_representation_size, device=self.device)
        self.classification_layer = nn.Linear(
            fused_representation_size, self.num_classes, device=self.device)

    def forward(self, text_inputs, image_inputs, audio_inputs):
        # Ensure inputs are on the correct device
        text_inputs = text_inputs.to(self.device)
        image_inputs = image_inputs.to(self.device)
        audio_inputs = audio_inputs.to(self.device)

        # Ensure inputs have the correct shape
        if len(image_inputs.shape) == 2:
            image_inputs = image_inputs.unsqueeze(1)  # (batch_size, seq_len=1, embed_dim)
        if len(text_inputs.shape) == 2:
            text_inputs = text_inputs.unsqueeze(1)
        if len(audio_inputs.shape) == 2:
            audio_inputs = audio_inputs.unsqueeze(1)

        # Apply attention
        image_attention_output, _ = self.image_attention(
            image_inputs, image_inputs, image_inputs)
        text_attention_output, _ = self.text_attention(
            text_inputs, text_inputs, text_inputs)
        audio_attention_output, _ = self.audio_attention(
            audio_inputs, audio_inputs, audio_inputs)

        # Apply Linear layers to attention outputs
        image_features = self.image_attention_Linear(image_attention_output)
        text_features = self.text_attention_Linear(text_attention_output)
        audio_features = self.audio_attention_Linear(audio_attention_output)

        # Aggregate over the sequence dimension if seq_len > 1
        image_features = image_features.mean(dim=1)  # (batch_size, fused_representation_size)
        text_features = text_features.mean(dim=1)
        audio_features = audio_features.mean(dim=1)

        # Concatenate features
        fused_features = torch.cat(
            (image_features, text_features, audio_features), dim=1)  # (batch_size, 1536)

        # Pass through fusion and non-linear layers
        fused_features = self.fusion_layer(fused_features)
        fused_features = F.relu(fused_features)
        fused_features = self.nonlinear_layer_1(fused_features)
        fused_features = F.relu(fused_features)
        fused_features = self.nonlinear_layer_2(fused_features)
        fused_features = F.relu(fused_features)

        # Classification
        output = self.classification_layer(fused_features)
        probabilities = F.softmax(output, dim=-1)

        return probabilities
