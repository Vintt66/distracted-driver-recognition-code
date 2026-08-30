"""The three image-classification models used in the project."""

from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

from config import NUM_CLASSES


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        input_channels = 3

        for output_channels in [32, 64, 128, 256]:
            layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            input_channels = output_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, images):
        features = self.features(images)
        pooled_features = self.pool(features)
        return self.classifier(pooled_features)


def create_model(model_name, pretrained):
    if model_name == "custom_cnn":
        return CustomCNN()

    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    if model_name == "mobilenet_v3_small":
        weights = (
            MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        model = mobilenet_v3_small(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, NUM_CLASSES)
        return model

    valid_names = "custom_cnn, resnet18, mobilenet_v3_small"
    raise ValueError(f"Unknown model: {model_name}. Choose from {valid_names}.")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
