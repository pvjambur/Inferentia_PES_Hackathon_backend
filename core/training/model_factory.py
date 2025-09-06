import torch
import torch.nn as nn
from typing import Dict, Any
from utils.logging import get_logger

logger = get_logger(__name__)

def create_model(model_type: str, params: Dict[str, Any]):
    """
    A factory function to create and return a machine learning or deep learning model using PyTorch.

    Args:
        model_type (str): The type of model to create (e.g., 'LogisticRegression', 'SimpleNN', 'ImageClassifier').
        params (Dict[str, Any]): A dictionary of hyperparameters for the model.

    Returns:
        A PyTorch model instance.

    Raises:
        ValueError: If the specified model_type is not supported.
    """
    logger.info(f"Creating model of type: {model_type} with params: {params}")

    model_type = model_type.lower()

    if model_type == 'logisticregression':
        num_features = params.pop('num_features', None)
        num_classes = params.pop('num_classes', 1)
        if num_features is None:
            raise ValueError("For 'LogisticRegression', 'num_features' must be provided in the params.")
        
        # A simple linear model for binary or multi-class classification
        model = nn.Linear(num_features, num_classes)
        return model

    elif model_type == 'simplenn':
        num_features = params.pop('num_features', None)
        num_classes = params.pop('num_classes', 1)
        if num_features is None:
            raise ValueError("For 'SimpleNN', 'num_features' must be provided in the params.")
        
        class SimpleNN(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(SimpleNN, self).__init__()
                self.layer1 = nn.Linear(input_dim, 64)
                self.relu1 = nn.ReLU()
                self.layer2 = nn.Linear(64, 32)
                self.relu2 = nn.ReLU()
                self.output_layer = nn.Linear(32, output_dim)
            
            def forward(self, x):
                x = self.relu1(self.layer1(x))
                x = self.relu2(self.layer2(x))
                x = self.output_layer(x)
                return x

        model = SimpleNN(num_features, num_classes)
        return model

    elif model_type == 'imageclassifier':
        num_classes = params.pop('num_classes', 1)
        
        class SimpleImageClassifier(nn.Module):
            def __init__(self, num_classes):
                super(SimpleImageClassifier, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                self.classifier = nn.Sequential(
                    nn.Linear(32 * 7 * 7, 128),  # Assuming 28x28 input image
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = SimpleImageClassifier(num_classes)
        return model

    else:
        logger.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")