import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import librosa
import logging
from typing import Dict, Any, Tuple
from config import settings

logger = logging.getLogger(__name__)

class DLTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def train(self, data_loader: DataLoader, model_type: str, 
                   hyperparams: Dict, job_id: str) -> Dict[str, Any]:
        """Train deep learning model"""
        try:
            logger.info(f"Starting DL training for {model_type} with job ID {job_id}")
            
            # Determine data type from model type
            if 'cnn' in model_type.lower() or 'image' in model_type.lower():
                return await self._train_image_model(data_loader, model_type, hyperparams, job_id)
            elif 'audio' in model_type.lower() or 'spectrogram' in model_type.lower():
                return await self._train_audio_model(data_loader, model_type, hyperparams, job_id)
            elif 'text' in model_type.lower() or 'lstm' in model_type.lower():
                return await self._train_text_model(data_loader, model_type, hyperparams, job_id)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"DL training failed: {e}")
            raise
            
    async def _train_image_model(self, data_loader: DataLoader, model_type: str,
                                hyperparams: Dict, job_id: str) -> Dict[str, Any]:
        """Train CNN for image classification"""
        # Get data info
        sample_batch = next(iter(data_loader))
        input_shape = sample_batch[0].shape[1:]  # Skip batch dimension
        num_classes = len(torch.unique(sample_batch[1]))
        
        # Create model
        model = self._create_cnn_model(input_shape, num_classes, hyperparams)
        model = model.to(self.device)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), 
                              lr=hyperparams.get('learning_rate', 0.001))
        epochs = hyperparams.get('epochs', 50)
        
        # Training loop
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            avg_loss = epoch_loss / len(data_loader)
            accuracy = 100 * correct / total
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Save model
        model_path = await self._save_pytorch_model(model, job_id, model_type)
        
        # Calculate metrics
        metrics = {
            'final_loss': float(train_losses[-1]),
            'final_accuracy': float(train_accuracies[-1]),
            'best_accuracy': float(max(train_accuracies)),
            'epochs_trained': epochs
        }
        
        return {
            'file_path': model_path,
            'metrics': metrics,
            'model_type': model_type,
            'hyperparameters': hyperparams
        }
        
    def _create_cnn_model(self, input_shape: Tuple, num_classes: int, 
                         hyperparams: Dict) -> nn.Module:
        """Create CNN model architecture"""
        class CNNModel(nn.Module):
            def __init__(self):
                super(CNNModel, self).__init__()
                
                # Calculate input channels
                channels = input_shape[0] if len(input_shape) == 3 else 1
                
                self.features = nn.Sequential(
                    nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout(0.5)
                )
                
                # Calculate flattened size
                with torch.no_grad():
                    dummy_input = torch.zeros(1, *input_shape)
                    dummy_output = self.features(dummy_input)
                    flattened_size = dummy_output.numel()
                
                self.classifier = nn.Sequential(
                    nn.Linear(flattened_size, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
                
        return CNNModel()
        
    async def _train_audio_model(self, data_loader: DataLoader, model_type: str,
                                hyperparams: Dict, job_id: str) -> Dict[str, Any]:
        """Train model for audio/spectrogram data"""
        # Similar structure to image training but with audio-specific preprocessing
        sample_batch = next(iter(data_loader))
        input_shape = sample_batch[0].shape[1:]
        num_classes = len(torch.unique(sample_batch[1]))
        
        # Create audio model (1D CNN or spectrogram CNN)
        model = self._create_audio_model(input_shape, num_classes, hyperparams)
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), 
                              lr=hyperparams.get('learning_rate', 0.001))
        epochs = hyperparams.get('epochs', 50)
        
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
            avg_loss = epoch_loss / len(data_loader)
            accuracy = 100 * correct / total
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            if epoch % 10 == 0:
                logger.info(f'Audio Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        model_path = await self._save_pytorch_model(model, job_id, model_type)
        
        metrics = {
            'final_loss': float(train_losses[-1]),
            'final_accuracy': float(train_accuracies[-1]),
            'best_accuracy': float(max(train_accuracies)),
            'epochs_trained': epochs
        }
        
        return {
            'file_path': model_path,
            'metrics': metrics,
            'model_type': model_type,
            'hyperparameters': hyperparams
        }
        
    def _create_audio_model(self, input_shape: Tuple, num_classes: int, 
                           hyperparams: Dict) -> nn.Module:
        """Create audio classification model"""
        class AudioModel(nn.Module):
            def __init__(self):
                super(AudioModel, self).__init__()
                
                if len(input_shape) == 3:  # Spectrogram (C, H, W)
                    channels = input_shape[0]
                    self.features = nn.Sequential(
                        nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1))
                    )
                    feature_size = 128
                else:  # 1D audio signal
                    self.features = nn.Sequential(
                        nn.Conv1d(1, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(2),
                        nn.Conv1d(32, 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(2),
                        nn.Conv1d(64, 128, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool1d(1)
                    )
                    feature_size = 128
                
                self.classifier = nn.Sequential(
                    nn.Linear(feature_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
                
        return AudioModel()
        
    async def _train_text_model(self, data_loader: DataLoader, model_type: str,
                               hyperparams: Dict, job_id: str) -> Dict[str, Any]:
        """Train text classification model using TensorFlow/Keras"""
        # Use TensorFlow for text models (LSTM/GRU)
        
        # Collect all data
        all_data = []
        all_labels = []
        
        for batch_data, batch_labels in data_loader:
            all_data.extend(batch_data.numpy())
            all_labels.extend(batch_labels.numpy())
        
        X = np.array(all_data)
        y = np.array(all_labels)
        
        # Create text model
        vocab_size = hyperparams.get('vocab_size', 10000)
        embedding_dim = hyperparams.get('embedding_dim', 128)
        max_length = hyperparams.get('max_length', 100)
        num_classes = len(np.unique(y))
        
        model = keras.Sequential([
            keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        epochs = hyperparams.get('epochs', 50)
        batch_size = hyperparams.get('batch_size', 32)
        
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        # Save model
        model_path = await self._save_tensorflow_model(model, job_id, model_type)
        
        # Calculate metrics
        final_loss = float(history.history['loss'][-1])
        final_accuracy = float(history.history['accuracy'][-1])
        val_accuracy = float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else 0.0
        
        metrics = {
            'final_loss': final_loss,
            'final_accuracy': final_accuracy,
            'val_accuracy': val_accuracy,
            'best_val_accuracy': float(max(history.history['val_accuracy'])) if 'val_accuracy' in history.history else 0.0,
            'epochs_trained': epochs
        }
        
        return {
            'file_path': model_path,
            'metrics': metrics,
            'model_type': model_type,
            'hyperparameters': hyperparams
        }
        
    async def _save_pytorch_model(self, model: nn.Module, job_id: str, model_type: str) -> str:
        """Save PyTorch model"""
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        
        model_filename = f"{job_id}_{model_type}.pth"
        model_path = os.path.join(models_dir, model_filename)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_type': model_type
        }, model_path)
        
        logger.info(f"PyTorch model saved to {model_path}")
        return model_path
        
    async def _save_tensorflow_model(self, model, job_id: str, model_type: str) -> str:
        """Save TensorFlow model"""
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        
        model_filename = f"{job_id}_{model_type}.h5"
        model_path = os.path.join(models_dir, model_filename)
        
        model.save(model_path)
        logger.info(f"TensorFlow model saved to {model_path}")
        return model_path
        
    async def load_pytorch_model(self, model_path: str, model_class, input_shape: Tuple) -> nn.Module:
        """Load PyTorch model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = model_class()
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading PyTorch model from {model_path}: {e}")
            raise
            
    async def load_tensorflow_model(self, model_path: str):
        """Load TensorFlow model"""
        try:
            model = keras.models.load_model(model_path)
            return model
        except Exception as e:
            logger.error(f"Error loading TensorFlow model from {model_path}: {e}")
            raise
            
    async def predict_pytorch(self, model: nn.Module, data: torch.Tensor) -> np.ndarray:
        """Make predictions with PyTorch model"""
        try:
            model.eval()
            with torch.no_grad():
                data = data.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
        except Exception as e:
            logger.error(f"Error making PyTorch predictions: {e}")
            raise
            
    async def predict_tensorflow(self, model, data: np.ndarray) -> np.ndarray:
        """Make predictions with TensorFlow model"""
        try:
            predictions = model.predict(data)
            return np.argmax(predictions, axis=1)
        except Exception as e:
            logger.error(f"Error making TensorFlow predictions: {e}")
            raise