import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, silhouette_score
)
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class MLTrainer:
    def __init__(self):
        self.models = {
            'linear': LinearRegression,
            'logistic': LogisticRegression,
            'svm_classifier': SVC,
            'svm_regressor': SVR,
            'random_forest_classifier': RandomForestClassifier,
            'random_forest_regressor': RandomForestRegressor,
            'kmeans': KMeans
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    async def train(self, X: np.ndarray, y: np.ndarray, model_type: str, 
                   hyperparams: Dict, job_id: str) -> Dict[str, Any]:
        """Train ML model"""
        try:
            logger.info(f"Starting training for {model_type} with job ID {job_id}")
            
            # Determine if classification or regression
            is_clustering = model_type == 'kmeans'
            is_classification = self._is_classification_task(y) if not is_clustering else False
            
            # Adjust model type based on task type
            if model_type == 'svm':
                model_type = 'svm_classifier' if is_classification else 'svm_regressor'
            elif model_type == 'random_forest':
                model_type = 'random_forest_classifier' if is_classification else 'random_forest_regressor'
            
            # Preprocess data
            X_processed, y_processed = await self._preprocess_data(X, y, is_clustering)
            
            # Initialize model
            model_class = self.models[model_type]
            model = model_class(**hyperparams)
            
            # Train model
            if is_clustering:
                model.fit(X_processed)
                metrics = await self._evaluate_clustering(model, X_processed)
            else:
                # Split data for supervised learning
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=0.2, random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                metrics = await self._evaluate_model(
                    y_test, y_pred, is_classification
                )
                
                # Add cross-validation score
                cv_scores = cross_val_score(model, X_processed, y_processed, cv=5)
                metrics['cv_mean'] = float(np.mean(cv_scores))
                metrics['cv_std'] = float(np.std(cv_scores))
            
            # Save model
            model_path = await self._save_model(model, job_id, model_type)
            
            return {
                'file_path': model_path,
                'metrics': metrics,
                'model_type': model_type,
                'hyperparameters': hyperparams
            }
            
        except Exception as e:
            logger.error(f"Training failed for {model_type}: {e}")
            raise
            
    def _is_classification_task(self, y: np.ndarray) -> bool:
        """Determine if task is classification or regression"""
        # Check if target is categorical or has few unique values
        unique_values = len(np.unique(y))
        total_values = len(y)
        
        # If less than 10% unique values or non-numeric, likely classification
        if unique_values < 20 or unique_values / total_values < 0.1:
            return True
            
        # Check if all values are integers
        if np.all(y == y.astype(int)):
            return True
            
        return False
        
    async def _preprocess_data(self, X: np.ndarray, y: np.ndarray, 
                              is_clustering: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for training"""
        # Handle missing values
        if np.isnan(X).any():
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if is_clustering:
            return X_scaled, None
            
        # Encode labels if necessary
        if y.dtype == 'object' or y.dtype.kind in {'U', 'S'}:
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
            
        return X_scaled, y_encoded
        
    async def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             is_classification: bool) -> Dict[str, float]:
        """Evaluate model performance"""
        metrics = {}
        
        if is_classification:
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted'))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted'))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted'))
        else:
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics['r2_score'] = float(r2_score(y_true, y_pred))
            
        return metrics
        
    async def _evaluate_clustering(self, model, X: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering model"""
        cluster_labels = model.labels_
        
        metrics = {
            'n_clusters': int(model.n_clusters),
            'silhouette_score': float(silhouette_score(X, cluster_labels)),
            'inertia': float(model.inertia_)
        }
        
        return metrics
        
    async def _save_model(self, model, job_id: str, model_type: str) -> str:
        """Save trained model"""
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        
        model_filename = f"{job_id}_{model_type}.joblib"
        model_path = os.path.join(models_dir, model_filename)
        
        # Save model along with preprocessors
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder if hasattr(self.label_encoder, 'classes_') else None,
            'model_type': model_type
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
        
    async def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load trained model"""
        try:
            model_data = joblib.load(model_path)
            return model_data
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
            
    async def predict(self, model_path: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model"""
        try:
            model_data = await self.load_model(model_path)
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Preprocess input data
            X_scaled = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            
            # Decode labels if classification
            if model_data['label_encoder'] is not None:
                predictions = model_data['label_encoder'].inverse_transform(predictions)
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise