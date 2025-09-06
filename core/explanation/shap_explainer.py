import shap
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
import logging
from config import settings

logger = logging.getLogger(__name__)

class SHAPExplainer:
    def __init__(self):
        self.explainer = None
        self.shap_values = None
        
    async def create_explainer(self, model_path: str, X_background: np.ndarray) -> Dict[str, Any]:
        """Create SHAP explainer for the model"""
        try:
            # Load model
            model_data = joblib.load(model_path)
            model = model_data['model']
            scaler = model_data['scaler']
            model_type = model_data['model_type']
            
            # Preprocess background data
            X_background_scaled = scaler.transform(X_background)
            
            # Limit background samples for performance
            if len(X_background_scaled) > settings.SHAP_MAX_SAMPLES:
                indices = np.random.choice(
                    len(X_background_scaled), 
                    settings.SHAP_MAX_SAMPLES, 
                    replace=False
                )
                X_background_scaled = X_background_scaled[indices]
            
            # Choose appropriate explainer based on model type
            if model_type in ['linear', 'logistic']:
                self.explainer = shap.LinearExplainer(model, X_background_scaled)
            elif model_type in ['svm_classifier', 'svm_regressor']:
                self.explainer = shap.KernelExplainer(model.predict, X_background_scaled)
            elif model_type in ['random_forest_classifier', 'random_forest_regressor']:
                self.explainer = shap.TreeExplainer(model)
            elif model_type == 'kmeans':
                # For clustering, we'll use KernelExplainer with a custom function
                def cluster_predict(X):
                    distances = model.transform(X)
                    return np.argmin(distances, axis=1)
                self.explainer = shap.KernelExplainer(cluster_predict, X_background_scaled)
            else:
                # Default to KernelExplainer
                self.explainer = shap.KernelExplainer(model.predict, X_background_scaled)
            
            return {
                'status': 'success',
                'explainer_type': type(self.explainer).__name__,
                'model_type': model_type,
                'background_samples': len(X_background_scaled)
            }
            
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
            raise
            
    async def explain_instance(self, X_instance: np.ndarray, model_path: str, 
                              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Explain a single prediction instance"""
        try:
            # Load model for preprocessing
            model_data = joblib.load(model_path)
            scaler = model_data['scaler']
            
            # Preprocess instance
            X_instance_scaled = scaler.transform(X_instance.reshape(1, -1))
            
            # Generate SHAP values
            if self.explainer is None:
                raise ValueError("Explainer not initialized. Call create_explainer first.")
                
            shap_values = self.explainer.shap_values(X_instance_scaled)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_values_dict = {}
                for i, class_shap in enumerate(shap_values):
                    shap_values_dict[f'class_{i}'] = class_shap[0].tolist()
            else:
                # Binary classification or regression
                shap_values_dict = {'values': shap_values[0].tolist()}
            
            # Get feature names
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(X_instance))]
            
            # Get expected value
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, np.ndarray):
                expected_value = expected_value.tolist()
            elif not isinstance(expected_value, list):
                expected_value = float(expected_value)
            
            return {
                'shap_values': shap_values_dict,
                'expected_value': expected_value,
                'feature_names': feature_names,
                'instance_values': X_instance.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error explaining instance: {e}")
            raise
            
    async def explain_dataset(self, X_data: np.ndarray, model_path: str,
                             feature_names: Optional[List[str]] = None,
                             max_samples: int = 100) -> Dict[str, Any]:
        """Explain predictions for a dataset"""
        try:
            # Load model for preprocessing
            model_data = joblib.load(model_path)
            scaler = model_data['scaler']
            
            # Limit samples for performance
            if len(X_data) > max_samples:
                indices = np.random.choice(len(X_data), max_samples, replace=False)
                X_data_sample = X_data[indices]
            else:
                X_data_sample = X_data
            
            # Preprocess data
            X_data_scaled = scaler.transform(X_data_sample)
            
            # Generate SHAP values
            if self.explainer is None:
                raise ValueError("Explainer not initialized. Call create_explainer first.")
                
            shap_values = self.explainer.shap_values(X_data_scaled)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class classification - take first class for summary
                summary_shap_values = shap_values[0]
            else:
                summary_shap_values = shap_values
            
            # Get feature names
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X_data_sample.shape[1])]
            
            # Calculate feature importance
            feature_importance = np.mean(np.abs(summary_shap_values), axis=0)
            
            # Create feature importance ranking
            importance_ranking = []
            for i, importance in enumerate(feature_importance):
                importance_ranking.append({
                    'feature': feature_names[i],
                    'importance': float(importance),
                    'rank': int(np.argsort(feature_importance)[::-1].tolist().index(i) + 1)
                })
            
            importance_ranking.sort(key=lambda x: x['importance'], reverse=True)
            
            return {
                'feature_importance': importance_ranking,
                'shap_values_shape': summary_shap_values.shape,
                'samples_analyzed': len(X_data_sample),
                'feature_names': feature_names
            }
            
        except Exception as e:
            logger.error(f"Error explaining dataset: {e}")
            raise
            
    async def generate_plots(self, X_data: np.ndarray, model_path: str,
                            feature_names: Optional[List[str]] = None,
                            plot_types: List[str] = ['summary', 'waterfall', 'force']) -> Dict[str, str]:
        """Generate SHAP visualization plots"""
        try:
            # Load model for preprocessing
            model_data = joblib.load(model_path)
            scaler = model_data['scaler']
            
            # Preprocess data (limit samples for visualization)
            max_viz_samples = min(50, len(X_data))
            X_viz = X_data[:max_viz_samples]
            X_viz_scaled = scaler.transform(X_viz)
            
            if self.explainer is None:
                raise ValueError("Explainer not initialized. Call create_explainer first.")
            
            # Generate SHAP values
            shap_values = self.explainer.shap_values(X_viz_scaled)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[0]  # Use first class
            else:
                shap_values_plot = shap_values
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X_viz.shape[1])]
            
            plots = {}
            
            # Summary plot
            if 'summary' in plot_types:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values_plot, X_viz_scaled, 
                                feature_names=feature_names, show=False)
                plots['summary'] = await self._plot_to_base64()
                plt.close()
            
            # Waterfall plot for first instance
            if 'waterfall' in plot_types and len(X_viz_scaled) > 0:
                plt.figure(figsize=(10, 6))
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = expected_value[0]
                    
                shap.waterfall_plot(
                    expected_value, 
                    shap_values_plot[0], 
                    X_viz_scaled[0],
                    feature_names=feature_names,
                    show=False
                )
                plots['waterfall'] = await self._plot_to_base64()
                plt.close()
            
            # Force plot for first instance
            if 'force' in plot_types and len(X_viz_scaled) > 0:
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = expected_value[0]
                    
                # Create force plot as matplotlib figure
                plt.figure(figsize=(12, 4))
                shap.force_plot(
                    expected_value,
                    shap_values_plot[0],
                    X_viz_scaled[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                plots['force'] = await self._plot_to_base64()
                plt.close()
            
            return plots
            
        except Exception as e:
            logger.error(f"Error generating SHAP plots: {e}")
            raise
            
    async def _plot_to_base64(self) -> str:
        """Convert matplotlib plot to base64 string"""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        
        # Convert to base64
        plot_base64 = base64.b64encode(plot_data).decode()
        return f"data:image/png;base64,{plot_base64}"
        
    async def get_top_features(self, X_data: np.ndarray, model_path: str,
                              feature_names: Optional[List[str]] = None,
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top K most important features"""
        try:
            explanation = await self.explain_dataset(X_data, model_path, feature_names)
            top_features = explanation['feature_importance'][:top_k]
            return top_features
            
        except Exception as e:
            logger.error(f"Error getting top features: {e}")
            raise
            
    async def explain_prediction_change(self, X_original: np.ndarray, X_modified: np.ndarray,
                                       model_path: str, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Explain how prediction changes between original and modified instances"""
        try:
            # Get explanations for both instances
            original_explanation = await self.explain_instance(X_original, model_path, feature_names)
            modified_explanation = await self.explain_instance(X_modified, model_path, feature_names)
            
            # Calculate differences
            if 'values' in original_explanation['shap_values']:
                original_shap = np.array(original_explanation['shap_values']['values'])
                modified_shap = np.array(modified_explanation['shap_values']['values'])
            else:
                # Multi-class case - use first class
                class_key = list(original_explanation['shap_values'].keys())[0]
                original_shap = np.array(original_explanation['shap_values'][class_key])
                modified_shap = np.array(modified_explanation['shap_values'][class_key])
            
            shap_diff = modified_shap - original_shap
            
            # Get feature names
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(X_original))]
            
            # Create difference summary
            changes = []
            for i, (feature, diff) in enumerate(zip(feature_names, shap_diff)):
                changes.append({
                    'feature': feature,
                    'original_value': float(X_original[i]),
                    'modified_value': float(X_modified[i]),
                    'shap_change': float(diff),
                    'abs_shap_change': float(abs(diff))
                })
            
            # Sort by absolute SHAP change
            changes.sort(key=lambda x: x['abs_shap_change'], reverse=True)
            
            return {
                'changes': changes,
                'total_shap_change': float(np.sum(shap_diff)),
                'original_explanation': original_explanation,
                'modified_explanation': modified_explanation
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction change: {e}")
            raise