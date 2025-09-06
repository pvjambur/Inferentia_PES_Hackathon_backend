import os
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
import logging
from groq import Groq
from config import settings

logger = logging.getLogger(__name__)

class TextSyntheticGenerator:
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
        
    async def load_original_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load original text data"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                return df.to_dict('records')
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading original data: {e}")
            raise
            
    async def generate_synthetic_data(self, original_data: List[Dict], 
                                    num_samples: int, generation_method: str = "groq",
                                    feedback: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Generate synthetic text data"""
        try:
            if generation_method == "groq" and self.groq_client:
                return await self._generate_with_groq(original_data, num_samples, feedback)
            else:
                return await self._generate_with_patterns(original_data, num_samples, feedback)
                
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            raise
            
    async def _generate_with_groq(self, original_data: List[Dict], 
                                 num_samples: int, feedback: Optional[Dict] = None) -> List[Dict]:
        """Generate synthetic data using Groq API"""
        try:
            # Analyze original data structure
            if not original_data:
                raise ValueError("No original data provided")
                
            sample_record = original_data[0]
            columns = list(sample_record.keys())
            
            # Create prompt for data generation
            prompt = self._create_generation_prompt(original_data[:10], columns, feedback)
            
            synthetic_data = []
            batch_size = 10  # Generate in batches
            
            for batch in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - batch)
                
                response = self.groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": "You are a data synthesis expert. Generate realistic synthetic data based on the patterns shown."},
                        {"role": "user", "content": f"{prompt}\n\nGenerate {current_batch_size} new synthetic records in the same JSON format."}
                    ],
                    temperature=0.8,
                    max_tokens=2000
                )
                
                # Parse response
                try:
                    batch_data = json.loads(response.choices[0].message.content)
                    if isinstance(batch_data, list):
                        synthetic_data.extend(batch_data)
                    else:
                        synthetic_data.append(batch_data)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Groq response, using pattern generation")
                    fallback_data = await self._generate_with_patterns(
                        original_data, current_batch_size, feedback
                    )
                    synthetic_data.extend(fallback_data)
            
            return synthetic_data[:num_samples]
            
        except Exception as e:
            logger.error(f"Error with Groq generation: {e}")
            # Fallback to pattern generation
            return await self._generate_with_patterns(original_data, num_samples, feedback)
            
    def _create_generation_prompt(self, sample_data: List[Dict], 
                                 columns: List[str], feedback: Optional[Dict] = None) -> str:
        """Create prompt for synthetic data generation"""
        prompt = f"Generate synthetic data with the following structure and patterns:\n\n"
        prompt += f"Columns: {', '.join(columns)}\n\n"
        prompt += f"Sample data:\n{json.dumps(sample_data, indent=2)}\n\n"
        
        if feedback:
            prompt += f"Consider this feedback for generation:\n"
            if 'positive_examples' in feedback:
                prompt += f"Good examples: {feedback['positive_examples']}\n"
            if 'negative_examples' in feedback:
                prompt += f"Avoid patterns like: {feedback['negative_examples']}\n"
            if 'requirements' in feedback:
                prompt += f"Requirements: {feedback['requirements']}\n"
        
        prompt += "Generate realistic, diverse synthetic data that follows these patterns but is not identical to the examples."
        
        return prompt
        
    async def _generate_with_patterns(self, original_data: List[Dict], 
                                    num_samples: int, feedback: Optional[Dict] = None) -> List[Dict]:
        """Generate synthetic data using pattern analysis"""
        try:
            if not original_data:
                return []
                
            synthetic_data = []
            df = pd.DataFrame(original_data)
            
            for _ in range(num_samples):
                synthetic_record = {}
                
                for column in df.columns:
                    if df[column].dtype == 'object':
                        # Text/categorical data
                        synthetic_record[column] = await self._generate_text_field(
                            df[column].tolist(), feedback
                        )
                    else:
                        # Numeric data
                        synthetic_record[column] = await self._generate_numeric_field(
                            df[column].tolist(), feedback
                        )
                
                synthetic_data.append(synthetic_record)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error with pattern generation: {e}")
            raise
            
    async def _generate_text_field(self, values: List[Any], feedback: Optional[Dict] = None) -> str:
        """Generate synthetic text field"""
        try:
            # Remove null values
            valid_values = [v for v in values if pd.notna(v)]
            
            if not valid_values:
                return "synthetic_value"
            
            # For categorical data, sample with variation
            if len(set(valid_values)) < len(valid_values) * 0.5:
                # Categorical - sample with some noise
                base_value = np.random.choice(valid_values)
                if feedback and 'text_variations' in feedback:
                    return self._add_text_variation(base_value, feedback['text_variations'])
                return base_value
            else:
                # Free text - create variations
                base_text = np.random.choice(valid_values)
                return self._create_text_variation(base_text)
                
        except Exception:
            return "synthetic_text"
            
    async def _generate_numeric_field(self, values: List[Any], feedback: Optional[Dict] = None) -> float:
        """Generate synthetic numeric field"""
        try:
            valid_values = [v for v in values if pd.notna(v)]
            
            if not valid_values:
                return 0.0
            
            # Use normal distribution based on original data
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values) if len(valid_values) > 1 else abs(mean_val * 0.1)
            
            # Add noise based on feedback
            noise_factor = 1.0
            if feedback and 'numeric_noise' in feedback:
                noise_factor = feedback['numeric_noise']
            
            synthetic_value = np.random.normal(mean_val, std_val * noise_factor)
            
            # Keep within reasonable bounds
            min_val, max_val = min(valid_values), max(valid_values)
            synthetic_value = np.clip(synthetic_value, min_val * 0.5, max_val * 1.5)
            
            return float(synthetic_value)
            
        except Exception:
            return 0.0
            
    def _add_text_variation(self, base_text: str, variations: List[str]) -> str:
        """Add variation to text based on feedback"""
        if np.random.random() < 0.3:  # 30% chance of variation
            variation = np.random.choice(variations)
            return f"{base_text}_{variation}"
        return base_text
        
    def _create_text_variation(self, base_text: str) -> str:
        """Create variation of text"""
        if len(base_text.split()) > 1:
            words = base_text.split()
            # Randomly shuffle some words or add prefixes/suffixes
            if np.random.random() < 0.3:
                return ' '.join(np.random.permutation(words))
            elif np.random.random() < 0.3:
                return f"synthetic_{base_text}"
            else:
                return f"{base_text}_variant"
        else:
            return f"{base_text}_synthetic"
            
    async def save_synthetic_data(self, synthetic_data: List[Dict], 
                                 agent_id: str, dataset_id: str) -> str:
        """Save synthetic data to file"""
        try:
            # Create synthetic data directory
            synthetic_dir = f"data/datasets/text/synthetic"
            os.makedirs(synthetic_dir, exist_ok=True)
            
            # Create filename
            filename = f"synthetic_{agent_id}_{dataset_id}.csv"
            file_path = os.path.join(synthetic_dir, filename)

        except Exception as e:
            logger.error(f"Error saving synthetic data: {e}")
            raise
            
    async def save_synthetic_data(self, synthetic_data: List[Dict], 
                                 agent_id: str, dataset_id: str) -> str:
        """Save synthetic data to file"""
        try:
            # Create synthetic data directory
            synthetic_dir = f"data/datasets/text/synthetic"
            os.makedirs(synthetic_dir, exist_ok=True)
            
            # Create filename
            filename = f"synthetic_{agent_id}_{dataset_id}.csv"
            file_path = os.path.join(synthetic_dir, filename)
            
            # Save as CSV
            df = pd.DataFrame(synthetic_data)
            df.to_csv(file_path, index=False)
            
            logger.info(f"Synthetic data saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving synthetic data: {e}")
            raise
            
    async def merge_datasets(self, original_path: str, synthetic_path: str, 
                           merge_ratio: float = 0.3) -> str:
        """Merge original and synthetic datasets"""
        try:
            # Load both datasets
            original_df = pd.read_csv(original_path)
            synthetic_df = pd.read_csv(synthetic_path)
            
            # Calculate samples to include
            original_samples = int(len(original_df) * (1 - merge_ratio))
            synthetic_samples = int(len(synthetic_df) * merge_ratio / (1 - merge_ratio))
            
            # Sample from both datasets
            original_sample = original_df.sample(n=min(original_samples, len(original_df)))
            synthetic_sample = synthetic_df.sample(n=min(synthetic_samples, len(synthetic_df)))
            
            # Merge datasets
            merged_df = pd.concat([original_sample, synthetic_sample], ignore_index=True)
            
            # Shuffle the merged dataset
            merged_df = merged_df.sample(frac=1).reset_index(drop=True)
            
            # Save merged dataset
            merged_dir = f"data/datasets/text/merged"
            os.makedirs(merged_dir, exist_ok=True)
            
            merged_filename = f"merged_{os.path.basename(original_path)}"
            merged_path = os.path.join(merged_dir, merged_filename)
            
            merged_df.to_csv(merged_path, index=False)
            
            logger.info(f"Merged dataset saved to {merged_path}")
            return merged_path
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            raise
            
    async def analyze_data_quality(self, original_data: List[Dict], 
                                 synthetic_data: List[Dict]) -> Dict[str, Any]:
        """Analyze quality of synthetic data compared to original"""
        try:
            original_df = pd.DataFrame(original_data)
            synthetic_df = pd.DataFrame(synthetic_data)
            
            quality_metrics = {
                'data_shape_match': original_df.shape[1] == synthetic_df.shape[1],
                'column_match': set(original_df.columns) == set(synthetic_df.columns),
                'statistical_similarity': {},
                'diversity_score': 0.0
            }
            
            # Compare statistical properties for numeric columns
            for column in original_df.columns:
                if original_df[column].dtype in ['int64', 'float64']:
                    orig_mean = original_df[column].mean()
                    synth_mean = synthetic_df[column].mean()
                    orig_std = original_df[column].std()
                    synth_std = synthetic_df[column].std()
                    
                    quality_metrics['statistical_similarity'][column] = {
                        'mean_similarity': 1 - abs(orig_mean - synth_mean) / abs(orig_mean) if orig_mean != 0 else 1.0,
                        'std_similarity': 1 - abs(orig_std - synth_std) / abs(orig_std) if orig_std != 0 else 1.0
                    }
            
            # Calculate diversity score
            synthetic_unique_ratio = len(synthetic_df.drop_duplicates()) / len(synthetic_df)
            original_unique_ratio = len(original_df.drop_duplicates()) / len(original_df)
            quality_metrics['diversity_score'] = min(synthetic_unique_ratio / original_unique_ratio, 1.0)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing data quality: {e}")
            return {'error': str(e)}