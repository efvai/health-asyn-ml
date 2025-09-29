"""
Production-Ready Motor Health Monitoring System

This module provides a simple interface for using the motor health monitoring
system in production environments.
"""

import sys
from pathlib import Path
import numpy as np
import pickle
from typing import Dict, Tuple, Optional, List
import logging

# Add the ml_toolbox to path
sys.path.append(str(Path(__file__).parent))

from ml_toolbox import (
    DataLoader, extract_features_for_ml, FeatureConfig
)

class MotorHealthMonitor:
    """
    Production-ready motor health monitoring system.
    
    This class provides a simple interface for:
    1. Training models on historical data
    2. Making real-time predictions
    3. Saving/loading trained models
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the motor health monitor.
        
        Args:
            model_path: Path to load a pre-trained model
        """
        self.model = None
        self.scaler = None
        self.feature_config = None
        self.feature_names = None
        self.classes = None
        
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def train(self, 
              dataset_path: Path, 
              max_windows_per_class: int = 100,
              window_size: int = 2048,
              test_size: float = 0.2) -> Dict:
        """
        Train the motor health monitoring model.
        
        Args:
            dataset_path: Path to dataset
            max_windows_per_class: Maximum windows per condition
            window_size: Size of analysis windows
            test_size: Fraction for testing
            
        Returns:
            Training results dictionary
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        from ml_toolbox import DatasetManager, create_windows_for_ml
        
        print("🔧 Training Motor Health Monitor")
        print("=" * 40)
        
        # 1. Load data
        print("📁 Loading dataset...")
        manager = DatasetManager(dataset_path)
        info = manager.get_dataset_info()
        
        data_loader = DataLoader(dataset_path)
        
        # Load data for each condition
        all_windows = []
        all_labels = []
        all_metadata = []
        
        for condition in info.conditions:
            if condition == 'cache':  # Skip cache files
                continue
                
            print(f"  Loading {condition}...")
            data, metadata = data_loader.load_batch(
                condition=condition,
                sensor_type="current",
                max_workers=2
            )
            
            if not data:
                continue
            
            windows, labels, win_metadata = create_windows_for_ml(
                data, metadata,
                window_size=window_size,
                overlap_ratio=0.5,
                balance_classes=True,
                max_windows_per_class=max_windows_per_class
            )
            
            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.extend([condition] * len(labels))
                all_metadata.extend(win_metadata)
                print(f"    ✓ {windows.shape[0]} windows")
        
        # Combine data
        X_windows = np.vstack(all_windows)
        y_labels = np.array(all_labels)
        
        print(f"Total: {X_windows.shape[0]} windows, {len(np.unique(y_labels))} classes")
        
        # 2. Extract features
        print("🔍 Extracting features...")
        
        self.feature_config = FeatureConfig(
            sampling_rate=10000,
            time_domain=True,
            frequency_domain=True,
            statistical_moments=True,
            shape_factors=True,
            entropy_features=False,  # Skip for performance
            spectral_features=True
        )
        
        X_features, self.feature_names = extract_features_for_ml(
            X_windows,
            sampling_rate=10000,
            sensor_type="current",
            feature_config=self.feature_config,
            metadata_list=all_metadata
        )
        
        # Remove zero-variance features
        feature_std = np.std(X_features, axis=0)
        good_features = feature_std > 1e-8
        X_features_clean = X_features[:, good_features]
        self.feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if good_features[i]]
        
        print(f"Features: {X_features_clean.shape[1]} (after cleaning)")
        
        # 3. Train model
        print("🤖 Training model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_features_clean, y_labels,
            test_size=test_size,
            random_state=42,
            stratify=y_labels
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        self.classes = self.model.classes_
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"✅ Training completed!")
        print(f"   Training accuracy: {train_score:.3f}")
        print(f"   Test accuracy: {test_score:.3f}")
        
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred),
            'n_features': len(self.feature_names),
            'n_samples': X_windows.shape[0]
        }
        
        return results
    
    def predict(self, 
                signal: np.ndarray, 
                sampling_rate: float = 10000.0,
                return_probabilities: bool = True) -> Dict:
        """
        Make a prediction on a signal.
        
        Args:
            signal: 2D array (samples, channels) or 1D array
            sampling_rate: Sampling rate in Hz
            return_probabilities: Whether to return probabilities
            
        Returns:
            Prediction results dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure 2D array
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        # Add batch dimension for feature extraction
        if signal.ndim == 2:
            signal = signal[np.newaxis, :, :]
        
        # Extract features
        features, _ = extract_features_for_ml(
            signal,
            sampling_rate=sampling_rate,
            sensor_type="current",
            feature_config=self.feature_config
        )
        
        # Handle feature dimension mismatch
        expected_features = len(self.feature_names)
        actual_features = features.shape[1]
        
        if actual_features < expected_features:
            # Pad with zeros for missing categorical features
            padding = np.zeros((features.shape[0], expected_features - actual_features))
            features_clean = np.hstack([features, padding])
        elif actual_features > expected_features:
            # Truncate to expected size
            features_clean = features[:, :expected_features]
        else:
            features_clean = features
        
        # Scale features
        features_scaled = self.scaler.transform(features_clean)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        result = {
            'condition': prediction,
            'confidence': None
        }
        
        if return_probabilities:
            probabilities = self.model.predict_proba(features_scaled)[0]
            result['probabilities'] = dict(zip(self.classes, probabilities))
            result['confidence'] = float(np.max(probabilities))
        
        return result
    
    def save_model(self, path: Path):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_config': self.feature_config,
            'feature_names': self.feature_names,
            'classes': self.classes
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load a trained model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_config = model_data['feature_config']
        self.feature_names = model_data['feature_names']
        self.classes = model_data['classes']
        
        print(f"✅ Model loaded from {path}")
    
    def get_feature_importance(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importances = self.model.feature_importances_
        feature_importance = list(zip(self.feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:top_n]


def demo_production_usage():
    """Demonstrate production usage of the motor health monitor."""
    print("🏭 Production Motor Health Monitor Demo")
    print("=" * 50)
    
    # 1. Train a model
    monitor = MotorHealthMonitor()
    
    try:
        results = monitor.train(
            dataset_path=Path("data_set"),
            max_windows_per_class=30,  # Small for demo
            window_size=1024  # Smaller window for faster training
        )
        
        print("\n📊 Training Results:")
        print(f"   • Test accuracy: {results['test_accuracy']:.1%}")
        print(f"   • Features used: {results['n_features']}")
        print(f"   • Training samples: {results['n_samples']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # 2. Save the model
    model_path = Path("motor_health_model.pkl")
    monitor.save_model(model_path)
    
    # 3. Load model and make predictions
    print("\n🔮 Making Predictions...")
    
    # Load some test data
    data_loader = DataLoader(Path("data_set"))
    test_data, _ = data_loader.load_batch(
        condition="healthy",
        sensor_type="current",
        max_workers=1
    )
    
    if test_data:
        # Take a sample from the test data
        sample = test_data[0][1000:2024]  # 1024 samples
        
        # Make prediction
        result = monitor.predict(sample)
        
        print(f"   Predicted condition: {result['condition']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print("   Probabilities:")
        for condition, prob in result['probabilities'].items():
            print(f"     {condition}: {prob:.3f}")
    
    # 4. Show feature importance
    print("\n📈 Top 5 Most Important Features:")
    top_features = monitor.get_feature_importance(5)
    for i, (feature, importance) in enumerate(top_features, 1):
        print(f"   {i}. {feature}: {importance:.4f}")
    
    print("\n✅ Production demo completed!")
    print("   The model is ready for deployment.")


if __name__ == "__main__":
    demo_production_usage()