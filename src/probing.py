"""
Linear probing techniques for extracting physical structure from JEPA latent representations.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings


class LatentProber:
    """
    Linear probing toolkit for extracting physical properties from latent representations.
    
    Trains simple linear classifiers/regressors on frozen latent representations
    to determine if physical properties are linearly accessible.
    """
    
    def __init__(
        self,
        latent_dim: int,
        random_state: int = 42,
        test_size: float = 0.2,
        cv_folds: int = 5
    ):
        """
        Initialize the LatentProber.
        
        Args:
            latent_dim: Dimensionality of latent representations
            random_state: Random seed for reproducibility
            test_size: Fraction of data to use for testing
            cv_folds: Number of cross-validation folds
        """
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.probes: Dict[str, Any] = {}
        
    def train_linear_probe(
        self,
        latent_representations: np.ndarray,
        labels: np.ndarray,
        property_name: str,
        task_type: str = "auto",
        regularization: float = 1.0
    ) -> Dict[str, float]:
        """
        Train a linear probe for a specific physical property.
        
        Args:
            latent_representations: Array of shape (n_samples, latent_dim)
            labels: Ground truth labels of shape (n_samples,) or (n_samples, n_outputs)
            property_name: Name of the property being probed
            task_type: 'classification', 'regression', or 'auto' (infer from labels)
            regularization: L2 regularization strength (C for classification, alpha for regression)
            
        Returns:
            Dictionary containing probe performance metrics
        """
        if latent_representations.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Mismatch in sample count: {latent_representations.shape[0]} vs {labels.shape[0]}"
            )
        
        if latent_representations.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got {latent_representations.shape[1]}"
            )
        
        # Infer task type if auto
        if task_type == "auto":
            task_type = self._infer_task_type(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            latent_representations,
            labels,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train probe
        if task_type == "classification":
            probe = LogisticRegression(
                C=regularization,
                random_state=self.random_state,
                max_iter=1000
            )
            probe.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = probe.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                probe, X_train_scaled, y_train,
                cv=self.cv_folds,
                scoring='accuracy'
            )
            
            results = {
                'property_name': property_name,
                'task_type': task_type,
                'accuracy': float(accuracy),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'n_train': len(X_train),
                'n_test': len(X_test)
            }
            
        else:  # regression
            probe = Ridge(alpha=regularization, random_state=self.random_state)
            probe.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = probe.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                probe, X_train_scaled, y_train,
                cv=self.cv_folds,
                scoring='r2'
            )
            
            results = {
                'property_name': property_name,
                'task_type': task_type,
                'r2_score': float(r2),
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'n_train': len(X_train),
                'n_test': len(X_test)
            }
        
        # Store probe
        self.probes[property_name] = {
            'model': probe,
            'scaler': self.scaler,
            'task_type': task_type,
            'results': results
        }
        
        return results
    
    def train_multi_probe(
        self,
        latent_representations: np.ndarray,
        properties: Dict[str, np.ndarray],
        regularization: float = 1.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Train linear probes for multiple physical properties simultaneously.
        
        Args:
            latent_representations: Array of shape (n_samples, latent_dim)
            properties: Dictionary mapping property names to label arrays
            regularization: L2 regularization strength
            
        Returns:
            Dictionary mapping property names to their probe results
        """
        results = {}
        
        for property_name, labels in properties.items():
            try:
                probe_results = self.train_linear_probe(
                    latent_representations,
                    labels,
                    property_name,
                    regularization=regularization
                )
                results[property_name] = probe_results
            except Exception as e:
                warnings.warn(f"Failed to probe {property_name}: {str(e)}")
                results[property_name] = {'error': str(e)}
        
        return results
    
    def predict(
        self,
        latent_representations: np.ndarray,
        property_name: str
    ) -> np.ndarray:
        """
        Use a trained probe to predict property values.
        
        Args:
            latent_representations: Array of shape (n_samples, latent_dim)
            property_name: Name of the property to predict
            
        Returns:
            Predicted property values
        """
        if property_name not in self.probes:
            raise ValueError(f"No probe trained for property '{property_name}'")
        
        probe_info = self.probes[property_name]
        X_scaled = probe_info['scaler'].transform(latent_representations)
        predictions = probe_info['model'].predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(
        self,
        property_name: str,
        top_k: Optional[int] = None
    ) -> np.ndarray:
        """
        Get feature importance scores for a trained probe.
        
        Args:
            property_name: Name of the property
            top_k: Return only top k features (None for all)
            
        Returns:
            Array of feature importance scores
        """
        if property_name not in self.probes:
            raise ValueError(f"No probe trained for property '{property_name}'")
        
        probe = self.probes[property_name]['model']
        
        # Get coefficients
        if hasattr(probe, 'coef_'):
            coef = probe.coef_
            if len(coef.shape) > 1:
                # Multi-class or multi-output: use L2 norm
                importance = np.linalg.norm(coef, axis=0)
            else:
                importance = np.abs(coef)
        else:
            raise ValueError(f"Probe for {property_name} has no coefficients")
        
        if top_k is not None:
            indices = np.argsort(importance)[-top_k:][::-1]
            return indices
        
        return importance
    
    def _infer_task_type(self, labels: np.ndarray) -> str:
        """
        Infer whether task is classification or regression from labels.
        
        Args:
            labels: Label array
            
        Returns:
            'classification' or 'regression'
        """
        # Check if labels are integers and have few unique values
        if labels.dtype in [np.int32, np.int64]:
            n_unique = len(np.unique(labels))
            if n_unique < 100:  # Arbitrary threshold
                return "classification"
        
        # Check if labels are continuous
        if labels.dtype in [np.float32, np.float64]:
            return "regression"
        
        # Default to classification for discrete values
        n_unique = len(np.unique(labels))
        if n_unique < len(labels) / 10:  # Less than 10% unique values
            return "classification"
        
        return "regression"
    
    def compute_probe_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all trained probes.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.probes:
            return {'n_probes': 0, 'message': 'No probes trained yet'}
        
        summary = {
            'n_probes': len(self.probes),
            'properties': list(self.probes.keys()),
            'classification_probes': [],
            'regression_probes': [],
            'mean_classification_accuracy': None,
            'mean_regression_r2': None
        }
        
        classification_scores = []
        regression_scores = []
        
        for prop_name, probe_info in self.probes.items():
            task_type = probe_info['task_type']
            results = probe_info['results']
            
            if task_type == 'classification':
                summary['classification_probes'].append(prop_name)
                if 'accuracy' in results:
                    classification_scores.append(results['accuracy'])
            else:
                summary['regression_probes'].append(prop_name)
                if 'r2_score' in results:
                    regression_scores.append(results['r2_score'])
        
        if classification_scores:
            summary['mean_classification_accuracy'] = float(np.mean(classification_scores))
        
        if regression_scores:
            summary['mean_regression_r2'] = float(np.mean(regression_scores))
        
        return summary


if __name__ == "__main__":
    # Demo: Probe synthetic latent representations
    print("=" * 60)
    print("JEPA Latent Probing Demo")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    latent_dim = 128
    
    # Latent representations with embedded structure
    latents = np.random.randn(n_samples, latent_dim)
    
    # Embed position information in first 2 dimensions
    positions = np.random.rand(n_samples, 2) * 100
    latents[:, 0:2] = positions + np.random.randn(n_samples, 2) * 0.1
    
    # Embed velocity in next 2 dimensions
    velocities = np.random.randn(n_samples, 2) * 10
    latents[:, 2:4] = velocities + np.random.randn(n_samples, 2) * 0.5
    
    # Embed object identity (5 classes) in dimensions 4-8
    object_ids = np.random.randint(0, 5, n_samples)
    for i in range(5):
        mask = object_ids == i
        latents[mask, 4:9] = np.random.randn(5) * 2 + i * 3
    
    print(f"\nGenerated {n_samples} synthetic latent representations")
    print(f"Latent dimension: {latent_dim}")
    
    # Initialize prober
    prober = LatentProber(latent_dim=latent_dim)
    
    # Probe position (regression)
    print("\n" + "-" * 60)
    print("Probing: Position (regression)")
    print("-" * 60)
    position_results = prober.train_linear_probe(
        latents,
        positions[:, 0],  # X coordinate
        property_name='position_x',
        task_type='regression'
    )
    print(f"R² Score: {position_results['r2_score']:.4f}")
    print(f"RMSE: {position_results['rmse']:.4f}")
    print(f"CV Mean ± Std: {position_results['cv_mean']:.4f} ± {position_results['cv_std']:.4f}")
    
    # Probe velocity (regression)
    print("\n" + "-" * 60)
    print("Probing: Velocity (regression)")
    print("-" * 60)
    velocity_results = prober.train_linear_probe(
        latents,
        velocities[:, 0],  # Vx
        property_name='velocity_x',
        task_type='regression'
    )
    print(f"R² Score: {velocity_results['r2_score']:.4f}")
    print(f"RMSE: {velocity_results['rmse']:.4f}")
    
    # Probe object identity (classification)
    print("\n" + "-" * 60)
    print("Probing: Object Identity (classification)")
    print("-" * 60)
    identity_results = prober.train_linear_probe(
        latents,
        object_ids,
        property_name='object_id',
        task_type='classification'
    )
    print(f"Accuracy: {identity_results['accuracy']:.4f}")
    print(f"CV Mean ± Std: {identity_results['cv_mean']:.4f} ± {identity_results['cv_std']:.4f}")
    
    # Multi-probe
    print("\n" + "-" * 60)
    print("Multi-Property Probing")
    print("-" * 60)
    multi_results = prober.train_multi_probe(
        latents,
        {
            'position_y': positions[:, 1],
            'velocity_y': velocities[:, 1],
        }
    )
    for prop, results in multi_results.items():
        if 'r2_score' in results:
            print(f"{prop}: R² = {results['r2_score']:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Probe Summary")
    print("=" * 60)
    summary = prober.compute_probe_summary()
    print(f"Total probes trained: {summary['n_probes']}")
    print(f"Classification probes: {summary['classification_probes']}")
    print(f"Regression probes: {summary['regression_probes']}")
    if summary['mean_classification_accuracy']:
        print(f"Mean classification accuracy: {summary['mean_classification_accuracy']:.4f}")
    if summary['mean_regression_r2']:
        print(f"Mean regression R²: {summary['mean_regression_r2']:.4f}")
    
    print("\n✓ Demo completed successfully!")