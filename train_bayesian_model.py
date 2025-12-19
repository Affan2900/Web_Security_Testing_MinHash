"""
Bayesian Model Training Script
Trains Gaussian Process Regression models for adaptive hyperparameter selection.
"""

import json
import numpy as np
import pickle
import os

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("ERROR: scikit-learn not installed!")
    print("Install with: pip install scikit-learn")
    exit(1)


def load_training_data(filepath='training_data/training_dataset.json'):
    """
    Load training dataset from JSON file.
    
    Args:
        filepath (str): Path to training dataset
        
    Returns:
        tuple: (features_array, params_dict)
    """
    print(f"Loading training data from: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training samples")
    
    # Extract features and parameters
    feature_keys = [
        'dom_size', 'dom_depth', 'branching_factor', 'unique_tags',
        'tag_diversity', 'tag_entropy', 'js_detected', 'form_elements',
        'ajax_indicators', 'html_size', 'external_resources', 'div_span_ratio'
    ]
    
    X = []
    y_k = []
    y_ell = []
    y_tau = []
    
    for sample in data:
        features = sample['features']
        params = sample['optimal_params']
        
        # Create feature vector
        feature_vector = [features.get(key, 0) for key in feature_keys]
        X.append(feature_vector)
        
        # Extract target values
        y_k.append(params['k'])
        y_ell.append(params['ell'])
        y_tau.append(params['tau'])
    
    X = np.array(X)
    y_k = np.array(y_k)
    y_ell = np.array(y_ell)
    y_tau = np.array(y_tau)
    
    return X, {'k': y_k, 'ell': y_ell, 'tau': y_tau}


def train_model(X_train, y_train, param_name):
    """
    Train a Gaussian Process Regressor for a specific hyperparameter.
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training targets
        param_name (str): Name of parameter ('k', 'ell', or 'tau')
        
    Returns:
        GaussianProcessRegressor: Trained model
    """
    print(f"\n  Training model for '{param_name}'...")
    
    # Define kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    
    # Create and train model
    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=42,
        alpha=1e-6
    )
    
    model.fit(X_train, y_train)
    
    print(f"    ✓ Model trained for '{param_name}'")
    
    return model


def evaluate_model(model, X_test, y_test, param_name):
    """
    Evaluate trained model.
    
    Args:
        model: Trained GP model
        X_test: Test features
        y_test: Test targets
        param_name: Parameter name
        
    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'param': param_name,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2
    }


def train_all_models(training_file='training_data/training_dataset.json',
                     model_dir='models',
                     test_size=0.2):
    """
    Train all three GP models.
    
    Args:
        training_file (str): Path to training dataset
        model_dir (str): Directory to save models
        test_size (float): Fraction of data for testing
        
    Returns:
        dict: Trained models
    """
    print("=" * 60)
    print("Bayesian Model Training")
    print("=" * 60)
    
    # Load data
    X, y_dict = load_training_data(training_file)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Sample count: {X.shape[0]}")
    
    # Train-test split
    print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train models
    print("\n" + "-" * 60)
    print("Training Gaussian Process models...")
    print("-" * 60)
    
    models = {}
    evaluation_results = []
    
    for param_name in ['k', 'ell', 'tau']:
        y_train = y_dict[param_name][train_idx]
        y_test = y_dict[param_name][test_idx]
        
        # Train
        model = train_model(X_train, y_train, param_name)
        models[param_name] = model
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, param_name)
        evaluation_results.append(metrics)
    
    # Print evaluation results
    print("\n" + "=" * 60)
    print("Model Evaluation Results")
    print("=" * 60)
    
    for result in evaluation_results:
        print(f"\n{result['param'].upper()}:")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  R² Score: {result['r2']:.4f}")
    
    # Save models
    print("\n" + "-" * 60)
    print("Saving models...")
    print("-" * 60)
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'bayesian_optimizer.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    
    print(f"✓ Models saved to: {model_path}")
    
    return models, evaluation_results


def test_prediction(models, sample_features=None):
    """
    Test prediction with sample features.
    
    Args:
        models (dict): Trained models
        sample_features (dict): Optional sample features
    """
    print("\n" + "=" * 60)
    print("Testing Predictions")
    print("=" * 60)
    
    if sample_features is None:
        # Create a sample feature vector (medium complexity)
        sample_features = {
            'dom_size': 1500,
            'dom_depth': 12,
            'branching_factor': 3.5,
            'unique_tags': 30,
            'tag_diversity': 28,
            'tag_entropy': 4.0,
            'js_detected': 1,
            'form_elements': 15,
            'ajax_indicators': 2,
            'html_size': 35000,
            'external_resources': 15,
            'div_span_ratio': 4.0
        }
    
    # Create feature vector
    feature_keys =  [
        'dom_size', 'dom_depth', 'branching_factor', 'unique_tags',
        'tag_diversity', 'tag_entropy', 'js_detected', 'form_elements',
        'ajax_indicators', 'html_size', 'external_resources', 'div_span_ratio'
    ]
    
    feature_vector = np.array([[sample_features[key] for key in feature_keys]])
    
    print("\nSample Features:")
    for key in ['dom_size', 'unique_tags', 'js_detected', 'tag_entropy']:
        print(f"  {key}: {sample_features[key]}")
    
    print("\nPredicted Hyperparameters:")
    for param_name, model in models.items():
        pred = model.predict(feature_vector)[0]
        
        # Clamp to valid ranges
        if param_name == 'k':
            pred = int(round(max(5, min(25, pred))))
        elif param_name == 'ell':
            pred = int(round(max(50, min(500, pred))))
        elif param_name == 'tau':
            pred = max(0.60, min(0.95, pred))
        
        print(f"  {param_name}: {pred}")


if __name__ == "__main__":
    # Check if training data exists
    training_file = 'training_data/training_dataset.json'
    
    if not os.path.exists(training_file):
        print("ERROR: training_dataset.json not found!")
        print("Please run 'python collect_training_data.py' first")
        exit(1)
    
    # Train models
    models, results = train_all_models(training_file)
    
    # Test predictions
    test_prediction(models)
    
    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)
    print("\nYour ML model is now ready to use!")
    print("The adaptive hyperparameter selection will now use")
    print("the trained Bayesian model instead of heuristics.")
