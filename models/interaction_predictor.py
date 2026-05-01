"""
Neural Network-based Interaction Predictor
Deep learning model for predicting drug interactions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugInteractionDataset(Dataset):
    """PyTorch Dataset for drug interactions"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DrugInteractionNN(nn.Module):
    """
    Neural Network for Drug Interaction Prediction
    """
    
    def __init__(self, input_dim, hidden_layers=[256, 128, 64], dropout=0.3):
        """
        Initialize neural network
        
        Args:
            input_dim: Input feature dimension
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
        """
        super(DrugInteractionNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
        logger.info(f"Initialized Neural Network with architecture: {hidden_layers}")
    
    def forward(self, x):
        return self.model(x)


class InteractionPredictor:
    """
    Wrapper class for training and prediction
    """
    
    def __init__(self, input_dim, config=None):
        """
        Initialize predictor
        
        Args:
            input_dim: Input feature dimension
            config: Model configuration dictionary
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Default config
        if config is None:
            config = {
                'hidden_layers': [256, 128, 64],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50
            }
        
        self.config = config
        
        # Initialize model
        self.model = DrugInteractionNN(
            input_dim,
            hidden_layers=config['hidden_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        # Create datasets
        train_dataset = DrugInteractionDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            drop_last=True
        )
        
        if X_val is not None and y_val is not None:
            val_dataset = DrugInteractionDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False
            )
        else:
            val_loader = None
        
        # Training loop
        logger.info("Starting training...")
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_features).squeeze()
                loss = self.criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = self.model(batch_features).squeeze()
                        loss = self.criterion(outputs, batch_labels)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                self.val_losses.append(avg_val_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.config['epochs']}] - "
                              f"Train Loss: {avg_train_loss:.4f}, "
                              f"Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.config['epochs']}] - "
                              f"Train Loss: {avg_train_loss:.4f}")
        
        logger.info("Training completed!")
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions
        
        Args:
            X: Feature matrix
            threshold: Classification threshold
            
        Returns:
            Predictions and probabilities
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probabilities = self.model(X_tensor).cpu().numpy().squeeze()
            predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def predict_interaction(self, drug1_features, drug2_features):
        """
        Predict interaction between two drugs
        
        Args:
            drug1_features: Features for drug 1
            drug2_features: Features for drug 2
            
        Returns:
            Prediction result dictionary
        """
        # Combine features (you can modify this based on your feature engineering)
        combined_features = np.concatenate([drug1_features, drug2_features])
        combined_features = combined_features.reshape(1, -1)
        
        # Predict
        prediction, probability = self.predict(combined_features)
        
        result = {
            'has_interaction': bool(prediction[0]),
            'confidence': float(probability[0]),
            'risk_level': self._get_risk_level(probability[0])
        }
        
        return result
    
    def _get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability >= 0.8:
            return "HIGH"
        elif probability >= 0.5:
            return "MEDIUM"
        elif probability >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def save_model(self, filepath):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        logger.info(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    
    input_dim = 20  # Feature dimension
    n_samples = 1000
    
    X = np.random.randn(n_samples, input_dim)
    y = (np.random.randn(n_samples) > 0).astype(float)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Initialize predictor
    predictor = InteractionPredictor(input_dim)
    
    # Train
    predictor.train(X_train, y_train, X_val, y_val)
    
    # Predict
    test_sample = X_val[:1]
    predictions, probabilities = predictor.predict(test_sample)
    print(f"\nPrediction: {predictions[0]}, Probability: {probabilities[0]:.4f}")
    
    # Save model
    predictor.save_model('data/models/interaction_predictor.pth')
