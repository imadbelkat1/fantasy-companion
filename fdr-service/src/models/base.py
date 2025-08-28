from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss, accuracy_score
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for FDR models"""

    def __init__(self, league: str, config: Dict[str, Any]):
        self.league = league
        self.config = config
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self.validation_metrics = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit the model to training data"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass

    def validate_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                       X_val: pd.DataFrame, y_val: pd.DataFrame,
                       target_col: str = None) -> Dict[str, float]:
        """Validate model performance using time-based split"""

        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return {}

        train_pred = self.predict(X_train)
        val_pred = self.predict(X_val)

        if target_col and target_col in y_train.columns:
            train_actual = y_train[target_col].values
            val_actual = y_val[target_col].values

            # Regression metrics
            if 'goal_diff' in target_col:
                train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
                val_rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
                train_mae = mean_absolute_error(train_actual, train_pred)
                val_mae = mean_absolute_error(val_actual, val_pred)
                train_r2 = r2_score(train_actual, train_pred)
                val_r2 = r2_score(val_actual, val_pred)

                metrics = {
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'overfitting_ratio': val_rmse / max(train_rmse, 0.01)
                }

            # Classification metrics
            else:
                # Convert probabilities to predictions if needed
                if len(train_pred.shape) > 1:
                    train_pred_class = np.argmax(train_pred, axis=1)
                    val_pred_class = np.argmax(val_pred, axis=1)
                else:
                    train_pred_class = (train_pred > 0.5).astype(int)
                    val_pred_class = (val_pred > 0.5).astype(int)

                train_acc = accuracy_score(train_actual, train_pred_class)
                val_acc = accuracy_score(val_actual, val_pred_class)

                metrics = {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'accuracy_drop': train_acc - val_acc
                }

                # Add log loss if probabilities available
                try:
                    if len(train_pred.shape) > 1 and train_pred.shape[1] > 1:
                        train_logloss = log_loss(train_actual, train_pred)
                        val_logloss = log_loss(val_actual, val_pred)
                        metrics['train_logloss'] = train_logloss
                        metrics['val_logloss'] = val_logloss
                except:
                    pass
        else:
            metrics = {'error': 'Target column not found or not specified'}

        self.validation_metrics = metrics
        logger.info(f"{self.league} model validation: {metrics}")

        return metrics

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        return {
            'league': self.league,
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'validation_metrics': self.validation_metrics,
            'config': self.config
        }