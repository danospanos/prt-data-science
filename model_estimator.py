"""
Defines a classes for model estimation, predictions and evaluation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb


class ModelEstimatorMixins:
    def fitted_model_plot(self, true_y):
        """
        Plots actual vs predicted values.

        Args:
            true_y (pd.Series): True values of the target variable.

        Returns:
            Creates plot within CWD with appropriate name identifier.
        """
        result = pd.DataFrame(true_y).rename(columns={true_y.name:'true'})
        result['predicted'] = self.predicted
        plot = result.plot()
        plt.title(f'Predictions of {true_y.name}')
        plt.savefig(f'predictions-{true_y.name}.png')

    def evaluate(self, true_y, naive_predictions):
        """
        Evaluates model performance.

        Args:
            true_y (pd.Series): True values of target variable.

        Prints evaluation metrics and improvements over naive model.
        """
        #r_squared = r2_score(true_y, self.predicted)
        mae = mean_absolute_error(true_y, self.predicted)
        mse = mean_squared_error(true_y, self.predicted)
        print('Results')
        print('-'*60)
        #print('R_squared: {:1.4f}'.format(r_squared))
        print('MAE: {:13.4f}'.format(mae))
        print('MSE: {:14.4f}'.format(mse))
        print('-'*60)

    def _sanity_check(self, y_train, std_multiple=3):
        """
        Sanity check of the forecsts,
        if higher than std_multiple*std of y_train etc.
        """
        uncond_expected_price = y_train.mean()
        std_price = y_train.std()
        upper_limit = uncond_expected_price+(std_price*std_multiple)
        lower_limit = uncond_expected_price-(std_price*std_multiple)
        self.predicted = np.where(
            self.predicted > upper_limit, upper_limit, self.predicted)
        self.predicted = np.where(
            self.predicted < lower_limit, lower_limit, self.predicted)


class LGBM(ModelEstimatorMixins):
    """
    LGBM estimator for time series data

    Attributes:
        max_depth (int): Maximum value of lgbm max_depth param.
        metric (str): Metric under which to train the model.
    """
    def __init__(self, metric='l1', max_depth=25):
        """
        Initializes ModelEstimator with specified metric and max_depth.

        Args:
            max_depth (int): Maximum value of lgbm max_depth param.
            metric (str): Metric under which to train the model.
        """
        self.max_depth = max_depth
        self.metric = metric

    def fit(self, X, y):
        """
        Fits Lgbm model to training data using validation set to
        stop at the "correct" number of iterations.

        Args:
            X (DataFrame): Feature data.
            y (Series): Target variable.

        Returns:
            ModelEstimator: Instance itself.
        """
        self.model = lgb.LGBMRegressor(
            boosting_type='gbdt',
            objective='regression',
            early_stopping_rounds=100,
            num_leaves=5,
            max_depth=self.max_depth,
            learning_rate=0.1,
        )
        X_train, X_val, y_train, y_val =\
            train_test_split(X, y, test_size=0.75)
        self.model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_val, y_val)],
          eval_metric=self.metric)
        return self

    def predict(self, X):
        """
        Makes predictions using fitted LGBM model.

        Args:
            X (DataFrame): Feature data for predictions.

        Returns:
            ndarray: Predicted values.
        """
        self.predicted = self.model.predict(X)
        return self.predicted


class LassoEst(ModelEstimatorMixins):
    """
    TimeSeries Cross validated Lasso estimator for time series data
    using Lasso regression with cross-validation.

    Attributes:
        max_lambda (int): Maximum value of lambda for Lasso.
        scaler (StandardScaler): Feature scaler.
        model (Lasso): Trained Lasso model.
    """
    def __init__(self, max_lambda=100):
        """
        Initializes ModelEstimator with specified max_lambda.

        Args:
            max_lambda (int): Maximum lambda for Lasso regularization.
        """
        self.max_lambda = max_lambda
        # For better LASSO convergence
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        Fits Lasso model to training data using cross-validation.

        Args:
            X (DataFrame): Feature data.
            y (Series): Target variable.

        Returns:
            ModelEstimator: Instance itself.
        """
        X_scaled = self.scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(
            Lasso(),
            {'alpha': np.linspace(0.01, self.max_lambda, 100)},
            cv=tscv,
            scoring='neg_mean_squared_error')
        grid_search.fit(X_scaled, y)
        self.model = self._fitted_best_model(grid_search.best_params_, X_scaled, y)
        return self

    def predict(self, X):
        """
        Makes predictions using fitted Lasso model.

        Args:
            X (DataFrame): Feature data for predictions.

        Returns:
            ndarray: Predicted values.
        """
        self.predicted = self.model.predict(self.scaler.transform(X))
        return self.predicted

    def _fitted_best_model(self, best_model_params, X, y):
        """
        Fits Lasso model with best parameters from cross-validation.

        Args:
            best_model_params (dict): Best parameters from cross-validation.
            X (DataFrame): Feature data.
            y (Series): Target variable.

        Returns:
            Lasso: Fitted Lasso model.
        """
        return Lasso(**best_model_params).fit(X, y)
