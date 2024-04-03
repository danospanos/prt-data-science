"""
Defines classes to process data and train some simple model.
"""
import pandas as pd
import numpy as np
import datetime as dt
import holidays
import utils_data as utils
import statsmodels.api as sm
# Not best but fast
from statsmodels.tsa.seasonal import seasonal_decompose

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class DataProcessingMixin:
    """Mixin for data processing functionalities.

    This mixin provides methods for missing data
    imputation, date feature engineering,
    and normalization of numerical features.
    """
    def dropna(self):
        """Removes rows from `self.data` with missing values."""
        self.data = self.data.dropna()

    def _us_holidays_dummy(self):
        """
        Adds a binary column to `self.data` indicating
        US federal holidays.

        The method checks each date in the DataFrame
        index against the US federal holidays and
        adds a binary (0 or 1) column, where 1 indicates a holiday.
        """
        us_holidays = holidays.US()
        self.data['US_holiday'] = self.data.index.map(
            lambda x: 1 if x in us_holidays else 0)

    def _day_of_week_dummies(self):
        """Adds dummy variables for the day of
        the week to `self.data`.

        This method creates dummy variables for
        each day of the week, adds them to `self.data`,
        and drops the original day of the week column
        along with the dummy variable for the last day
        of the week to avoid multicollinearity.
        """
        self.data['day_of_week'] = self.data.index.dayofweek
        dummies = pd.get_dummies(
            self.data['day_of_week'], prefix='day', drop_first=False)
        self.data = self.data.join(dummies.astype(int))
        # Dropping also day_6 as it would cause exact multicollinerity
        self.data.drop(['day_of_week', 'day_6'], axis=1, inplace=True)

    def _ar_model_predictions(self, column, rolling_window=500,
            lags=48, retrain_period=720):
        """Generates autoregressive model predictions
        for a specified column.

        Args:
            column (str): Name of the column to model.
            rolling_window (int): Size of the rolling window
                for the training set.
            lags (int): Number of lagged observations to include
                as features.
            retrain_period (int): Interval at which the model
                is retrained.

        The method uses a rolling window approach
        to fit an autoregressive model and predicts
        one-hour ahead values, retraining the model
        at specified intervals.
        """
        def ar_features(df, column, lags):
            """Generates lagged features for autoregressive modeling.

            Args:
                df (pd.DataFrame): DataFrame
                    containing the target column.
                column (str): Target column name.
                lags (int): Number of lags to generate.

            Returns:
                pd.DataFrame: Modified DataFrame with
                    lagged features.
                list: List of feature names.
            """
            df['target'] = df[column].shift(-1)
            features = [column]
            for lag in range(1, lags+1):
                df[f'{column}_l{lag}'] = df[column].shift(lag)
                features.append(f'{column}_l{lag}')
            df = df.dropna()
            return df, features
        tmp = self.data.copy(deep=True)
        tmp, features = ar_features(tmp, column, lags)
        tmp[f'forecast_{column}'] = np.nan

        end = rolling_window
        while end < len(tmp):
            train = tmp[end-rolling_window:end]
            X_train = train[features]
            y_train = train['target']
            model_fitted = sm.OLS(
                y_train, sm.add_constant(X_train)).fit()
            forecast = model_fitted.predict(
                    sm.add_constant(tmp).iloc[end:end+retrain_period][['const']+features])
            tmp.loc[forecast.index, f'forecast_{column}'] = forecast
            end += retrain_period
        self.data[f'forecast_{column}'] = tmp[f'forecast_{column}']

    def _forecast_errors(self, column):
        """Calculates forecast errors for a given column.

        Assuming the prediction accuracy for temperature
        and load is comparable to market standards,
        this method calculates the residuals between actual
        values and forecasted values. The notion is that
        surprises in predictions are significant for spot
        price analysis, though less so for the day-ahead market.

        Args:
            column (str): The column name for which to
                calculate forecast errors.
        """
        self.data[f'forecast_resid_{column}'] = \
            self.data[column]-self.data[f'forecast_{column}']

    def _rate_of_change(self, column, lag=1):
        """Calculates the rate of change for a given
        column based on a specified lag.

        Rapid changes in variables such as load and
        temperature can lead to unexpected behavior.
        This method computes the rate of change to
        capture such dynamics.

        Args:
            column (str): The column name for which
                to calculate the rate of change.
            lag (int): The lag period to use for
                calculating the rate of change. Defaults to 1.
        """
        def change(series):
            prev_values = series.shift(lag)
            return (series-prev_values)/lag
        self.data[f'roc_{column}_l{lag}'] = change(self.data[f'{column}'])

    def add_features(self):
        """Adds various features to the dataset based
            on temperature and load.

        This method enriches the dataset with additional
        features derived from existing data, including
        price lags, rate of change, rolling means,
        seasonal decompositions, and AR model predictions.

        Args:
            max_lag (int): Maximum number of lags to
                include for price. Defaults to 48.
        """
        for lag in range(1, self.max_lag):
            self.data[f'price_l{lag}'] = self.data['price'].shift(lag)
        for col in ['temperature', 'load']:
            self._rate_of_change(col)
            for roll in [12, 24, 168]:
                self.data[f'{col}_mean{roll}'] = self.data[col].rolling(roll).mean()
                self._rate_of_change(col, roll)
            for period in [24, 168, 720, 8640]:
                self.data[f'seasonal_decomp_{col}_p{period}'] = seasonal_decompose(
                    self.data[col], model='additive', period=period).seasonal
            self._ar_model_predictions(col)
            self._forecast_errors(col)
        self._day_of_week_dummies()
        self._us_holidays_dummy()



class DataProcessor(DataProcessingMixin):
    """
    Data processor class for preparing time series data.

    Attributes:
        data (DataFrame): DataFrame containing the time series data.
    """
    def __init__(self,
                 data,
                 base_features=['price', 'temperature', 'load'],
                 max_lag=24):
        self.data = data
        self.base_features = base_features
        self.max_lag = max_lag

    def resample_like_features(self, n_ahead, feature):
        """
        Add feature lags at given n_ahead frequency.
        This is meant to mimic something like if we use .resample()
        with n_ahead frequency.

        Args:
            n_ahead (int): Number of hours to predict ahead.
            dayhour (int): Hour of a day at which the prediction is made.
        """
        for lag in range(1, 24):
            adjusted_lag = n_ahead*lag
            self.data[f'{feature}_l{adjusted_lag}'] =\
                self.data[feature].shift(adjusted_lag)

    def time_of_day_features(self, dayhour, n_ahead, feature):
        """
        Make features to correspond with target shift time.
        In other words, if we predict tomorrows 17:00,
        we utilize past values of prices temp and load at 17:00.

        Args:
            n_ahead (int): Number of hours to predict ahead.
            dayhour (int): Hour of a day at which the prediction is made.
            feature (str): I give up with writing this :D
        """
        time_after_shift = (dayhour+n_ahead)%24
        self.data['time_of_day'] = self.data.index.hour
        available_times = self.data[self.data['time_of_day']==dayhour].index
        for time in available_times:
            sub_df = self.data[
                (self.data['time_of_day']==time_after_shift) &
                (self.data.index < time) &
                (self.data.index > time-dt.timedelta(days=self.max_lag+10))]\
                .sort_index(ascending=False)
            if len(sub_df)>self.max_lag:
                self.data.loc[time, f'{feature}_plag1'] = \
                    sub_df.iloc[0][feature]
        self.data[f'{feature}_plag1'] = self.data[f'{feature}_plag1'].ffill()
        self.data = self.data.drop(columns=['time_of_day'])

    def lag_time_of_day_features(self, feature):
        """
        Should be used after a day-time filter.
        """
        for lag in range(1, self.max_lag):
            self.data[f'{feature}_plag{lag+1}'] =\
                self.data[f'{feature}_plag1'].shift(lag)

    def filter_data_by_dayhour(self, hour_of_day):
        self.data = self.data[self.data.index.hour==hour_of_day].copy()

    def _shift_day_of_week_and_holidays(self, n_ahead):
        self.data['US_holiday'] = self.data['US_holiday'].shift(-1*n_ahead)
        for day_dummy in [col for col in self.data.columns  if 'day_' in col]:
            self.data[day_dummy] = self.data[day_dummy].shift(-1*n_ahead)

    def dayhour_data(self, n_ahead, dayhour):
        """
        Processes data by imputing missing values and adding features.

        Args:
            n_ahead (int): Number of hours to predict ahead.
            dayhour (int): Hour of a day at which the prediction is made.

        Returns:
            Tuple[DataFrame, Series]: Processed features and target variable.
        """
        for feature in self.base_features:
            self.resample_like_features(n_ahead, feature)
            self.time_of_day_features(dayhour, n_ahead, feature)
        # Missing data only at the beginning and at the end
        self.dropna()
        self.add_features()
        #
        self._shift_day_of_week_and_holidays(n_ahead)
        self.filter_data_by_dayhour(dayhour)
        for feature in self.base_features:
            self.lag_time_of_day_features(feature)
        # Define features and target
        features = self.data.columns
        self.data['target'] = self.data['price'].shift(-1*n_ahead)
        self.dropna()
        return self.data[features], self.data['target']

    @classmethod
    def from_csv(cls):
        """
        Creates DataProcessor instance from a CSV file.

        Args:
            target_variable (str): Name of the target variable.
            filename (str): Name of the CSV file.

        Returns:
            DataProcessor: Instance of DataProcessor.
        """
        # Prices
        price_df = utils.daily_to_hourly(
                utils.load_prt_legacy_file('Price.act')).dropna()
        price_df = price_df.rename(columns={0:'price'})
        # Temperature
        temperature_df = utils.daily_to_hourly(
                utils.load_prt_legacy_file('Temp.act')).dropna()
        temperature_df = temperature_df.rename(columns={0:'temperature'})
        # Load
        load_df = utils.daily_to_hourly(
                utils.load_prt_legacy_file('Load.act')).dropna()
        load_df = load_df.rename(columns={0:'load'})
        df = pd.concat([price_df, temperature_df, load_df], axis=1)
        return cls(df)
