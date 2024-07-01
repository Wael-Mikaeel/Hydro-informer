# utils.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import PowerTransformer, StandardScaler

def create_input_output(data, input_cols, output_cols, precipitation_col, input_length, output_length, aux_cols):
    """
    Converts DataFrame columns into NumPy arrays for fast processing and handles the creation of input and output sequences with a specific overlap strategy.
    Adds random noise to the precipitation array to account for variability.
    """
    input_data = data[input_cols].values
    output_data = data[output_cols].values
    aux_data = data[aux_cols].values
    precipitation_data = data[precipitation_col].values
    
    num_samples = (len(data) - input_length - output_length + 1) // (output_length)

    input_array = np.zeros((num_samples, input_length, len(input_cols)))
    output_array = np.zeros((num_samples, output_length, len(output_cols)))
    aux_array = np.zeros((num_samples, input_length, len(aux_cols)))
    precipitation_array = np.zeros((num_samples, output_length, len(precipitation_col)))
    
    for i in range(num_samples):
        start_idx = i * (output_length)
        input_array[i] = input_data[start_idx:start_idx + input_length]
        output_array[i] = output_data[start_idx + input_length:start_idx + input_length + output_length]
        aux_array[i] = aux_data[start_idx:start_idx + input_length]
        precipitation_array[i] = precipitation_data[i + input_length:i + input_length + output_length]
        
    precipitation_array += np.random.normal(0, 0.01, size=precipitation_array.shape)
    
    return input_array, output_array, precipitation_array, aux_array

class ColumnTransformer:
    """
    Applies different transformations to DataFrame columns based on their prefixes.
    """
    def __init__(self):
        self.transformers = {}
        self.col_mins = {}
    
    def fit_transform(self, df):
        transformed_df = df.copy()
        
        for col in df.columns:
            if col.startswith('H'):
                pt = PowerTransformer()
                transformed_data = pt.fit_transform(df[[col]]).flatten()
                transformed_df[col] = transformed_data
                self.transformers[col] = pt
                
            elif col.startswith('Q'):
                log_transformed = np.log1p(df[col])
                transformed_df[col] = log_transformed
                self.transformers[col] = 'log'
                
            elif col.startswith('P'):
                pt = PowerTransformer(method='box-cox')
                non_zero_mask = df[col] != 0
                transformed_data = np.zeros_like(df[col], dtype=float)
                non_zero_transformed = pt.fit_transform(df.loc[non_zero_mask, [col]]).flatten()
                min_value = non_zero_transformed.min()
                self.col_mins[col] = min_value
                transformed_data[non_zero_mask] = non_zero_transformed
                transformed_data[~non_zero_mask] = min_value - 0.001
                transformed_df[col] = transformed_data
                self.transformers[col] = pt
                
            elif col.startswith('T'):
                ss = StandardScaler()
                transformed_df[col] = ss.fit_transform(df[[col]]).flatten()
                self.transformers[col] = ss
        
        return transformed_df
    
    def inverse_transform(self, transformed_df):
        inverse_transformed_df = transformed_df.copy()
        
        for col, transformer in self.transformers.items():
            if col.startswith('H') and transformer != 'log':
                inverse_transformed_df[col] = transformer.inverse_transform(transformed_df[[col]]).flatten()
            elif transformer == 'log':
                inverse_transformed_df[col] = np.expm1(transformed_df[col])
            else:
                inverse_transformed_df[col] = transformer.inverse_transform(transformed_df[[col]]).flatten()
                
        return inverse_transformed_df
    
    def inverse_transform_column(self, transformed_array, col_name):
        transformer = self.transformers.get(col_name)
        
        if transformer:
            if col_name.startswith('H') and transformer != 'log':
                return transformer.inverse_transform(transformed_array)
            elif transformer == 'log':
                return np.expm1(transformed_array)
            else:
                return transformer.inverse_transform(transformed_array)
        else:
            raise ValueError(f"No transformer found for column {col_name}")

def custom_mse_with_threshold(y_true, y_pred, threshold=10):
    error = y_true - y_pred
    squared_error = tf.square(error)
    
    condition = tf.logical_and(tf.less(y_pred, y_true), tf.less_equal(error, threshold))
    
    adjusted_error = tf.where(condition, squared_error * 4, squared_error)
    
    return tf.reduce_mean(adjusted_error)

def custom_loss_extreme_focus(y_true, y_pred, threshold=10, extreme_threshold=200):
    error = y_true - y_pred
    squared_error = tf.square(error)
    
    underestimation_condition = tf.logical_and(tf.less(y_pred, y_true), tf.less_equal(error, threshold))
    extreme_condition = tf.greater(y_true, extreme_threshold)
    
    adjusted_error = tf.where(underestimation_condition, squared_error * 10, squared_error)
    adjusted_error = tf.where(extreme_condition, adjusted_error * 5, adjusted_error)
    
    return tf.reduce_mean(adjusted_error)
