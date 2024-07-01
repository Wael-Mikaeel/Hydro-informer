# utils.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error


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


def calculate_performance_metrics(y_test, y_pred):
    """
    Calculate various performance metrics for the model and return them as a DataFrame.

    Args:
    y_test (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.

    Returns:
    pd.DataFrame: DataFrame containing the performance metrics.
    """
    
    # Calculate the mean squared error
    mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
    print(f"Mean Squared Error: {mse}")

    # Calculate the root mean squared error
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")

    # Calculate the mean absolute error
    mae = np.mean(np.abs(y_test.flatten() - y_pred.flatten()))
    print(f"Mean Absolute Error: {mae}")

    # Calculate the mean absolute percentage error
    mape = np.mean(np.abs((y_test.flatten() - y_pred.flatten()) / y_test.flatten())) * 100
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    # Calculate the coefficient of determination (R^2)
    r2 = r2_score(y_test.flatten(), y_pred.flatten())
    print(f"Coefficient of Determination (R^2): {r2:.4f}")

    # Calculate the mean squared logarithmic error
    msle = mean_squared_log_error(y_test.flatten(), y_pred.flatten())
    print(f"Mean Squared Logarithmic Error: {msle:.4f}")

    # Calculate the root mean squared logarithmic error
    rmsle = np.sqrt(msle)
    print(f"Root Mean Squared Logarithmic Error: {rmsle:.4f}")

    # Calculate the symmetric mean absolute percentage error
    smape = 100 / len(y_test) * np.sum(2 * np.abs(y_pred.flatten() - y_test.flatten()) / (np.abs(y_test.flatten()) + np.abs(y_pred.flatten())))
    print(f"Symmetric Mean Absolute Percentage Error: {smape:.2f}%")
    
    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        "Metric": ["Mean Squared Error", "Root Mean Squared Error", "Mean Absolute Error", 
                   "Mean Absolute Percentage Error", "Coefficient of Determination (R^2)", 
                   "Mean Squared Logarithmic Error", "Root Mean Squared Logarithmic Error", 
                   "Symmetric Mean Absolute Percentage Error"],
        "Value": [mse, rmse, mae, mape, r2, msle, rmsle, smape]
    })
    
    return metrics_df

# utils.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error

def plot_actual_vs_predicted(actual_values, predicted_values, min_value=130, file_name=None):
    plt.figure(figsize=(10, 10))

    # Scatter plot of actual vs. predicted values
    plt.scatter(actual_values, predicted_values, alpha=0.6, edgecolors='k', linewidths=0.5, color='dodgerblue', s=40, label='Predicted vs Actual')

    # Plot the line y=x for reference
    plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'r--', lw=2, label='Perfect Prediction Line')

    # Shaded triangle regions starting from min_value
    max_value = max(max(actual_values), max(predicted_values))
    triangle_points = np.array([[min_value, min_value], [max_value, min_value], [max_value, max_value]])
    plt.fill(triangle_points[:, 0], triangle_points[:, 1], color='orange', alpha=0.3, label='Critical Underestimation')

    # Create additional shaded regions within the triangle
    mid_value = min_value + (max_value - min_value) / 2
    triangle_points_2 = np.array([[mid_value - 50, min_value], [max_value, min_value], [max_value, mid_value + 50]])
    plt.fill(triangle_points_2[:, 0], triangle_points_2[:, 1], color='yellow', alpha=0.3, label='Dangerous Underestimation')

    # Set axis limits to start from min_value
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)

    # Set axis labels
    plt.xlabel('Actual Water Levels (cm)', fontsize=13)
    plt.ylabel('Predicted Water Levels (cm)', fontsize=13)

    # Set title
    plt.title('Predicted vs. Actual Water Levels with Danger Zones', fontsize=15, fontweight='bold')

    # Set grid
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Add a legend
    plt.legend(fontsize=12)

    # Save the figure with high resolution
    if file_name:
        plt.savefig(file_name, dpi=300)

    # Show the plot
    plt.show()

def plot_predictions_with_confidence_intervals(y_test, y_pred, confidence_multiplier=1, file_name=None):
    y_pred_std = np.std(y_pred)
    confidence_interval = confidence_multiplier * y_pred_std
    lower_bound = y_pred.flatten() - confidence_interval
    upper_bound = y_pred.flatten() + confidence_interval

    plt.figure(figsize=(14, 7))

    # Plot actual and predicted values
    plt.plot(y_test.flatten(), label='Actual', color='blue')
    plt.plot(y_pred.flatten(), label='Predicted', color='orange', linestyle='dashed')

    # Fill between for confidence interval
    plt.fill_between(range(len(y_pred.flatten())), lower_bound, upper_bound, color='lightblue', alpha=0.4, label='95% Confidence Interval')

    # Add labels and title with larger font sizes
    plt.xlabel('Hours', fontsize=14, fontweight='bold')
    plt.ylabel('Water level (cm)', fontsize=14, fontweight='bold')
    plt.title('Predictions with 95% Confidence Intervals', fontsize=16, fontweight='bold')

    # Adding grid for better readability
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Adding legend
    plt.legend(fontsize=12)

    # Save the figure with high resolution
    if file_name:
        plt.savefig(file_name, dpi=300)

    # Show the plot
    plt.show()

def plot_peak_analysis(actual_data, predicted_data, start_idx, end_idx, file_name=None):
    # Extract the relevant portions of the data
    actual_data = actual_data[start_idx:end_idx]
    predicted_data = predicted_data[start_idx:end_idx]

    # Find the index of the actual peak
    actual_peak_index = np.argmax(actual_data)

    # Find the highest predicted peak before the actual peak
    predicted_peak_before_actual = np.argmax(predicted_data[:actual_peak_index + 1])

    # Get the peak values
    actual_peak_value = actual_data[actual_peak_index]
    predicted_peak_value_before_actual = predicted_data[predicted_peak_before_actual]

    plt.figure(figsize=(14, 8))

    # Plot the actual and predicted values
    plt.plot(actual_data, label='Actual', linewidth=2, color='blue')
    plt.plot(predicted_data, label='Predicted', linewidth=2, linestyle='--', color='orange')

    # Add labels and title
    plt.xlabel('Hours', fontsize=14)
    plt.ylabel('Water level (cm)', fontsize=14)
    plt.title('Actual vs Predicted Values', fontsize=16, fontweight='bold')

    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add a legend
    plt.legend(fontsize=12)

    # Annotate the peaks
    plt.annotate('Actual Peak', xy=(actual_peak_index, actual_peak_value), xytext=(actual_peak_index + 5, actual_peak_value + 5),
                 arrowprops=dict(facecolor='blue', arrowstyle='->'),
                 fontsize=12, color='blue')

    plt.annotate('Predicted Peak Before Actual', xy=(predicted_peak_before_actual, predicted_peak_value_before_actual), 
                 xytext=(predicted_peak_before_actual - 20, predicted_peak_value_before_actual),
                 arrowprops=dict(facecolor='orange', arrowstyle='->'),
                 fontsize=12, color='orange')

    # Add vertical lines to show peak differences
    plt.axvline(x=actual_peak_index, color='blue', linestyle=':', linewidth=1)
    plt.axvline(x=predicted_peak_before_actual, color='orange', linestyle=':', linewidth=1)

    # Show the difference in peak levels
    plt.text(predicted_peak_before_actual, predicted_peak_value_before_actual - 30, 
             f'Δ Peak Value = {actual_peak_value - predicted_peak_value_before_actual:.2f} cm', 
             fontsize=12, color='black', ha='center')

    # Show the difference in timing
    time_difference = actual_peak_index - predicted_peak_before_actual
    plt.text((actual_peak_index + predicted_peak_before_actual) / 2, 
             max(actual_peak_value, predicted_peak_value_before_actual) - 120, 
             f'Δ Time = {time_difference} hours', 
             fontsize=12, color='black', ha='center')

    # Save the figure with high resolution
    if file_name:
        plt.savefig(file_name, dpi=300)

    # Show the plot
    plt.show()

def plot_overall_with_peaks(y_test, y_pred, peaks, file_name=None):
    plt.figure(figsize=(20, 10))  # Adjusting the size for a larger and wider graph

    # Plot the actual and predicted values with thicker lines
    plt.plot(y_test.flatten(), label='Actual', color='blue', linewidth=2)
    plt.plot(y_pred.flatten(), label='Predicted', color='orange', linestyle='dashed', linewidth=2)

    # Highlight the peaks with arrows and shaded areas
    for peak_name, (start_idx, end_idx) in peaks.items():
        peak_idx = np.argmax(y_test.flatten()[start_idx:end_idx]) + start_idx
        peak_value = y_test.flatten()[peak_idx]
        plt.axvspan(start_idx, end_idx, color='red', alpha=0.1)
        plt.annotate(peak_name, xy=(peak_idx, peak_value), xytext=(peak_idx, peak_value - 40),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                     fontsize=16, fontweight='bold', color='black')

    # Add labels and title with larger font sizes
    plt.xlabel('Hours', fontsize=20, fontweight='bold')
    plt.ylabel('Water level (cm)', fontsize=20, fontweight='bold')
    plt.title('Actual vs Predicted Values', fontsize=24, fontweight='bold')

    # Add a legend with larger font size
    plt.legend(fontsize=16, loc='upper left')

    # Adjust y-axis limits to zoom in on the relevant range
    plt.ylim([100, 400])

    # Make the x and y tick labels larger
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')

    # Add grid for better readability
    plt.grid(True, linestyle='--', linewidth=0.5)

    # Save the figure with high resolution
    if file_name:
        plt.savefig(file_name, dpi=300)

    # Display the plot
    plt.show()
