import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['target'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app
st.title('Regression Model Evaluation App')

# Sidebar for user input
st.sidebar.header('Model Parameters')

# Select the feature for the X-axis
feature_x = st.sidebar.selectbox('Select Feature for X-axis:', X.columns)

# Select the target variable (y)
target_variable = st.sidebar.selectbox('Select Target Variable:', y.columns)

# Set plot title
plot_title = f'{target_variable} vs {feature_x}'

# Scatter plot with sliders
st.line_chart(X[feature_x])
st.write(f'## {plot_title}')

# Linear Regression model
model = LinearRegression()

# Train the Linear Regression model
model.fit(X_train[[feature_x]], y_train)

# Make predictions on the test set
y_pred = model.predict(X_test[[feature_x]])

# Display evaluation metrics
st.sidebar.subheader('Evaluation Metrics')

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
st.sidebar.write(f'Mean Squared Error: {mse:.4f}')

# R-squared (coefficient of determination)
r2 = r2_score(y_test, y_pred)
st.sidebar.write(f'R-squared (R2): {r2:.4f}')


