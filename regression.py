import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.title('Regression Model Evaluation App')

st.sidebar.header('Model Parameters')


feature_x = st.sidebar.selectbox('Select Feature for X-axis:', X.columns)


target_variable = st.sidebar.selectbox('Select Target Variable:', y.columns)


plot_title = f'{target_variable} vs {feature_x}'


st.line_chart(X[feature_x])
st.write(f'## {plot_title}')


model = LinearRegression()

model.fit(X_train[[feature_x]], y_train)

y_pred = model.predict(X_test[[feature_x]])


st.sidebar.subheader('Evaluation Metrics')

mse = mean_squared_error(y_test, y_pred)
st.sidebar.write(f'Mean Squared Error: {mse:.4f}')


r2 = r2_score(y_test, y_pred)
st.sidebar.write(f'R-squared (R2): {r2:.4f}')


