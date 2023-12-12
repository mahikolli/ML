import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_excel(r'C:\Users\Kartthik\Desktop\Personal\ML\Lab04\hello.xlsx')

df = pd.DataFrame(data)

X = df[['Unit price', 'Quantity', 'Tax 5%', 'Total', 'Rating']]
y = df['gross income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)
dt_predictions = dt_regressor.predict(X_test)

dt_mse = mean_squared_error(y_test, dt_predictions)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

rf_predictions = rf_regressor.predict(X_test)

# Step 7: Evaluate Random Forest Regressor
rf_mse = mean_squared_error(y_test, rf_predictions)

# Step 8: Compare Performance Metrics
print(f"Decision Tree MSE: {dt_mse}")
print(f"Random Forest MSE: {rf_mse}")