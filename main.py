import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Download the data (assuming this is already done or will be outside this script)
# !wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
# !wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
# import lab_setup_do_not_edit

# 2. Load the dataset
df = pd.read_csv('parkinsons.csv')

# 3. Select features (re-loading and dropping 'name' for correlation analysis)
df_numeric = df.drop(columns=['name'])
correlations = df_numeric.corr()['status'].sort_values(ascending=False)
print("Correlation of features with 'status':")
display(correlations)

# Based on the correlation analysis, using 'PPE' and 'spread1'
cols = ['PPE', 'spread1', 'status']
df = df[cols]
print("DataFrame head with selected features:")
display(df.head())

# 4. Split the data
X = df[['PPE', 'spread1']]
y = df['status']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
print("Scaled X_train head:")
display(X_train_scaled.head())
print("Scaled X_val head:")
display(X_val_scaled.head())

# 6. Choose a model
model = LogisticRegression(random_state=42)

# 7. Test the accuracy
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_pred)
print(f"Model Accuracy after selecting new features and scaling: {accuracy:.4f}")

# 8. Save the model
joblib.dump(model, 'my_model.joblib')
print("Model saved as 'my_model.joblib'")
