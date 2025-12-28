import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.head()

selected_features = ['MDVP:Fhi(Hz)', 'MDVP:Jitter(Abs)']
target = 'status'
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_for_analysis = scaler.fit_transform(df[selected_features])

from sklearn.model_selection import train_test_split
X_test, X_train, y_test, y_train = train_test_split(data_for_analysis, df[target], test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors= 1)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
import joblib

joblib.dump(model, 'adi.joblib')
