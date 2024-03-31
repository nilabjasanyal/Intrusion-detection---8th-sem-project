import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the CSV file into a DataFrame
data = pd.read_csv("TestbedSatJun12Flows.csv")

# Preprocessing
# Drop unnecessary columns
data = data.drop(['generated', 'startDateTime', 'stopDateTime'], axis=1)

# Encode categorical labels ('Normal' and 'Attack') to numerical values
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

# Drop columns with non-numeric values
data = data.drop(['appName', 'sourcePayloadAsBase64', 'sourcePayloadAsUTF', 
                  'destinationPayloadAsBase64', 'destinationPayloadAsUTF', 
                  'direction', 'sourceTCPFlagsDescription', 'destinationTCPFlagsDescription', 
                  'source', 'protocolName', 'destination'], axis=1)

# Split features and labels
X = data.drop('Label', axis=1)
y = data['Label']

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the K-Nearest Neighbors model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Predict labels for the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
