
# DECISION TREE
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

# Load dataset
dataset = pd.read_csv('PlayTennis.csv')

# Define features and target
X = dataset[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = dataset['PlayTennis']

# Encode categorical variables
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=100)

# Build decision tree model
dtree = DecisionTreeClassifier(criterion="entropy", random_state=100)
dtree.fit(X_train, y_train)

# Function to classify a new instance
def classify_new_instance(outlook, temperature, humidity, wind):
    instance = [[outlook, temperature, humidity, wind]]
    instance_encoded = encoder.transform(instance)
    prediction = dtree.predict(instance_encoded)
    return prediction[0]

# Predicting the class of a new instance
pred = classify_new_instance("Rain", "Mild", "High", "Strong")
print("Prediction:", pred)

# Evaluate model accuracy on test set
y_pred = dtree.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)