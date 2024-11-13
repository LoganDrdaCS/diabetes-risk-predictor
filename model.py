import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Assign the dataset to "data"
data = pd.read_csv('data/working_dataset_diabetes.csv')

# Remove any rows with missing values to clean the data
data.dropna(inplace=True)

# Assign features and target columns
X = data.drop(columns='Outcome')  # features
y = data['Outcome']  # target

# Split the data into training and testing sets using an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initialize the logistic regression model with L2 regularization
model = LogisticRegression(max_iter=1000, penalty='l2', solver='liblinear')

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_predictions = model.predict(X_test)

# Calculate the model's accuracy
resulting_accuracy = accuracy_score(y_test, y_predictions)

# Detail the performance metrics
print("\nClassification Report:\n", classification_report(y_test, y_predictions))

# Calculate medians of features
medians = np.median(X_train, axis=0)

# Prediction function
def make_prediction(user_input):
    try:
        # Convert user input into a DataFrame with the same columns as X_train
        user_input_df = pd.DataFrame([user_input], columns=X.columns)
        
        prediction = model.predict(user_input_df)
        prob = model.predict_proba(user_input_df)[0][1]

        return prediction, prob
    
    except ValueError:
        return "Invalid input", None

# Getter function to return the accuracy
def get_model_accuracy():
    return resulting_accuracy

# Getter function to return the medians
def get_medians():
    return medians