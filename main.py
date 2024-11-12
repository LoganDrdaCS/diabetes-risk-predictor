import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk

# Assigning the dataset to "data"
data = pd.read_csv('data/working_dataset_diabetes.csv')

# Removing any rows with missing values
data.dropna(inplace=True)

# Assigning features and target columns
X = data.drop(columns='Outcome')  # features
y = data['Outcome']  # target

# Splitting the data into training and testing sets using an 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initializing the logistic regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test data
y_predictions = model.predict(X_test)

# Calculating the model's accuracy
resulting_accuracy = accuracy_score(y_test, y_predictions)
print("Accuracy: ", f"{resulting_accuracy * 100:.2f}%")

# Detailing the performance metrics
print("\nClassification Report:\n", classification_report(y_test, y_predictions))

# Calculating medians of features
medians = np.median(X_train, axis=0)

# Function to make prediction based on user input
def make_prediction():
    try:
        # Retrieve values from the entry boxes, convert to float
        user_input = [
            float(entries[0].get()), float(entries[1].get()), float(entries[2].get()),
            float(entries[3].get()), float(entries[4].get()), float(entries[5].get())
        ]
        
        # Convert user input into a DataFrame with the same columns as X_train
        user_input_df = pd.DataFrame([user_input], columns=X.columns)
        
        # Make the prediction
        prediction = model.predict(user_input_df)
        prob = model.predict_proba(user_input_df)[0][1]  # Probability of the positive class (diabetes onset)

        # Display the result
        if prediction == 1:
            result_var.set("Likely")
        else:
            result_var.set("Unlikely")
        
        # Display probability as a percentage
        prob_var.set(f"{prob * 100:.2f}%")
    
    except ValueError:
        result_var.set("Invalid input")
        prob_var.set("")

# Function to reset input fields and clear results
def reset_inputs():
    # Reset all input fields to the median values
    for i, entry in enumerate(entries):
        entry.delete(0, tk.END)
        entry.insert(0, str(medians[i]))  # Set the median value as the default
    
    # Clear result and probability labels
    result_var.set("")
    prob_var.set("")

# Create the main window
root = tk.Tk()
root.title("Diabetes Onset Predictor")

# Display the model's overall accuracy
accuracy_var = tk.StringVar()
accuracy_var.set(f"Model Accuracy: {resulting_accuracy * 100:.2f}%")
accuracy_label = tk.Label(root, textvariable=accuracy_var, font=('Arial', 10, 'bold'))
accuracy_label.grid(row=0, column=0, columnspan=3, pady=10, sticky='n')  # Adjust pady for spacing and sticky='n' for top alignment

# Create labels and input fields for the 6 features
labels = ['Glucose', 'Blood Pressure', 'Skin Thickness', 'Serum Insulin', 'BMI', 'Age']
units = ['mg/dL', 'mm Hg', 'mm', 'mu U/ml', 'kg/m^2', 'years']
entries = []
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i+1, column=0)
    entry = tk.Entry(root, width=15)
    entry.insert(0, str(medians[i]))  # Set the default value to the median
    entry.grid(row=i+1, column=1)
    tk.Label(root, text=units[i]).grid(row=i+1, column=2)  # Place the unit label next to the entry
    entries.append(entry)

# Variables to display the result and probability
result_var = tk.StringVar()
prob_var = tk.StringVar()

# Labels for displaying the result
tk.Label(root, text="Diabetes Onset Prediction:", anchor='e').grid(row=7, column=0, sticky='e')
tk.Label(root, textvariable=result_var).grid(row=7, column=1)

tk.Label(root, text="Probability of Diabetes Onset:", anchor='e').grid(row=8, column=0, sticky='e')
tk.Label(root, textvariable=prob_var).grid(row=8, column=1)

# Submit button to make the prediction
submit_button = tk.Button(root, text="Submit", command=make_prediction)
submit_button.grid(row=10, columnspan=2)

# Reset button to clear inputs and results
reset_button = tk.Button(root, text="Reset", command=reset_inputs)
reset_button.grid(row=11, columnspan=2)

# Start the GUI loop
root.mainloop()