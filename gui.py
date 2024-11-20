import tkinter as tk
from model import make_prediction, get_model_accuracy, get_medians, data, y_test, y_predictions, model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Create the main GUI window
root = tk.Tk()
root.title("Diabetes Onset Predictor")

# Variables for model accuracy, results, and probability
accuracy_var = tk.StringVar()
result_var = tk.StringVar()
prob_var = tk.StringVar()

# Get the model's overall accuracy
resulting_accuracy = get_model_accuracy()

# Display the model's overall accuracy
accuracy_var.set(f"Model Accuracy: {resulting_accuracy * 100:.2f}%")
accuracy_label = tk.Label(root, textvariable=accuracy_var, font=('Arial', 10, 'bold'))
accuracy_label.grid(row=0, column=0, columnspan=3, pady=10, sticky='n')

# Labels and entry fields for features
labels = ['Glucose', 'Blood Pressure', 'Skin Thickness', 'Serum Insulin', 'BMI', 'Age']
units = ['mg/dL', 'mm Hg', 'mm', 'mu U/ml', 'kg/m^2', 'years']
entries = []
medians = get_medians()
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i+1, column=0)
    entry = tk.Entry(root, width=15)
    entry.insert(0, str(medians[i]))
    entry.grid(row=i+1, column=1)
    tk.Label(root, text=units[i]).grid(row=i+1, column=2)  # Unit labels
    entries.append(entry)

# Labels for displaying the prediction result and probability
tk.Label(root, text="Diabetes Onset Prediction:", anchor='e').grid(row=7, column=0, sticky='e')
tk.Label(root, textvariable=result_var).grid(row=7, column=1)
tk.Label(root, text="Probability of Diabetes Onset:", anchor='e').grid(row=8, column=0, sticky='e')
tk.Label(root, textvariable=prob_var).grid(row=8, column=1)

# Prediction function
def make_prediction_gui():
    try:
        # Convert user input to float values
        user_input = [
            float(entries[0].get()), float(entries[1].get()), float(entries[2].get()),
            float(entries[3].get()), float(entries[4].get()), float(entries[5].get())
        ]
        
        # Call make_prediction from model.py
        prediction, prob = make_prediction(user_input)

        # Update the result and probability labels
        if prediction == 1:
            result_var.set("Likely")
        else:
            result_var.set("Unlikely")
        
        if prob is not None:
            prob_var.set(f"{prob * 100:.2f}%")
        else:
            prob_var.set("")
    
    except ValueError:
        result_var.set("Invalid input")
        prob_var.set("")

# Submit button
submit_button = tk.Button(root, text="Submit", command=make_prediction_gui, width=20, bg="lightblue", fg="black")
submit_button.grid(row=10, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

# Reset button
def reset_inputs():
    for i, entry in enumerate(entries):
        entry.delete(0, tk.END)
        entry.insert(0, str(medians[i])) # Reset to median default values
    
    result_var.set("")
    prob_var.set("")

reset_button = tk.Button(root, text="Reset", command=reset_inputs, width=20, bg="lightcoral", fg="black")
reset_button.grid(row=10, column=2, columnspan=2, sticky="ew", padx=5, pady=5)

# Visualization functions for correlation matrix, glucose histogram, and confusion matrix
def show_correlation_matrix():
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def show_glucose_histogram():
    plt.figure(figsize=(8, 6))
    sns.histplot(data['Glucose'], bins=20, kde=True)
    plt.title("Distribution of Glucose Levels")
    plt.xlabel("Glucose (mg/dL)")
    plt.ylabel("Frequency")
    plt.show()

def show_confusion_matrix():
    cm = confusion_matrix(y_test, y_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix for Diabetes Prediction Model")
    plt.show()

# Buttons for opening each visualization
correlation_button = tk.Button(root, text="Correlation Matrix", command=show_correlation_matrix)
correlation_button.grid(row=11, column=0, sticky="ew", padx=5, pady=5)

glucose_histogram_button = tk.Button(root, text="Glucose Histogram", command=show_glucose_histogram)
glucose_histogram_button.grid(row=11, column=1, sticky="ew", padx=5, pady=5)

confusion_matrix_button = tk.Button(root, text="Confusion Matrix", command=show_confusion_matrix)
confusion_matrix_button.grid(row=11, column=2, sticky="ew", padx=5, pady=5)

# Configure the grid
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)