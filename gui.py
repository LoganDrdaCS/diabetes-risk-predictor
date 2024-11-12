import tkinter as tk
from model import make_prediction, get_model_accuracy

# Create the main window
root = tk.Tk()
root.title("Diabetes Onset Predictor")

# Get the model's accuracy
resulting_accuracy = get_model_accuracy()

# Display the model's overall accuracy
accuracy_var = tk.StringVar()
accuracy_var.set(f"Model Accuracy: {resulting_accuracy * 100:.2f}%")
accuracy_label = tk.Label(root, textvariable=accuracy_var, font=('Arial', 10, 'bold'))
accuracy_label.grid(row=0, column=0, columnspan=3, pady=10, sticky='n')

# Labels and entry fields for features
labels = ['Glucose', 'Blood Pressure', 'Skin Thickness', 'Serum Insulin', 'BMI', 'Age']
units = ['mg/dL', 'mm Hg', 'mm', 'mu U/ml', 'kg/m^2', 'years']
entries = []
medians = [120, 80, 20, 100, 25, 50]  # Example median values, adjust based on your data

for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i+1, column=0)
    entry = tk.Entry(root, width=15)
    entry.insert(0, str(medians[i]))  # Set default to median
    entry.grid(row=i+1, column=1)
    tk.Label(root, text=units[i]).grid(row=i+1, column=2)  # Units next to the entry
    entries.append(entry)

# Variables for displaying result and probability
result_var = tk.StringVar()
prob_var = tk.StringVar()

# Labels for displaying the result
tk.Label(root, text="Diabetes Onset Prediction:", anchor='e').grid(row=7, column=0, sticky='e')
tk.Label(root, textvariable=result_var).grid(row=7, column=1)

tk.Label(root, text="Probability of Diabetes Onset:", anchor='e').grid(row=8, column=0, sticky='e')
tk.Label(root, textvariable=prob_var).grid(row=8, column=1)

# Submit button to make the prediction
def make_prediction_gui():
    try:
        # Retrieve values from the entry boxes and convert to float
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

submit_button = tk.Button(root, text="Submit", command=make_prediction_gui)
submit_button.grid(row=10, columnspan=2)

# Reset button to clear inputs and results
def reset_inputs():
    for i, entry in enumerate(entries):
        entry.delete(0, tk.END)
        entry.insert(0, str(medians[i]))
    
    result_var.set("")
    prob_var.set("")

reset_button = tk.Button(root, text="Reset", command=reset_inputs)
reset_button.grid(row=11, columnspan=2)
