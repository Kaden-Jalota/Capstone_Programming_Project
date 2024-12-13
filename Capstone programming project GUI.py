# *********************************************
# Author: 3276045
# Assessment: Capstone programming project (GUI)
# Date: 13/12/24
# *********************************************

import tkinter as tk
from tkinter import messagebox
from capstone_programming_project import *
import warnings


# Code adapted from:
# https://www.geeksforgeeks.org/how-to-disable-python-warnings/
# Retrieved on 12/12/24

# Filters warning messages
warnings.filterwarnings('ignore')

# Code adapted from:
# https://uclearn.canberra.edu.au/courses/17088/pages/week-12-programming-project-final-stage
# Retrieved on 11/12/24

# Function which calls the final model to make a prediction on new information
def pred():
    try:

        # Gets input values then calls model to make and display a prediction
        POD = float(entryPOD.get())
        SystemLoadEA = float(entrySystemLoadEA.get())
        SystemLoadEP2 = float(entrySystemLoadEP2.get())
        SMPEA = float(entrySMPEA.get())
        prediction = finalModel.predict([[POD, SystemLoadEA, SystemLoadEP2, SMPEA]])
        messagebox.showinfo("Prediction", f"The predicted price of electricity is: {prediction[0]:.2f}")

    except Exception as error:
        # Error message when failing to get inputs, calling model or displaying a predection
        messagebox.showerror("Error", f"An error occurred: {str(error)}")


# Main window
root = tk.Tk()
root.title("Electricity Price Predictor")
root.geometry("400x200")

# Tkinter labels and entry boxes
tk.Label(root, text="The period of day as a number from 1-48:").place(x=10, y=10)
entryPOD = tk.Entry(root)
entryPOD.place(x=250, y=10)

tk.Label(root, text="The forecasted system load:").place(x=10, y=50)
entrySystemLoadEA = tk.Entry(root)
entrySystemLoadEA.place(x=250, y=50)

tk.Label(root, text="The actual system load:").place(x=10, y=90)
entrySystemLoadEP2 = tk.Entry(root)
entrySystemLoadEP2.place(x=250, y=90)

tk.Label(root, text="The forecasted price of electricity:").place(x=10, y=130)
entrySMPEA = tk.Entry(root)
entrySMPEA.place(x=250, y=130)


btnPred = tk.Button(root, text="Make prediction", command=pred)
btnPred.place(x=250, y=170)

# Main loop
root.mainloop()