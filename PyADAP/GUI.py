"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

GUI

Author: Kannmu
Date: 2024/4/2
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""

import tkinter as tk

class Interface:
    """
    Interface class to create a graphical user interface for PyADAP.
    """

    def __init__(self, title="PyADAP Interface", width=600, height=400) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")

        # Initialize variables lists
        self.independentVars = []
        self.dependentVars = []
        self.unassignedVars = []

        self.alphaBox = tk.DoubleVar()   # Initialize alpha variable
        self.apiKeyBox = tk.StringVar()  # Initialize apiKey variable

        

        self.isClean = False  # Initialize clean variable
        self.enableWriting = False  # Initialize EnableWriting variable
        self.enableBoxCox = False

    def ParametersSettingPage(self, Vars: list):
        self.unassignedVars = Vars.copy()

        # Function to move selected variable to independent list
        def move_to_independent():
            selected_var = vars_listbox.get(tk.ACTIVE)
            if selected_var:
                self.unassignedVars.remove(selected_var)
                self.independentVars.append(selected_var)
                update_lists()

        # Function to move selected variable to dependent list
        def move_to_dependent():
            selected_var = vars_listbox.get(tk.ACTIVE)
            if selected_var:
                self.unassignedVars.remove(selected_var)
                self.dependentVars.append(selected_var)
                update_lists()

        # Function to reset lists to initial state
        def reset_lists():
            self.independentVars.clear()
            self.dependentVars.clear()
            self.unassignedVars = Vars.copy()
            update_lists()

        # Function to update listboxes
        def update_lists():
            independent_listbox.delete(0, tk.END)
            dependent_listbox.delete(0, tk.END)
            vars_listbox.delete(0, tk.END)
            for item in self.independentVars:
                independent_listbox.insert(tk.END, item)
            for item in self.dependentVars:
                dependent_listbox.insert(tk.END, item)
            for item in self.unassignedVars:
                vars_listbox.insert(tk.END, item)

        # Function to close the GUI
        def close_gui():
            self.root.destroy()

        # Function to toggle clean variable
        def toggle_clean():
            self.isClean = not self.isClean

        # Function to toggle EnableWriting variable
        def toggle_enable_writing():
            self.enableWriting = not self.enableWriting

        # Function to toggle EnableWriting variable
        def toggle_enable_boxcox():
            self.enableBoxCox = not self.enableBoxCox

        # Create UI elements
        frame = tk.Frame(self.root)
        frame.pack(pady=10)
        vars_label = tk.Label(frame, text="Variables")
        vars_label.grid(row=0, column=0)
        vars_listbox = tk.Listbox(frame)
        vars_listbox.grid(row=1, column=0)
        for var in Vars:
            vars_listbox.insert(tk.END, var)

        buttons_frame = tk.Frame(frame)
        buttons_frame.grid(row=1, column=1, padx=10)

        move_to_independent_button = tk.Button(
            buttons_frame, text="Independent", command=move_to_independent
        )
        move_to_independent_button.pack()

        move_to_dependent_button = tk.Button(
            buttons_frame, text="Dependent", command=move_to_dependent
        )
        move_to_dependent_button.pack()

        reset_button = tk.Button(buttons_frame, text="Reset", command=reset_lists)
        reset_button.pack()

        clean_checkbox = tk.Checkbutton(frame, text="Clean Data", command=toggle_clean)
        clean_checkbox.grid(row=2, column=0)

        enable_writing_checkbox = tk.Checkbutton(frame, text="Enable Writing", command=toggle_enable_writing)
        enable_writing_checkbox.grid(row=2, column=1)
        
        enable_writing_checkbox = tk.Checkbutton(frame, text="Enable Box-Cox", command=toggle_enable_boxcox)
        enable_writing_checkbox.grid(row=2, column=2)

        independent_label = tk.Label(frame, text="Independent Variables")
        independent_label.grid(row=0, column=2)
        independent_listbox = tk.Listbox(frame)
        independent_listbox.grid(row=1, column=2)

        dependent_label = tk.Label(frame, text="Dependent Variables")
        dependent_label.grid(row=0, column=3)
        dependent_listbox = tk.Listbox(frame)
        dependent_listbox.grid(row=1, column=3)

        finish_button = tk.Button(frame, text="Finish", command=close_gui)
        finish_button.grid(row=3, column=3, columnspan=2, pady=10)

        # Function to set alpha value
        def set_alpha(value):
            self.alphaBox = value

        # Create dropdown menu for alpha selection
        alpha_label = tk.Label(frame, text="Select Alpha:")
        alpha_label.grid(row=3, column=0)
        alpha_options = [0.05, 0.01, 0.001]
        
        self.alphaBox.set(alpha_options[0])  # default value
        alpha_dropdown = tk.OptionMenu(
            frame, self.alphaBox, *alpha_options, command=set_alpha
        )
        alpha_dropdown.grid(row=3, column=1)

        # Create apiKey input field
        apiKey_label = tk.Label(frame, text="API Key:")
        apiKey_label.grid(row=4, column=0)
        apiKey_entry = tk.Entry(frame, textvariable=self.apiKeyBox)
        apiKey_entry.grid(row=4, column=1)

        # Initial update of lists
        update_lists()
        self.root.mainloop()

        self.alpha = self.alphaBox.get()
        self.apiKey = self.apiKeyBox.get()
