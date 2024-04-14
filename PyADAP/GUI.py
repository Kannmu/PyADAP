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

from cv2 import sort

import PyADAP.File as file


class Interface:
    """
    Interface class to create a graphical user interface for PyADAP.
    """

    def __init__(self, title="PyADAP Interface", width=600, height=400) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")

        # Initialize variables lists
        self.IndependentVars = []
        self.DependentVars = []
        self.UnassignedVars = []
        self.Clean = False  # Initialize clean variable
        self.Alpha = 0.05

    def ParametersSettingPage(self, Vars: list):
        self.UnassignedVars = Vars.copy()

        # Function to move selected variable to independent list
        def move_to_independent():
            selected_var = vars_listbox.get(tk.ACTIVE)
            if selected_var:
                self.UnassignedVars.remove(selected_var)
                self.IndependentVars.append(selected_var)
                update_lists()

        # Function to move selected variable to dependent list
        def move_to_dependent():
            selected_var = vars_listbox.get(tk.ACTIVE)
            if selected_var:
                self.UnassignedVars.remove(selected_var)
                self.DependentVars.append(selected_var)
                update_lists()

        # Function to reset lists to initial state
        def reset_lists():
            self.IndependentVars.clear()
            self.DependentVars.clear()
            self.UnassignedVars = Vars.copy()
            update_lists()

        # Function to update listboxes
        def update_lists():
            independent_listbox.delete(0, tk.END)
            dependent_listbox.delete(0, tk.END)
            vars_listbox.delete(0, tk.END)
            for item in self.IndependentVars:
                independent_listbox.insert(tk.END, item)
            for item in self.DependentVars:
                dependent_listbox.insert(tk.END, item)
            for item in self.UnassignedVars:
                vars_listbox.insert(tk.END, item)

        # Function to close the GUI
        def close_gui():
            self.root.destroy()

        # Function to toggle clean variable
        def toggle_clean():
            self.Clean = not self.Clean

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
            self.Alpha = float(value)

        # Create dropdown menu for alpha selection
        alpha_label = tk.Label(frame, text="Select Alpha:")
        alpha_label.grid(row=3, column=0)
        alpha_options = ["0.05", "0.01", "0.001"]
        self.Alpha = tk.StringVar()
        self.Alpha.set(alpha_options[0])  # default value
        alpha_dropdown = tk.OptionMenu(
            frame, self.Alpha, *alpha_options, command=set_alpha
        )
        alpha_dropdown.grid(row=3, column=1)

        # Initial update of lists
        update_lists()
        self.root.mainloop()

        return self.IndependentVars, self.DependentVars, self.Clean, self.Alpha
