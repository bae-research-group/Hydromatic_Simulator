import tkinter as tk
from tkinter import PhotoImage
import sys

from Hydromatic_Simulator.ui.ui_app import BinaryPredictorApp
from Hydromatic_Simulator.utils import mac_icon

def run_gui():
    root = tk.Tk()
    root.title("Hydromatic Simulator")
    if sys.platform == "darwin":
        mac_icon.set_dock_icon("./Hydromatic_Simulator/icon.png")
        
    icon = PhotoImage(file='./Hydromatic_Simulator/icon.png')
    root.iconphoto(True, icon)
    app = BinaryPredictorApp(root)
    root.mainloop()
