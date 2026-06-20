import os
import sys
import tkinter as tk
from tkinter import PhotoImage

from Hydromatic_Simulator.ui.ui_app import BinaryPredictorApp
from Hydromatic_Simulator.utils import mac_icon

def run_gui():
    root = tk.Tk()
    root.title("Hydromatic Simulator")
    
    icon_path = os.path.join(".", "Hydromatic_Simulator", "icon.png")

    if sys.platform == "darwin":
        mac_icon.set_dock_icon(icon_path)
        
    try:
        icon = PhotoImage(file=icon_path)
        root.iconphoto(True, icon)
    except tk.TclError:
        print(f"Notice: Icon is only loadable in Mac environment. Using default.")

    app = BinaryPredictorApp(root)
    root.mainloop()
