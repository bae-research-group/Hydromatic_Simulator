import os
import threading
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow.keras as keras

import tkinter as tk
from tkinter import ttk, messagebox

from Hydromatic_Simulator.model.main import load_trained_model
from Hydromatic_Simulator.model.inference import run_inference
from Hydromatic_Simulator.utils.visualization import save_sequence_plot
from Hydromatic_Simulator.utils.config import config
from Hydromatic_Simulator.utils import mac_icon

S = config['structure-dim']

class BitPatternEditor(tk.Toplevel):
    def __init__(self, root, on_submit):
        super().__init__(root)
        
        self.title("Configuring double/triple AHC structures")
        self.on_submit = on_submit 

        self.grid_size = 10
        self.num_bits = S
        
        self.rect_xoffset = 10
        
        self.canvas_width = self.num_bits * self.grid_size + self.rect_xoffset
        self.canvas_height = 200

        self.patterns = {
            1: '10101010101',  
            2: '1100110011',
            3: '111111',       
        }
        self.colors = {
            0: "light gray",
            1: "deep sky blue",
            2: "deep sky blue",
            3: "deep sky blue"
        }
        self.AHC_labels = {
            1: 'AHC-1',
            2: 'AHC-2',
            3: 'AHC-6'
        }

        self.block_values = [0] * self.num_bits

        self.canvas = tk.Canvas(self, width=self.canvas_width + 150, height=self.canvas_height, bg="white")
        self.canvas.pack()

        
        self.canvas.create_rectangle(self.rect_xoffset, 50, self.canvas_width, 80,
                                     fill="lightgray", outline="gray", tags="strip_bg")
        self.canvas.create_text((0 + self.canvas_width) / 2 + 80, 30,
                            text='Hydromatic Actuator', fill="black", font=("Arial", 20, "bold"))
        
        self.column_label = self.canvas.create_text(10, 85, text="", anchor="nw", font=("Arial", 12), fill="black")

        self.message_label = tk.Label(self, text="", fg="red")
        self.message_label.pack()
        
        self.sample_blocks = {}
        for i, (val, pattern) in enumerate(self.patterns.items(), start=1):
            x1 = self.canvas_width + 30
            y1 = 50 + (i - 1) * 55
            bit_length = len(pattern)
            x2 = x1 + bit_length * self.grid_size
            y2 = y1 + 25

            rect_set = []
            for j, bit in enumerate(pattern):
                cx = x1 + j*self.grid_size
                rect = self.canvas.create_rectangle(cx, y1, cx + self.grid_size, y2,
                                                 fill=self.colors[val] if bit == '1' else "gray",
                                                 outline="", tags=("sample", f"sample{val}"))
                rect_set.append(rect)
            
            self.canvas.create_text((x1 + x2) / 2, y1 - 12,
                            text=self.AHC_labels[i], fill="black", font=("Arial", 15, "bold"))
            
            self.sample_blocks[val] = (val, pattern, rect_set)


        tk.Button(self, text="Print Code", command=self.print_code).pack()

        self.reset_btn = tk.Button(self, text="Reset", command=self.reset_all)
        self.reset_btn.pack(pady=2)
        self.reset_btn.place(x=10, y=10)

        tk.Button(self, text="Submit Code", command=self.submit_code).pack(pady=5)

        self.canvas.tag_bind("sample", "<ButtonPress-1>", self.start_drag)
        self.canvas.tag_bind("sample", "<B1-Motion>", self.do_drag)
        self.canvas.tag_bind("sample", "<ButtonRelease-1>", self.end_drag)

        self.drag_data = {"item": None, "value": None, "pattern": None, "x": None, "y": None}
        self.placed_blocks = []
        self.visual_blocks = []

        self.draw_x_scale()

        self.centroid_circle = None
        self.centroid_line = None

    def draw_x_scale(self):
        step = self.grid_size
        x_offset = self.rect_xoffset
        y_offset = 120
        self.canvas.create_line(x_offset, y_offset - 7, self.canvas_width + 15,
                                y_offset - 7, fill='black', arrow=tk.LAST)
        self.canvas.create_text(
            self.canvas_width + 10, y_offset + 8, 
            text="x", fill="black", font=("Arial",20)
        )
        for x in range(x_offset, self.canvas_width + step, step):
            self.canvas.create_line(x, y_offset, x, y_offset - 14, fill="black") 
            if (x - x_offset) % (step * 10) == 0:
                self.canvas.create_text(x, y_offset + 10, text=str((x - x_offset)//step), font=("Arial", 15))
        
    def submit_code(self):
        if len(self.placed_blocks) <= 1:
            self.message_label.config(text="Error: Double/triple structure requires more input.")
            return False
        code_str = ''.join(str(b) for b in self.block_values)
        
        self.on_submit(code_str)
        self.destroy()
        
    def reset_all(self):
        self.block_values = [0] * self.num_bits
    
        self.block_counts = {1: 0, 2: 0, 3: 0}
        self.placed_blocks = []
        
        if hasattr(self, 'visual_blocks'):
            for canvas_ids, _, _, _ in self.visual_blocks:
                for cid in canvas_ids:
                    self.canvas.delete(cid)
            self.visual_blocks = []
        
        self.message_label.config(text="")
        self.canvas.itemconfig(self.column_label, text="")
        
        self.canvas.delete("strip_bg")
        self.canvas.create_rectangle(self.rect_xoffset, 50, self.canvas_width, 80,
                                     fill="lightgray", outline="gray", tags="strip_bg")
        
        messagebox.showinfo("Reset", "All AHC structures have been cleared and the strip is reset.")
    
    def start_drag(self, event):
        self.canvas.itemconfig("strip_bg", fill="lightgray")
        
        clicked = self.canvas.find_closest(event.x, event.y)[0]

        val = None
        pattern = None
        rect_set = None
        for v, (_, pat, rects) in self.sample_blocks.items():
            if clicked in rects:
                val = v
                pattern = pat
                rect_set = rects
                break

        if val is None:
            return

        self.drag_data["item"] = []
        for src_id in rect_set:
            coords = self.canvas.coords(src_id)
            fill = self.canvas.itemcget(src_id, "fill")
            drag_rect = self.canvas.create_rectangle(*coords, fill=fill, outline="")
            self.drag_data["item"].append(drag_rect)

        self.drag_data["value"] = val
        self.drag_data["pattern"] = pattern
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.message_label.config(text="")

    def _bbox_for_id_list(self, id_list):
        xs = []
        ys = []
        xs2 = []
        ys2 = []
        for cid in id_list:
            coords = self.canvas.coords(cid)
            if not coords:
                continue
            
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            xs.append(x1); ys.append(y1); xs2.append(x2); ys2.append(y2)
        if not xs:
            return None
        return (min(xs), min(ys), max(xs2), max(ys2))

    def do_drag(self, event):
        if not self.drag_data["item"]:
            return
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        
        for cid in self.drag_data["item"]:
            self.canvas.move(cid, dx, dy)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

        if self.centroid_circle:
            self.canvas.delete(self.centroid_circle)
            self.centroid_circle = None
        if self.centroid_line:
            self.canvas.delete(self.centroid_line)
            self.centroid_line = None

        bbox = self._bbox_for_id_list(self.drag_data["item"])
        if bbox:
            x1, y1, x2, y2 = bbox
            cx = x1
            cy = (y1 + y2) / 2
            
            r = 9
            self.centroid_circle = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="red", width=2
            )

            self.centroid_line = self.canvas.create_line(
                cx, 0, cx, self.canvas_height,
                fill="red", dash=(4, 2), width=2
            )

            x_offset = self.rect_xoffset
            if 40 <= cy <= 80 and x_offset <= cx <= self.canvas_width:
                col = round((cx - x_offset) / float(self.grid_size))
                self.canvas.itemconfig(self.column_label, text=f"Starting Position: {col} mm")
            else:
                self.canvas.itemconfig(self.column_label, text="")

    def can_place_block(self, new_start, block_type):
        new_end = new_start + len(self.patterns[block_type]) - 1
    
        if len(self.placed_blocks) >= 3:
            self.message_label.config(text="Error: Max 3 AHC structures to place")
            return False
        for (start, end, _) in self.placed_blocks:
            if abs(new_start - end) < 5 or abs(start - new_end) < 5:
                self.message_label.config(text="Error: Too close to another AHC structure (less than 5 mm gap)")
                return False
    
        if new_end >= S:
            self.message_label.config(text="Error: AHC exceeds actuator length")
            return False
    
        return True
    
    def end_drag(self, event):
        if not self.drag_data["item"]:
            return

        if self.centroid_circle:
            self.canvas.delete(self.centroid_circle)
            self.centroid_circle = None
        if self.centroid_line:
            self.canvas.delete(self.centroid_line)
            self.centroid_line = None
        
        bbox = self._bbox_for_id_list(self.drag_data["item"])
        if not bbox:
            for cid in self.drag_data["item"]:
                self.canvas.delete(cid)
            self.drag_data = {"item": None, "value": None, "pattern": None, "x": None, "y": None}
            return

        x1, y1, x2, y2 = bbox
        x = x1
        y = y1
        x_offset = self.rect_xoffset
        col = round((x - x_offset) / float(self.grid_size))
        pattern = self.drag_data["pattern"]
        val = self.drag_data["value"]
        length = len(pattern)

        if 0 <= col < self.num_bits and 30 <= y <= 90:
            if col + length > self.num_bits:
                self.message_label.config(text=f"Error: AHC structure too long for starting position {col} mm (needs {length - (self.num_bits-col)} mm)")
                for cid in self.drag_data["item"]:
                    self.canvas.delete(cid)
            elif self.can_place_block(col, val):
                self.placed_blocks.append((col, col + length - 1, val))
                
                for i, bit in enumerate(pattern):
                    self.block_values[col + i] = int(bit)
                    
                    cx = self.rect_xoffset + (col + i) * self.grid_size + self.grid_size / 2
                    self.canvas.create_rectangle(cx - 5, 50, cx + 5, 70,
                                                 fill=self.colors[val] if bit == '1' else "gray", outline="")
                self.message_label.config(text="")
            else:
                for cid in self.drag_data["item"]:
                    self.canvas.delete(cid)
                return
        else:
            self.message_label.config(text="Error: Dropped outside the actuator")
            for cid in self.drag_data["item"]:
                self.canvas.delete(cid)
            return

        
        for cid in self.drag_data["item"]:
            self.canvas.delete(cid)
        self.canvas.itemconfig(self.column_label, text="")
        self.drag_data = {"item": None, "value": None, "pattern": None, "x": None, "y": None}
        return
        
    def print_code(self):
        if len(self.placed_blocks) <= 1:
            self.message_label.config(text="Error: Double/triple structure requires more input.")
            return
        bit_str = ''.join(str(v) for v in self.block_values)
        self.message_label.config(text="Last valid actuator design: {}".format(bit_str))
        
        self.canvas.itemconfig("strip_bg", fill="gray")


class BinaryPredictorApp:
    def __init__(self, root):
        
        self.root = root
        self.root.title("Hydromatic Simulator")
        self.root.geometry("600x700")

        self.loading_label = tk.Label(root, text="", fg="blue")
        self.loading_label.pack()
        
        self.progress = ttk.Progressbar(root, mode="indeterminate", length=200)
        self.progress.pack_forget()

        self.image_label_load = tk.Label(root)
        self.image_label_load.pack_forget()
        
        self.load_btn = ttk.Button(root, text="Load Models", command=self.start_model_loading)
        self.load_btn.pack(pady=5)

        self.reset_btn = ttk.Button(self.root, text="Reset", command=self.reset)
        self.reset_btn.place(x=10, y=10)
        self.reset_btn.config(state="disabled")

        self.label = ttk.Label(root, text=f"Enter {S}-bit Binary Code:")
        self.label.pack(pady=10)
        self.label.config(state="disabled")

        self.entry = tk.Entry(root, width=60)
        self.entry.pack(pady=5)
        self.entry.config(state="disabled")

        self.editor_btn = tk.Button(root, text="Open Design Editor", command=self.open_bit_editor)
        self.editor_btn.pack()
        self.editor_btn.config(state="disabled")

        self.predict_btn = tk.Button(root, text="Predict", command=self.predict)
        self.predict_btn.pack(pady=5)
        self.predict_btn.config(state="disabled")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.animation_running = False
        self.image_paths = []
        self.frame_idx = 0

        self.start_btn = ttk.Button(root, text="▶ Start Deformation", command=self.start_animation)
        self.start_btn.pack(pady=5)
        self.start_btn.config(state="disabled")

        self.stop_btn = ttk.Button(root, text="⏸ Stop Deformation", command=self.stop_animation)
        self.stop_btn.pack(pady=5)
        self.stop_btn.config(state="disabled")

    
    def open_bit_editor(self):
        def receive_code(code_str):
            print("Received code:", code_str)
            
            self.entry.delete(0, tk.END) 
            self.entry.insert(0, code_str)
    
        editor = BitPatternEditor(self.root, receive_code)
        self.entry.config(state="normal")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        
    def show_loading_image(self):
        img_path = './Hydromatic_Simulator/load_img.tif'
        img = Image.open(img_path)
        img.thumbnail((540, 540))
        self.load_photo = ImageTk.PhotoImage(img)
        self.image_label_load.configure(image=self.load_photo, anchor='w')
        self.image_label_load.image = self.load_photo 
        self.image_label_load.pack()

    def hide_loading_image(self):
        self.image_label_load.pack_forget()
    
    def start_model_loading(self):
        self.show_loading_image() 
        
        self.loading_label.config(text="Loading models...")
        self.progress.start()
        self.progress.pack()
        self.load_btn.config(state="disabled")
    
        threading.Thread(target=self.load_model_thread).start()

    def load_model_thread(self):
        models = load_trained_model()
        
        self.root.after(0, lambda: self.finish_model_loading(models))

    def finish_model_loading(self, models):
        self.models = models
        self.hide_loading_image() 
        
        self.progress.stop()
        self.progress.pack_forget()
        self.loading_label.config(text="Models loaded ✅")

        self.label.config(state="normal")
        self.entry.config(state="disabled")
        self.predict_btn.config(state="normal")

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        self.reset_btn.config(state="normal")
        self.editor_btn.config(state="normal")

    def predict(self):
        self.stop_animation()
        
        self.predict_btn.config(state="disabled")
        self.start_btn.config(state="enabled")
        self.stop_btn.config(state="disabled")
        self.editor_btn.config(state="disabled")
        self.entry.config(state="disabled")

        if sys.platform == "darwin":
            self.root.bind("<FocusIn>", mac_icon.refresh_dock_icon)
        
        binary_str = self.entry.get().strip()
        if len(binary_str) != S or not all(c in "01" for c in binary_str):
            messagebox.showerror("Invalid Input", f"Enter exactly {S} binary digits (0 or 1).")
            return self.reset()

        if binary_str == '0'*S:
            messagebox.showerror("Invalid Input", f"Enter valid design.")
            return self.reset()
        
        pred = run_inference(self.models, binary_str)
        
        self.image_paths = save_sequence_plot(pred)
        
        self.frame_idx = 0
        self.start_animation()
    
    def start_animation(self):
        if not self.image_paths:
            return
            
        self.animation_running = True
        self.animate()
        
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        return

    def stop_animation(self):
        self.animation_running = False

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        return

    def animate(self):
        if not self.animation_running:
            return
        if sys.platform == "darwin":
            self.root.bind("<FocusIn>", mac_icon.refresh_dock_icon)
            
        img_path = self.image_paths[self.frame_idx % (config['num-timesteps']+1)]
        img = Image.open(img_path)
        img.thumbnail((370, 370))
        
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo, anchor='w')
        self.image_label.image = self.photo
        
        self.frame_idx += 1

        if self.frame_idx % (config['num-timesteps']+1) == 0:
            self.root.after(400, self.animate) 
            
        else:
            self.root.after(200, self.animate) 
        
        return

    def reset(self):
        self.image_label_load.pack_forget()
    
        if hasattr(self, 'image_label'):
            self.image_label.configure(image='')
            self.image_label.image = None
        
        self.frame_idx = 0
        self.animation_running = False

        self.predict_btn.config(state="normal")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")

        self.entry.delete(0, tk.END)
        self.entry.config(state="disabled")

        self.editor_btn.config(state="normal")
        return
