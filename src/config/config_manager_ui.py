import tkinter as tk
from tkinter import ttk, messagebox
import json
from pathlib import Path
from .validator import ConfigValidator, ValidationLevel
from .env_manager import EnvManager

class ConfigManagerUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("thInk Configuration Manager")
        self.root.geometry("800x600")
        
        self.validator = ConfigValidator()
        self.env_manager = EnvManager()
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Create main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        tab_control = ttk.Notebook(main_frame)
        env_tab = ttk.Frame(tab_control)
        validation_tab = ttk.Frame(tab_control)
        
        tab_control.add(env_tab, text='Environment Variables')
        tab_control.add(validation_tab, text='Validation')
        tab_control.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self._create_env_tab(env_tab)
        self._create_validation_tab(validation_tab)
        
    def _create_env_tab(self, parent):
        # Create scrollable frame for variables
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add environment variable entries
        self.env_vars = {}
        for var_name, rules in self.validator.validation_rules.items():
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=var_name).pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=50)
            entry.pack(side=tk.LEFT, padx=5)
            entry.insert(0, self.env_manager.get(var_name, rules.get('default', '')))
            
            self.env_vars[var_name] = entry
            
        # Add save button
        ttk.Button(
            scrollable_frame,
            text="Save Changes",
            command=self._save_changes
        ).pack(pady=10)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def _create_validation_tab(self, parent):
        # Add validate button
        ttk.Button(
            parent,
            text="Validate Configuration",
            command=self._validate_config
        ).pack(pady=10)
        
        # Add validation results area
        self.validation_text = tk.Text(parent, height=20, width=70)
        self.validation_text.pack(padx=10, pady=5)
        
    def _save_changes(self):
        for var_name, entry in self.env_vars.items():
            self.env_manager.set(var_name, entry.get())
        messagebox.showinfo("Success", "Configuration saved successfully!")
        
    def _validate_config(self):
        self.validation_text.delete(1.0, tk.END)
        issues = self.validator.validate_all()
        
        if not issues:
            self.validation_text.insert(tk.END, "✓ All configurations are valid!")
            return
            
        for issue in issues:
            color = {
                ValidationLevel.ERROR.value: "red",
                ValidationLevel.WARNING.value: "orange",
                ValidationLevel.INFO.value: "blue"
            }.get(issue['level'], "black")
            
            self.validation_text.insert(
                tk.END,
                f"[{issue['level'].upper()}] {issue['message']}\n",
                issue['level']
            )
            self.validation_text.tag_config(issue['level'], foreground=color)
            
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ConfigManagerUI()
    app.run()