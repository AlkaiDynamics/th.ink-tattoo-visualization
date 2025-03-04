import os
from pathlib import Path

def list_files():
    root = Path("c:/devdrive/thInk")
    exclude = {'.git', '__pycache__', 'node_modules', 'build', 'dist', '*.egg-info', 
              '.pytest_cache', '.mypy_cache', '.coverage', '.venv', '.env'}
    
    for path in root.rglob('*'):
        if path.is_file():
            # Skip excluded directories
            if any(x in str(path) for x in exclude):
                continue
                
            # Skip compiled Python files
            if path.suffix in {'.pyc', '.pyo', '.pyd'}:
                continue
                
            print(path.relative_to(root))

if __name__ == "__main__":
    list_files()