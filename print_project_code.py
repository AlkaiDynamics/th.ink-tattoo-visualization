import os
from pathlib import Path

# Directories to always exclude (auto-generated, VCS, IDE files)
EXCLUDED_DIRS = {
    ".git", "node_modules", "build", "dist", ".vscode", ".idea",
    "__pycache__", ".pytest_cache", ".venv", "venv", ".mypy_cache", "docs", "infra", "output", "reports", "logs"
}

# Only include files that are within these top-level directories (relative to project root)
# Change this list to include only the directories where you have written your core application code.
INCLUDED_ROOT_DIRS = {"src", "server", "tools"}

# Allowed file extensions for source code files
ALLOWED_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".json", ".yml", ".yaml", ".md"}

def print_project_code(root_dir: Path):
    """
    Recursively prints the relative file path and contents of all files that are in 
    the INCLUDED_ROOT_DIRS (i.e. core application code), while skipping files and directories
    that are auto-generated or not part of the written app functionality.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Get relative path parts from the project root
        rel_path = Path(dirpath).relative_to(root_dir)
        if rel_path.parts:
            # Only process files if the top-level directory is in the inclusion list.
            if rel_path.parts[0] not in INCLUDED_ROOT_DIRS:
                # Skip this entire subtree
                dirnames[:] = []  # do not walk subdirectories
                continue

        # Exclude any subdirectories that are in the EXCLUDED_DIRS
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for filename in filenames:
            file_path = Path(dirpath) / filename

            # Only consider allowed file types
            if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            # Print header for file
            rel_file_path = file_path.relative_to(root_dir)
            separator = "=" * 80
            print(separator)
            print(f"FILE: {rel_file_path}")
            print(separator)

            # Read and print file content
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                print(content)
            except Exception as e:
                print(f"[Error reading file: {e}]")
            print("\n")  # Blank line between files

if __name__ == "__main__":
    project_root = Path.cwd()  # or set this to your project root explicitly
    print(f"Printing project code for: {project_root}\n")
    print_project_code(project_root)
