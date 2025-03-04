import os
from pathlib import Path

EXCLUDED_DIRS = {
    ".git", "node_modules", "build", "dist", ".vscode", ".idea",
    "__pycache__", ".pytest_cache", ".venv", "venv", ".mypy_cache"
}

BUILD_RELATED_FILES = {
    'setup.py', 'setup.cfg', 'pyproject.toml', 'requirements.txt',
    'Dockerfile', 'docker-compose.yml', '.dockerignore', 'MANIFEST.in',
    'conda_export.txt', 'pytest.ini'
}

def print_tree_and_contents(root_dir: Path, indent: str = "") -> None:
    try:
        entries = sorted(os.listdir(root_dir))
    except PermissionError:
        print(f"{indent}[Permission Denied]: {root_dir}")
        return

    # First, print all build-related files in current directory
    for entry in entries:
        if entry in BUILD_RELATED_FILES:
            entry_path = root_dir / entry
            if entry_path.is_file():
                print(f"{indent}📄 {entry} (Build Config)")
                try:
                    with open(entry_path, "r", encoding="utf-8", errors="ignore") as file:
                        content = file.read().splitlines()
                        for line in content:
                            print(f"{indent}    {line}")
                except Exception as e:
                    print(f"{indent}    [Error reading file: {e}]")

    # Then process directories
    for entry in entries:
        entry_path = root_dir / entry
        if entry_path.is_dir() and entry not in EXCLUDED_DIRS:
            print(f"{indent}📂 {entry}/")
            print_tree_and_contents(entry_path, indent + "    ")

if __name__ == "__main__":
    root_directory = Path("c:/devdrive/thInk")
    print(f"📂 Project Root: {root_directory}\n")
    print_tree_and_contents(root_directory)