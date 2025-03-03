import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

class ReleaseManager:
    def __init__(self):
        self.root_dir = Path("c:/devdrive/thInk")
        self.version_file = self.root_dir / "src" / "version.py"
        self.changelog_file = self.root_dir / "CHANGELOG.md"
        
    def bump_version(self, version_type: str):
        """Bump version number (major, minor, or patch)"""
        current = self._read_version()
        major, minor, patch = map(int, current.split('.'))
        
        if version_type == "major":
            major += 1
            minor = patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
            
        new_version = f"{major}.{minor}.{patch}"
        self._write_version(new_version)
        return new_version
        
    def create_release(self, version_type: str):
        """Create a new release"""
        new_version = self.bump_version(version_type)
        self._update_changelog(new_version)
        self._create_release_branch(new_version)
        self._build_release()
        return new_version
        
    def _read_version(self) -> str:
        with open(self.version_file) as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
        return "0.0.0"
        
    def _write_version(self, version: str):
        with open(self.version_file, 'w') as f:
            f.write(f'__version__ = "{version}"\n')
            f.write(f'__build__ = "release"\n')
            f.write(f'__release_date__ = "{datetime.now().strftime("%Y-%m-%d")}"\n')
            
    def _update_changelog(self, version: str):
        with open(self.changelog_file, 'r') as f:
            content = f.read()
            
        header = f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}\n"
        with open(self.changelog_file, 'w') as f:
            f.write(header + content)
            
    def _create_release_branch(self, version: str):
        subprocess.run(['git', 'checkout', '-b', f'release/{version}'])
        subprocess.run(['git', 'add', '.'])
        subprocess.run(['git', 'commit', '-m', f'Release version {version}'])
        
    def _build_release(self):
        subprocess.run(['python', '-m', 'build'])

if __name__ == "__main__":
    manager = ReleaseManager()
    if len(sys.argv) != 2:
        print("Usage: python release_manager.py [major|minor|patch]")
        sys.exit(1)
        
    version = manager.create_release(sys.argv[1])
    print(f"Created release {version}")