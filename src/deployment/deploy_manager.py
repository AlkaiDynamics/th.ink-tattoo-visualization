import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional

class DeploymentManager:
    def __init__(self):
        self.root_dir = Path("c:/devdrive/thInk")
        self.dist_dir = self.root_dir / "dist"
        self.backup_dir = self.root_dir / "backups"
        self.log_dir = self.root_dir / "logs" / "deployment"
        
        self._setup_directories()
        self._setup_logging()
        
    def _setup_directories(self):
        """Create necessary directories"""
        for directory in [self.dist_dir, self.backup_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def _setup_logging(self):
        """Configure deployment logging"""
        log_file = self.log_dir / f"deployment_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def create_deployment(self, version: str) -> bool:
        """Create deployment package"""
        try:
            logging.info(f"Starting deployment creation for version {version}")
            
            # Create backup
            self._create_backup()
            
            # Build application
            if not self._build_application():
                return False
                
            # Package assets
            self._package_assets()
            
            # Create deployment manifest
            self._create_manifest(version)
            
            logging.info("Deployment creation completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Deployment creation failed: {str(e)}")
            return False
            
    def _create_backup(self):
        """Create backup of current deployment"""
        backup_path = self.backup_dir / f"backup_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.copytree(self.dist_dir, backup_path, dirs_exist_ok=True)
        logging.info(f"Created backup at {backup_path}")
        
    def _build_application(self) -> bool:
        """Build application using PyInstaller"""
        try:
            subprocess.run([
                "pyinstaller",
                "--clean",
                "--windowed",
                "--onefile",
                "--icon=assets/icon.ico",
                "--add-data=config;config",
                "--add-data=models;models",
                "src/main.py"
            ], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Build failed: {str(e)}")
            return False
            
    def _package_assets(self):
        """Package required assets with deployment"""
        assets_dir = self.dist_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        # Copy required assets
        asset_paths = [
            ("models", "*.pth"),
            ("config", "*.json"),
            ("assets", "*.ico"),
        ]
        
        for path, pattern in asset_paths:
            source_dir = self.root_dir / path
            if source_dir.exists():
                for file in source_dir.glob(pattern):
                    shutil.copy2(file, assets_dir)
                    
    def _create_manifest(self, version: str):
        """Create deployment manifest"""
        manifest = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "files": self._get_file_manifest(),
            "requirements": self._get_requirements()
        }
        
        with open(self.dist_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
    def _get_file_manifest(self) -> Dict[str, str]:
        """Generate file manifest with checksums"""
        import hashlib
        manifest = {}
        
        for file in self.dist_dir.rglob("*"):
            if file.is_file():
                with open(file, "rb") as f:
                    manifest[str(file.relative_to(self.dist_dir))] = hashlib.sha256(f.read()).hexdigest()
                    
        return manifest