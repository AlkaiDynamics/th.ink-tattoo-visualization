import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import sys

class DeploymentVerifier:
    def __init__(self, deployment_path: Path):
        self.deployment_path = deployment_path
        self.manifest_path = deployment_path / "manifest.json"
        self.verification_results = []
        
    def verify_deployment(self) -> bool:
        """Perform comprehensive deployment verification"""
        try:
            manifest = self._load_manifest()
            if not manifest:
                return False
                
            checks = [
                self._verify_file_integrity(manifest['files']),
                self._verify_dependencies(manifest.get('requirements', {})),
                self._verify_executable(),
                self._verify_assets()
            ]
            
            return all(checks)
            
        except Exception as e:
            logging.error(f"Deployment verification failed: {str(e)}")
            return False
            
    def _load_manifest(self) -> Dict:
        """Load deployment manifest"""
        try:
            with open(self.manifest_path) as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load manifest: {str(e)}")
            return {}
            
    def _verify_file_integrity(self, manifest_files: Dict[str, str]) -> bool:
        """Verify integrity of deployment files"""
        for file_path, expected_hash in manifest_files.items():
            full_path = self.deployment_path / file_path
            if not full_path.exists():
                self.verification_results.append({
                    'type': 'file_missing',
                    'path': str(file_path),
                    'status': 'error'
                })
                return False
                
            with open(full_path, 'rb') as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
                
            if actual_hash != expected_hash:
                self.verification_results.append({
                    'type': 'hash_mismatch',
                    'path': str(file_path),
                    'status': 'error'
                })
                return False
                
        return True
        
    def _verify_dependencies(self, requirements: Dict[str, str]) -> bool:
        """Verify required dependencies"""
        try:
            import pkg_resources
            installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
            
            for package, version in requirements.items():
                if package not in installed:
                    self.verification_results.append({
                        'type': 'missing_dependency',
                        'package': package,
                        'status': 'error'
                    })
                    return False
                    
                if installed[package] != version:
                    self.verification_results.append({
                        'type': 'version_mismatch',
                        'package': package,
                        'installed': installed[package],
                        'required': version,
                        'status': 'warning'
                    })
                    
            return True
            
        except Exception as e:
            logging.error(f"Dependency verification failed: {str(e)}")
            return False
            
    def _verify_executable(self) -> bool:
        """Verify executable integrity and permissions"""
        exe_path = self.deployment_path / "main.exe"
        if not exe_path.exists():
            self.verification_results.append({
                'type': 'executable_missing',
                'status': 'error'
            })
            return False
            
        try:
            # Test executable launch with --version flag
            result = subprocess.run(
                [str(exe_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
            
        except Exception as e:
            self.verification_results.append({
                'type': 'executable_error',
                'error': str(e),
                'status': 'error'
            })
            return False
            
    def _verify_assets(self) -> bool:
        """Verify required assets are present and valid"""
        required_assets = [
            ('assets', '*.ico'),
            ('models', '*.pth'),
            ('config', '*.json')
        ]
        
        for folder, pattern in required_assets:
            asset_path = self.deployment_path / folder
            if not asset_path.exists() or not list(asset_path.glob(pattern)):
                self.verification_results.append({
                    'type': 'missing_assets',
                    'folder': folder,
                    'pattern': pattern,
                    'status': 'error'
                })
                return False
                
        return True