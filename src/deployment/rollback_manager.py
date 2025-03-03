import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from .deploy_verifier import DeploymentVerifier

class RollbackManager:
    def __init__(self):
        self.root_dir = Path("c:/devdrive/thInk")
        self.backup_dir = self.root_dir / "backups"
        self.dist_dir = self.root_dir / "dist"
        self.log_dir = self.root_dir / "logs" / "rollback"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def rollback_to_version(self, version: str) -> bool:
        """Rollback to a specific version"""
        backup = self._find_backup_by_version(version)
        if not backup:
            logging.error(f"No backup found for version {version}")
            return False
            
        return self._perform_rollback(backup)
        
    def rollback_to_last_stable(self) -> bool:
        """Rollback to the last known stable version"""
        backup = self._find_last_stable_backup()
        if not backup:
            logging.error("No stable backup found")
            return False
            
        return self._perform_rollback(backup)
        
    def _find_backup_by_version(self, version: str) -> Optional[Path]:
        """Find backup directory for specific version"""
        for backup_dir in self.backup_dir.iterdir():
            manifest_path = backup_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    if manifest.get('version') == version:
                        return backup_dir
        return None
        
    def _find_last_stable_backup(self) -> Optional[Path]:
        """Find the most recent stable backup"""
        stable_backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            verifier = DeploymentVerifier(backup_dir)
            if verifier.verify_deployment():
                stable_backups.append(backup_dir)
                
        if not stable_backups:
            return None
            
        # Sort by creation time and return most recent
        return sorted(stable_backups, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        
    def _perform_rollback(self, backup_path: Path) -> bool:
        """Execute rollback process"""
        try:
            # Create backup of current state
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_backup = self.backup_dir / f"pre_rollback_{timestamp}"
            shutil.copytree(self.dist_dir, current_backup)
            
            # Clear current distribution
            shutil.rmtree(self.dist_dir)
            self.dist_dir.mkdir(parents=True)
            
            # Restore from backup
            shutil.copytree(backup_path, self.dist_dir, dirs_exist_ok=True)
            
            # Verify restored deployment
            verifier = DeploymentVerifier(self.dist_dir)
            if not verifier.verify_deployment():
                # If verification fails, restore from pre-rollback backup
                shutil.rmtree(self.dist_dir)
                shutil.copytree(current_backup, self.dist_dir)
                logging.error("Rollback verification failed, restored previous state")
                return False
                
            logging.info(f"Successfully rolled back to {backup_path.name}")
            return True
            
        except Exception as e:
            logging.error(f"Rollback failed: {str(e)}")
            return False
            
    def get_available_versions(self) -> List[Dict[str, str]]:
        """Get list of available backup versions"""
        versions = []
        
        for backup_dir in self.backup_dir.iterdir():
            manifest_path = backup_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    versions.append({
                        'version': manifest.get('version'),
                        'timestamp': manifest.get('timestamp'),
                        'path': str(backup_dir)
                    })
                    
        return sorted(versions, key=lambda x: x['timestamp'], reverse=True)