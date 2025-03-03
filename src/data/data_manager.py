import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import shutil

class DataManager:
    def __init__(self, base_path: str = "c:/devdrive/thInk/data"):
        self.base_path = base_path
        self.user_data_path = os.path.join(base_path, "users")
        os.makedirs(self.user_data_path, exist_ok=True)
        
    async def export_user_data(self, user_id: str) -> Optional[Dict]:
        """Export all user data in a portable format"""
        try:
            user_path = os.path.join(self.user_data_path, user_id)
            if not os.path.exists(user_path):
                return None
                
            data = {
                'designs': self._get_user_designs(user_id),
                'sessions': self._get_user_sessions(user_id),
                'preferences': self._get_user_preferences(user_id),
                'export_date': datetime.now().isoformat()
            }
            
            export_path = os.path.join(user_path, 'exports')
            os.makedirs(export_path, exist_ok=True)
            
            export_file = os.path.join(export_path, f'data_export_{datetime.now().strftime("%Y%m%d")}.json')
            with open(export_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            return data
            
        except Exception as e:
            print(f"Export failed: {e}")
            return None
    
    async def delete_user_data(self, user_id: str, partial: bool = False) -> bool:
        """Delete user data (partial or complete)"""
        try:
            user_path = os.path.join(self.user_data_path, user_id)
            if not os.path.exists(user_path):
                return False
                
            if partial:
                # Keep essential account data, remove personal data
                self._remove_personal_data(user_id)
            else:
                # Complete deletion
                shutil.rmtree(user_path)
                
            return True
            
        except Exception as e:
            print(f"Deletion failed: {e}")
            return False
    
    def _get_user_designs(self, user_id: str) -> List[Dict]:
        """Retrieve user's tattoo designs"""
        designs_path = os.path.join(self.user_data_path, user_id, 'designs')
        if not os.path.exists(designs_path):
            return []
            
        designs = []
        for design_file in os.listdir(designs_path):
            if design_file.endswith('.json'):
                with open(os.path.join(designs_path, design_file)) as f:
                    designs.append(json.load(f))
        return designs
    
    def _get_user_sessions(self, user_id: str) -> List[Dict]:
        """Retrieve user's session history"""
        sessions_path = os.path.join(self.user_data_path, user_id, 'sessions')
        if not os.path.exists(sessions_path):
            return []
            
        sessions = []
        for session_file in os.listdir(sessions_path):
            if session_file.endswith('.json'):
                with open(os.path.join(sessions_path, session_file)) as f:
                    sessions.append(json.load(f))
        return sessions
    
    def _get_user_preferences(self, user_id: str) -> Dict:
        """Retrieve user's preferences"""
        prefs_file = os.path.join(self.user_data_path, user_id, 'preferences.json')
        if not os.path.exists(prefs_file):
            return {}
            
        with open(prefs_file) as f:
            return json.load(f)