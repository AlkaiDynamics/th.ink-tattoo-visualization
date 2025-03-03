from datetime import datetime, timedelta
import json
import os
from typing import Dict, Optional

class PrivacyManager:
    def __init__(self, retention_period_days: int = 365):
        self.retention_period = timedelta(days=retention_period_days)
        self.user_preferences = {}
        self.data_access_log = {}
        
    def set_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Set user's privacy preferences"""
        try:
            self.user_preferences[user_id] = {
                'ai_training_opt_out': preferences.get('ai_training_opt_out', False),
                'data_retention_days': preferences.get('data_retention_days', 365),
                'last_updated': datetime.now().isoformat()
            }
            return True
        except Exception as e:
            print(f"Failed to set privacy preferences: {e}")
            return False
    
    def should_retain_data(self, user_id: str, data_timestamp: datetime) -> bool:
        """Check if data should be retained based on user preferences"""
        if user_id not in self.user_preferences:
            return True
            
        retention_days = self.user_preferences[user_id]['data_retention_days']
        retention_period = timedelta(days=retention_days)
        return datetime.now() - data_timestamp < retention_period
    
    def can_use_for_training(self, user_id: str) -> bool:
        """Check if user data can be used for AI training"""
        if user_id not in self.user_preferences:
            return True
        return not self.user_preferences[user_id].get('ai_training_opt_out', False)
    
    def log_data_access(self, user_id: str, data_type: str, purpose: str):
        """Log data access for audit purposes"""
        if user_id not in self.data_access_log:
            self.data_access_log[user_id] = []
            
        self.data_access_log[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'purpose': purpose
        })