"""
Privacy management module for the Th.ink AR application.

This module provides functionality for managing user privacy preferences,
data retention policies, and GDPR compliance features.
"""

from datetime import datetime, timedelta
import json
import os
import logging
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import uuid

# Configure logger
logger = logging.getLogger(__name__)


class PrivacyManager:
    """
    Manages user privacy preferences and data retention policies.
    
    This class provides methods for handling privacy settings, consent
    management, and data retention in compliance with privacy regulations.
    """
    
    def __init__(self, retention_period_days: int = 365, storage_dir: Optional[str] = None):
        """
        Initialize the privacy manager.
        
        Args:
            retention_period_days: Default data retention period in days
            storage_dir: Directory to store privacy settings
        """
        self.default_retention_period = timedelta(days=retention_period_days)
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.data_access_log: Dict[str, List[Dict[str, Any]]] = {}
        self.deletion_requests: Dict[str, datetime] = {}
        
        # Set up storage
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_preferences()
        
        logger.info(f"Privacy manager initialized with {retention_period_days} day retention period")
    
    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Set a user's privacy preferences.
        
        Args:
            user_id: User identifier
            preferences: Dictionary of privacy preferences
            
        Returns:
            True if preferences were set successfully
        """
        try:
            # Validate preference keys
            valid_keys = {
                'ai_training_opt_out', 'data_retention_days', 
                'marketing_opt_out', 'analytics_opt_out',
                'third_party_sharing_opt_out'
            }
            
            for key in preferences.keys():
                if key not in valid_keys:
                    logger.warning(f"Ignoring invalid preference key: {key}")
            
            # Filter to only valid keys
            valid_preferences = {k: v for k, v in preferences.items() if k in valid_keys}
            
            # Set default values for missing keys
            if 'ai_training_opt_out' not in valid_preferences:
                valid_preferences['ai_training_opt_out'] = False
                
            if 'data_retention_days' not in valid_preferences:
                valid_preferences['data_retention_days'] = self.default_retention_period.days
                
            if 'marketing_opt_out' not in valid_preferences:
                valid_preferences['marketing_opt_out'] = False
                
            if 'analytics_opt_out' not in valid_preferences:
                valid_preferences['analytics_opt_out'] = False
                
            if 'third_party_sharing_opt_out' not in valid_preferences:
                valid_preferences['third_party_sharing_opt_out'] = False
            
            # Add timestamp
            valid_preferences['last_updated'] = datetime.now().isoformat()
            
            # Store preferences
            self.user_preferences[user_id] = valid_preferences
            
            # Save to storage if enabled
            if self.storage_dir:
                self._save_preferences(user_id)
            
            logger.info(f"Privacy preferences updated for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set privacy preferences: {str(e)}")
            return False
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get a user's privacy preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of privacy preferences
        """
        # Return copy of preferences or default values
        if user_id in self.user_preferences:
            return self.user_preferences[user_id].copy()
        else:
            # Return default preferences
            return {
                'ai_training_opt_out': False,
                'data_retention_days': self.default_retention_period.days,
                'marketing_opt_out': False,
                'analytics_opt_out': False,
                'third_party_sharing_opt_out': False,
                'last_updated': None
            }
    
    def should_retain_data(self, user_id: str, data_timestamp: datetime) -> bool:
        """
        Check if data should be retained based on user preferences.
        
        Args:
            user_id: User identifier
            data_timestamp: Timestamp of the data
            
        Returns:
            True if data should be retained, False if it should be deleted
        """
        # If user has a pending deletion request, don't retain any data
        if user_id in self.deletion_requests:
            return False
        
        # Get user retention period
        preferences = self.get_user_preferences(user_id)
        retention_days = preferences.get('data_retention_days', self.default_retention_period.days)
        retention_period = timedelta(days=retention_days)
        
        # Check if data is within retention period
        return datetime.now() - data_timestamp < retention_period
    
    def can_use_for_training(self, user_id: str) -> bool:
        """
        Check if user data can be used for AI training.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if data can be used for training, False otherwise
        """
        preferences = self.get_user_preferences(user_id)
        return not preferences.get('ai_training_opt_out', False)
    
    def can_use_for_marketing(self, user_id: str) -> bool:
        """
        Check if user data can be used for marketing purposes.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if data can be used for marketing, False otherwise
        """
        preferences = self.get_user_preferences(user_id)
        return not preferences.get('marketing_opt_out', False)
    
    def can_use_for_analytics(self, user_id: str) -> bool:
        """
        Check if user data can be used for analytics.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if data can be used for analytics, False otherwise
        """
        preferences = self.get_user_preferences(user_id)
        return not preferences.get('analytics_opt_out', False)
    
    def can_share_with_third_parties(self, user_id: str) -> bool:
        """
        Check if user data can be shared with third parties.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if data can be shared with third parties, False otherwise
        """
        preferences = self.get_user_preferences(user_id)
        return not preferences.get('third_party_sharing_opt_out', False)
    
    def log_data_access(self, user_id: str, data_type: str, purpose: str, accessed_by: str) -> str:
        """
        Log data access for audit purposes.
        
        Args:
            user_id: User identifier
            data_type: Type of data accessed
            purpose: Purpose of the access
            accessed_by: Identifier of the accessing entity
            
        Returns:
            Access log entry ID
        """
        if user_id not in self.data_access_log:
            self.data_access_log[user_id] = []
        
        # Create log entry
        log_id = str(uuid.uuid4())
        entry = {
            'id': log_id,
            'timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'purpose': purpose,
            'accessed_by': accessed_by,
            'ip_address': None  # Would be set in a real implementation
        }
        
        # Store entry
        self.data_access_log[user_id].append(entry)
        
        # Save to storage if enabled
        if self.storage_dir:
            self._save_access_log(user_id)
        
        return log_id
    
    def get_access_log(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get data access log for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of access log entries
        """
        return self.data_access_log.get(user_id, [])
    
    def request_data_deletion(self, user_id: str) -> bool:
        """
        Request deletion of all user data.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if request was recorded successfully
        """
        try:
            # Record deletion request
            self.deletion_requests[user_id] = datetime.now()
            
            # Save to storage if enabled
            if self.storage_dir:
                self._save_deletion_requests()
            
            logger.info(f"Data deletion request received for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process deletion request: {str(e)}")
            return False
    
    def get_deletion_status(self, user_id: str) -> Optional[datetime]:
        """
        Get status of a data deletion request.
        
        Args:
            user_id: User identifier
            
        Returns:
            Timestamp of deletion request or None if no request exists
        """
        return self.deletion_requests.get(user_id)
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Export all data for a user in a portable format.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing all user data
        """
        # This is a simplified implementation
        # A real implementation would gather all user data from various sources
        
        data = {
            'preferences': self.get_user_preferences(user_id),
            'access_log': self.get_access_log(user_id),
            'export_date': datetime.now().isoformat()
        }
        
        return data
    
    def _save_preferences(self, user_id: str) -> None:
        """
        Save user preferences to persistent storage.
        
        Args:
            user_id: User identifier
        """
        if not self.storage_dir:
            return
        
        # Get preferences
        preferences = self.user_preferences.get(user_id, {})
        
        if not preferences:
            return
        
        # Save to file
        preferences_file = self.storage_dir / f"preferences_{user_id}.json"
        
        try:
            with open(preferences_file, 'w') as f:
                json.dump(preferences, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save preferences for user {user_id}: {str(e)}")
    
    def _save_access_log(self, user_id: str) -> None:
        """
        Save access log to persistent storage.
        
        Args:
            user_id: User identifier
        """
        if not self.storage_dir:
            return
        
        # Get access log
        access_log = self.data_access_log.get(user_id, [])
        
        if not access_log:
            return
        
        # Save to file
        log_file = self.storage_dir / f"access_log_{user_id}.json"
        
        try:
            with open(log_file, 'w') as f:
                json.dump(access_log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save access log for user {user_id}: {str(e)}")
    
    def _save_deletion_requests(self) -> None:
        """Save deletion requests to persistent storage."""
        if not self.storage_dir:
            return
        
        # Convert datetime objects to strings
        serializable_requests = {
            user_id: timestamp.isoformat()
            for user_id, timestamp in self.deletion_requests.items()
        }
        
        # Save to file
        requests_file = self.storage_dir / "deletion_requests.json"
        
        try:
            with open(requests_file, 'w') as f:
                json.dump(serializable_requests, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save deletion requests: {str(e)}")
    
    def _load_preferences(self) -> None:
        """Load privacy data from persistent storage."""
        if not self.storage_dir:
            return
        
        # Load user preferences
        preferences_files = list(self.storage_dir.glob("preferences_*.json"))
        
        for file in preferences_files:
            try:
                with open(file, 'r') as f:
                    preferences = json.load(f)
                
                # Extract user ID from filename
                filename = file.name
                if filename.startswith("preferences_") and filename.endswith(".json"):
                    user_id = filename[12:-5]  # Remove "preferences_" prefix and ".json" suffix
                    
                    # Add to preferences
                    self.user_preferences[user_id] = preferences
            except Exception as e:
                logger.error(f"Failed to load preferences from {file}: {str(e)}")
        
        # Load access logs
        log_files = list(self.storage_dir.glob("access_log_*.json"))
        
        for file in log_files:
            try:
                with open(file, 'r') as f:
                    log_entries = json.load(f)
                
                # Extract user ID from filename
                filename = file.name
                if filename.startswith("access_log_") and filename.endswith(".json"):
                    user_id = filename[11:-5]  # Remove "access_log_" prefix and ".json" suffix
                    
                    # Add to access log
                    self.data_access_log[user_id] = log_entries
            except Exception as e:
                logger.error(f"Failed to load access log from {file}: {str(e)}")
        
        # Load deletion requests
        requests_file = self.storage_dir / "deletion_requests.json"
        
        if requests_file.exists():
            try:
                with open(requests_file, 'r') as f:
                    serialized_requests = json.load(f)
                
                # Convert string timestamps to datetime
                self.deletion_requests = {
                    user_id: datetime.fromisoformat(timestamp)
                    for user_id, timestamp in serialized_requests.items()
                }
            except Exception as e:
                logger.error(f"Failed to load deletion requests: {str(e)}")
        
        logger.info(f"Loaded privacy data for {len(self.user_preferences)} users")


# Create a singleton instance
_privacy_manager = None

def get_privacy_manager(retention_period_days: int = 365, storage_dir: Optional[str] = None) -> PrivacyManager:
    """
    Get the singleton privacy manager instance.
    
    Args:
        retention_period_days: Default retention period in days
        storage_dir: Directory for persistent storage
        
    Returns:
        PrivacyManager: The privacy manager instance
    """
    global _privacy_manager
    if _privacy_manager is None:
        _privacy_manager = PrivacyManager(retention_period_days, storage_dir)
    return _privacy_manager