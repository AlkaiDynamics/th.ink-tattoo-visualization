"""
Session management for the Th.ink AR application.

This module provides functionality for creating and managing user sessions,
tracking session metrics, and ensuring session security.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import uuid
import logging
import os
import json
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages user sessions and their metadata in the AR application.
    
    This class provides methods to create, update, and validate sessions,
    as well as track performance metrics and user actions.
    """
    
    def __init__(self, session_timeout: int = 30, storage_dir: Optional[str] = None):
        """
        Initialize the session manager.
        
        Args:
            session_timeout: Timeout in minutes for inactive sessions
            storage_dir: Directory to store session data (if persistent storage is needed)
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(minutes=session_timeout)
        self.session_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Set up storage if specified
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_sessions()
        
        logger.info(f"Session manager initialized with {session_timeout} minute timeout")
    
    def create_session(self, user_id: str, device_info: Dict[str, Any]) -> str:
        """
        Create a new session for a user.
        
        Args:
            user_id: User identifier
            device_info: Information about the user's device
            
        Returns:
            Session ID
        """
        session_id = self._generate_session_id(user_id)
        
        self.sessions[session_id] = {
            'user_id': user_id,
            'start_time': datetime.now(),
            'last_activity': datetime.now(),
            'device_info': device_info,
            'metrics': {
                'designs_generated': 0,
                'designs_viewed': 0,
                'frames_processed': 0,
                'total_processing_time': 0,
                'ar_sessions': 0,
                'ar_duration': 0,
                'tattoo_adjustments': 0
            },
            'geo_location': device_info.get('geo_location', None),
            'user_agent': device_info.get('user_agent', None),
            'app_version': device_info.get('app_version', None),
            'platform': device_info.get('platform', None)
        }
        
        logger.info(f"Created session {session_id} for user {user_id}")
        
        # Save session if storage is configured
        if self.storage_dir:
            self._save_session(session_id)
        
        return session_id
    
    def update_session(self, session_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Update session metrics and activity timestamp.
        
        Args:
            session_id: Session identifier
            metrics: Metrics to update
            
        Returns:
            True if session was updated, False if not found
        """
        if session_id not in self.sessions:
            logger.warning(f"Attempted to update non-existent session: {session_id}")
            return False
        
        session = self.sessions[session_id]
        session['last_activity'] = datetime.now()
        
        # Update metrics
        for key, value in metrics.items():
            if key in session['metrics']:
                # Accumulate numeric values, replace others
                if isinstance(value, (int, float)) and isinstance(session['metrics'][key], (int, float)):
                    session['metrics'][key] += value
                else:
                    session['metrics'][key] = value
        
        # Save session if storage is configured
        if self.storage_dir:
            self._save_session(session_id)
        
        return True
    
    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        End a session and archive metrics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        if session_id not in self.sessions:
            logger.warning(f"Attempted to end non-existent session: {session_id}")
            return None
        
        session = self.sessions.pop(session_id)
        user_id = session['user_id']
        
        # Add end time
        session['end_time'] = datetime.now()
        session['duration'] = (session['end_time'] - session['start_time']).total_seconds()
        
        # Archive session in history
        if user_id not in self.session_history:
            self.session_history[user_id] = []
        
        self.session_history[user_id].append(session)
        
        # Limit history size
        max_history = 100  # Maximum sessions per user
        if len(self.session_history[user_id]) > max_history:
            # Remove oldest sessions
            self.session_history[user_id] = self.session_history[user_id][-max_history:]
        
        # Save history if storage is configured
        if self.storage_dir:
            self._save_history(user_id)
            self._remove_session(session_id)
        
        logger.info(f"Ended session {session_id} for user {user_id}")
        
        return session
    
    def is_session_valid(self, session_id: str) -> bool:
        """
        Check if a session is valid and not timed out.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session is valid, False otherwise
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Check if session has timed out
        if datetime.now() - session['last_activity'] > self.session_timeout:
            # Session has timed out - end it
            self.end_session(session_id)
            return False
        
        return True
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        if not self.is_session_valid(session_id):
            return None
        
        return self.sessions[session_id]
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of active sessions
        """
        return [
            {"session_id": session_id, **session}
            for session_id, session in self.sessions.items()
            if session['user_id'] == user_id
        ]
    
    def get_user_session_history(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get session history for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of past sessions
        """
        return self.session_history.get(user_id, [])
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions removed
        """
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if datetime.now() - session['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        # End each expired session
        for session_id in expired_sessions:
            self.end_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)
    
    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session metrics
        """
        if session_id not in self.sessions:
            return {}
        # Get session
        session = self.sessions[session_id]
        
        # Get metrics
        metrics = session['metrics'].copy()
        
        # Add session duration
        duration = (datetime.now() - session['start_time']).total_seconds()
        metrics['session_duration'] = duration
        
        # Add session start time
        metrics['session_start'] = session['start_time'].isoformat()
        
        # Add last activity time
        metrics['last_activity'] = session['last_activity'].isoformat()
        
        return metrics
    
    def get_aggregate_metrics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregate metrics across all sessions or for a specific user.
        
        Args:
            user_id: Optional user identifier to filter sessions
            
        Returns:
            Aggregate metrics
        """
        # Filter sessions by user if specified
        if user_id:
            sessions = [s for s in self.sessions.values() if s['user_id'] == user_id]
        else:
            sessions = list(self.sessions.values())
        
        # Initialize aggregates
        aggregates = {
            'active_sessions': len(sessions),
            'designs_generated': 0,
            'designs_viewed': 0,
            'frames_processed': 0,
            'total_processing_time': 0,
            'ar_sessions': 0,
            'ar_duration': 0,
            'tattoo_adjustments': 0,
            'average_session_duration': 0
        }
        
        # Calculate total session duration
        total_duration = 0
        
        # Aggregate metrics
        for session in sessions:
            metrics = session['metrics']
            
            # Sum up metrics
            for key in aggregates.keys():
                if key in metrics and key != 'active_sessions':
                    aggregates[key] += metrics[key]
            
            # Calculate session duration
            duration = (datetime.now() - session['start_time']).total_seconds()
            total_duration += duration
        
        # Calculate average session duration
        if sessions:
            aggregates['average_session_duration'] = total_duration / len(sessions)
        
        return aggregates
    
    def _generate_session_id(self, user_id: str) -> str:
        """
        Generate a unique session ID.
        
        Args:
            user_id: User identifier
            
        Returns:
            Unique session ID
        """
        # Combine user ID, timestamp and random UUID for uniqueness
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_part = str(uuid.uuid4())[:8]
        return f"{user_id}_{timestamp}_{random_part}"
    
    def _save_session(self, session_id: str) -> None:
        """
        Save session data to persistent storage.
        
        Args:
            session_id: Session identifier
        """
        if not self.storage_dir:
            return
        
        # Get session
        session = self.sessions[session_id]
        
        # Convert datetime objects to strings
        serializable_session = self._prepare_for_serialization(session)
        
        # Save to file
        session_file = self.storage_dir / f"session_{session_id}.json"
        
        try:
            with open(session_file, 'w') as f:
                json.dump(serializable_session, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {str(e)}")
    
    def _save_history(self, user_id: str) -> None:
        """
        Save user session history to persistent storage.
        
        Args:
            user_id: User identifier
        """
        if not self.storage_dir:
            return
        
        # Get user history
        history = self.session_history.get(user_id, [])
        
        if not history:
            return
        
        # Convert datetime objects to strings
        serializable_history = [self._prepare_for_serialization(session) for session in history]
        
        # Save to file
        history_file = self.storage_dir / f"history_{user_id}.json"
        
        try:
            with open(history_file, 'w') as f:
                json.dump(serializable_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history for user {user_id}: {str(e)}")
    
    def _remove_session(self, session_id: str) -> None:
        """
        Remove session file from storage.
        
        Args:
            session_id: Session identifier
        """
        if not self.storage_dir:
            return
        
        session_file = self.storage_dir / f"session_{session_id}.json"
        
        try:
            if session_file.exists():
                session_file.unlink()
        except Exception as e:
            logger.error(f"Failed to remove session file {session_id}: {str(e)}")
    
    def _load_sessions(self) -> None:
        """Load sessions from persistent storage."""
        if not self.storage_dir:
            return
        
        # Load active sessions
        session_files = list(self.storage_dir.glob("session_*.json"))
        
        for file in session_files:
            try:
                with open(file, 'r') as f:
                    session_data = json.load(f)
                
                # Convert string timestamps back to datetime
                session_data = self._prepare_from_serialization(session_data)
                
                # Extract session ID from filename
                filename = file.name
                if filename.startswith("session_") and filename.endswith(".json"):
                    session_id = filename[8:-5]  # Remove "session_" prefix and ".json" suffix
                    
                    # Add to sessions
                    self.sessions[session_id] = session_data
            except Exception as e:
                logger.error(f"Failed to load session from {file}: {str(e)}")
        
        # Load session history
        history_files = list(self.storage_dir.glob("history_*.json"))
        
        for file in history_files:
            try:
                with open(file, 'r') as f:
                    history_data = json.load(f)
                
                # Convert string timestamps back to datetime
                history_data = [self._prepare_from_serialization(session) for session in history_data]
                
                # Extract user ID from filename
                filename = file.name
                if filename.startswith("history_") and filename.endswith(".json"):
                    user_id = filename[8:-5]  # Remove "history_" prefix and ".json" suffix
                    
                    # Add to history
                    self.session_history[user_id] = history_data
            except Exception as e:
                logger.error(f"Failed to load history from {file}: {str(e)}")
        
        logger.info(f"Loaded {len(self.sessions)} active sessions and history for {len(self.session_history)} users")
    
    def _prepare_for_serialization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for JSON serialization by converting datetime objects to strings.
        
        Args:
            data: Session data
            
        Returns:
            Serializable data
        """
        serializable = {}
        
        for key, value in data.items():
            if isinstance(value, datetime):
                serializable[key] = value.isoformat()
            elif isinstance(value, dict):
                serializable[key] = self._prepare_for_serialization(value)
            else:
                serializable[key] = value
        
        return serializable
    
    def _prepare_from_serialization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data from JSON deserialization by converting strings to datetime objects.
        
        Args:
            data: Serialized session data
            
        Returns:
            Deserialized data
        """
        deserialized = {}
        
        for key, value in data.items():
            if key in ['start_time', 'last_activity', 'end_time'] and isinstance(value, str):
                try:
                    deserialized[key] = datetime.fromisoformat(value)
                except ValueError:
                    # Fallback if parsing fails
                    deserialized[key] = datetime.now()
            elif isinstance(value, dict):
                deserialized[key] = self._prepare_from_serialization(value)
            else:
                deserialized[key] = value
        
        return deserialized


# Create a singleton instance
_session_manager = None

def get_session_manager(session_timeout: int = 30, storage_dir: Optional[str] = None) -> SessionManager:
    """
    Get the singleton session manager instance.
    
    Args:
        session_timeout: Timeout in minutes for inactive sessions
        storage_dir: Directory to store session data
        
    Returns:
        SessionManager: The session manager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(session_timeout, storage_dir)
    return _session_manager