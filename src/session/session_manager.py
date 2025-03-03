from datetime import datetime, timedelta
from typing import Dict, Optional, List
import json
import os

class SessionManager:
    def __init__(self, session_timeout: int = 30):
        self.sessions = {}
        self.session_timeout = timedelta(minutes=session_timeout)
        self.session_history = {}
        
    def create_session(self, user_id: str, device_info: Dict) -> str:
        """Create new session for user"""
        session_id = self._generate_session_id(user_id)
        self.sessions[session_id] = {
            'user_id': user_id,
            'start_time': datetime.now(),
            'last_activity': datetime.now(),
            'device_info': device_info,
            'metrics': {
                'designs_generated': 0,
                'frames_processed': 0,
                'total_processing_time': 0
            }
        }
        return session_id
    
    def update_session(self, session_id: str, metrics: Dict) -> bool:
        """Update session metrics and activity timestamp"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        session['last_activity'] = datetime.now()
        session['metrics'].update(metrics)
        return True
    
    def end_session(self, session_id: str) -> Optional[Dict]:
        """End session and archive metrics"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions.pop(session_id)
        user_id = session['user_id']
        
        if user_id not in self.session_history:
            self.session_history[user_id] = []
            
        session['end_time'] = datetime.now()
        self.session_history[user_id].append(session)
        return session
    
    def is_session_valid(self, session_id: str) -> bool:
        """Check if session is valid and not timed out"""
        if session_id not in self.sessions:
            return False
            
        session = self.sessions[session_id]
        if datetime.now() - session['last_activity'] > self.session_timeout:
            self.end_session(session_id)
            return False
        return True
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{user_id}_{timestamp}"