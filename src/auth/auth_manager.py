from datetime import datetime, timedelta
import jwt
import bcrypt
from typing import Dict, Optional

class AuthManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.active_sessions = {}
        self.token_expiry = timedelta(days=7)
        
    async def register_user(self, email: str, password: str) -> Optional[str]:
        """Register a new user and return user_id"""
        try:
            user_id = self._generate_user_id(email)
            hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
            
            # Store user data (placeholder for database integration)
            self.active_sessions[user_id] = {
                'email': email,
                'password_hash': hashed_password,
                'created_at': datetime.now()
            }
            
            return user_id
        except Exception as e:
            print(f"Registration failed: {e}")
            return None
    
    async def authenticate(self, email: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token"""
        user_id = self._generate_user_id(email)
        user_data = self.active_sessions.get(user_id)
        
        if not user_data:
            return None
            
        if bcrypt.checkpw(password.encode(), user_data['password_hash']):
            return self._generate_token(user_id)
        return None
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify JWT token and return user_id"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            if datetime.fromtimestamp(payload['exp']) > datetime.now():
                return payload['user_id']
        except:
            pass
        return None
    
    def _generate_token(self, user_id: str) -> str:
        """Generate JWT token"""
        expiry = datetime.now() + self.token_expiry
        return jwt.encode(
            {
                'user_id': user_id,
                'exp': expiry.timestamp()
            },
            self.secret_key,
            algorithm='HS256'
        )
    
    def _generate_user_id(self, email: str) -> str:
        """Generate unique user ID from email"""
        return bcrypt.hashpw(email.encode(), bcrypt.gensalt()).hex()[:24]