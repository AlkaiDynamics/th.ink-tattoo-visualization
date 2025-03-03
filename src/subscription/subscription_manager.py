from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

class SubscriptionFeatures:
    def __init__(self, tier: str):
        self.tier = tier
        self.features = {
            "free": {
                "daily_generations": 2,
                "preview_limit": 5,
                "watermark": True,
                "export_quality": "standard"
            },
            "premium": {
                "daily_generations": 20,
                "preview_limit": -1,
                "watermark": False,
                "export_quality": "high"
            },
            "pro": {
                "daily_generations": -1,
                "preview_limit": -1,
                "watermark": False,
                "export_quality": "maximum",
                "marketplace_commission": 0.10
            }
        }
    
    def get_limit(self, feature: str) -> int:
        return self.features.get(self.tier, {}).get(feature, 0)
    
    def has_feature(self, feature: str) -> bool:
        return feature in self.features.get(self.tier, {})

class SubscriptionManager:
    def __init__(self):
        self.user_subscriptions = {}
        self.usage_tracking = {}
    
    def check_limit(self, user_id: str, feature: str) -> bool:
        """Check if user has reached their feature limit"""
        if user_id not in self.usage_tracking:
            self.usage_tracking[user_id] = {}
        
        tier = self.user_subscriptions.get(user_id, "free")
        features = SubscriptionFeatures(tier)
        limit = features.get_limit(feature)
        
        if limit == -1:  # Unlimited
            return True
            
        current_usage = self.usage_tracking[user_id].get(feature, 0)
        return current_usage < limit
    
    def increment_usage(self, user_id: str, feature: str) -> bool:
        """Increment feature usage counter"""
        if not self.check_limit(user_id, feature):
            return False
            
        if user_id not in self.usage_tracking:
            self.usage_tracking[user_id] = {}
            
        self.usage_tracking[user_id][feature] = \
            self.usage_tracking[user_id].get(feature, 0) + 1
        return True