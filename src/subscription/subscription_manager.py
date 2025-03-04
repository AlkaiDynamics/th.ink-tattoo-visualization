"""
Subscription management module for the Th.ink AR application.

This module provides functionality for managing user subscription tiers,
feature access control, and usage limits based on subscription level.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import logging
import json
from pathlib import Path
import time

# Configure logger
logger = logging.getLogger(__name__)


class SubscriptionTier(Enum):
    """Subscription tier levels."""
    FREE = "free"
    PREMIUM = "premium"
    PRO = "pro"


@dataclass
class SubscriptionFeatures:
    """Features and limits for each subscription tier."""
    
    # Generation limits
    daily_generations: int
    
    # Preview limits (-1 for unlimited)
    preview_limit: int
    
    # Quality settings
    max_resolution: str
    export_quality: str
    
    # Feature access
    watermark: bool
    nerf_avatar: bool
    advanced_styles: bool
    commercial_use: bool
    
    # Marketplace settings
    marketplace_commission: float = 0.0


@dataclass
class UsageData:
    """User usage tracking data."""
    
    # Daily counters with reset timestamps
    daily_generations: int = 0
    daily_previews: int = 0
    daily_reset: datetime = field(default_factory=datetime.now)
    
    # Monthly counters with reset timestamps
    monthly_exports: int = 0
    monthly_reset: datetime = field(default_factory=lambda: datetime.now().replace(day=1))
    
    # Lifetime statistics
    total_generations: int = 0
    total_previews: int = 0
    total_exports: int = 0
    join_date: datetime = field(default_factory=datetime.now)


class SubscriptionManager:
    """
    Manages user subscriptions, feature access, and usage limits.
    
    This class provides methods to check feature availability based on
    subscription tier and track usage against limits.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the subscription manager.
        
        Args:
            storage_dir: Directory for persistent storage
        """
        # Initialize subscription features for each tier
        self.features = {
            SubscriptionTier.FREE: SubscriptionFeatures(
                daily_generations=3,
                preview_limit=5,
                max_resolution="standard",
                export_quality="standard",
                watermark=True,
                nerf_avatar=False,
                advanced_styles=False,
                commercial_use=False
            ),
            SubscriptionTier.PREMIUM: SubscriptionFeatures(
                daily_generations=20,
                preview_limit=-1,  # Unlimited
                max_resolution="high",
                export_quality="high",
                watermark=False,
                nerf_avatar=True,
                advanced_styles=True,
                commercial_use=False,
                marketplace_commission=0.15  # 15% commission
            ),
            SubscriptionTier.PRO: SubscriptionFeatures(
                daily_generations=-1,  # Unlimited
                preview_limit=-1,      # Unlimited
                max_resolution="ultra",
                export_quality="maximum",
                watermark=False,
                nerf_avatar=True,
                advanced_styles=True,
                commercial_use=True,
                marketplace_commission=0.10  # 10% commission
            )
        }
        
        # User subscriptions (user_id -> tier)
        self.user_subscriptions: Dict[str, SubscriptionTier] = {}
        
        # User subscription expiry dates (user_id -> expiry datetime)
        self.expiry_dates: Dict[str, datetime] = {}
        
        # User usage tracking (user_id -> usage data)
        self.usage_tracking: Dict[str, UsageData] = {}
        
        # Set up storage
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_data()
        
        logger.info("Subscription manager initialized")
    
    def set_user_subscription(self, user_id: str, tier: SubscriptionTier, 
                             duration_days: int = 30) -> bool:
        """
        Set or update a user's subscription tier.
        
        Args:
            user_id: User identifier
            tier: Subscription tier
            duration_days: Subscription duration in days
            
        Returns:
            True if subscription was set successfully
        """
        try:
            # Update subscription
            self.user_subscriptions[user_id] = tier
            
            # Set expiry date
            expiry = datetime.now() + timedelta(days=duration_days)
            self.expiry_dates[user_id] = expiry
            
            # Initialize usage tracking if not exists
            if user_id not in self.usage_tracking:
                self.usage_tracking[user_id] = UsageData()
            
            # Save to storage if enabled
            if self.storage_dir:
                self._save_data()
            
            logger.info(f"Set subscription for user {user_id} to {tier.value} for {duration_days} days")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set subscription: {str(e)}")
            return False
    
    def get_user_subscription(self, user_id: str) -> SubscriptionTier:
        """
        Get a user's current subscription tier.
        
        Args:
            user_id: User identifier
            
        Returns:
            Current subscription tier (defaults to FREE)
        """
        # Check if subscription is expired
        if user_id in self.expiry_dates and datetime.now() > self.expiry_dates[user_id]:
            # Subscription expired, revert to FREE
            if user_id in self.user_subscriptions and self.user_subscriptions[user_id] != SubscriptionTier.FREE:
                logger.info(f"Subscription expired for user {user_id}")
                self.user_subscriptions[user_id] = SubscriptionTier.FREE
                
                # Save change to storage
                if self.storage_dir:
                    self._save_data()
        
        # Return current tier or default to FREE
        return self.user_subscriptions.get(user_id, SubscriptionTier.FREE)
    
    def get_expiry_date(self, user_id: str) -> Optional[datetime]:
        """
        Get the expiry date of a user's subscription.
        
        Args:
            user_id: User identifier
            
        Returns:
            Expiry date or None for FREE tier
        """
        tier = self.get_user_subscription(user_id)
        if tier == SubscriptionTier.FREE:
            return None
        
        return self.expiry_dates.get(user_id)
    
def get_features(self, user_id: str) -> SubscriptionFeatures:
        """
        Get features available for a user's subscription tier.
        
        Args:
            user_id: User identifier
            
        Returns:
            Features available for the user's subscription tier
        """
        tier = self.get_user_subscription(user_id)
        return self.features[tier]
    
    def check_limit(self, user_id: str, feature: str) -> bool:
        """
        Check if user has reached their feature limit.
        
        Args:
            user_id: User identifier
            feature: Feature to check (daily_generations, preview_limit, etc.)
            
        Returns:
            True if limit not reached, False if limit reached
        """
        # Get user's subscription tier and features
        tier = self.get_user_subscription(user_id)
        features = self.features[tier]
        
        # Initialize usage tracking if not exists
        if user_id not in self.usage_tracking:
            self.usage_tracking[user_id] = UsageData()
        
        # Reset daily counters if needed
        self._check_daily_reset(user_id)
        
        # Reset monthly counters if needed
        self._check_monthly_reset(user_id)
        
        # Get usage data
        usage = self.usage_tracking[user_id]
        
        # Check specific feature limit
        if feature == "daily_generations":
            limit = features.daily_generations
            current = usage.daily_generations
        elif feature == "preview_limit":
            limit = features.preview_limit
            current = usage.daily_previews
        elif feature == "monthly_exports":
            limit = -1  # No export limit currently
            current = usage.monthly_exports
        else:
            logger.warning(f"Unknown feature limit: {feature}")
            return True  # Default to allowing if feature unknown
        
        # -1 means unlimited
        if limit == -1:
            return True
        
        # Check if current usage is below limit
        return current < limit
    
    def increment_usage(self, user_id: str, feature: str) -> bool:
        """
        Increment feature usage counter.
        
        Args:
            user_id: User identifier
            feature: Feature to increment (daily_generations, preview_limit, etc.)
            
        Returns:
            True if increment successful and under limit, False otherwise
        """
        # Check if under limit first
        if not self.check_limit(user_id, feature):
            return False
        
        # Initialize usage tracking if not exists
        if user_id not in self.usage_tracking:
            self.usage_tracking[user_id] = UsageData()
        
        # Get usage data
        usage = self.usage_tracking[user_id]
        
        # Increment specific counter
        if feature == "daily_generations":
            usage.daily_generations += 1
            usage.total_generations += 1
        elif feature == "preview_limit":
            usage.daily_previews += 1
            usage.total_previews += 1
        elif feature == "monthly_exports":
            usage.monthly_exports += 1
            usage.total_exports += 1
        else:
            logger.warning(f"Unknown feature counter: {feature}")
            return False
        
        # Save to storage if enabled
        if self.storage_dir:
            self._save_data()
        
        return True
    
    def has_feature(self, user_id: str, feature: str) -> bool:
        """
        Check if user has access to a specific feature.
        
        Args:
            user_id: User identifier
            feature: Feature to check (nerf_avatar, advanced_styles, etc.)
            
        Returns:
            True if feature is available, False otherwise
        """
        # Get user's subscription tier and features
        tier = self.get_user_subscription(user_id)
        features = self.features[tier]
        
        # Check for specific features
        if feature == "nerf_avatar":
            return features.nerf_avatar
        elif feature == "advanced_styles":
            return features.advanced_styles
        elif feature == "no_watermark":
            return not features.watermark
        elif feature == "commercial_use":
            return features.commercial_use
        else:
            logger.warning(f"Unknown feature check: {feature}")
            return False
    
    def get_max_resolution(self, user_id: str) -> str:
        """
        Get maximum resolution available for user's subscription tier.
        
        Args:
            user_id: User identifier
            
        Returns:
            Maximum resolution string
        """
        # Get user's subscription tier and features
        tier = self.get_user_subscription(user_id)
        features = self.features[tier]
        
        return features.max_resolution
    
    def get_export_quality(self, user_id: str) -> str:
        """
        Get export quality available for user's subscription tier.
        
        Args:
            user_id: User identifier
            
        Returns:
            Export quality string
        """
        # Get user's subscription tier and features
        tier = self.get_user_subscription(user_id)
        features = self.features[tier]
        
        return features.export_quality
    
    def get_marketplace_commission(self, user_id: str) -> float:
        """
        Get marketplace commission rate for user's subscription tier.
        
        Args:
            user_id: User identifier
            
        Returns:
            Commission rate as a decimal (0.15 = 15%)
        """
        # Get user's subscription tier and features
        tier = self.get_user_subscription(user_id)
        features = self.features[tier]
        
        return features.marketplace_commission
    
    def get_usage_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get usage statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary of usage statistics
        """
        # Initialize usage tracking if not exists
        if user_id not in self.usage_tracking:
            self.usage_tracking[user_id] = UsageData()
        
        # Reset counters if needed
        self._check_daily_reset(user_id)
        self._check_monthly_reset(user_id)
        
        # Get usage data
        usage = self.usage_tracking[user_id]
        tier = self.get_user_subscription(user_id)
        features = self.features[tier]
        
        # Get limits
        daily_gen_limit = features.daily_generations
        preview_limit = features.preview_limit
        
        # Prepare statistics
        stats = {
            "tier": tier.value,
            "daily_generations": usage.daily_generations,
            "daily_generations_limit": daily_gen_limit,
            "daily_generations_left": daily_gen_limit - usage.daily_generations if daily_gen_limit >= 0 else -1,
            "daily_generations_unlimited": daily_gen_limit < 0,
            
            "daily_previews": usage.daily_previews,
            "daily_previews_limit": preview_limit,
            "daily_previews_left": preview_limit - usage.daily_previews if preview_limit >= 0 else -1,
            "daily_previews_unlimited": preview_limit < 0,
            
            "monthly_exports": usage.monthly_exports,
            
            "total_generations": usage.total_generations,
            "total_previews": usage.total_previews,
            "total_exports": usage.total_exports,
            
            "join_date": usage.join_date.isoformat(),
            "next_daily_reset": (usage.daily_reset + timedelta(days=1)).isoformat(),
            "next_monthly_reset": (usage.monthly_reset.replace(day=1) + timedelta(days=32)).replace(day=1).isoformat()
        }
        
        # Add subscription expiry if applicable
        if tier != SubscriptionTier.FREE and user_id in self.expiry_dates:
            stats["expiry_date"] = self.expiry_dates[user_id].isoformat()
            stats["days_left"] = (self.expiry_dates[user_id] - datetime.now()).days
        
        return stats
    
    def _check_daily_reset(self, user_id: str) -> None:
        """
        Check and reset daily counters if needed.
        
        Args:
            user_id: User identifier
        """
        usage = self.usage_tracking[user_id]
        now = datetime.now()
        
        # Check if day has changed since last reset
        if now.date() > usage.daily_reset.date():
            # Reset daily counters
            usage.daily_generations = 0
            usage.daily_previews = 0
            usage.daily_reset = now
            
            logger.debug(f"Daily counters reset for user {user_id}")
    
    def _check_monthly_reset(self, user_id: str) -> None:
        """
        Check and reset monthly counters if needed.
        
        Args:
            user_id: User identifier
        """
        usage = self.usage_tracking[user_id]
        now = datetime.now()
        
        # Check if month has changed since last reset
        if now.year > usage.monthly_reset.year or (now.year == usage.monthly_reset.year and now.month > usage.monthly_reset.month):
            # Reset monthly counters
            usage.monthly_exports = 0
            usage.monthly_reset = now.replace(day=1)  # First day of current month
            
            logger.debug(f"Monthly counters reset for user {user_id}")
    
    def _save_data(self) -> None:
        """Save subscription data to persistent storage."""
        if not self.storage_dir:
            return
        
        try:
            # Save user subscriptions
            subscriptions_data = {
                user_id: tier.value 
                for user_id, tier in self.user_subscriptions.items()
            }
            
            subscriptions_file = self.storage_dir / "subscriptions.json"
            with open(subscriptions_file, 'w') as f:
                json.dump(subscriptions_data, f, indent=2)
            
            # Save expiry dates
            expiry_data = {
                user_id: expiry.isoformat() 
                for user_id, expiry in self.expiry_dates.items()
            }
            
            expiry_file = self.storage_dir / "expiry_dates.json"
            with open(expiry_file, 'w') as f:
                json.dump(expiry_data, f, indent=2)
            
            # Save usage tracking
            usage_data = {}
            for user_id, usage in self.usage_tracking.items():
                usage_data[user_id] = {
                    "daily_generations": usage.daily_generations,
                    "daily_previews": usage.daily_previews,
                    "daily_reset": usage.daily_reset.isoformat(),
                    "monthly_exports": usage.monthly_exports,
                    "monthly_reset": usage.monthly_reset.isoformat(),
                    "total_generations": usage.total_generations,
                    "total_previews": usage.total_previews,
                    "total_exports": usage.total_exports,
                    "join_date": usage.join_date.isoformat()
                }
            
            usage_file = self.storage_dir / "usage_tracking.json"
            with open(usage_file, 'w') as f:
                json.dump(usage_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save subscription data: {str(e)}")
    
    def _load_data(self) -> None:
        """Load subscription data from persistent storage."""
        if not self.storage_dir:
            return
        
        try:
            # Load user subscriptions
            subscriptions_file = self.storage_dir / "subscriptions.json"
            if subscriptions_file.exists():
                with open(subscriptions_file, 'r') as f:
                    subscriptions_data = json.load(f)
                
                self.user_subscriptions = {
                    user_id: SubscriptionTier(tier_value)
                    for user_id, tier_value in subscriptions_data.items()
                }
            
            # Load expiry dates
            expiry_file = self.storage_dir / "expiry_dates.json"
            if expiry_file.exists():
                with open(expiry_file, 'r') as f:
                    expiry_data = json.load(f)
                
                self.expiry_dates = {
                    user_id: datetime.fromisoformat(expiry_str)
                    for user_id, expiry_str in expiry_data.items()
                }
            
            # Load usage tracking
            usage_file = self.storage_dir / "usage_tracking.json"
            if usage_file.exists():
                with open(usage_file, 'r') as f:
                    usage_data = json.load(f)
                
                for user_id, usage_dict in usage_data.items():
                    usage = UsageData(
                        daily_generations=usage_dict.get("daily_generations", 0),
                        daily_previews=usage_dict.get("daily_previews", 0),
                        daily_reset=datetime.fromisoformat(usage_dict.get("daily_reset")),
                        monthly_exports=usage_dict.get("monthly_exports", 0),
                        monthly_reset=datetime.fromisoformat(usage_dict.get("monthly_reset")),
                        total_generations=usage_dict.get("total_generations", 0),
                        total_previews=usage_dict.get("total_previews", 0),
                        total_exports=usage_dict.get("total_exports", 0),
                        join_date=datetime.fromisoformat(usage_dict.get("join_date"))
                    )
                    
                    self.usage_tracking[user_id] = usage
            
            logger.info(f"Loaded subscription data for {len(self.user_subscriptions)} users")
            
        except Exception as e:
            logger.error(f"Failed to load subscription data: {str(e)}")


# Create a singleton instance
_subscription_manager = None

def get_subscription_manager(storage_dir: Optional[str] = None) -> SubscriptionManager:
    """
    Get the singleton subscription manager instance.
    
    Args:
        storage_dir: Directory for persistent storage
        
    Returns:
        SubscriptionManager: The subscription manager instance
    """
    global _subscription_manager
    if _subscription_manager is None:
        _subscription_manager = SubscriptionManager(storage_dir)
    return _subscription_manager