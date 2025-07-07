#enhanced-tts-project\services\cache_services.py
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import threading

class CacheService:
    def __init__(self, cache_duration_minutes: int = 30):
        self.cache_file = Path("cache.json")
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.cache_data = self._load_cache()
        self.lock = threading.Lock()

    def _load_cache(self) -> Dict:
        """Load cache from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f, default=str, indent=2)
        except Exception as e:
            print(f"Cache save error: {e}")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if not expired"""
        with self.lock:
            if key in self.cache_data:
                entry = self.cache_data[key]
                cached_time = datetime.fromisoformat(entry['timestamp'])
                
                if datetime.now() - cached_time < self.cache_duration:
                    return entry['data']
                else:
                    # Remove expired entry
                    del self.cache_data[key]
                    self._save_cache()
        return None

    def set(self, key: str, data: Dict[str, Any]):
        """Cache data with timestamp"""
        with self.lock:
            self.cache_data[key] = {
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()

    def clear_expired(self):
        """Remove all expired cache entries"""
        with self.lock:
            current_time = datetime.now()
            expired_keys = []
            
            for key, entry in self.cache_data.items():
                cached_time = datetime.fromisoformat(entry['timestamp'])
                if current_time - cached_time >= self.cache_duration:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                del self.cache_data[key]
                
            if expired_keys:
                self._save_cache()

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache_data)

    def clear_all(self):
        """Clear all cache"""
        with self.lock:
            self.cache_data.clear()
            self._save_cache()