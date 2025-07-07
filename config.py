#config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management"""
    
    # =============================================================================
    # SERVER SETTINGS
    # =============================================================================
    BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", "1234"))
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))
    API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{BACKEND_PORT}")
    
    # =============================================================================
    # API KEYS
    # =============================================================================
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    
    # =============================================================================
    # CACHE SETTINGS
    # =============================================================================
    CACHE_DURATION_MINUTES = int(os.getenv("CACHE_DURATION_MINUTES", "30"))
    CACHE_FILE_PATH = os.getenv("CACHE_FILE_PATH", "cache.json")
    CACHE_AUTO_CLEANUP = os.getenv("CACHE_AUTO_CLEANUP", "true").lower() == "true"
    
    # =============================================================================
    # AUDIO SETTINGS
    # =============================================================================
    AUDIO_OUTPUT_DIR = Path(os.getenv("AUDIO_OUTPUT_DIR", "audio"))
    DEFAULT_TTS_LANGUAGE = os.getenv("DEFAULT_TTS_LANGUAGE", "en")
    AUDIO_CLEANUP_DAYS = int(os.getenv("AUDIO_CLEANUP_DAYS", "1"))
    MAX_AUDIO_LENGTH = int(os.getenv("MAX_AUDIO_LENGTH", "5000"))
    
    # =============================================================================
    # REQUEST SETTINGS
    # =============================================================================
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    MAX_TOPICS_PER_REQUEST = int(os.getenv("MAX_TOPICS_PER_REQUEST", "5"))
    RATE_LIMIT_DELAY = int(os.getenv("RATE_LIMIT_DELAY", "2"))
    USER_AGENT = os.getenv("USER_AGENT", "NewsNinja/2.0 (Educational Use)")
    
    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================
    ENABLE_REDDIT_SCRAPING = os.getenv("ENABLE_REDDIT_SCRAPING", "true").lower() == "true"
    ENABLE_NEWS_SCRAPING = os.getenv("ENABLE_NEWS_SCRAPING", "true").lower() == "true"
    ENABLE_SENTIMENT_ANALYSIS = os.getenv("ENABLE_SENTIMENT_ANALYSIS", "true").lower() == "true"
    ENABLE_TRENDING_TOPICS = os.getenv("ENABLE_TRENDING_TOPICS", "true").lower() == "true"
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # =============================================================================
    # LANGUAGE SUPPORT
    # =============================================================================
    SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "en,es,fr,de,it,pt,ru,ja,ko,zh,hi,ar").split(",")
    FALLBACK_LANGUAGE = os.getenv("FALLBACK_LANGUAGE", "en")
    
    # =============================================================================
    # SECURITY SETTINGS
    # =============================================================================
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    @classmethod
    def get_languages_dict(cls) -> Dict[str, str]:
        """Get supported languages as display dictionary"""
        lang_names = {
            'en': '🇺🇸 English', 'es': '🇪🇸 Spanish', 'fr': '🇫🇷 French',
            'de': '🇩🇪 German', 'it': '🇮🇹 Italian', 'pt': '🇵🇹 Portuguese',
            'ru': '🇷🇺 Russian', 'ja': '🇯🇵 Japanese', 'ko': '🇰🇷 Korean',
            'zh': '🇨🇳 Chinese', 'hi': '🇮🇳 Hindi', 'ar': '🇸🇦 Arabic'
        }
        return {lang: lang_names.get(lang, f"🌐 {lang.upper()}") for lang in cls.SUPPORTED_LANGUAGES}
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return any warnings"""
        warnings = []
        
        # Check required directories
        if not cls.AUDIO_OUTPUT_DIR.exists():
            cls.AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)
            
        # Check API keys
        if not cls.HUGGINGFACE_API_KEY:
            warnings.append("HUGGINGFACE_API_KEY not set - using simple summarization")
            
        # Check language support
        if cls.DEFAULT_TTS_LANGUAGE not in cls.SUPPORTED_LANGUAGES:
            warnings.append(f"DEFAULT_TTS_LANGUAGE '{cls.DEFAULT_TTS_LANGUAGE}' not in supported languages")
            
        return warnings
    
    @classmethod
    def print_config_summary(cls):
        """Print configuration summary for debugging"""
        if cls.DEBUG_MODE:
            print("🔧 NewsNinja Configuration Summary:")
            print(f"   Backend: {cls.BACKEND_HOST}:{cls.BACKEND_PORT}")
            print(f"   Frontend: localhost:{cls.FRONTEND_PORT}")
            print(f"   Cache: {'Enabled' if cls.ENABLE_CACHE else 'Disabled'} ({cls.CACHE_DURATION_MINUTES}min)")
            print(f"   Languages: {len(cls.SUPPORTED_LANGUAGES)} supported")
            print(f"   Environment: {cls.ENVIRONMENT}")
            
            warnings = cls.validate_config()
            if warnings:
                print("⚠️  Warnings:")
                for warning in warnings:
                    print(f"   - {warning}")

# Initialize configuration
config = Config()
config.print_config_summary()