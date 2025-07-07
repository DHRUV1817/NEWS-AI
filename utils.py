from urllib.parse import quote_plus
from dotenv import load_dotenv
import requests
import os
from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from gtts import gTTS
import feedparser
import time

load_dotenv()

class MCPOverloadedError(Exception):
    """Custom exception for MCP service overloads"""
    pass

def generate_valid_news_url(keyword: str) -> str:
    """Generate a Google News RSS URL for a keyword"""
    q = quote_plus(keyword)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

def scrape_news_free(keyword: str) -> str:
    """Scrape news using free Google News RSS"""
    try:
        url = generate_valid_news_url(keyword)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse RSS feed
        feed = feedparser.parse(response.content)
        headlines = []
        
        for entry in feed.entries[:10]:  # Get top 10 headlines
            headlines.append(entry.title)
            
        return "\n".join(headlines)
        
    except Exception as e:
        return f"Error fetching news for {keyword}: {str(e)}"

def summarize_with_free_api(headlines: str) -> str:
    """Summarize using free Hugging Face API"""
    try:
        # Using free Hugging Face Inference API
        api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY', '')}"}
        
        # Truncate if too long
        max_length = 1000
        if len(headlines) > max_length:
            headlines = headlines[:max_length]
            
        payload = {"inputs": headlines}
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('summary_text', headlines)
        
        # Fallback to simple processing if API fails
        return create_simple_summary(headlines)
        
    except Exception as e:
        return create_simple_summary(headlines)

def create_simple_summary(headlines: str) -> str:
    """Create a simple summary from headlines"""
    lines = [line.strip() for line in headlines.split('\n') if line.strip()]
    
    if not lines:
        return "No news available"
        
    # Take first 5 headlines and format them
    top_headlines = lines[:5]
    summary = "Here are today's top news stories. "
    
    for i, headline in enumerate(top_headlines, 1):
        summary += f"Story {i}: {headline}. "
        
    summary += "That concludes our news summary."
    return summary

def generate_broadcast_news(api_key, news_data, reddit_data, topics):
    """Generate broadcast news using available data"""
    try:
        broadcast_segments = []
        
        for topic in topics:
            news_content = news_data.get("news_analysis", {}).get(topic, '') if news_data else ''
            reddit_content = reddit_data.get("reddit_analysis", {}).get(topic, '') if reddit_data else ''
            
            segment = f"Now reporting on {topic}. "
            
            if news_content and not news_content.startswith("Error"):
                segment += f"According to recent reports, {news_content} "
                
            if reddit_content and not reddit_content.startswith("Error"):
                segment += f"Meanwhile, online discussions reveal {reddit_content} "
                
            if not news_content and not reddit_content:
                segment += f"No recent updates available for this topic. "
                
            broadcast_segments.append(segment)
            
        final_script = " ".join(broadcast_segments)
        final_script += " This concludes our news update."
        
        return final_script
        
    except Exception as e:
        return f"Error generating broadcast: {str(e)}"

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

def tts_to_audio(text: str, language: str = 'en') -> str:
    """Convert text to speech using gTTS with language support"""
    try:
        # Validate language - fallback to English if unsupported
        supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'hi', 'ar']
        if language not in supported_languages:
            language = 'en'
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = AUDIO_DIR / f"tts_{language}_{timestamp}.mp3"
        
        # Create TTS object and save
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(str(filename))
        
        return str(filename)
    except Exception as e:
        print(f"gTTS Error: {str(e)}")
        # Fallback to English if language fails
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = AUDIO_DIR / f"tts_en_{timestamp}.mp3"
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(str(filename))
            return str(filename)
        except:
            return None

# Keep the original ElevenLabs function as fallback but make it optional
def text_to_audio_elevenlabs_sdk(text: str, **kwargs) -> str:
    """Fallback to gTTS if ElevenLabs not available"""
    language = kwargs.get('language', 'en')
    return tts_to_audio(text, language)