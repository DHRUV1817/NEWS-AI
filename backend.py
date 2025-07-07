from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "NewsNinja API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/generate-news-audio")
async def generate_news_audio(request: dict):
    try:
        topics = request.get("topics", [])
        source_type = request.get("source_type", "both")
        language = request.get("language", "en")
        
        # Validate input
        if not topics:
            raise HTTPException(status_code=400, detail="No topics provided")
        
        # Import your existing modules
        from models import NewsRequest
        from utils import generate_broadcast_news, tts_to_audio
        from news_scraper import NewsScraper
        from reddit_scraper import scrape_reddit_topics
        
        # Create NewsRequest object
        news_request = NewsRequest(
            topics=topics,
            source_type=source_type,
            language=language
        )
        
        results = {}
        
        if source_type in ["news", "both"]:
            print(f"Scraping news for topics: {topics}")
            news_scraper = NewsScraper()
            results["news"] = await news_scraper.scrape_news(topics)
        
        if source_type in ["reddit", "both"]:
            print(f"Scraping Reddit for topics: {topics}")
            results["reddit"] = await scrape_reddit_topics(topics)

        news_data = results.get("news", {})
        reddit_data = results.get("reddit", {})
        
        print("Generating broadcast script...")
        news_summary = generate_broadcast_news(
            api_key=None,
            news_data=news_data,
            reddit_data=reddit_data,
            topics=topics
        )

        print("Converting to audio...")
        audio_path = tts_to_audio(text=news_summary, language=language)

        if audio_path and Path(audio_path).exists():
            return FileResponse(
                path=audio_path,
                media_type="audio/mpeg",
                filename="news-summary.mp3"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio file")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trending")
async def get_trending_topics():
    """Get trending topics"""
    trending = ["artificial intelligence", "climate change", "cryptocurrency", 
               "space exploration", "renewable energy", "cybersecurity"]
    return {"trending_topics": trending}

@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    audio_dir = Path("audio")
    audio_files = len(list(audio_dir.glob("*.mp3"))) if audio_dir.exists() else 0
    
    return {
        "cache_size": 0,
        "audio_files": audio_files,
        "supported_languages": ["en", "es", "fr", "de", "it", "pt", "hi", "ja", "ko"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=1234,
        reload=True
    )