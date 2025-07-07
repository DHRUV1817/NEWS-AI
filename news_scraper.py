#news_scraper.py
import asyncio
import os
from typing import Dict, List
from dotenv import load_dotenv
from utils import scrape_news_free, summarize_with_free_api

load_dotenv()

class NewsScraper:
    
    async def scrape_news(self, topics: List[str]) -> Dict[str, str]:
        """Scrape and analyze news articles using free resources"""
        results = {}
        
        for topic in topics:
            try:
                # Get headlines using free RSS
                headlines = scrape_news_free(topic)
                
                if headlines and not headlines.startswith("Error"):
                    # Summarize using free API or simple processing
                    summary = summarize_with_free_api(headlines)
                    results[topic] = summary
                else:
                    results[topic] = f"No recent news found for {topic}"
                    
                # Add delay to be respectful to free services
                await asyncio.sleep(2)
                
            except Exception as e:
                results[topic] = f"Error: {str(e)}"

        return {"news_analysis": results}