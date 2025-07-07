from typing import List
import asyncio
import requests
import json
from datetime import datetime, timedelta

def scrape_reddit_free(topic: str) -> str:
    """Scrape Reddit using free public JSON API"""
    try:
        # Reddit allows accessing JSON by adding .json to URLs
        search_url = f"https://www.reddit.com/search.json?q={topic}&sort=hot&limit=5&t=week"
        
        headers = {
            'User-Agent': 'NewsNinja/1.0 (Educational Use)'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        posts = data.get('data', {}).get('children', [])
        
        summary_parts = []
        
        if posts:
            summary_parts.append(f"Recent Reddit discussions about {topic} show:")
            
            for post in posts[:3]:  # Top 3 posts
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                score = post_data.get('score', 0)
                num_comments = post_data.get('num_comments', 0)
                
                if title:
                    summary_parts.append(f"A post titled '{title}' received {score} upvotes and {num_comments} comments.")
            
            # Simple sentiment analysis based on score
            avg_score = sum(post.get('data', {}).get('score', 0) for post in posts[:3]) / min(3, len(posts))
            if avg_score > 100:
                sentiment = "positive reception"
            elif avg_score > 10:
                sentiment = "moderate interest"
            else:
                sentiment = "mixed reactions"
                
            summary_parts.append(f"Overall sentiment appears to show {sentiment}.")
        else:
            summary_parts.append(f"Limited recent Reddit activity found for {topic}.")
            
        return " ".join(summary_parts)
        
    except Exception as e:
        return f"Error accessing Reddit data for {topic}: {str(e)}"

async def scrape_reddit_topics(topics: List[str]) -> dict:
    """Process list of topics and return analysis results"""
    reddit_results = {}
    
    for topic in topics:
        reddit_results[topic] = scrape_reddit_free(topic)
        await asyncio.sleep(3)  # Be respectful to Reddit's servers
        
    return {"reddit_analysis": reddit_results}