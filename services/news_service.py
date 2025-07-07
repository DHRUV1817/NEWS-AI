#enhanced-tts-project\services\news_service.py
import asyncio
import requests
import feedparser
from typing import Dict, List
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import re
from datetime import datetime
from models import TopicAnalysis

class NewsService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NewsNinja/2.0 (Educational)'
        })

    async def analyze_topics(self, topics: List[str], source_type: str) -> Dict:
        """Enhanced topic analysis with sentiment"""
        results = []
        
        for topic in topics:
            analysis = TopicAnalysis(topic=topic)
            
            if source_type in ["news", "both"]:
                analysis.news_summary = await self._get_news_summary(topic)
                
            if source_type in ["reddit", "both"]:
                analysis.reddit_summary = await self._get_reddit_summary(topic)
                
            # Analyze sentiment and extract key points
            analysis.sentiment = self._analyze_sentiment(analysis.news_summary, analysis.reddit_summary)
            analysis.key_points = self._extract_key_points(analysis.news_summary, analysis.reddit_summary)
            
            results.append(analysis)
            await asyncio.sleep(1)  # Rate limiting
            
        return {"topics": results}

    async def _get_news_summary(self, topic: str) -> str:
        """Get news with better parsing"""
        try:
            url = f"https://news.google.com/rss/search?q={quote_plus(topic)}&hl=en-US&gl=US&ceid=US:en"
            response = self.session.get(url, timeout=10)
            
            feed = feedparser.parse(response.content)
            headlines = []
            
            for entry in feed.entries[:8]:
                # Clean headline
                title = BeautifulSoup(entry.title, "html.parser").get_text()
                headlines.append(title)
                
            return self._create_smart_summary(headlines, topic)
            
        except Exception as e:
            return f"News unavailable: {str(e)}"

    async def _get_reddit_summary(self, topic: str) -> str:
        """Enhanced Reddit analysis"""
        try:
            url = f"https://www.reddit.com/search.json?q={quote_plus(topic)}&sort=hot&limit=10&t=week"
            response = self.session.get(url, timeout=10)
            data = response.json()
            
            posts = data.get('data', {}).get('children', [])
            if not posts:
                return f"No Reddit discussions found for {topic}"
                
            total_score = sum(p.get('data', {}).get('score', 0) for p in posts)
            total_comments = sum(p.get('data', {}).get('num_comments', 0) for p in posts)
            
            engagement = "high" if total_score > 500 else "moderate" if total_score > 100 else "low"
            
            return f"Reddit shows {engagement} engagement with {len(posts)} discussions, {total_score} total upvotes, and {total_comments} comments about {topic}"
            
        except Exception as e:
            return f"Reddit data unavailable: {str(e)}"

    def _create_smart_summary(self, headlines: List[str], topic: str) -> str:
        """Create intelligent summary from headlines"""
        if not headlines:
            return f"No recent news for {topic}"
            
        # Count keyword occurrences
        keywords = {}
        for headline in headlines:
            words = re.findall(r'\b\w+\b', headline.lower())
            for word in words:
                if len(word) > 3 and word != topic.lower():
                    keywords[word] = keywords.get(word, 0) + 1
                    
        # Get top keywords
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary = f"Current {topic} news highlights: "
        if top_keywords:
            summary += f"Key themes include {', '.join([k[0] for k in top_keywords])}. "
            
        summary += f"Based on {len(headlines)} recent articles."
        return summary

    def _analyze_sentiment(self, news: str, reddit: str) -> str:
        """Simple sentiment analysis"""
        text = f"{news or ''} {reddit or ''}".lower()
        
        positive_words = ['good', 'great', 'positive', 'success', 'growth', 'improvement']
        negative_words = ['bad', 'negative', 'decline', 'problem', 'crisis', 'concern']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"

    def _extract_key_points(self, news: str, reddit: str) -> List[str]:
        """Extract key points from summaries"""
        points = []
        
        if news and not news.startswith("News unavailable"):
            points.append(f"📰 {news[:100]}...")
            
        if reddit and not reddit.startswith("Reddit data unavailable"):
            points.append(f"💬 {reddit[:100]}...")
            
        return points

    def create_broadcast_script(self, analysis: Dict, language: str = "en") -> str:
        """Create engaging broadcast script"""
        script_parts = [
            "Welcome to NewsNinja, your AI-powered news briefing.",
            f"Here's your analysis for {datetime.now().strftime('%B %d, %Y')}."
        ]
        
        for topic_data in analysis.get("topics", []):
            topic = topic_data.topic
            sentiment = topic_data.sentiment
            
            script_parts.append(f"Now covering {topic}.")
            
            if topic_data.news_summary:
                script_parts.append(topic_data.news_summary)
                
            if topic_data.reddit_summary:
                script_parts.append(f"Social media perspective: {topic_data.reddit_summary}")
                
            script_parts.append(f"Overall sentiment for {topic} appears {sentiment}.")
            
        script_parts.append("That concludes your NewsNinja briefing. Stay informed!")
        
        return " ".join(script_parts)

    async def get_trending_topics(self) -> List[str]:
        """Get trending topics from Google Trends"""
        try:
            # Simple trending topics - you can enhance with actual Google Trends API
            default_trending = [
                "artificial intelligence", "climate change", "cryptocurrency", 
                "space exploration", "renewable energy", "cybersecurity"
            ]
            return default_trending
        except:
            return []