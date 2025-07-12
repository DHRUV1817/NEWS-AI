import streamlit as st
import requests
from urllib.parse import quote_plus
import feedparser
from gtts import gTTS
from datetime import datetime
import io
import time
import asyncio
from typing import List, Dict
import json
import re

# Page config
st.set_page_config(
    page_title="NewsNinja 2.0",
    page_icon="🥷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .topic-card {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
        border-radius: 5px;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Language options
LANGUAGES = {
    'en': '🇺🇸 English', 'es': '🇪🇸 Spanish', 'fr': '🇫🇷 French',
    'de': '🇩🇪 German', 'it': '🇮🇹 Italian', 'pt': '🇵🇹 Portuguese',
    'hi': '🇮🇳 Hindi', 'ja': '🇯🇵 Japanese', 'ko': '🇰🇷 Korean'
}

# =============================================================================
# BACKEND FUNCTIONALITY (Built into Streamlit)
# =============================================================================

def scrape_news_advanced(keyword: str) -> str:
    """Advanced news scraping with better parsing"""
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(keyword)}&hl=en-US&gl=US&ceid=US:en"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse RSS feed
        feed = feedparser.parse(response.content)
        headlines = []
        
        for entry in feed.entries[:8]:
            # Clean HTML from title
            title = re.sub(r'<[^>]+>', '', entry.title)
            headlines.append(title)
            
        if headlines:
            return create_smart_summary(headlines, keyword)
        else:
            return f"No recent news found for {keyword}"
            
    except Exception as e:
        return f"News temporarily unavailable for {keyword}"

def create_smart_summary(headlines: List[str], topic: str) -> str:
    """Create intelligent summary with keyword analysis"""
    if not headlines:
        return f"No recent news for {topic}"
        
    # Extract keywords using simple TF-IDF approach
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
        
    summary += f"Analysis based on {len(headlines)} recent articles."
    return summary

def scrape_reddit_advanced(topic: str) -> str:
    """Advanced Reddit analysis with engagement metrics"""
    try:
        url = f"https://www.reddit.com/search.json?q={quote_plus(topic)}&sort=hot&limit=10&t=week"
        headers = {'User-Agent': 'NewsNinja/2.0 (Educational)'}
        
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        posts = data.get('data', {}).get('children', [])
        
        if not posts:
            return f"No Reddit discussions found for {topic}"
            
        total_score = sum(p.get('data', {}).get('score', 0) for p in posts)
        total_comments = sum(p.get('data', {}).get('num_comments', 0) for p in posts)
        
        # Advanced engagement analysis
        if total_score > 500:
            engagement = "high"
        elif total_score > 100:
            engagement = "moderate"
        else:
            engagement = "low"
            
        # Sentiment based on engagement
        avg_score = total_score / len(posts) if posts else 0
        if avg_score > 50:
            sentiment = "positive"
        elif avg_score > 10:
            sentiment = "neutral"
        else:
            sentiment = "mixed"
            
        return f"Reddit shows {engagement} engagement for {topic} with {sentiment} sentiment. {len(posts)} discussions, {total_score} upvotes, {total_comments} comments total."
        
    except Exception as e:
        return f"Reddit data currently unavailable for {topic}"

def analyze_sentiment_advanced(news: str, reddit: str) -> str:
    """Advanced sentiment analysis"""
    text = f"{news or ''} {reddit or ''}".lower()
    
    positive_words = ['good', 'great', 'positive', 'success', 'growth', 'improvement', 'breakthrough', 'excellent', 'amazing', 'wonderful']
    negative_words = ['bad', 'negative', 'decline', 'problem', 'crisis', 'concern', 'failure', 'terrible', 'awful', 'disaster']
    
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    # Weighted analysis
    if pos_count > neg_count + 1:
        return "positive"
    elif neg_count > pos_count + 1:
        return "negative"
    return "neutral"

def generate_broadcast_script(results: List[Dict], language: str = "en") -> str:
    """Generate professional broadcast script"""
    script_parts = [
        "Welcome to NewsNinja, your AI-powered news briefing.",
        f"Here's your comprehensive analysis for {datetime.now().strftime('%B %d, %Y')}."
    ]
    
    for result in results:
        topic = result['topic']
        script_parts.append(f"Now covering {topic}.")
        
        if result['news_summary']:
            script_parts.append(result['news_summary'])
            
        if result['reddit_summary']:
            script_parts.append(f"Social media perspective: {result['reddit_summary']}")
            
        script_parts.append(f"Overall sentiment for {topic} appears {result['sentiment']}.")
        
    script_parts.append("That concludes your NewsNinja briefing. Stay informed!")
    
    return " ".join(script_parts)

# =============================================================================
# FRONTEND FUNCTIONALITY
# =============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🥷 NewsNinja 2.0")
    st.markdown("#### *Super-Powered AI News & Social Media Analyzer*")
    st.markdown("**🚀 Standalone Edition - Full Backend Integration**")
    st.markdown('</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.success("🟢 Integrated Backend - Online!")
        
        # Settings
        source_type = st.selectbox(
            "📊 Data Sources",
            options=["both", "news", "reddit"],
            format_func=lambda x: {"both": "🌐 News + Reddit", "news": "📰 News Only", "reddit": "💬 Reddit Only"}[x]
        )
        
        language = st.selectbox("🌍 Language", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
        
        st.markdown("---")
        
        # Features
        st.subheader("🎯 Features")
        st.info("✅ Real-time news scraping")
        st.info("✅ Reddit sentiment analysis") 
        st.info("✅ Multi-language TTS")
        st.info("✅ AI keyword extraction")
        st.info("✅ Advanced sentiment analysis")
        st.info("✅ Professional audio generation")
        
        # Trending topics
        if st.button("📈 Load Trending"):
            trending = ["Artificial Intelligence", "Climate Change", "Cryptocurrency", "Space Exploration", "Renewable Energy"]
            st.session_state.topics = trending[:3]
            st.rerun()

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Topic management
        st.subheader("📝 Topic Analysis")
        
        # Add topic
        new_topic = st.text_input(
            "Enter topic to analyze:",
            placeholder="e.g., Artificial Intelligence, Climate Change"
        )
        
        col_add, col_clear = st.columns([1, 1])
        with col_add:
            if st.button("➕ Add Topic", disabled=not new_topic.strip() or len(st.session_state.topics) >= 5):
                st.session_state.topics.append(new_topic.strip())
                st.rerun()
                
        with col_clear:
            if st.button("🗑️ Clear All", disabled=not st.session_state.topics):
                st.session_state.topics = []
                st.session_state.last_analysis = None
                st.rerun()

        # Display topics
        if st.session_state.topics:
            st.write("**Selected Topics:**")
            for i, topic in enumerate(st.session_state.topics):
                col_topic, col_remove = st.columns([4, 1])
                with col_topic:
                    st.markdown(f'<div class="topic-card">📌 {topic}</div>', unsafe_allow_html=True)
                with col_remove:
                    if st.button("❌", key=f"remove_{i}"):
                        st.session_state.topics.pop(i)
                        st.rerun()

    with col2:
        # Quick stats
        st.subheader("📊 Session Stats")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Topics", len(st.session_state.topics))
        with col_b:
            analyzed = len(st.session_state.last_analysis) if st.session_state.last_analysis else 0
            st.metric("Analyzed", analyzed)

    # Analysis section
    st.markdown("---")
    
    col_analyze, col_audio = st.columns([1, 1])
    
    with col_analyze:
        if st.button("🔍 Analyze Topics", disabled=not st.session_state.topics, type="primary"):
            analyze_topics_comprehensive(source_type, language)
            
    with col_audio:
        if st.button("🎵 Generate Audio", disabled=not st.session_state.last_analysis):
            generate_audio_summary(language)

    # Results section
    if st.session_state.last_analysis:
        show_analysis_results()

def analyze_topics_comprehensive(source_type, language):
    """Comprehensive topic analysis with progress tracking"""
    
    with st.spinner("🔍 Performing comprehensive analysis..."):
        results = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = len(st.session_state.topics) * 2  # News + Reddit for each topic
        current_step = 0
        
        for topic in st.session_state.topics:
            status_text.text(f"🔍 Analyzing {topic}...")
            
            news_summary = ""
            reddit_summary = ""
            
            # News analysis
            if source_type in ["news", "both"]:
                status_text.text(f"📰 Fetching news for {topic}...")
                news_summary = scrape_news_advanced(topic)
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                time.sleep(1)  # Rate limiting
            
            # Reddit analysis
            if source_type in ["reddit", "both"]:
                status_text.text(f"💬 Analyzing Reddit discussions for {topic}...")
                reddit_summary = scrape_reddit_advanced(topic)
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                time.sleep(1)  # Rate limiting
                
            # Sentiment analysis
            sentiment = analyze_sentiment_advanced(news_summary, reddit_summary)
            
            # Extract key points
            key_points = []
            if news_summary and not news_summary.startswith("News temporarily"):
                key_points.append(f"📰 {news_summary[:100]}...")
            if reddit_summary and not reddit_summary.startswith("Reddit data"):
                key_points.append(f"💬 {reddit_summary[:100]}...")
            
            results.append({
                'topic': topic,
                'news_summary': news_summary,
                'reddit_summary': reddit_summary,
                'sentiment': sentiment,
                'key_points': key_points
            })
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis completed!")
        
        # Store results
        st.session_state.last_analysis = results
        
        st.success(f"🎉 Successfully analyzed {len(results)} topics!")

def generate_audio_summary(language):
    """Generate professional audio summary"""
    
    with st.spinner("🎵 Generating professional audio briefing..."):
        try:
            # Generate script
            script = generate_broadcast_script(st.session_state.last_analysis, language)
            
            # Generate audio with progress
            progress_bar = st.progress(0)
            progress_bar.progress(0.3)
            
            # Create TTS
            tts = gTTS(text=script, lang=language, slow=False)
            fp = io.BytesIO()
            
            progress_bar.progress(0.7)
            
            tts.write_to_fp(fp)
            fp.seek(0)
            
            progress_bar.progress(1.0)
            
            st.success("🎵 Audio briefing generated successfully!")
            st.audio(fp.getvalue(), format="audio/mp3")
            
            # Download button
            st.download_button(
                "⬇️ Download Audio Briefing",
                data=fp.getvalue(),
                file_name=f"newsninja-briefing-{datetime.now().strftime('%Y%m%d-%H%M')}.mp3",
                mime="audio/mp3",
                type="secondary"
            )
            
        except Exception as e:
            st.error(f"🚨 Audio generation error: {e}")

def show_analysis_results():
    """Display comprehensive analysis results"""
    
    st.subheader("📊 Analysis Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    sentiments = [r['sentiment'] for r in st.session_state.last_analysis]
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
    
    with col1:
        st.metric("Topics Analyzed", len(st.session_state.last_analysis))
    with col2:
        most_common = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "None"
        st.metric("Dominant Sentiment", most_common.title())
    with col3:
        st.metric("Generated", datetime.now().strftime("%H:%M"))
    
    # Detailed results
    st.markdown("### 📋 Detailed Analysis")
    
    for i, result in enumerate(st.session_state.last_analysis):
        with st.expander(f"📌 {result['topic']} - {result['sentiment'].title()} Sentiment", expanded=i==0):
            
            # Sentiment indicator
            sentiment_colors = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}
            st.markdown(f"**Overall Sentiment:** {sentiment_colors.get(result['sentiment'], '🟡')} {result['sentiment'].title()}")
            
            # News analysis
            if result['news_summary']:
                st.markdown("**📰 News Analysis:**")
                st.info(result['news_summary'])
            
            # Reddit analysis
            if result['reddit_summary']:
                st.markdown("**💬 Social Media Analysis:**")
                st.info(result['reddit_summary'])
            
            # Key insights
            if result['key_points']:
                st.markdown("**🔑 Key Insights:**")
                for point in result['key_points']:
                    st.markdown(f"• {point}")

if __name__ == "__main__":
    main()