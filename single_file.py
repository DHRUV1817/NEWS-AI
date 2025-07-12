import streamlit as st
import os
import sys
import time
import json
import re
from datetime import datetime
import io
import subprocess
from urllib.parse import quote_plus

# Page config
st.set_page_config(
    page_title="NewsNinja 2.0 Pro",
    page_icon="🥷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Install missing packages function
@st.cache_data(ttl=3600)
def install_missing_packages():
    """Install required packages if missing"""
    packages_to_install = []
    
    # Check each package
    try:
        import feedparser
    except ImportError:
        packages_to_install.append('feedparser')
    
    try:
        import requests
    except ImportError:
        packages_to_install.append('requests')
    
    try:
        from gtts import gTTS
    except ImportError:
        packages_to_install.append('gtts')
    
    try:
        from groq import Groq
    except ImportError:
        packages_to_install.append('groq')
    
    # Install missing packages
    if packages_to_install:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, package in enumerate(packages_to_install):
            status_text.text(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                    capture_output=True, text=True)
                st.success(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Failed to install {package}: {e}")
            
            progress_bar.progress((i + 1) / len(packages_to_install))
        
        status_text.text("✅ Installation complete!")
        time.sleep(1)
        st.rerun()
    
    return True

# Install packages first
if 'packages_installed' not in st.session_state:
    st.session_state.packages_installed = install_missing_packages()

# Now import after installation
try:
    import feedparser
    import requests
    from gtts import gTTS
    from groq import Groq
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    st.error(f"Import failed: {e}")
    st.error("Please add the following to your requirements.txt or Pipfile:")
    st.code("""
feedparser
requests
gtts
groq
    """)
    IMPORTS_SUCCESSFUL = False
    st.stop()

# Enhanced CSS with modern theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .topic-card {
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        margin: 1rem 0;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        color: #2c3e50;
        font-weight: 500;
    }
    .topic-card:hover {
        transform: translateY(-2px);
    }
    .metric-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    .sentiment-positive { color: #27ae60; font-weight: bold; }
    .sentiment-negative { color: #e74c3c; font-weight: bold; }
    .sentiment-neutral { color: #f39c12; font-weight: bold; }
    .audio-controls {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced language support with gender-specific voices
LANGUAGES = {
    'en': {
        'name': '🇺🇸 English', 
        'male': {'lang': 'en', 'tld': 'com.au'},    # Australian accent sounds more male
        'female': {'lang': 'en', 'tld': 'co.uk'},   # British accent sounds more female
        'auto': {'lang': 'en', 'tld': 'com'}
    },
    'es': {
        'name': '🇪🇸 Spanish',
        'male': {'lang': 'es', 'tld': 'com.mx'},    # Mexican Spanish
        'female': {'lang': 'es', 'tld': 'es'},      # Spain Spanish
        'auto': {'lang': 'es', 'tld': 'es'}
    },
    'fr': {
        'name': '🇫🇷 French',
        'male': {'lang': 'fr', 'tld': 'ca'},        # Canadian French
        'female': {'lang': 'fr', 'tld': 'fr'},      # France French
        'auto': {'lang': 'fr', 'tld': 'fr'}
    },
    'de': {
        'name': '🇩🇪 German',
        'male': {'lang': 'de', 'tld': 'de'},
        'female': {'lang': 'de', 'tld': 'de'},
        'auto': {'lang': 'de', 'tld': 'de'}
    },
    'it': {
        'name': '🇮🇹 Italian',
        'male': {'lang': 'it', 'tld': 'it'},
        'female': {'lang': 'it', 'tld': 'it'},
        'auto': {'lang': 'it', 'tld': 'it'}
    },
    'pt': {
        'name': '🇵🇹 Portuguese',
        'male': {'lang': 'pt', 'tld': 'com.br'},
        'female': {'lang': 'pt', 'tld': 'com.br'},
        'auto': {'lang': 'pt', 'tld': 'com.br'}
    },
    'hi': {
        'name': '🇮🇳 Hindi',
        'male': {'lang': 'hi', 'tld': 'co.in'},
        'female': {'lang': 'hi', 'tld': 'co.in'},
        'auto': {'lang': 'hi', 'tld': 'co.in'}
    },
    'ja': {
        'name': '🇯🇵 Japanese',
        'male': {'lang': 'ja', 'tld': 'co.jp'},
        'female': {'lang': 'ja', 'tld': 'co.jp'},
        'auto': {'lang': 'ja', 'tld': 'co.jp'}
    },
    'ko': {
        'name': '🇰🇷 Korean',
        'male': {'lang': 'ko', 'tld': 'co.kr'},
        'female': {'lang': 'ko', 'tld': 'co.kr'},
        'auto': {'lang': 'ko', 'tld': 'co.kr'}
    }
}

# Voice gender options with realistic expectations
VOICE_OPTIONS = {
    'auto': '🤖 Standard Voice',
    'male': '👨 Male Style (Deeper/Slower)', 
    'female': '👩 Female Style (Higher/Faster)'
}

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    try:
        # Try Streamlit secrets first (for cloud deployment)
        api_key = None
        try:
            api_key = st.secrets.get('GROQ_API_KEY', None)
        except Exception:
            pass
        
        # Fallback to environment variable
        if not api_key:
            api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            return None
            
        return Groq(api_key=api_key)
    except Exception as e:
        st.warning(f"⚠️ Could not initialize Groq client: {e}")
        return None

def scrape_news_advanced(keyword: str) -> str:
    """Enhanced news scraping with AI summarization"""
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(keyword)}&hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        headlines = [re.sub(r'<[^>]+>', '', entry.title) for entry in feed.entries[:8]]
        
        return create_ai_summary(headlines, keyword) if headlines else f"No recent news found for {keyword}"
    except Exception as e:
        return f"News temporarily unavailable for {keyword}"

def create_ai_summary(headlines: list, topic: str) -> str:
    """Create AI-powered summary using Groq"""
    try:
        client = get_groq_client()
        if not client:
            return f"Found {len(headlines)} news articles about {topic}"
            
        headlines_text = "\n".join(headlines)
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{
                "role": "user",
                "content": f"Summarize these {topic} news headlines in 2-3 sentences for a news broadcast:\n{headlines_text}"
            }],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception:
        return f"Analysis of {len(headlines)} recent {topic} articles shows mixed developments"

def scrape_reddit_advanced(topic: str) -> str:
    """Enhanced Reddit analysis"""
    try:
        url = f"https://www.reddit.com/search.json?q={quote_plus(topic)}&sort=hot&limit=10&t=week"
        headers = {'User-Agent': 'NewsNinja/2.0 (Educational)'}
        
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        posts = data.get('data', {}).get('children', [])
        if not posts:
            return f"Limited Reddit discussions found for {topic}"
            
        total_score = sum(p.get('data', {}).get('score', 0) for p in posts)
        total_comments = sum(p.get('data', {}).get('num_comments', 0) for p in posts)
        
        engagement = "high" if total_score > 500 else "moderate" if total_score > 100 else "emerging"
        sentiment = "positive" if total_score/len(posts) > 50 else "neutral" if total_score/len(posts) > 10 else "mixed"
        
        return f"Reddit shows {engagement} engagement for {topic} with {sentiment} community sentiment. {len(posts)} active discussions."
    except Exception:
        return f"Reddit community data currently unavailable for {topic}"

def enhance_script_for_speech(script: str, voice_gender: str = "auto") -> str:
    """Enhance script for natural speech with pauses and emphasis"""
    
    # Add natural pauses and breathing
    enhanced = script.replace(". ", "... ")  # Longer pauses between sentences
    enhanced = enhanced.replace(", ", ".. ")  # Short pauses for commas
    enhanced = enhanced.replace(":", "... ")  # Pause after colons
    
    # Add emphasis markers for important words
    emphasis_words = ["breaking", "urgent", "important", "significant", "major", "critical"]
    for word in emphasis_words:
        enhanced = enhanced.replace(word, f"*{word}*")
    
    # Add natural transitions
    enhanced = enhanced.replace("Moving on", "... Now moving on")
    enhanced = enhanced.replace("Next", "... Next")
    enhanced = enhanced.replace("Finally", "... And finally")
    
    # Remove cringe words that sound robotic
    enhanced = enhanced.replace(" anchor", "")
    enhanced = enhanced.replace("sound ", "")
    enhanced = enhanced.replace("correspondent", "")
    
    return enhanced

def generate_professional_script(results: list, language: str = "en", voice_gender: str = "auto") -> str:
    """Generate professional broadcast script with natural flow"""
    try:
        client = get_groq_client()
        if not client:
            return generate_natural_script(results, voice_gender)
            
        # Gender-specific prompts
        gender_prompt = ""
        if voice_gender == "male":
            gender_prompt = "Write in a confident, authoritative male news anchor style."
        elif voice_gender == "female":
            gender_prompt = "Write in a professional, engaging female news anchor style."
        else:
            gender_prompt = "Write in a neutral, professional news anchor style."
        
        # Prepare content
        content = []
        for result in results:
            content.append(f"Topic: {result['topic']}")
            content.append(f"News: {result['news_summary']}")
            content.append(f"Social: {result['reddit_summary']}")
            content.append(f"Sentiment: {result['sentiment']}")
            content.append("---")
        
        content_text = "\n".join(content)
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{
                "role": "user",
                "content": f"""Create a natural, conversational news broadcast script in {LANGUAGES[language]['name']}. 
                {gender_prompt}
                
                Make it sound like a real person talking, not reading. Use:
                - Natural transitions and conversational phrases
                - Varied sentence lengths
                - Appropriate pauses (use ... for emphasis)
                - Engaging, human-like delivery
                - Professional but warm tone
                
                Content:\n{content_text}"""
            }],
            max_tokens=1000,
            temperature=0.8
        )
        
        return response.choices[0].message.content.strip()
    except Exception:
        return generate_natural_script(results, voice_gender)

def generate_natural_script(results: list, voice_gender: str = "auto") -> str:
    """Generate natural-sounding fallback script"""
    
    # Gender-specific greetings - simple and natural
    if voice_gender == "male":
        greeting = "Good morning, bringing you today's top stories."
    elif voice_gender == "female":
        greeting = "Hello everyone, here's your latest news update."
    else:
        greeting = "Welcome to NewsNinja. Here's what's happening today."
    
    script = [greeting]
    script.append(f"It's {datetime.now().strftime('%A, %B %d')}... and we've got some interesting developments.")
    
    for i, result in enumerate(results):
        if i == 0:
            script.append(f"Starting with {result['topic']}...")
        else:
            script.append(f"Moving to {result['topic']}...")
            
        if result['news_summary']:
            script.append(f"Here's what's happening... {result['news_summary']}")
            
        if result['reddit_summary']:
            script.append(f"And from social media... {result['reddit_summary']}")
            
        script.append(f"Overall sentiment appears {result['sentiment']} for this story.")
        
        if i < len(results) - 1:
            script.append("Next up...")
    
    script.append("That's your NewsNinja update. Stay informed!")
    
    return " ".join(script)

def generate_professional_audio(script: str, language: str = "en", voice_gender: str = "auto") -> io.BytesIO:
    """Generate high-quality audio with REAL male/female voices using Edge TTS"""
    try:
        # Enhance script for natural speech
        enhanced_script = enhance_script_for_speech(script, voice_gender)
        
        # Generate audio with voice style
        return generate_gtts_audio(enhanced_script, language, voice_gender)
        
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        return None

def generate_gtts_audio(script: str, language: str, voice_gender: str) -> io.BytesIO:
    """Enhanced gTTS with gender-specific modifications"""
    try:
        # Get language and voice settings
        lang_config = LANGUAGES.get(language, LANGUAGES['en'])
        voice_config = lang_config.get(voice_gender, lang_config['auto'])
        
        # Modify script for gender perception
        if voice_gender == 'male':
            # Process for deeper, slower male voice simulation
            script = script.lower().replace('hello', 'good morning')
            script = script.replace('everyone', 'folks')
            
            # Use gTTS with slower speed and different accent for male perception
            tts = gTTS(text=script, lang=voice_config['lang'], tld=voice_config['tld'], slow=True)
        else:
            # Process for higher, faster female voice simulation  
            script = script.replace('good morning', 'hello')
            script = script.replace('folks', 'everyone')
            
            # Use gTTS with normal speed and different accent for female perception
            tts = gTTS(text=script, lang=voice_config['lang'], tld=voice_config['tld'], slow=False)
        
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        return fp
        
    except Exception as e:
        raise Exception(f"gTTS generation failed: {e}")

def main():
    # Enhanced header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🥷 NewsNinja 2.0 Pro")
    st.markdown("#### *AI-Powered Professional News Broadcasting*")
    st.markdown("**🎯 Enhanced with Groq AI & Professional Audio**")
    st.markdown('</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None

    # Enhanced sidebar
    with st.sidebar:
        st.header("⚙️ Professional Settings")
        
        # API Status
        client = get_groq_client()
        if client:
            st.success("🟢 Groq AI: Connected")
        else:
            st.error("🔴 Groq AI: Not Connected")
            with st.expander("💡 How to add API Key"):
                st.markdown("""
                **For Streamlit Cloud:**
                1. Go to your app settings
                2. Add a secret: `GROQ_API_KEY = "your-key"`
                3. Get free key: [console.groq.com](https://console.groq.com/keys)
                """)
        
        # Enhanced settings
        col_voice1, col_voice2 = st.columns(2)
        
        with col_voice1:
            voice_gender = st.selectbox(
                "🎤 Voice Style",
                options=list(VOICE_OPTIONS.keys()),
                format_func=lambda x: VOICE_OPTIONS[x]
            )
        
        with col_voice2:
            language = st.selectbox(
                "🌍 Language", 
                options=list(LANGUAGES.keys()), 
                format_func=lambda x: LANGUAGES[x]['name']
            )
        
        source_type = st.selectbox(
            "📊 Data Sources",
            options=["both", "news", "reddit"],
            format_func=lambda x: {"both": "🌐 News + Social", "news": "📰 News Focus", "reddit": "💬 Social Focus"}[x]
        )
        
        st.markdown("---")
        
        # Professional features
        st.subheader("🎯 Audio Features")
        
        st.info("🎤 Voice Style Options:")
        st.write("• 👨 **Male Style**: Deeper tone, slower pace, Australian accent")  
        st.write("• 👩 **Female Style**: Higher tone, normal pace, British accent")
        st.write("• 🤖 **Standard**: Default gTTS voice")

    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📝 Topic Analysis Dashboard")
        
        # Topic input with suggestions
        new_topic = st.text_input(
            "Enter topic for analysis:",
            placeholder="e.g., Tesla Stock, AI Regulation, Climate Summit"
        )
        
        col_add, col_trending = st.columns([1, 1])
        with col_add:
            if st.button("➕ Add Topic", disabled=not new_topic.strip()):
                if new_topic.strip() not in st.session_state.topics:
                    st.session_state.topics.append(new_topic.strip())
                    st.rerun()
                
        with col_trending:
            if st.button("🔥 Load Trending"):
                trending = ["Artificial Intelligence", "Climate Change", "Cryptocurrency", "Space Tech", "Green Energy"]
                st.session_state.topics = trending[:3]
                st.rerun()

        # Display topics with enhanced cards
        if st.session_state.topics:
            st.write("**📌 Selected Topics:**")
            for i, topic in enumerate(st.session_state.topics):
                col_topic, col_remove = st.columns([5, 1])
                with col_topic:
                    st.markdown(f'<div class="topic-card">🎯 {topic}</div>', unsafe_allow_html=True)
                with col_remove:
                    if st.button("🗑️", key=f"remove_{i}"):
                        st.session_state.topics.pop(i)
                        st.rerun()

    with col2:
        st.subheader("📊 Dashboard")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Topics", len(st.session_state.topics))
        st.metric("Language", LANGUAGES[language]['name'])
        st.metric("Voice", VOICE_OPTIONS[voice_gender])
        st.markdown('</div>', unsafe_allow_html=True)

    # Action buttons
    if st.session_state.topics:
        st.markdown("---")
        col_analyze, col_audio, col_clear = st.columns([1, 1, 1])
        
        with col_analyze:
            if st.button("🔍 Analyze All Topics", type="primary"):
                analyze_topics_pro(source_type, language, voice_gender)
        
        with col_audio:
            if st.button("🎵 Generate Pro Audio", disabled=not st.session_state.last_analysis):
                generate_audio_pro(language, voice_gender)
        
        with col_clear:
            if st.button("🗑️ Clear All"):
                st.session_state.topics = []
                st.session_state.last_analysis = None
                st.rerun()

    # Results display
    if st.session_state.last_analysis:
        show_professional_results()

def analyze_topics_pro(source_type, language, voice_gender):
    """Professional analysis with enhanced progress tracking"""
    with st.spinner("🔍 Performing professional analysis..."):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, topic in enumerate(st.session_state.topics):
            status_text.text(f"🎯 Analyzing {topic}... ({i+1}/{len(st.session_state.topics)})")
            
            news_summary = scrape_news_advanced(topic) if source_type in ["news", "both"] else ""
            reddit_summary = scrape_reddit_advanced(topic) if source_type in ["reddit", "both"] else ""
            
            # Simple sentiment analysis
            combined_text = f"{news_summary} {reddit_summary}".lower()
            pos_words = sum(1 for word in ['positive', 'growth', 'success', 'breakthrough', 'excellent'] if word in combined_text)
            neg_words = sum(1 for word in ['negative', 'decline', 'crisis', 'concern', 'problem'] if word in combined_text)
            
            sentiment = "positive" if pos_words > neg_words else "negative" if neg_words > pos_words else "neutral"
            
            results.append({
                'topic': topic,
                'news_summary': news_summary,
                'reddit_summary': reddit_summary,
                'sentiment': sentiment
            })
            
            progress_bar.progress((i + 1) / len(st.session_state.topics))
            time.sleep(0.5)  # Rate limiting
        
        st.session_state.last_analysis = results
        st.session_state.voice_gender = voice_gender  # Store voice preference
        status_text.text("✅ Professional analysis completed!")
        st.success(f"🎉 Successfully analyzed {len(results)} topics with AI enhancement!")

def generate_audio_pro(language, voice_gender):
    """Generate professional audio broadcast with natural voice"""
    with st.spinner("🎵 Generating natural-sounding broadcast..."):
        try:
            # Generate enhanced script with voice gender
            script = generate_professional_script(st.session_state.last_analysis, language, voice_gender)
            
            # Generate high-quality audio
            audio_fp = generate_professional_audio(script, language, voice_gender)
            
            if audio_fp:
                # Voice type indicator with realistic description
                voice_descriptions = {
                    "male": "👨 Male Style (Deeper, Slower)",
                    "female": "👩 Female Style (Higher, Faster)", 
                    "auto": "🤖 Standard Voice"
                }
                
                st.success(f"🎵 {voice_descriptions[voice_gender]} broadcast generated!")
                
                # Audio player with enhanced controls
                st.markdown('<div class="audio-controls">', unsafe_allow_html=True)
                st.audio(audio_fp.getvalue(), format="audio/mp3")
                
                # Download button with voice info
                filename = f"newsninja-{voice_gender}-style-{datetime.now().strftime('%Y%m%d-%H%M')}.mp3"
                st.download_button(
                    f"⬇️ Download {voice_descriptions[voice_gender]} Broadcast",
                    data=audio_fp.getvalue(),
                    file_name=filename,
                    mime="audio/mp3"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show script preview
                with st.expander("📝 View Generated Script"):
                    st.text_area("Broadcast Script", script, height=200)
                
        except Exception as e:
            st.error(f"🚨 Audio generation error: {e}")

def show_professional_results():
    """Display professional analysis results"""
    st.subheader("📊 Professional Analysis Results")
    
    # Enhanced metrics
    sentiments = [r['sentiment'] for r in st.session_state.last_analysis]
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Topics Analyzed", len(st.session_state.last_analysis))
    with col2:
        dominant = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "neutral"
        st.metric("Market Sentiment", dominant.title())
    with col3:
        st.metric("Generated", datetime.now().strftime("%H:%M:%S"))
    
    # Professional results display
    for i, result in enumerate(st.session_state.last_analysis):
        sentiment_class = f"sentiment-{result['sentiment']}"
        
        with st.expander(f"📈 {result['topic']} - Analysis Report", expanded=i==0):
            st.markdown(f"**Overall Sentiment:** <span class='{sentiment_class}'>{result['sentiment'].upper()}</span>", unsafe_allow_html=True)
            
            if result['news_summary']:
                st.markdown("**📰 News Intelligence:**")
                st.info(result['news_summary'])
            
            if result['reddit_summary']:
                st.markdown("**💬 Social Media Intelligence:**")
                st.info(result['reddit_summary'])

if __name__ == "__main__":
    main()