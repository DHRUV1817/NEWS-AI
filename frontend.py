import streamlit as st
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="NewsNinja 2.0",
    page_icon="🥷",
    layout="wide"
)

BACKEND_URL = "http://localhost:1234"

def main():
    st.title("🥷 NewsNinja 2.0")
    st.markdown("#### *Super-Powered AI News & Social Media Analyzer*")

    # Initialize session state
    if 'topics' not in st.session_state:
        st.session_state.topics = []

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Status Check
        api_status = check_api_status()
        if api_status:
            st.success("🟢 API Connected")
            show_stats()
        else:
            st.error("🔴 API Disconnected")
            st.info("Make sure backend is running on port 1234")
            
        # Settings
        source_type = st.selectbox(
            "📊 Data Sources",
            options=["both", "news", "reddit"],
            format_func=lambda x: {"both": "🌐 News + Reddit", "news": "📰 News Only", "reddit": "💬 Reddit Only"}[x]
        )
        
        language = st.selectbox(
            "🌍 Language", 
            options=["en", "es", "fr", "de"],
            format_func=lambda x: {"en": "🇺🇸 English", "es": "🇪🇸 Spanish", "fr": "🇫🇷 French", "de": "🇩🇪 German"}[x]
        )

    # Main content
    st.subheader("📝 Topic Analysis")
    
    # Add topic
    col1, col2 = st.columns([3, 1])
    with col1:
        new_topic = st.text_input("Enter topic to analyze:", placeholder="e.g., Artificial Intelligence")
    with col2:
        if st.button("➕ Add", disabled=not new_topic.strip() or len(st.session_state.topics) >= 5):
            st.session_state.topics.append(new_topic.strip())
            st.rerun()

    # Display topics
    if st.session_state.topics:
        st.write("**Selected Topics:**")
        for i, topic in enumerate(st.session_state.topics):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📌 {topic}")
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.topics.pop(i)
                    st.rerun()

    # Generate audio
    st.markdown("---")
    if st.button("🎵 Generate Audio Summary", disabled=not st.session_state.topics, type="primary"):
        generate_audio(source_type, language)

def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def show_stats():
    """Show API statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            st.metric("Audio Files", stats.get('audio_files', 0))
    except:
        pass

def generate_audio(source_type, language):
    """Generate audio summary"""
    with st.spinner("🎵 Generating audio summary..."):
        try:
            payload = {
                "topics": st.session_state.topics,
                "source_type": source_type,
                "language": language
            }
            
            response = requests.post(f"{BACKEND_URL}/generate-news-audio", json=payload, timeout=60)
            
            if response.status_code == 200:
                st.success("🎵 Audio generated successfully!")
                st.audio(response.content, format="audio/mpeg")
                
                # Download button
                st.download_button(
                    "⬇️ Download Audio",
                    data=response.content,
                    file_name=f"newsninja-{datetime.now().strftime('%Y%m%d-%H%M')}.mp3",
                    mime="audio/mpeg",
                    type="secondary"
                )
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.headers.get('content-type', '').startswith('application/json') else response.text
                st.error(f"Audio generation failed: {error_detail}")
                
        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection Error: Could not reach the backend server")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    main()