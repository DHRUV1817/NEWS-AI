# 🥷 NewsNinja 2.0 - Super-Powered AI News Analyzer

> *Transform news consumption with AI-powered analysis, multi-language support, and intelligent audio summaries*

## ✨ Features

### 🚀 Core Capabilities
- **Multi-Source Analysis**: News + Reddit discussions in one place
- **12 Language Support**: Generate audio in English, Spanish, French, German, and more
- **Smart Caching**: Lightning-fast results with intelligent caching system
- **Sentiment Analysis**: Understand the mood around any topic
- **Audio Generation**: High-quality text-to-speech in multiple languages
- **Trending Topics**: Discover what's hot right now

### 🔧 Technical Highlights
- **Modern FastAPI Backend**: High-performance async API
- **Beautiful Streamlit UI**: Interactive web interface with charts
- **Modular Architecture**: Clean, maintainable code structure
- **Free Resources**: No API keys required for basic functionality
- **Auto Cleanup**: Intelligent file management
- **Real-time Updates**: Live data processing

## 🚀 Quick Start

### One-Click Launch
```bash
python start.py
```

That's it! The script will:
- ✅ Check and install requirements
- 📁 Set up directories
- 🚀 Start backend (localhost:1234)
- 🖥️ Launch frontend (localhost:8501)
- 🌐 Open your browser automatically

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start backend
python backend.py

# Start frontend (new terminal)
streamlit run frontend.py
```

## 📁 Project Structure

```
newsninja-main/
├── 🚀 start.py              # One-click startup
├── 🔧 backend.py            # FastAPI server
├── 🖥️ frontend.py           # Streamlit UI
├── 📊 models.py             # Data models
├── services/
│   ├── 📰 news_service.py   # News analysis
│   ├── 🎵 audio_service.py  # TTS generation
│   └── 💾 cache_service.py  # Intelligent caching
├── 📁 audio/                # Generated audio files
└── 📋 requirements.txt      # Dependencies
```

## 🎯 How to Use

### 1. Add Topics
- Enter any topic (AI, climate change, crypto, etc.)
- Use trending topics for inspiration
- Analyze up to 5 topics simultaneously

### 2. Choose Sources
- **News Only**: Latest headlines and summaries
- **Reddit Only**: Social media discussions
- **Both**: Complete picture from all sources

### 3. Select Language
- 🇺🇸 English, 🇪🇸 Spanish, 🇫🇷 French, 🇩🇪 German
- 🇮🇹 Italian, 🇵🇹 Portuguese, 🇮🇳 Hindi, 🇯🇵 Japanese
- 🇰🇷 Korean, and more!

### 4. Generate Content
- **Analyze**: Get detailed text analysis with sentiment
- **Audio**: Create professional news briefings
- **Download**: Save audio for offline listening

## 🛠️ API Endpoints

### Core Analysis
- `POST /analyze` - Analyze topics with caching
- `POST /generate-audio` - Create audio summaries
- `GET /trending` - Get trending topics

### Utilities
- `GET /health` - Check service status
- `GET /stats` - Usage statistics
- `GET /docs` - Interactive API documentation

## 🌟 Advanced Features

### Smart Caching
- 30-minute cache duration
- Automatic cleanup of expired data
- Force refresh option available

### Multi-Language TTS
- Automatic text chunking for long content
- Language-specific audio optimization
- Background file cleanup

### Sentiment Analysis
- Real-time mood detection
- Visual sentiment charts
- Topic-specific insights

## 🔧 Configuration

### Environment Variables (Optional)
```bash
# .env file
HUGGINGFACE_API_KEY="your_key_here"  # For enhanced summarization
```

### Cache Settings
- Default: 30 minutes
- Configurable in `cache_service.py`
- Manual cache clearing available

## 📊 Performance

- **Response Time**: < 3 seconds with cache
- **Languages**: 12+ supported
- **Sources**: News + Reddit
- **Concurrent Users**: Optimized for multiple users
- **Audio Quality**: High-quality MP3 output

## 🆙 What's New in 2.0

### 🎯 Enhanced Features
- ✅ Multi-language support (12 languages)
- ✅ Intelligent caching system
- ✅ Sentiment analysis with visualizations
- ✅ Trending topics discovery
- ✅ Background task processing
- ✅ One-click startup script

### 🔧 Technical Improvements
- ✅ Modular service architecture
- ✅ Enhanced error handling
- ✅ Automatic file cleanup
- ✅ Better rate limiting
- ✅ CORS support for web deployment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - feel free to use in your projects!

## 🆘 Troubleshooting

### Common Issues

**Backend won't start?**
- Check if port 1234 is available
- Run `pip install -r requirements.txt`

**Audio generation fails?**
- Ensure internet connection for gTTS
- Check audio directory permissions

**Frontend connection error?**
- Verify backend is running (localhost:1234/health)
- Clear browser cache

### Getting Help
- Check the `/docs` endpoint for API documentation
- Review logs in terminal output
- Ensure all requirements are installed

---

Made with ❤️ by NewsNinja Team | *Stay informed, stay ahead!*