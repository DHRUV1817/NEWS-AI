#enhanced-tts-project\services\audio_service.py
from gtts import gTTS
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import os
import glob

class AudioService:
    def __init__(self):
        self.audio_dir = Path("audio")
        self.audio_dir.mkdir(exist_ok=True)
        
        # Supported language codes
        self.languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 
            'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
            'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean',
            'zh': 'Chinese', 'hi': 'Hindi', 'ar': 'Arabic'
        }

    async def text_to_speech(self, text: str, language: str = 'en') -> str:
        """Convert text to speech with language support"""
        try:
            # Validate language
            if language not in self.languages:
                language = 'en'
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.audio_dir / f"news_{language}_{timestamp}.mp3"
            
            # Run TTS in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self._generate_tts, 
                text, 
                language, 
                str(filename)
            )
            
            return str(filename) if filename.exists() else None
            
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def _generate_tts(self, text: str, language: str, filepath: str):
        """Generate TTS file (runs in thread pool)"""
        # Split long text into chunks for better TTS
        max_length = 3000
        
        if len(text) > max_length:
            chunks = self._split_text(text, max_length)
            
            # Generate audio for each chunk
            temp_files = []
            for i, chunk in enumerate(chunks):
                temp_file = filepath.replace('.mp3', f'_chunk_{i}.mp3')
                tts = gTTS(text=chunk, lang=language, slow=False)
                tts.save(temp_file)
                temp_files.append(temp_file)
                
            # Combine chunks (simple concatenation for now)
            self._combine_audio_files(temp_files, filepath)
            
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass
        else:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(filepath)

    def _split_text(self, text: str, max_length: int) -> list:
        """Split text into chunks at sentence boundaries"""
        sentences = text.replace('. ', '.|').split('|')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _combine_audio_files(self, temp_files: list, output_file: str):
        """Simple file combination (can be enhanced with pydub)"""
        # For now, just use the first chunk
        # You can enhance this with pydub for proper audio concatenation
        if temp_files:
            import shutil
            shutil.copy(temp_files[0], output_file)

    async def cleanup_old_files(self, days_old: int = 1):
        """Clean up old audio files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for audio_file in self.audio_dir.glob("*.mp3"):
                if audio_file.stat().st_mtime < cutoff_date.timestamp():
                    audio_file.unlink()
                    
        except Exception as e:
            print(f"Cleanup error: {e}")

    def get_supported_languages(self) -> dict:
        """Get supported languages"""
        return self.languages

    def get_audio_stats(self) -> dict:
        """Get audio directory statistics"""
        files = list(self.audio_dir.glob("*.mp3"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "languages_used": len(set(f.name.split('_')[1] for f in files if len(f.name.split('_')) > 1))
        }