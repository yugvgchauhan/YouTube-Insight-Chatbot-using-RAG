"""
YouTube Transcript Fetcher Module
"""
from typing import Optional, List
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from src.utils import extract_video_id, validate_video_id


class TranscriptFetcher:
    """Handles fetching and processing YouTube video transcripts."""
    
    def __init__(self):
        # Use an instance of the client; older releases expose list/fetch on instances.
        self.api = YouTubeTranscriptApi()
    
    def fetch_transcript(self, url_or_id: str, language_codes: List[str] = None) -> Optional[str]:
        """
        Fetch transcript from YouTube video.
        
        Args:
            url_or_id: YouTube URL or video ID
            language_codes: Preferred language codes (default: ['en'])
            
        Returns:
            Transcript text as a single string, or None if failed
        """
        if language_codes is None:
            language_codes = ['en']
        
        # Extract video ID from URL
        video_id = extract_video_id(url_or_id)
        
        if not video_id:
            raise ValueError(f"Invalid YouTube URL or video ID: {url_or_id}")
        
        if not validate_video_id(video_id):
            raise ValueError(f"Invalid video ID format: {video_id}")
        
        try:
            # Try to get transcript in preferred languages
            transcript_list = self.api.list(video_id)
            
            # Try to get transcript in preferred language
            transcript = None
            for lang_code in language_codes:
                try:
                    transcript = transcript_list.find_transcript([lang_code])
                    break
                except NoTranscriptFound:
                    continue
            
            # If no preferred language found, try to get any available transcript
            if transcript is None:
                transcript = transcript_list.find_manually_created_transcript(language_codes)
            
            # If still no transcript, get the first available one
            if transcript is None:
                transcript = transcript_list.find_generated_transcript(language_codes)
            
            # Fetch the transcript
            transcript_data = transcript.fetch()
            
            # Combine all transcript snippets into a single string. Depending on
            # youtube-transcript-api version the snippets can be dicts or
            # FetchedTranscriptSnippet objects, so handle both.
            transcript_parts = []
            for item in transcript_data:
                if isinstance(item, dict):
                    text = item.get('text', '')
                else:
                    text = getattr(item, 'text', '')
                if text:
                    transcript_parts.append(text.strip())
            
            transcript_text = " ".join(transcript_parts)
            
            return transcript_text
            
        except TranscriptsDisabled:
            raise ValueError(f"Transcripts are disabled for video: {video_id}")
        except NoTranscriptFound:
            raise ValueError(f"No transcript found for video: {video_id}")
        except Exception as e:
            raise Exception(f"Error fetching transcript: {str(e)}")
    
    def get_available_transcripts(self, url_or_id: str) -> List[dict]:
        """
        Get list of available transcripts for a video.
        
        Args:
            url_or_id: YouTube URL or video ID
            
        Returns:
            List of available transcript information
        """
        video_id = extract_video_id(url_or_id)
        
        if not video_id:
            raise ValueError(f"Invalid YouTube URL or video ID: {url_or_id}")
        
        try:
            transcript_list = self.api.list(video_id)
            transcripts = []
            
            for transcript in transcript_list:
                transcripts.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                })
            
            return transcripts
        except Exception as e:
            raise Exception(f"Error listing transcripts: {str(e)}")

