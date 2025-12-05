"""
YouTube Transcript Fetcher Module
"""
from typing import Optional, List
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from src.utils import extract_video_id, validate_video_id


class TranscriptFetcher:
    """Handles fetching and processing YouTube video transcripts."""
    
    def __init__(self):
        # Use an instance of the client; older releases expose list/fetch on instances.
        self.api = YouTubeTranscriptApi()

    def fetch_transcript(
        self,
        url_or_id: str,
        language_codes: Optional[List[str]] = None,
        target_language: Optional[str] = None,
    ) -> Optional[str]:
        """
        Fetch transcript from YouTube video.

        This follows the usage pattern from the official youtube-transcript-api
        documentation: first retrieve the available transcripts via .list(),
        then pick/translate the best match.

        Args:
            url_or_id: YouTube URL or video ID
            language_codes: Preferred language codes ordered by preference
            target_language: Optional language code to translate the transcript to

        Returns:
            Transcript text as a single string, or None if failed
        """
        # If no language preference provided, default to English preference
        if language_codes is None:
            language_codes = ["en"]

        # Extract video ID from URL
        video_id = extract_video_id(url_or_id)
        
        if not video_id:
            raise ValueError(f"Invalid YouTube URL or video ID: {url_or_id}")
        
        if not validate_video_id(video_id):
            raise ValueError(f"Invalid video ID format: {video_id}")
        
        try:
            # Retrieve the available transcripts
            transcript_list = self.api.list(video_id)

            transcript = None

            # 1) Try to directly filter for preferred languages
            try:
                transcript = transcript_list.find_transcript(language_codes)
            except NoTranscriptFound:
                transcript = None

            # 2) If none found, try manually created transcripts
            if transcript is None:
                try:
                    transcript = transcript_list.find_manually_created_transcript(
                        language_codes
                    )
                except NoTranscriptFound:
                    transcript = None

            # 3) If still none, try automatically generated ones
            if transcript is None:
                try:
                    transcript = transcript_list.find_generated_transcript(
                        language_codes
                    )
                except NoTranscriptFound:
                    transcript = None

            # 4) As a very last resort, just take the first available transcript
            if transcript is None:
                try:
                    transcript = next(iter(transcript_list))
                except StopIteration:
                    transcript = None

            if transcript is None:
                raise NoTranscriptFound(video_id)

            # Optionally translate the transcript to the target language
            if (
                target_language
                and getattr(transcript, "is_translatable", False)
                and target_language != getattr(transcript, "language_code", None)
            ):
                try:
                    transcript = transcript.translate(target_language)
                except Exception:
                    # If translation fails, fall back to the original transcript
                    pass

            # Fetch the actual transcript data
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

