"""
Utility functions for YouTube RAG Chatbot
"""
import re
from typing import Optional


def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Extract YouTube video ID from URL or return the ID if already provided.
    
    Supports various YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    - VIDEO_ID (direct ID)
    
    Args:
        url_or_id: YouTube URL or video ID
        
    Returns:
        Video ID if found, None otherwise
    """
    # If it's already a valid video ID (11 characters, alphanumeric)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id
    
    # Pattern for standard YouTube URLs
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    return None


def validate_video_id(video_id: str) -> bool:
    """
    Validate if the string is a valid YouTube video ID.
    
    Args:
        video_id: Video ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))

