#!/usr/bin/env python3
"""
AI-SUB v6.0 - Library Manager with TMDB Integration
Features: Auto-profile detection via TMDB, library monitoring, SQLite state tracking
Replaces Bazarr with intelligent subtitle generation
"""

import os, sys, time, subprocess, shutil, json, warnings, gc, re, logging
import tempfile, threading, random, string, sqlite3, hashlib
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import requests

# Environment setup - MUST be before torch import
os.environ["MKL_VERBOSE"] = "0"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCH_LOGS_FORMAT", None)
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

TORCH_HUB_DIR = os.environ.get("TORCH_HUB_DIR", "/root/.cache/torch/hub")
os.environ["TORCH_HOME"] = os.path.dirname(TORCH_HUB_DIR)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The usage of an integral type.*")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("AI-SUB")

import torch
torch.hub.set_dir(TORCH_HUB_DIR)
logger.info(f"Torch hub directory: {TORCH_HUB_DIR}")

import numpy as np

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
    logger.info("Intel Extension for PyTorch (IPEX) available")
except ImportError:
    IPEX_AVAILABLE = False

import stable_whisper

try:
    import faster_whisper
    FASTER_WHISPER_AVAILABLE = True
    logger.info("faster-whisper backend available")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

from fastapi import FastAPI, File, UploadFile, Query, Body, Header, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse, Response, HTMLResponse
import uvicorn

VERSION = "6.0.0"

# Configuration
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
DATABASE_PATH = os.path.join(DATA_DIR, "ai-sub.db")
MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v2")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "auto")
WEBHOOK_PORT = int(os.environ.get("WEBHOOKPORT", "9000"))
DEBUG = os.environ.get("DEBUG", "true").lower() in {"1", "true", "yes"}
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v", ".ts")

# TMDB Configuration
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Our subtitle tag - used to identify AI-generated subs
SUBTITLE_TAG = "AI-SUB"

# Transcription settings
LANG_SAMPLE_SECONDS = float(os.environ.get("LANG_SAMPLE_SECONDS", "30"))
LANG_SAMPLE_OFFSET = float(os.environ.get("LANG_SAMPLE_OFFSET", "60"))
ALLOW_UNKNOWN_AS_ENGLISH = os.environ.get("ALLOW_UNKNOWN_AS_ENGLISH", "true").lower() in {"1", "true", "yes"}
VAD_ENABLED = os.environ.get("VAD_ENABLED", "true").lower() in {"1", "true", "yes"}
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.35"))
SUPPRESS_SILENCE = os.environ.get("SUPPRESS_SILENCE", "true").lower() in {"1", "true", "yes"}
SUPPRESS_WORD_TS = os.environ.get("SUPPRESS_WORD_TS", "true").lower() in {"1", "true", "yes"}
CUSTOM_REGROUP = os.environ.get("CUSTOM_REGROUP", "cm_sl=84_sl=42++++++1")
USE_REGROUP = os.environ.get("USE_REGROUP", "true").lower() in {"1", "true", "yes"}
USE_MODEL_PROMPT = os.environ.get("USE_MODEL_PROMPT", "true").lower() in {"1", "true", "yes"}
MODEL_PROMPT = os.environ.get("MODEL_PROMPT", "Hello, welcome to my video.")
USE_DEMUCS = os.environ.get("USE_DEMUCS", "false").lower() in {"1", "true", "yes"}
ONLY_VOICE_FREQ = os.environ.get("ONLY_VOICE_FREQ", "false").lower() in {"1", "true", "yes"}
MIN_SEGMENT_DURATION = float(os.environ.get("MIN_SEGMENT_DURATION", "0.5"))
MAX_CHARS_PER_LINE = int(os.environ.get("MAX_CHARS_PER_LINE", "42"))
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "4"))
CONCURRENT_TRANSCRIPTIONS = int(os.environ.get("CONCURRENT_TRANSCRIPTIONS", "1"))
CLEAR_VRAM_ON_COMPLETE = os.environ.get("CLEAR_VRAM_ON_COMPLETE", "true").lower() in {"1", "true", "yes"}

# Scan interval in seconds
LIBRARY_SCAN_INTERVAL = int(os.environ.get("LIBRARY_SCAN_INTERVAL", "300"))

# Muxing options
MUX_SUBTITLES = os.environ.get("MUX_SUBTITLES", "true").lower() in {"1", "true", "yes"}
DELETE_SRT_AFTER_MUX = os.environ.get("DELETE_SRT_AFTER_MUX", "true").lower() in {"1", "true", "yes"}
REPLACE_ORIGINAL = os.environ.get("REPLACE_ORIGINAL", "true").lower() in {"1", "true", "yes"}

# Subtitle sync - DISABLED by default as ffsubsync can cause worse timing issues
# If you have consistent timing drift, use SUBTITLE_OFFSET_MS instead
SYNC_SUBTITLES = os.environ.get("SYNC_SUBTITLES", "false").lower() in {"1", "true", "yes"}

# Simple timing offset in milliseconds (positive = delay subtitles, negative = earlier)
# Use this if subtitles are consistently early/late across all files
# Example: If subs appear 200ms early, set SUBTITLE_OFFSET_MS=200
SUBTITLE_OFFSET_MS = int(os.environ.get("SUBTITLE_OFFSET_MS", "0"))

PROFILES = {
    "anime": {"description": "Anime with background music", "vad_threshold": 0.4, "custom_regroup": "cm_sl=84_sl=42++++++1", "use_demucs": False, "initial_prompt": "Anime dialogue."},
    "tv": {"description": "Live-action TV shows", "vad_threshold": 0.35, "custom_regroup": "cm_sl=84_sl=42++++++1", "use_demucs": False, "initial_prompt": "Television dialogue."},
    "cartoon": {"description": "Western cartoons", "vad_threshold": 0.35, "custom_regroup": "cm_sl=80_sl=40++++++1", "use_demucs": False, "initial_prompt": "Cartoon dialogue."},
    "movie": {"description": "Movies", "vad_threshold": 0.35, "custom_regroup": "cm_sl=84_sl=42++++++1", "use_demucs": False, "initial_prompt": "Movie dialogue."},
    "default": {"description": "Balanced settings", "vad_threshold": 0.35, "custom_regroup": "cm_sl=84_sl=42++++++1", "use_demucs": False, "initial_prompt": "Hello, welcome to my video."},
}

WHISPER_LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian", "ko": "korean",
    "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish", "pl": "polish", "nl": "dutch",
    "ar": "arabic", "sv": "swedish", "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish",
    "vi": "vietnamese", "he": "hebrew", "uk": "ukrainian", "el": "greek", "cs": "czech", "ro": "romanian",
    "da": "danish", "hu": "hungarian", "no": "norwegian", "th": "thai", "hr": "croatian", "bg": "bulgarian",
}

def log(msg: str, level: str = "info"):
    getattr(logger, level)(msg)


# =============================================================================
# TMDB Client
# =============================================================================

class TMDBClient:
    """Client for The Movie Database API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
    
    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to TMDB API"""
        if not self.api_key:
            log("TMDB API key not configured", "warning")
            return None
        
        params = params or {}
        params["api_key"] = self.api_key
        
        try:
            response = self.session.get(f"{TMDB_BASE_URL}{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log(f"TMDB API error: {e}", "error")
            return None
    
    def search_tv(self, query: str, year: Optional[int] = None) -> List[Dict]:
        """Search for TV shows"""
        params = {"query": query}
        if year:
            params["first_air_date_year"] = year
        
        result = self._request("/search/tv", params)
        if not result:
            return []
        
        shows = []
        for item in result.get("results", [])[:10]:
            shows.append({
                "tmdb_id": item["id"],
                "title": item["name"],
                "original_title": item.get("original_name", ""),
                "year": item.get("first_air_date", "")[:4] if item.get("first_air_date") else None,
                "overview": item.get("overview", "")[:200],
                "poster_path": item.get("poster_path"),
                "media_type": "tv"
            })
        return shows
    
    def search_movie(self, query: str, year: Optional[int] = None) -> List[Dict]:
        """Search for movies"""
        params = {"query": query}
        if year:
            params["year"] = year
        
        result = self._request("/search/movie", params)
        if not result:
            return []
        
        movies = []
        for item in result.get("results", [])[:10]:
            movies.append({
                "tmdb_id": item["id"],
                "title": item["title"],
                "original_title": item.get("original_title", ""),
                "year": item.get("release_date", "")[:4] if item.get("release_date") else None,
                "overview": item.get("overview", "")[:200],
                "poster_path": item.get("poster_path"),
                "media_type": "movie"
            })
        return movies
    
    def search_multi(self, query: str, year: Optional[int] = None) -> List[Dict]:
        """Search across both movies and TV shows simultaneously.
        TMDB's multi-search often gives better results than separate searches."""
        params = {"query": query}
        
        result = self._request("/search/multi", params)
        if not result:
            return []
        
        items = []
        for item in result.get("results", [])[:10]:
            media_type = item.get("media_type")
            if media_type not in ("movie", "tv"):
                continue  # Skip person results etc.
            
            if media_type == "movie":
                items.append({
                    "tmdb_id": item["id"],
                    "title": item.get("title", ""),
                    "original_title": item.get("original_title", ""),
                    "year": item.get("release_date", "")[:4] if item.get("release_date") else None,
                    "overview": item.get("overview", "")[:200],
                    "poster_path": item.get("poster_path"),
                    "media_type": "movie"
                })
            elif media_type == "tv":
                items.append({
                    "tmdb_id": item["id"],
                    "title": item.get("name", ""),
                    "original_title": item.get("original_name", ""),
                    "year": item.get("first_air_date", "")[:4] if item.get("first_air_date") else None,
                    "overview": item.get("overview", "")[:200],
                    "poster_path": item.get("poster_path"),
                    "media_type": "tv"
                })
        
        # If year provided, sort matches with matching year first
        if year:
            year_str = str(year)
            items.sort(key=lambda x: (0 if x.get("year") == year_str else 1))
        
        return items
    
    @staticmethod
    def clean_search_query(folder_name: str) -> str:
        """Clean up a folder name for better TMDB search results.
        
        Removes common release group tags, quality indicators, codec info, etc.
        """
        name = folder_name
        
        # Remove anything in square brackets: [Group], [1080p], etc.
        name = re.sub(r'\[.*?\]', '', name)
        
        # Remove common release tags and quality markers
        noise_patterns = [
            r'\b(?:BluRay|BDRip|BRRip|WEBRip|WEB-DL|WEBDL|HDRip|DVDRip|HDTV)\b',
            r'\b(?:720p|1080p|1080i|2160p|4K|UHD)\b',
            r'\b(?:x264|x265|H\.?264|H\.?265|HEVC|AVC|10bit|10-bit)\b',
            r'\b(?:AAC|AC3|DTS|DTS-HD|TrueHD|FLAC|EAC3|Atmos|MA\.?\d+\.?\d*)\b',
            r'\b(?:REMUX|PROPER|REPACK|INTERNAL)\b',
            r'\b(?:MULTI|MULTi|DUAL|DUBBED)\b',
            r'-\w+$',  # Release group at end: -SPARKS, -iVy, -FGT, etc.
        ]
        for pattern in noise_patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        # Replace dots and underscores with spaces
        name = name.replace('.', ' ').replace('_', ' ')
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def get_tv_details(self, tmdb_id: int) -> Optional[Dict]:
        """Get detailed TV show info including genres and origin country"""
        result = self._request(f"/tv/{tmdb_id}")
        if not result:
            return None
        
        return {
            "tmdb_id": result["id"],
            "title": result["name"],
            "original_title": result.get("original_name", ""),
            "year": result.get("first_air_date", "")[:4] if result.get("first_air_date") else None,
            "genres": [g["id"] for g in result.get("genres", [])],
            "genre_names": [g["name"] for g in result.get("genres", [])],
            "origin_country": result.get("origin_country", []),
            "original_language": result.get("original_language", ""),
            "poster_path": result.get("poster_path"),
            "media_type": "tv"
        }
    
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Get detailed movie info"""
        result = self._request(f"/movie/{tmdb_id}")
        if not result:
            return None
        
        countries = [c["iso_3166_1"] for c in result.get("production_countries", [])]
        
        return {
            "tmdb_id": result["id"],
            "title": result["title"],
            "original_title": result.get("original_title", ""),
            "year": result.get("release_date", "")[:4] if result.get("release_date") else None,
            "genres": [g["id"] for g in result.get("genres", [])],
            "genre_names": [g["name"] for g in result.get("genres", [])],
            "origin_country": countries,
            "original_language": result.get("original_language", ""),
            "poster_path": result.get("poster_path"),
            "media_type": "movie"
        }
    
    def detect_profile(self, details: Dict) -> str:
        """Determine the best profile based on TMDB metadata"""
        genres = set(details.get("genres", []))
        countries = set(details.get("origin_country", []))
        original_lang = details.get("original_language", "")
        media_type = details.get("media_type", "tv")
        
        # Animation genre ID = 16
        is_animation = 16 in genres
        is_japanese = "JP" in countries or original_lang == "ja"
        
        if is_animation and is_japanese:
            return "anime"
        
        if is_animation:
            return "cartoon"
        
        if media_type == "movie":
            return "movie"
        
        return "tv"
    
    def get_credits(self, tmdb_id: int, media_type: str = "movie") -> Optional[Dict]:
        """Get cast and crew for a movie or TV show"""
        endpoint = f"/{media_type}/{tmdb_id}/credits"
        result = self._request(endpoint)
        if not result:
            return None
        
        # Extract top cast (character names + actor names)
        cast = []
        for member in result.get("cast", [])[:15]:  # Top 15 cast
            entry = {
                "name": member.get("name", ""),
                "character": member.get("character", ""),
                "order": member.get("order", 99)
            }
            if entry["character"] and entry["name"]:
                cast.append(entry)
        
        return {"cast": cast}
    
    def generate_whisper_prompt(self, tmdb_id: int, media_type: str, title: str, 
                                 profile: str, genre_names: List[str] = None) -> str:
        """Generate an optimized Whisper initial_prompt using TMDB data.
        
        The prompt provides context about character names, genre, and setting
        to help Whisper accurately transcribe proper nouns and domain-specific terms.
        """
        parts = []
        
        # Start with title context
        parts.append(f"Dialogue from \"{title}\".")
        
        # Add genre context
        if genre_names:
            genre_str = ", ".join(genre_names[:3])
            parts.append(f"Genre: {genre_str}.")
        
        # Profile-specific context
        profile_context = {
            "anime": "Anime-style dialogue. Characters may use Japanese honorifics like -san, -kun, -chan, and -sama.",
            "cartoon": "Animated cartoon dialogue with expressive characters.",
            "movie": "Feature film dialogue with clear pronunciation.",
            "tv": "Television series dialogue.",
        }
        if profile in profile_context:
            parts.append(profile_context[profile])
        
        # Get cast data from TMDB
        if tmdb_id and tmdb_id > 0:
            try:
                credits = self.get_credits(tmdb_id, media_type)
                if credits and credits.get("cast"):
                    # Build character name list
                    character_names = []
                    for member in credits["cast"]:
                        char = member.get("character", "")
                        # Clean up character names
                        # Remove "(voice)" and similar tags
                        char = re.sub(r'\s*\(.*?\)', '', char).strip()
                        # Skip very long names (usually descriptions)
                        if char and len(char) < 40 and '/' not in char:
                            character_names.append(char)
                    
                    if character_names:
                        # Keep to ~10 most important characters
                        names = character_names[:10]
                        parts.append(f"Characters: {', '.join(names)}.")
            except Exception as e:
                log(f"Could not fetch credits for prompt: {e}", "warning")
        
        prompt = " ".join(parts)
        
        # Whisper works best with prompts under ~200 tokens
        # Rough estimate: ~4 chars per token, so cap at ~800 chars
        if len(prompt) > 800:
            prompt = prompt[:800].rsplit(',', 1)[0] + "."
        
        return prompt


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """SQLite database for tracking media and processed files"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        self._migrate_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database schema"""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS media (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tmdb_id INTEGER NOT NULL,
                    media_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    year INTEGER,
                    genres TEXT,
                    origin_country TEXT,
                    profile TEXT NOT NULL,
                    folder_path TEXT NOT NULL UNIQUE,
                    monitored INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id INTEGER REFERENCES media(id) ON DELETE CASCADE,
                    file_path TEXT NOT NULL UNIQUE,
                    file_size INTEGER,
                    status TEXT DEFAULT 'pending',
                    progress REAL DEFAULT 0,
                    subtitle_path TEXT,
                    error_message TEXT,
                    processed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS libraries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    enabled INTEGER DEFAULT 1,
                    auto_import INTEGER DEFAULT 0,
                    default_profile TEXT DEFAULT 'default',
                    last_scan TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
                CREATE INDEX IF NOT EXISTS idx_files_media_id ON files(media_id);
                CREATE INDEX IF NOT EXISTS idx_media_folder ON media(folder_path);
            """)
            conn.commit()
            log("Database initialized")
        finally:
            conn.close()
    
    def _migrate_db(self):
        """Add missing columns to existing databases"""
        conn = self._get_conn()
        try:
            # Check and add missing columns to files table
            cursor = conn.execute("PRAGMA table_info(files)")
            file_columns = {row["name"] for row in cursor.fetchall()}
            
            if "progress" not in file_columns:
                conn.execute("ALTER TABLE files ADD COLUMN progress REAL DEFAULT 0")
                log("Migration: Added 'progress' column to files table")
            
            # Check and add missing columns to libraries table
            cursor = conn.execute("PRAGMA table_info(libraries)")
            lib_columns = {row["name"] for row in cursor.fetchall()}
            
            if "auto_import" not in lib_columns:
                conn.execute("ALTER TABLE libraries ADD COLUMN auto_import INTEGER DEFAULT 0")
                log("Migration: Added 'auto_import' column to libraries table")
            
            if "default_profile" not in lib_columns:
                conn.execute("ALTER TABLE libraries ADD COLUMN default_profile TEXT DEFAULT 'default'")
                log("Migration: Added 'default_profile' column to libraries table")
            
            if "last_scan" not in lib_columns:
                conn.execute("ALTER TABLE libraries ADD COLUMN last_scan TIMESTAMP")
                log("Migration: Added 'last_scan' column to libraries table")
            
            # Check and add missing columns to media table
            cursor = conn.execute("PRAGMA table_info(media)")
            media_columns = {row["name"] for row in cursor.fetchall()}
            
            if "poster_path" not in media_columns:
                conn.execute("ALTER TABLE media ADD COLUMN poster_path TEXT")
                log("Migration: Added 'poster_path' column to media table")
            
            if "library_id" not in media_columns:
                conn.execute("ALTER TABLE media ADD COLUMN library_id INTEGER")
                log("Migration: Added 'library_id' column to media table")
                # Backfill library_id based on folder_path matching library paths
                libraries = conn.execute("SELECT id, path FROM libraries").fetchall()
                for lib in libraries:
                    conn.execute(
                        "UPDATE media SET library_id = ? WHERE folder_path LIKE ? AND library_id IS NULL",
                        (lib[0], lib[1] + "%")
                    )
                log("Migration: Backfilled library_id for existing media")
            
            conn.commit()
        except Exception as e:
            log(f"Migration error (may be harmless): {e}", "warning")
        finally:
            conn.close()
    
    def add_library(self, path: str, auto_import: bool = False, default_profile: str = "default") -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT OR REPLACE INTO libraries (path, enabled, auto_import, default_profile) VALUES (?, 1, ?, ?)",
                (path, 1 if auto_import else 0, default_profile)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def update_library(self, library_id: int, auto_import: Optional[bool] = None, default_profile: Optional[str] = None):
        """Update library settings"""
        conn = self._get_conn()
        try:
            if auto_import is not None:
                conn.execute("UPDATE libraries SET auto_import = ? WHERE id = ?", (1 if auto_import else 0, library_id))
            if default_profile is not None:
                conn.execute("UPDATE libraries SET default_profile = ? WHERE id = ?", (default_profile, library_id))
            conn.commit()
        finally:
            conn.close()
    
    def update_library_scan_time(self, library_id: int):
        """Update last scan time for a library"""
        conn = self._get_conn()
        try:
            conn.execute("UPDATE libraries SET last_scan = CURRENT_TIMESTAMP WHERE id = ?", (library_id,))
            conn.commit()
        finally:
            conn.close()
    
    def get_libraries(self) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM libraries WHERE enabled = 1").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def remove_library(self, library_id: int):
        """Remove a library and all associated media/files"""
        conn = self._get_conn()
        try:
            lib = conn.execute("SELECT path FROM libraries WHERE id = ?", (library_id,)).fetchone()
            if lib:
                lib_path = lib[0]
                
                # First, disable monitoring on all media (stops scanner immediately)
                conn.execute("UPDATE media SET monitored = 0 WHERE library_id = ?", (library_id,))
                conn.execute("UPDATE media SET monitored = 0 WHERE folder_path LIKE ? AND library_id IS NULL", (lib_path + "%",))
                
                # Delete all files for this library's media
                conn.execute("""
                    DELETE FROM files WHERE media_id IN (
                        SELECT id FROM media WHERE library_id = ? OR (folder_path LIKE ? AND library_id IS NULL)
                    )
                """, (library_id, lib_path + "%"))
                
                # Delete media entries
                conn.execute("DELETE FROM media WHERE library_id = ?", (library_id,))
                conn.execute("DELETE FROM media WHERE folder_path LIKE ? AND library_id IS NULL", (lib_path + "%",))
                
                log(f"Removed library and all associated media/files: {lib_path}")
            
            # Delete the library itself  
            conn.execute("DELETE FROM libraries WHERE id = ?", (library_id,))
            conn.commit()
        finally:
            conn.close()
    
    def add_media(self, tmdb_id: int, media_type: str, title: str, year: Optional[int],
                  genres: List, origin_country: List, profile: str, folder_path: str,
                  poster_path: str = None, library_id: int = None) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                INSERT INTO media (tmdb_id, media_type, title, year, genres, origin_country, profile, folder_path, poster_path, library_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(folder_path) DO UPDATE SET
                    tmdb_id = excluded.tmdb_id,
                    title = excluded.title,
                    year = excluded.year,
                    genres = excluded.genres,
                    origin_country = excluded.origin_country,
                    profile = excluded.profile,
                    poster_path = COALESCE(excluded.poster_path, poster_path),
                    library_id = COALESCE(excluded.library_id, library_id),
                    updated_at = CURRENT_TIMESTAMP
            """, (tmdb_id, media_type, title, year, json.dumps(genres), json.dumps(origin_country), profile, folder_path, poster_path, library_id))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def get_media_by_folder(self, folder_path: str) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM media WHERE folder_path = ?", (folder_path,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    def get_media_by_id(self, media_id: int) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    def get_all_media(self) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM media ORDER BY title").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_monitored_media(self) -> List[Dict]:
        """Get only media whose library still exists and is enabled.
        Excludes media with NULL library_id whose folder path would have been under a now-deleted library."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT m.* FROM media m
                WHERE m.monitored = 1
                AND m.library_id IN (SELECT id FROM libraries WHERE enabled = 1)
                ORDER BY m.title
            """).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_media_by_library(self, library_id: int) -> List[Dict]:
        """Get all media belonging to a specific library"""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT m.*,
                    COUNT(f.id) as file_count,
                    SUM(CASE WHEN f.status = 'completed' THEN 1 ELSE 0 END) as completed_count,
                    SUM(CASE WHEN f.status = 'pending' THEN 1 ELSE 0 END) as pending_count,
                    SUM(CASE WHEN f.status = 'failed' THEN 1 ELSE 0 END) as failed_count,
                    SUM(CASE WHEN f.status = 'processing' THEN 1 ELSE 0 END) as processing_count,
                    SUM(CASE WHEN f.status = 'skipped' THEN 1 ELSE 0 END) as skipped_count
                FROM media m
                LEFT JOIN files f ON f.media_id = m.id
                WHERE m.library_id = ?
                GROUP BY m.id
                ORDER BY m.title
            """, (library_id,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_all_media_with_stats(self) -> List[Dict]:
        """Get all media with file count and status summary"""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT m.*,
                    COUNT(f.id) as file_count,
                    SUM(CASE WHEN f.status = 'completed' THEN 1 ELSE 0 END) as completed_count,
                    SUM(CASE WHEN f.status = 'pending' THEN 1 ELSE 0 END) as pending_count,
                    SUM(CASE WHEN f.status = 'failed' THEN 1 ELSE 0 END) as failed_count,
                    SUM(CASE WHEN f.status = 'processing' THEN 1 ELSE 0 END) as processing_count,
                    SUM(CASE WHEN f.status = 'skipped' THEN 1 ELSE 0 END) as skipped_count
                FROM media m
                LEFT JOIN files f ON f.media_id = m.id
                GROUP BY m.id
                ORDER BY m.title
            """).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def get_media_files(self, media_id: int) -> List[Dict]:
        """Get all files for a specific media"""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM files WHERE media_id = ?
                ORDER BY file_path
            """, (media_id,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def update_media_profile(self, media_id: int, profile: str):
        conn = self._get_conn()
        try:
            conn.execute("UPDATE media SET profile = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (profile, media_id))
            conn.commit()
        finally:
            conn.close()

    def update_media_match(
        self,
        media_id: int,
        tmdb_id: int,
        media_type: str,
        title: str,
        year: Optional[int],
        genres: List,
        origin_country: List,
        profile: str,
        poster_path: Optional[str] = None,
    ):
        conn = self._get_conn()
        try:
            conn.execute(
                """
                UPDATE media
                SET tmdb_id = ?,
                    media_type = ?,
                    title = ?,
                    year = ?,
                    genres = ?,
                    origin_country = ?,
                    profile = ?,
                    poster_path = COALESCE(?, poster_path),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    tmdb_id,
                    media_type,
                    title,
                    year,
                    json.dumps(genres),
                    json.dumps(origin_country),
                    profile,
                    poster_path,
                    media_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    
    def delete_media(self, media_id: int):
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM files WHERE media_id = ?", (media_id,))
            conn.execute("DELETE FROM media WHERE id = ?", (media_id,))
            conn.commit()
        finally:
            conn.close()
    
    def queue_file(self, file_id: int) -> bool:
        """Force a file back to pending (works regardless of current status)"""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "UPDATE files SET status = 'pending', error_message = NULL, progress = 0 WHERE id = ?",
                (file_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def queue_media_files(self, media_id: int, status_filter: Optional[str] = None) -> int:
        """Queue all files for a media entry. Returns count of files queued.
        status_filter: if set, only queue files with this status. If None, queue ALL files.
        """
        conn = self._get_conn()
        try:
            if status_filter:
                cursor = conn.execute(
                    "UPDATE files SET status = 'pending', error_message = NULL, progress = 0 WHERE media_id = ? AND status = ?",
                    (media_id, status_filter)
                )
            else:
                cursor = conn.execute(
                    "UPDATE files SET status = 'pending', error_message = NULL, progress = 0 WHERE media_id = ?",
                    (media_id,)
                )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
    
    def add_file(self, media_id: int, file_path: str, file_size: int) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                INSERT INTO files (media_id, file_path, file_size, status)
                VALUES (?, ?, ?, 'pending')
                ON CONFLICT(file_path) DO UPDATE SET file_size = excluded.file_size
            """, (media_id, file_path, file_size))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def get_file(self, file_path: str) -> Optional[Dict]:
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM files WHERE file_path = ?", (file_path,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    def get_pending_files(self, limit: int = 100) -> List[Dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT f.*, m.profile, m.title as media_title, 
                       m.tmdb_id, m.media_type, m.genres as genre_ids
                FROM files f
                JOIN media m ON f.media_id = m.id
                WHERE f.status = 'pending' AND m.monitored = 1
                ORDER BY f.created_at
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def update_file_status(self, file_path: str, status: str, subtitle_path: str = None, error: str = None):
        conn = self._get_conn()
        try:
            if status == "completed":
                conn.execute("""
                    UPDATE files SET status = ?, subtitle_path = ?, processed_at = CURRENT_TIMESTAMP
                    WHERE file_path = ?
                """, (status, subtitle_path, file_path))
            elif status == "failed":
                conn.execute("""
                    UPDATE files SET status = ?, error_message = ?, processed_at = CURRENT_TIMESTAMP
                    WHERE file_path = ?
                """, (status, error, file_path))
            else:
                conn.execute("UPDATE files SET status = ? WHERE file_path = ?", (status, file_path))
            conn.commit()
        finally:
            conn.close()
    
    def get_stats(self) -> Dict:
        conn = self._get_conn()
        try:
            stats = {}
            stats["total_media"] = conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
            stats["total_files"] = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            stats["pending"] = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'pending'").fetchone()[0]
            stats["processing"] = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'processing'").fetchone()[0]
            stats["completed"] = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'completed'").fetchone()[0]
            stats["failed"] = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'failed'").fetchone()[0]
            stats["skipped"] = conn.execute("SELECT COUNT(*) FROM files WHERE status = 'skipped'").fetchone()[0]
            return stats
        finally:
            conn.close()
    
    def get_files_by_status(self, status: str, limit: int = 100) -> List[Dict]:
        """Get files filtered by status"""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT f.*, m.profile, m.title as media_title
                FROM files f
                JOIN media m ON f.media_id = m.id
                WHERE f.status = ?
                ORDER BY f.processed_at DESC, f.created_at DESC
                LIMIT ?
            """, (status, limit)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def update_file_progress(self, file_path: str, progress: float):
        """Update processing progress (0-100)"""
        conn = self._get_conn()
        try:
            # Check if progress column exists
            cursor = conn.execute("PRAGMA table_info(files)")
            columns = {row["name"] for row in cursor.fetchall()}
            if "progress" in columns:
                conn.execute("UPDATE files SET progress = ? WHERE file_path = ?", (progress, file_path))
                conn.commit()
        except Exception as e:
            log(f"Could not update progress: {e}", "warning")
        finally:
            conn.close()
    
    def reset_stuck_processing(self):
        """Reset any files stuck in 'processing' state back to 'pending'"""
        conn = self._get_conn()
        try:
            cursor = conn.execute("UPDATE files SET status = 'pending' WHERE status = 'processing'")
            if cursor.rowcount > 0:
                log(f"Reset {cursor.rowcount} stuck processing files to pending")
            conn.commit()
        finally:
            conn.close()
    
    def get_current_processing(self) -> Optional[Dict]:
        """Get currently processing file with progress"""
        conn = self._get_conn()
        try:
            row = conn.execute("""
                SELECT f.*, m.profile, m.title as media_title
                FROM files f
                JOIN media m ON f.media_id = m.id
                WHERE f.status = 'processing'
                LIMIT 1
            """).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


# =============================================================================
# File Parser (Sonarr/Radarr naming)
# =============================================================================

class MediaParser:
    """Parse media file paths to extract show/movie info"""
    
    # Standard: "Title (2020)" or "Title (2020) {tmdb-12345}"
    FOLDER_PATTERN = re.compile(r"^(?P<title>.+?)\s*\((?P<year>\d{4})\).*$")
    
    # Dotted: "Title.2020.1080p.BluRay" or "Title.S01.2020"
    DOTTED_PATTERN = re.compile(r"^(?P<title>.+?)[\.\s](?P<year>(?:19|20)\d{2})[\.\s]")
    
    # Year in brackets: "Title [2020]"
    BRACKET_YEAR = re.compile(r"^(?P<title>.+?)\s*\[(?P<year>\d{4})\]")
    
    @classmethod
    def parse_folder_name(cls, folder_name: str) -> Tuple[str, Optional[int]]:
        """Extract title and year from folder name.
        
        Handles many common naming conventions:
        - Title (2020)
        - Title (2020) {tmdb-12345}
        - Title.2020.1080p.BluRay-GROUP
        - Title [2020]
        - Title 2020
        - Just Title
        """
        # Try standard pattern first: "Title (2020)"
        match = cls.FOLDER_PATTERN.match(folder_name)
        if match:
            return match.group("title").strip(), int(match.group("year"))
        
        # Try bracket year: "Title [2020]"
        match = cls.BRACKET_YEAR.match(folder_name)
        if match:
            return match.group("title").strip(), int(match.group("year"))
        
        # Try dotted format: "Title.2020.1080p"
        match = cls.DOTTED_PATTERN.match(folder_name)
        if match:
            title = match.group("title").replace('.', ' ').replace('_', ' ').strip()
            return title, int(match.group("year"))
        
        # Try to find a year anywhere: "Title 2020 ..."
        year_match = re.search(r'\b((?:19|20)\d{2})\b', folder_name)
        if year_match:
            year = int(year_match.group(1))
            # Only use if it looks like a real year (1920-2029)
            if 1920 <= year <= 2029:
                title = folder_name[:year_match.start()].strip()
                # Clean up trailing separators
                title = re.sub(r'[\.\-_]+$', '', title)
                title = title.replace('.', ' ').replace('_', ' ').strip()
                if title:
                    return title, year
        
        # No year found - clean up the name and return
        clean = TMDBClient.clean_search_query(folder_name)
        return clean if clean else folder_name.strip(), None


# =============================================================================
# Subtitle Detection
# =============================================================================

def has_ai_subtitle(video_path: str) -> bool:
    """Check if video already has our AI-generated subtitle track"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return False
        
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "subtitle":
                tags = stream.get("tags", {})
                title = tags.get("title", "") or tags.get("TITLE", "")
                if SUBTITLE_TAG in title:
                    return True
        return False
    except Exception as e:
        log(f"Error checking subtitles in {video_path}: {e}", "warning")
        return False


def get_audio_streams(filepath: str) -> List[Dict]:
    """Get audio streams from video file"""
    try:
        # Check file exists first
        if not os.path.exists(filepath):
            log(f"File not found: {filepath}", "error")
            return []
        
        # Get file size for debugging
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            log(f"File is empty (0 bytes): {filepath}", "error")
            return []
        
        # Run ffprobe - don't use -select_streams as it can cause issues with some files
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", filepath],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode != 0:
            # Try again with more verbose error output
            result2 = subprocess.run(
                ["ffprobe", "-v", "error", filepath],
                capture_output=True, text=True, timeout=60
            )
            error_msg = result2.stderr[:500] if result2.stderr else "unknown error"
            log(f"ffprobe failed for {os.path.basename(filepath)}: {error_msg}", "error")
            return []
        
        if not result.stdout.strip():
            log(f"ffprobe returned empty output for {os.path.basename(filepath)} ({file_size:,} bytes)", "error")
            return []
        
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            log(f"ffprobe JSON parse error for {os.path.basename(filepath)}: {e}", "error")
            log(f"ffprobe output was: {result.stdout[:500]}", "error")
            return []
        
        all_streams = data.get("streams", [])
        if not all_streams:
            log(f"No streams found in {os.path.basename(filepath)} - file may be corrupted or unsupported format", "error")
            return []
        
        audio_streams = []
        for stream in all_streams:
            if stream.get("codec_type") == "audio":
                tags = stream.get("tags", {})
                # Check multiple possible language tag keys (case-insensitive)
                language = None
                for key in ["language", "LANGUAGE", "lang", "LANG"]:
                    if key in tags and tags[key]:
                        language = tags[key]
                        break
                if not language:
                    language = "und"
                
                title = tags.get("title") or tags.get("TITLE") or tags.get("handler_name") or ""
                
                audio_streams.append({
                    "index": stream.get("index"),  # Use actual stream index from ffprobe
                    "codec": stream.get("codec_name", "unknown"),
                    "channels": stream.get("channels", 2),
                    "language": language.lower() if language else "und",
                    "title": title
                })
        
        if not audio_streams:
            # Log what streams we DID find for debugging
            stream_types = [s.get("codec_type", "unknown") for s in all_streams]
            log(f"No AUDIO streams in {os.path.basename(filepath)}. Found streams: {stream_types}", "error")
        
        return audio_streams
        
    except subprocess.TimeoutExpired:
        log(f"ffprobe timed out for {os.path.basename(filepath)} (file may be on slow storage or very large)", "error")
        return []
    except Exception as e:
        log(f"Error getting audio streams from {os.path.basename(filepath)}: {e}", "error")
        import traceback
        traceback.print_exc()
        return []


def get_video_duration(filepath: str) -> float:
    """Get video duration in seconds"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return 0


def list_directories(path: str) -> List[Dict]:
    """List directories in a path for the file browser"""
    try:
        if not os.path.exists(path):
            return []
        
        items = []
        for name in sorted(os.listdir(path)):
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path) and not name.startswith('.'):
                items.append({
                    "name": name,
                    "path": full_path,
                    "is_dir": True
                })
        return items
    except PermissionError:
        return []
    except Exception as e:
        log(f"Error listing directory {path}: {e}", "warning")
        return []


# =============================================================================
# Whisper Model & Transcription
# =============================================================================

@dataclass
class ProfileSettings:
    name: str
    description: str
    vad_threshold: float
    custom_regroup: str
    use_demucs: bool
    initial_prompt: str

def get_profile(name: str) -> ProfileSettings:
    config = PROFILES.get(name, PROFILES["default"])
    return ProfileSettings(
        name=name,
        description=config["description"],
        vad_threshold=config["vad_threshold"],
        custom_regroup=config["custom_regroup"],
        use_demucs=config["use_demucs"],
        initial_prompt=config["initial_prompt"]
    )

@dataclass
class TranscriptionResult:
    success: bool
    srt_path: Optional[str] = None
    language: str = "en"
    detected_language: Optional[str] = None
    segments_count: int = 0
    duration_seconds: float = 0
    error: Optional[str] = None

def detect_device() -> Tuple[str, str]:
    if IPEX_AVAILABLE and torch.xpu.is_available():
        return "xpu", "openai-whisper"
    if torch.cuda.is_available():
        return "cuda", "faster-whisper" if FASTER_WHISPER_AVAILABLE else "openai-whisper"
    return "cpu", "faster-whisper" if FASTER_WHISPER_AVAILABLE else "openai-whisper"

def get_compute_type(device: str) -> str:
    if COMPUTE_TYPE != "auto":
        return COMPUTE_TYPE
    if device in ("xpu", "cuda"):
        return "float16"
    return "int8"

def cleanup_device(device: str):
    gc.collect()
    if device == "xpu" and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

class WhisperModel:
    def __init__(self, model_name: str, device: str, backend: str):
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.model = None
        self._load_model()
    
    def _load_model(self):
        compute_type = get_compute_type(self.device)
        if self.backend == "faster-whisper":
            log(f"Loading faster-whisper model: {self.model_name} ({compute_type})")
            self.model = stable_whisper.load_faster_whisper(
                self.model_name, device=self.device, compute_type=compute_type,
                cpu_threads=WHISPER_THREADS, num_workers=CONCURRENT_TRANSCRIPTIONS
            )
        else:
            log(f"Loading openai-whisper model: {self.model_name}")
            self.model = stable_whisper.load_model(self.model_name)
            try:
                self.model = self.model.to(self.device)
                log(f"Model moved to {self.device}")
            except Exception as e:
                log(f"Could not move to {self.device}: {e}", "warning")
                self.device = "cpu"
                self.model = self.model.to("cpu")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None, 
                   profile: Optional[ProfileSettings] = None) -> Any:
        kwargs = {"verbose": None, "word_timestamps": True, "task": "transcribe"}
        if language:
            kwargs["language"] = language
        if VAD_ENABLED:
            kwargs["vad"] = True
            kwargs["vad_threshold"] = profile.vad_threshold if profile else VAD_THRESHOLD
        if SUPPRESS_SILENCE:
            kwargs["suppress_silence"] = True
            kwargs["suppress_word_ts"] = SUPPRESS_WORD_TS
        if USE_REGROUP:
            regroup = profile.custom_regroup if profile else CUSTOM_REGROUP
            kwargs["regroup"] = regroup if regroup else True
        if USE_MODEL_PROMPT:
            kwargs["initial_prompt"] = (profile.initial_prompt if profile else MODEL_PROMPT) or None
        if USE_DEMUCS or (profile and profile.use_demucs):
            kwargs["demucs"] = True
        if ONLY_VOICE_FREQ:
            kwargs["only_voice_freq"] = True
        
        log(f"Transcription: lang={language}, vad={kwargs.get('vad')}, profile={profile.name if profile else 'default'}")
        with torch.inference_mode():
            return self.model.transcribe(audio_path, **kwargs)
    
    def detect_language(self, audio_path: str) -> Tuple[Optional[str], float]:
        kwargs = {"verbose": None, "word_timestamps": False}
        if VAD_ENABLED:
            kwargs["vad"] = True
            kwargs["vad_threshold"] = VAD_THRESHOLD
        
        with torch.inference_mode():
            result = self.model.transcribe(audio_path, **kwargs)
        
        lang = getattr(result, "language", None)
        confidence = 0.8
        segments = getattr(result, "segments", [])
        if segments and len("".join(getattr(s, "text", "") for s in segments[:3]).strip()) > 10:
            confidence = 0.9
        return lang, confidence


# =============================================================================
# Audio/Subtitle Utilities
# =============================================================================

def extract_audio(video_path: str, stream_index: int, output_path: str) -> bool:
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-map", f"0:{stream_index}",
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        log(f"Audio extraction failed: {e}", "error")
        return False

def extract_audio_sample(video_path: str, output_path: str, offset: float, duration: float, stream_index: Optional[int] = None) -> bool:
    """Extract a sample of audio for language detection.
    
    Args:
        video_path: Path to video file
        output_path: Where to save the audio sample
        offset: Start position in seconds
        duration: Sample duration in seconds
        stream_index: Specific audio stream index, or None for default
    """
    try:
        cmd = ["ffmpeg", "-y", "-ss", str(offset), "-i", video_path, "-t", str(duration)]
        
        # If stream_index specified, extract from that specific stream
        if stream_index is not None:
            cmd.extend(["-map", f"0:{stream_index}"])
        
        cmd.extend(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output_path])
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        log(f"Audio sample extraction failed: {e}", "error")
        return False


def result_to_srt_string(result) -> str:
    lines = []
    for i, seg in enumerate(result.segments, 1):
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        text = seg.text.strip()
        if text:
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def apply_subtitle_offset(srt_path: str, offset_ms: int) -> bool:
    """Apply a timing offset to all subtitles in an SRT file.
    
    Args:
        srt_path: Path to SRT file (modified in-place)
        offset_ms: Offset in milliseconds (positive = delay/later, negative = earlier)
        
    Returns:
        True if successful
    """
    if offset_ms == 0:
        return True
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse timestamp: 00:01:23,456 --> 00:01:25,789
        import re
        
        def shift_timestamp(match):
            time_str = match.group(0)
            # Parse: HH:MM:SS,mmm
            parts = time_str.replace(',', ':').split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            millis = int(parts[3])
            
            # Convert to total milliseconds
            total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + millis
            
            # Apply offset
            total_ms += offset_ms
            
            # Don't go negative
            if total_ms < 0:
                total_ms = 0
            
            # Convert back
            new_hours = total_ms // 3600000
            total_ms %= 3600000
            new_minutes = total_ms // 60000
            total_ms %= 60000
            new_seconds = total_ms // 1000
            new_millis = total_ms % 1000
            
            return f"{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d},{new_millis:03d}"
        
        # Match timestamps in format HH:MM:SS,mmm
        new_content = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}', shift_timestamp, content)
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        log(f"Applied {offset_ms:+d}ms offset to subtitles")
        return True
        
    except Exception as e:
        log(f"Failed to apply subtitle offset: {e}", "error")
        return False

def save_subtitles(result, output_path: str) -> bool:
    try:
        srt_content = result_to_srt_string(result)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        return True
    except Exception as e:
        log(f"Failed to save subtitles: {e}", "error")
        return False


def mux_subtitle(video_path: str, subtitle_path: str, replace_original: bool = True) -> Tuple[bool, Optional[str]]:
    """Mux subtitle into video file. Uses mkvmerge for MKV files (faster), ffmpeg for others."""
    temp_output = None
    try:
        if not os.path.exists(video_path) or not os.path.exists(subtitle_path):
            log(f"Mux skipped - files missing", "warning")
            return False, None
        
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_ext = os.path.splitext(video_path)[1].lower()
        
        # Create temp output file in system temp directory (not in media folder!)
        # This prevents leftover temp files in the media library
        temp_output = tempfile.mktemp(suffix=video_ext, prefix=f"mux_{video_name[:30]}_")
        
        # Use mkvmerge for MKV files (much faster, preserves everything)
        if video_ext in ('.mkv', '.mka', '.mks', '.webm'):
            success = mux_with_mkvmerge(video_path, subtitle_path, temp_output)
        else:
            success = mux_with_ffmpeg(video_path, subtitle_path, temp_output)
        
        if not success:
            if temp_output and os.path.exists(temp_output):
                os.remove(temp_output)
            return False, None
        
        # Verify output exists and has reasonable size
        if not os.path.exists(temp_output):
            log("Mux output file not created", "error")
            return False, None
        
        original_size = os.path.getsize(video_path)
        new_size = os.path.getsize(temp_output)
        
        # Sanity check - new file should be similar size (within 5%)
        if new_size < original_size * 0.95:
            log(f"Muxed file too small ({new_size:,} vs {original_size:,} bytes), aborting", "error")
            os.remove(temp_output)
            return False, None
        
        if replace_original:
            # Replace original with muxed version
            backup_path = video_path + ".bak"
            
            try:
                os.rename(video_path, backup_path)
                # Copy from temp to final location (may be different filesystem)
                shutil.copy2(temp_output, video_path)
                os.remove(backup_path)
                os.remove(temp_output)
                log(f"✓ Muxed subtitle into: {os.path.basename(video_path)}")
                return True, video_path
            except Exception as e:
                log(f"Error replacing original: {e}", "error")
                if os.path.exists(backup_path) and not os.path.exists(video_path):
                    os.rename(backup_path, video_path)
                if temp_output and os.path.exists(temp_output):
                    os.remove(temp_output)
                return False, None
        else:
            final_output = os.path.join(video_dir, f"{video_name}_subs{video_ext}")
            shutil.copy2(temp_output, final_output)
            os.remove(temp_output)
            log(f"✓ Created muxed file: {os.path.basename(final_output)}")
            return True, final_output
            
    except Exception as e:
        log(f"Mux error: {e}", "error")
        import traceback
        traceback.print_exc()
        # Clean up temp file on any error
        if temp_output and os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        return False, None


def mux_with_mkvmerge(video_path: str, subtitle_path: str, output_path: str) -> bool:
    """Mux using mkvmerge - fast, preserves all tracks and tags"""
    try:
        cmd = [
            "mkvmerge", "-o", output_path,
            video_path,
            "--language", "0:eng",
            "--track-name", f"0:English [{SUBTITLE_TAG}]",
            "--default-track", "0:no",
            subtitle_path
        ]
        
        log(f"Muxing with mkvmerge: {os.path.basename(video_path)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # mkvmerge returns 0 for success, 1 for warnings, 2 for errors
        if result.returncode == 2:
            error_msg = result.stdout[:500] if result.stdout else result.stderr[:500]
            log(f"mkvmerge failed: {error_msg}", "error")
            return False
        
        if result.returncode == 1:
            log(f"mkvmerge warning (continuing): {result.stdout[:200] if result.stdout else ''}", "warning")
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
    except FileNotFoundError:
        log("mkvmerge not found, install mkvtoolnix package", "error")
        return False
    except subprocess.TimeoutExpired:
        log("mkvmerge timed out", "error")
        return False
    except Exception as e:
        log(f"mkvmerge error: {e}", "error")
        return False


def mux_with_ffmpeg(video_path: str, subtitle_path: str, output_path: str) -> bool:
    """Mux using ffmpeg - fallback for non-MKV files"""
    try:
        # Get existing subtitle count
        probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        
        existing_subs = 0
        if probe_result.returncode == 0:
            try:
                data = json.loads(probe_result.stdout)
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "subtitle":
                        existing_subs += 1
            except:
                pass
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", subtitle_path,
            "-map", "0",
            "-map", "1:0",
            "-c", "copy",
            "-c:s", "srt",
            f"-metadata:s:s:{existing_subs}", f"title=English [{SUBTITLE_TAG}]",
            f"-metadata:s:s:{existing_subs}", "language=eng",
            output_path
        ]
        
        log(f"Muxing with ffmpeg: {os.path.basename(video_path)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            error_msg = result.stderr[-1000:] if result.stderr else "Unknown error"
            log(f"ffmpeg mux failed: {error_msg}", "error")
            return False
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
    except subprocess.TimeoutExpired:
        log("ffmpeg mux timed out", "error")
        return False
    except Exception as e:
        log(f"ffmpeg mux error: {e}", "error")
        return False


def sync_subtitles(video_path: str, subtitle_path: str, output_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Use ffsubsync to align subtitles with audio.
    
    Args:
        video_path: Path to video file (used as reference for audio)
        subtitle_path: Path to SRT file to sync
        output_path: Output path for synced subtitles (if None, overwrites original)
        
    Returns:
        Tuple of (success, output_path)
    """
    if output_path is None:
        output_path = subtitle_path  # Overwrite original
    
    try:
        if not os.path.exists(video_path):
            log(f"Sync failed: video not found: {video_path}", "error")
            return False, None
        
        if not os.path.exists(subtitle_path):
            log(f"Sync failed: subtitle not found: {subtitle_path}", "error")
            return False, None
        
        # Create temp file for output to avoid corrupting original on failure
        temp_output = tempfile.mktemp(suffix=".srt", prefix="sync_")
        
        log(f"Syncing subtitles: {os.path.basename(subtitle_path)}")
        
        cmd = [
            "ffsubsync",
            video_path,             # Reference video (audio track)
            "-i", subtitle_path,    # Input subtitle
            "-o", temp_output,      # Output subtitle
            "--no-fix-framerate"    # Keep original framerate
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            log(f"ffsubsync failed: {error_msg}", "error")
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False, None
        
        # Verify output exists and has content
        if not os.path.exists(temp_output) or os.path.getsize(temp_output) == 0:
            log("ffsubsync produced empty output", "error")
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False, None
        
        # Move temp to final location
        shutil.move(temp_output, output_path)
        log(f"✓ Synced subtitles: {os.path.basename(output_path)}")
        return True, output_path
        
    except subprocess.TimeoutExpired:
        log("ffsubsync timed out (>5 minutes)", "error")
        return False, None
    except Exception as e:
        log(f"Sync error: {e}", "error")
        return False, None


def extract_subtitle_from_video(video_path: str, output_srt: str, track_name_contains: str = SUBTITLE_TAG) -> bool:
    """Extract a subtitle track from video file.
    
    Args:
        video_path: Path to video file
        output_srt: Output SRT file path
        track_name_contains: Only extract track with this string in its title
        
    Returns:
        True if extraction successful
    """
    try:
        # First, find which subtitle track to extract
        probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        
        if probe_result.returncode != 0:
            log(f"Could not probe video: {video_path}", "error")
            return False
        
        data = json.loads(probe_result.stdout)
        
        sub_stream_index = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "subtitle":
                tags = stream.get("tags", {})
                title = tags.get("title", "") or tags.get("TITLE", "")
                if track_name_contains in title:
                    sub_stream_index = stream.get("index")
                    log(f"Found AI-SUB track at index {sub_stream_index}: {title}")
                    break
        
        if sub_stream_index is None:
            log(f"No subtitle track containing '{track_name_contains}' found", "warning")
            return False
        
        # Extract the subtitle
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-map", f"0:{sub_stream_index}",
            "-c:s", "srt",
            output_srt
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            log(f"Subtitle extraction failed: {result.stderr[-200:] if result.stderr else 'unknown'}", "error")
            return False
        
        return os.path.exists(output_srt) and os.path.getsize(output_srt) > 0
        
    except Exception as e:
        log(f"Extract subtitle error: {e}", "error")
        return False


def resync_completed_file(video_path: str) -> Tuple[bool, str]:
    """Resync subtitles for a completed file.
    
    This extracts the AI-SUB subtitle, runs ffsubsync, and re-muxes it.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        if not os.path.exists(video_path):
            return False, "Video file not found"
        
        # Create temp files
        temp_srt = tempfile.mktemp(suffix=".srt", prefix="resync_orig_")
        synced_srt = tempfile.mktemp(suffix=".srt", prefix="resync_synced_")
        
        try:
            # Step 1: Extract existing AI-SUB subtitle
            log(f"Extracting subtitle from: {os.path.basename(video_path)}")
            if not extract_subtitle_from_video(video_path, temp_srt):
                return False, "Could not extract AI-SUB subtitle from video"
            
            # Step 2: Sync it with ffsubsync
            success, _ = sync_subtitles(video_path, temp_srt, synced_srt)
            if not success:
                return False, "ffsubsync failed"
            
            # Step 3: Re-mux the synced subtitle
            log(f"Re-muxing synced subtitle into: {os.path.basename(video_path)}")
            
            # We need to remove the old AI-SUB track and add the new one
            # For MKV, mkvmerge can do this efficiently
            video_ext = os.path.splitext(video_path)[1].lower()
            
            if video_ext in ('.mkv', '.mka', '.mks', '.webm'):
                success, _ = remux_with_synced_subtitle_mkvmerge(video_path, synced_srt)
            else:
                success, _ = remux_with_synced_subtitle_ffmpeg(video_path, synced_srt)
            
            if success:
                return True, "Subtitles resynced successfully"
            else:
                return False, "Re-muxing failed"
                
        finally:
            # Cleanup temp files
            for f in [temp_srt, synced_srt]:
                if os.path.exists(f):
                    os.remove(f)
                    
    except Exception as e:
        log(f"Resync error: {e}", "error")
        return False, str(e)


def remux_with_synced_subtitle_mkvmerge(video_path: str, subtitle_path: str) -> Tuple[bool, Optional[str]]:
    """Remove old AI-SUB track and add new synced one using mkvmerge."""
    temp_output = tempfile.mktemp(suffix=".mkv", prefix="remux_")
    
    try:
        # First, find which track to remove (the AI-SUB one)
        probe_cmd = ["mkvmerge", "-J", video_path]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        
        tracks_to_remove = []
        if probe_result.returncode == 0:
            try:
                data = json.loads(probe_result.stdout)
                for track in data.get("tracks", []):
                    if track.get("type") == "subtitles":
                        props = track.get("properties", {})
                        track_name = props.get("track_name", "")
                        if SUBTITLE_TAG in track_name:
                            tracks_to_remove.append(track.get("id"))
            except:
                pass
        
        # Build mkvmerge command
        cmd = ["mkvmerge", "-o", temp_output]
        
        # Exclude old AI-SUB tracks
        if tracks_to_remove:
            track_ids = ",".join(str(t) for t in tracks_to_remove)
            cmd.extend(["--subtitle-tracks", f"!{track_ids}"])
        
        cmd.append(video_path)
        
        # Add new synced subtitle
        cmd.extend([
            "--language", "0:eng",
            "--track-name", f"0:English [{SUBTITLE_TAG}]",
            "--default-track", "0:no",
            subtitle_path
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode not in (0, 1):  # 0=success, 1=warnings
            log(f"mkvmerge remux failed: {result.stderr[-300:] if result.stderr else 'unknown'}", "error")
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False, None
        
        # Verify and replace original
        if os.path.exists(temp_output) and os.path.getsize(temp_output) > os.path.getsize(video_path) * 0.95:
            backup_path = video_path + ".bak"
            os.rename(video_path, backup_path)
            shutil.move(temp_output, video_path)
            os.remove(backup_path)
            log(f"✓ Remuxed with synced subtitle: {os.path.basename(video_path)}")
            return True, video_path
        else:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False, None
            
    except Exception as e:
        log(f"mkvmerge remux error: {e}", "error")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False, None


def remux_with_synced_subtitle_ffmpeg(video_path: str, subtitle_path: str) -> Tuple[bool, Optional[str]]:
    """Remove old AI-SUB track and add new synced one using ffmpeg."""
    temp_output = tempfile.mktemp(suffix=os.path.splitext(video_path)[1], prefix="remux_")
    
    try:
        # This is more complex with ffmpeg - we'll just add the subtitle as a new track
        # Users with non-MKV files will have to accept this limitation
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", subtitle_path,
            "-map", "0",
            "-map", "1:0",
            "-c", "copy",
            "-c:s", "srt",
            "-metadata:s:s:0", f"title=English [{SUBTITLE_TAG}] (synced)",
            "-metadata:s:s:0", "language=eng",
            temp_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            log(f"ffmpeg remux failed: {result.stderr[-300:] if result.stderr else 'unknown'}", "error")
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False, None
        
        # Verify and replace original
        if os.path.exists(temp_output) and os.path.getsize(temp_output) > os.path.getsize(video_path) * 0.95:
            backup_path = video_path + ".bak"
            os.rename(video_path, backup_path)
            shutil.move(temp_output, video_path)
            os.remove(backup_path)
            log(f"✓ Remuxed with synced subtitle: {os.path.basename(video_path)}")
            return True, video_path
        else:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False, None
            
    except Exception as e:
        log(f"ffmpeg remux error: {e}", "error")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False, None


# =============================================================================
# Library Scanner
# =============================================================================

class LibraryScanner:
    def __init__(self, db: DatabaseManager, tmdb: TMDBClient):
        self.db = db
        self.tmdb = tmdb
    
    def scan_library(self, library_path: str) -> Dict:
        if not os.path.exists(library_path):
            return {"error": f"Path does not exist: {library_path}"}
        
        found = {"new_folders": [], "existing": 0, "errors": []}
        
        try:
            for item in os.listdir(library_path):
                folder_path = os.path.join(library_path, item)
                if not os.path.isdir(folder_path):
                    continue
                
                existing = self.db.get_media_by_folder(folder_path)
                if existing:
                    found["existing"] += 1
                    continue
                
                title, year = MediaParser.parse_folder_name(item)
                found["new_folders"].append({
                    "path": folder_path,
                    "folder_name": item,
                    "title": title,
                    "year": year
                })
        except Exception as e:
            found["errors"].append(str(e))
        
        return found
    
    def scan_media_files(self, media_id: int) -> Dict:
        media = self.db.get_media_by_id(media_id)
        if not media:
            return {"error": "Media not found"}
        
        folder_path = media["folder_path"]
        if not os.path.exists(folder_path):
            return {"error": f"Folder not found: {folder_path}"}
        
        found = {"new": 0, "existing": 0, "skipped": 0}
        
        # Patterns to exclude (temp files, samples, etc.)
        exclude_patterns = ['_muxtemp', '_muxed', '.sample', '.sample.', '_temp', '.part']
        
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if not filename.lower().endswith(VIDEO_EXTENSIONS):
                    continue
                
                # Skip temp files and samples
                filename_lower = filename.lower()
                if any(pattern in filename_lower for pattern in exclude_patterns):
                    continue
                
                file_path = os.path.join(root, filename)
                
                existing = self.db.get_file(file_path)
                if existing:
                    found["existing"] += 1
                    continue
                
                if has_ai_subtitle(file_path):
                    self.db.add_file(media_id, file_path, os.path.getsize(file_path))
                    self.db.update_file_status(file_path, "skipped")
                    found["skipped"] += 1
                    continue
                
                file_size = os.path.getsize(file_path)
                self.db.add_file(media_id, file_path, file_size)
                found["new"] += 1
        
        return found


# =============================================================================
# Processing Engine
# =============================================================================

class ProcessingEngine:
    def __init__(self, db: DatabaseManager, whisper_model: WhisperModel):
        self.db = db
        self.model = whisper_model
        self.running = False
        self.current_file = None
        self.current_progress = 0
        self.current_stage = ""
        self.cancel_requested = False
    
    def cancel_current(self):
        """Request cancellation of current processing"""
        if self.current_file:
            self.cancel_requested = True
            log(f"Cancellation requested for: {os.path.basename(self.current_file)}")
            return True
        return False
    
    def _update_progress(self, file_path: str, progress: float, stage: str = ""):
        """Update progress for current file"""
        self.current_progress = progress
        self.current_stage = stage
        try:
            self.db.update_file_progress(file_path, progress)
        except Exception as e:
            log(f"Failed to update progress: {e}", "warning")
    
    def _check_cancelled(self) -> bool:
        """Check if cancellation was requested"""
        if self.cancel_requested:
            self.cancel_requested = False
            return True
        return False
    
    def process_file(self, file_info: Dict) -> TranscriptionResult:
        file_path = file_info["file_path"]
        profile_name = file_info.get("profile", "default")
        profile = get_profile(profile_name)
        
        # Generate dynamic prompt from TMDB data
        tmdb_id = file_info.get("tmdb_id", 0)
        media_type = file_info.get("media_type", "movie")
        media_title = file_info.get("media_title", "")
        
        if tmdb_id and tmdb_id > 0 and media_title:
            try:
                # Get genre names from TMDB details
                genre_names = []
                if media_type == "tv":
                    details = tmdb.get_tv_details(tmdb_id)
                else:
                    details = tmdb.get_movie_details(tmdb_id)
                if details:
                    genre_names = details.get("genre_names", [])
                
                dynamic_prompt = tmdb.generate_whisper_prompt(
                    tmdb_id=tmdb_id,
                    media_type=media_type,
                    title=media_title,
                    profile=profile_name,
                    genre_names=genre_names
                )
                # Override profile prompt with dynamic one
                profile = ProfileSettings(
                    name=profile.name,
                    description=profile.description,
                    vad_threshold=profile.vad_threshold,
                    custom_regroup=profile.custom_regroup,
                    use_demucs=profile.use_demucs,
                    initial_prompt=dynamic_prompt
                )
                log(f"Dynamic prompt: {dynamic_prompt[:120]}...")
            except Exception as e:
                log(f"Could not generate dynamic prompt: {e}", "warning")
        
        log(f"Processing: {os.path.basename(file_path)} (profile: {profile_name})")
        
        self.current_file = file_path
        self.cancel_requested = False
        self.db.update_file_status(file_path, "processing")
        self._update_progress(file_path, 5, "Analyzing audio tracks")
        
        audio_wav = tempfile.mktemp(suffix=".wav")
        
        try:
            # Check if file exists first
            if not os.path.exists(file_path):
                log(f"FILE NOT FOUND: {file_path}", "error")
                return TranscriptionResult(success=False, error=f"File not found: {file_path}")
            
            # Check for cancellation
            if self._check_cancelled():
                return TranscriptionResult(success=False, error="Cancelled by user")
            
            # Get video duration for progress estimation
            video_duration = get_video_duration(file_path)
            
            # Get ALL audio streams
            streams = get_audio_streams(file_path)
            if not streams:
                log(f"No audio streams found in: {file_path}", "error")
                return TranscriptionResult(success=False, error="No audio streams found")
            
            # Log all available audio tracks
            log(f"Found {len(streams)} audio track(s):")
            for s in streams:
                log(f"  Track {s['index']}: {s['language']} - {s.get('title', 'no title')} ({s['codec']}, {s['channels']}ch)")
            
            self._update_progress(file_path, 10, "Finding English track")
            
            if self._check_cancelled():
                return TranscriptionResult(success=False, error="Cancelled by user")
            
            # Find English audio track
            english_stream = self._find_english_audio_track(file_path, streams)
            
            if english_stream is None:
                # List all tracks in the error for debugging
                track_langs = ", ".join([f"{s['index']}:{s['language']}" for s in streams])
                log(f"No English audio track found. Available: {track_langs}", "warning")
                return TranscriptionResult(success=False, error=f"No English audio (tracks: {track_langs})")
            
            log(f"Using audio track {english_stream['index']}: {english_stream['language']} - {english_stream.get('title', '')}")
            
            self._update_progress(file_path, 20, "Extracting audio")
            
            if self._check_cancelled():
                return TranscriptionResult(success=False, error="Cancelled by user")
            
            if not extract_audio(file_path, english_stream["index"], audio_wav):
                return TranscriptionResult(success=False, error="Audio extraction failed")
            
            self._update_progress(file_path, 30, "Transcribing")
            
            if self._check_cancelled():
                return TranscriptionResult(success=False, error="Cancelled by user")
            
            start_time = time.time()
            result = self.model.transcribe(audio_wav, language="en", profile=profile)
            elapsed = time.time() - start_time
            
            self._update_progress(file_path, 85, "Saving subtitles")
            
            segment_count = len(result.segments) if hasattr(result, "segments") else 0
            if segment_count == 0:
                return TranscriptionResult(success=False, error="No speech detected")
            
            # Save subtitle file next to video
            video_dir = os.path.dirname(file_path)
            video_name = os.path.splitext(os.path.basename(file_path))[0]
            srt_path = os.path.join(video_dir, f"{video_name}.en.srt")
            
            if not save_subtitles(result, srt_path):
                return TranscriptionResult(success=False, error="Failed to save subtitles")
            
            log(f"✓ Transcribed: {segment_count} segments in {elapsed:.1f}s")
            
            # Sync subtitles with audio if enabled
            if SYNC_SUBTITLES:
                self._update_progress(file_path, 88, "Syncing subtitles")
                sync_success, _ = sync_subtitles(file_path, srt_path)
                if not sync_success:
                    log("Subtitle sync failed, continuing with original timing", "warning")
            
            # Mux subtitle into video if enabled
            final_srt_path = srt_path
            if MUX_SUBTITLES:
                self._update_progress(file_path, 92, "Muxing subtitles")
                mux_success, muxed_path = mux_subtitle(file_path, srt_path, REPLACE_ORIGINAL)
                
                if mux_success:
                    # Delete external SRT after successful mux
                    if DELETE_SRT_AFTER_MUX and os.path.exists(srt_path):
                        try:
                            os.remove(srt_path)
                            log(f"✓ Deleted external SRT after muxing")
                            final_srt_path = None  # SRT is now embedded
                        except Exception as e:
                            log(f"Could not delete SRT: {e}", "warning")
                else:
                    log(f"Muxing failed, keeping external SRT", "warning")
            
            self._update_progress(file_path, 100, "Complete")
            
            return TranscriptionResult(
                success=True,
                srt_path=final_srt_path,
                language="en",
                detected_language="en",
                segments_count=segment_count,
                duration_seconds=elapsed
            )
            
        except Exception as e:
            log(f"Processing error: {e}", "error")
            import traceback
            traceback.print_exc()
            return TranscriptionResult(success=False, error=str(e))
        finally:
            if os.path.exists(audio_wav):
                os.remove(audio_wav)
            cleanup_device(self.model.device)
            self.current_file = None
    
    def _find_english_audio_track(self, file_path: str, streams: List[Dict]) -> Optional[Dict]:
        """Find the English audio track from available streams.
        
        Strategy:
        1. Look for tracks tagged as English (eng, en, english)
        2. Check track titles for "English"
        3. If only one track with undefined language, use it
        4. Sample tracks with undefined language to detect
        5. If ALLOW_UNKNOWN_AS_ENGLISH and single track, use it
        
        Returns the stream dict for the English track, or None if not found.
        """
        if not streams:
            return None
        
        # English language tags (case-insensitive)
        english_tags = {'eng', 'en', 'english'}
        
        # Priority 1: Tracks explicitly tagged as English
        for stream in streams:
            lang = stream.get('language', '').lower()
            if lang in english_tags:
                log(f"Found English track by language tag: stream {stream['index']}")
                return stream
        
        # Priority 2: Check track titles for "English"
        for stream in streams:
            title = stream.get('title', '').lower()
            if 'english' in title or 'eng ' in title or title.startswith('eng'):
                log(f"Found English track by title: stream {stream['index']} ({stream.get('title', '')})")
                return stream
        
        # Priority 3: If only one track and it's undefined, assume English (common case)
        undefined_langs = {'und', 'unk', ''}
        if len(streams) == 1:
            lang = streams[0].get('language', 'und').lower()
            if lang in undefined_langs or ALLOW_UNKNOWN_AS_ENGLISH:
                log(f"Single audio track (lang={lang}), using as English")
                return streams[0]
        
        # Priority 4: Sample undefined tracks to detect language
        undefined_streams = [s for s in streams if s.get('language', 'und').lower() in undefined_langs]
        
        for stream in undefined_streams:
            detected = self._detect_track_language(file_path, stream['index'])
            if detected == 'en':
                log(f"Found English track by detection: stream {stream['index']}")
                return stream
            elif detected:
                log(f"Track {stream['index']} detected as: {detected}")
        
        # Priority 5: If ALLOW_UNKNOWN_AS_ENGLISH, use first undefined track
        if ALLOW_UNKNOWN_AS_ENGLISH and undefined_streams:
            stream = undefined_streams[0]
            log(f"Using first undefined track as English (ALLOW_UNKNOWN_AS_ENGLISH=true): stream {stream['index']}")
            return stream
        
        # No English track found
        return None
    
    def _detect_track_language(self, file_path: str, stream_index: int) -> Optional[str]:
        """Detect language of a specific audio track by sampling."""
        sample_positions = [60, 120, 180, 300]  # Try multiple positions
        
        for offset in sample_positions:
            sample_wav = tempfile.mktemp(suffix="_lang.wav")
            try:
                if extract_audio_sample(file_path, sample_wav, offset, LANG_SAMPLE_SECONDS, stream_index):
                    detected_lang, confidence = self.model.detect_language(sample_wav)
                    if detected_lang is not None:
                        log(f"Track {stream_index} language detected: {detected_lang} (at {offset}s)")
                        return detected_lang
            except Exception as e:
                log(f"Language detection error for stream {stream_index}: {e}", "warning")
            finally:
                if os.path.exists(sample_wav):
                    os.remove(sample_wav)
        
        return None
    
    def process_pending(self, limit: int = 1) -> int:
        processed = 0
        pending = self.db.get_pending_files(limit)
        
        for file_info in pending:
            result = self.process_file(file_info)
            
            if result.success:
                self.db.update_file_status(file_info["file_path"], "completed", result.srt_path)
            elif result.error and "Cancelled" in result.error:
                # Cancelled by user - set back to pending so they can retry
                self.db.update_file_status(file_info["file_path"], "pending")
                log(f"File returned to pending: {os.path.basename(file_info['file_path'])}")
            else:
                self.db.update_file_status(file_info["file_path"], "failed", error=result.error)
            
            processed += 1
        
        return processed


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(title="AI-SUB", version=VERSION, description="Library Manager with TMDB Integration")

db: Optional[DatabaseManager] = None
tmdb: Optional[TMDBClient] = None
scanner: Optional[LibraryScanner] = None
engine: Optional[ProcessingEngine] = None
whisper_model: Optional[WhisperModel] = None

# Web UI HTML
WEB_UI_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-SUB</title>
    <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body { background: #0f172a; }
        .card { background: #1e293b; border: 1px solid #334155; }
        .btn-primary { background: #3b82f6; }
        .btn-primary:hover { background: #2563eb; }
        .btn-danger { background: #ef4444; }
        .btn-success { background: #22c55e; }
        .btn-secondary { background: #4b5563; }
        .btn-secondary:hover { background: #374151; }
        .profile-anime { background: #ec4899; color: white; }
        .profile-cartoon { background: #f59e0b; color: white; }
        .profile-tv { background: #3b82f6; color: white; }
        .profile-movie { background: #8b5cf6; color: white; }
        .profile-default { background: #6b7280; color: white; }
        .stat-card { cursor: pointer; transition: transform 0.2s, border-color 0.2s; }
        .stat-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
        .stat-card.active { border-color: #3b82f6; border-width: 2px; }
        .progress-bar { transition: width 0.5s ease; }
        .modal { background: rgba(0,0,0,0.7); }
        .match-good { border-left: 4px solid #22c55e; }
        .match-none { border-left: 4px solid #f59e0b; }
        .dropdown-menu { position: absolute; right: 0; top: 100%; z-index: 50; min-width: 200px; }
        .toggle-switch { position: relative; width: 44px; height: 24px; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .toggle-slider { position: absolute; cursor: pointer; inset: 0; background: #4b5563; border-radius: 24px; transition: 0.3s; }
        .toggle-slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background: white; border-radius: 50%; transition: 0.3s; }
        input:checked + .toggle-slider { background: #22c55e; }
        input:checked + .toggle-slider:before { transform: translateX(20px); }
        .poster-card { position: relative; overflow: hidden; border-radius: 0.5rem; transition: all 0.2s; }
        .poster-card:hover { transform: scale(1.05); border-color: #3b82f6; z-index: 10; }
        .poster-card .poster-wrap { position: relative; padding-top: 150%; background: #1e293b; }
        .poster-card .poster-wrap img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
        .poster-card .poster-wrap .no-poster { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; padding: 0.5rem; text-align: center; font-size: 0.75rem; color: #6b7280; }
        .poster-card .overlay { position: absolute; bottom: 0; left: 0; right: 0; padding: 0.5rem; padding-top: 2rem; background: linear-gradient(to top, rgba(0,0,0,0.95), rgba(0,0,0,0.6) 60%, transparent); }
        .scrollbar-thin::-webkit-scrollbar { width: 6px; }
        .scrollbar-thin::-webkit-scrollbar-track { background: transparent; }
        .scrollbar-thin::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 3px; }
    </style>
</head>
<body class="text-gray-100 min-h-screen">
    <div id="app" class="container mx-auto px-4 py-6 max-w-7xl">
        <!-- Header -->
        <header class="mb-6">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-3xl font-bold text-white">🎬 AI-SUB</h1>
                    <p class="text-gray-400 text-sm">Automated Subtitle Generation</p>
                </div>
                <div class="flex items-center gap-4">
                    <div class="text-right text-sm">
                        <span class="text-green-400">{{ stats.completed || 0 }}✓</span>
                        <span class="text-yellow-400 ml-2">{{ stats.pending || 0 }}⏳</span>
                        <span class="text-red-400 ml-2">{{ stats.failed || 0 }}✗</span>
                    </div>
                    <div class="text-lg" :class="status.current_file ? 'text-yellow-400' : 'text-green-400'">
                        {{ status.current_file ? '● Processing' : '● Idle' }}
                    </div>
                </div>
            </div>
        </header>

        <!-- Progress Bar -->
        <div v-if="status.current_file" class="card rounded-lg p-4 mb-4">
            <div class="flex justify-between items-center mb-2">
                <span class="text-white font-medium truncate mr-4">{{ status.current_filename }}</span>
                <div class="flex items-center gap-3 flex-shrink-0">
                    <span class="text-gray-400 text-sm">{{ status.stage }} - {{ Math.round(status.progress) }}%</span>
                    <button @click="cancelProcessing" class="btn-danger px-3 py-1 rounded text-xs">Cancel</button>
                </div>
            </div>
            <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="progress-bar bg-blue-500 h-2 rounded-full" :style="{width: status.progress + '%'}"></div>
            </div>
        </div>

        <!-- Stats Cards (clickable) -->
        <div class="grid grid-cols-5 gap-3 mb-4">
            <div class="card rounded-lg p-3 text-center stat-card" :class="{active: filesFilter === 'pending'}" @click="showFiles('pending')">
                <div class="text-xl font-bold text-yellow-400">{{ stats.pending || 0 }}</div>
                <div class="text-xs text-gray-400">Pending</div>
            </div>
            <div class="card rounded-lg p-3 text-center stat-card" :class="{active: filesFilter === 'processing'}" @click="showFiles('processing')">
                <div class="text-xl font-bold text-blue-400">{{ stats.processing || 0 }}</div>
                <div class="text-xs text-gray-400">Processing</div>
            </div>
            <div class="card rounded-lg p-3 text-center stat-card" :class="{active: filesFilter === 'completed'}" @click="showFiles('completed')">
                <div class="text-xl font-bold text-green-400">{{ stats.completed || 0 }}</div>
                <div class="text-xs text-gray-400">Completed</div>
            </div>
            <div class="card rounded-lg p-3 text-center stat-card" :class="{active: filesFilter === 'failed'}" @click="showFiles('failed')">
                <div class="text-xl font-bold text-red-400">{{ stats.failed || 0 }}</div>
                <div class="text-xs text-gray-400">Failed</div>
            </div>
            <div class="card rounded-lg p-3 text-center stat-card" :class="{active: filesFilter === 'skipped'}" @click="showFiles('skipped')">
                <div class="text-xl font-bold text-gray-400">{{ stats.skipped || 0 }}</div>
                <div class="text-xs text-gray-400">Skipped</div>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="flex border-b border-gray-700 mb-4 space-x-1">
            <button @click="tab = 'home'; filesFilter = ''" :class="tab === 'home' ? 'border-b-2 border-blue-500 text-white' : 'text-gray-400'" class="pb-2 px-4">Home</button>
            <button @click="tab = 'import'; filesFilter = ''" :class="tab === 'import' ? 'border-b-2 border-blue-500 text-white' : 'text-gray-400'" class="pb-2 px-4">Import</button>
            <button @click="tab = 'libraries'; filesFilter = ''" :class="tab === 'libraries' ? 'border-b-2 border-blue-500 text-white' : 'text-gray-400'" class="pb-2 px-4">Libraries</button>
            <button @click="tab = 'files'" :class="tab === 'files' ? 'border-b-2 border-blue-500 text-white' : 'text-gray-400'" class="pb-2 px-4">Files</button>
        </div>

        <!-- ==================== HOME TAB ==================== -->
        <div v-if="tab === 'home'">
            <!-- Manage Content -->
            <div class="flex items-center justify-between mb-4 flex-wrap gap-2">
                <label class="flex items-center gap-2 text-sm text-gray-300">
                    <input type="checkbox" v-model="manageContent" @change="resetSelectionOnToggle" class="rounded">
                    Manage content
                </label>
                <div v-if="manageContent" class="flex items-center gap-2 flex-wrap">
                    <span class="text-xs text-gray-400">Selected {{ selectedMediaCount }}</span>
                    <button @click="selectAllVisible" class="btn-secondary px-3 py-1 rounded text-sm">Select All</button>
                    <button @click="clearSelectedMedia" class="btn-secondary px-3 py-1 rounded text-sm">Clear</button>
                    <button @click="fixMatchSelected" class="btn-primary px-3 py-1 rounded text-sm">🔎 Fix Match</button>
                    <button @click="rescanSelected" class="btn-secondary px-3 py-1 rounded text-sm">🔄 Rescan</button>
                    <button @click="deleteSelectedMedia" class="btn-danger px-3 py-1 rounded text-sm">🗑 Delete</button>
                </div>
            </div>
            <!-- Library Filter Tabs -->
            <div class="flex items-center gap-2 mb-4 flex-wrap">
                <button @click="mediaLibFilter = 'all'" 
                        :class="mediaLibFilter === 'all' ? 'btn-primary' : 'btn-secondary'" 
                        class="px-3 py-1 rounded text-sm">All ({{ media.length }})</button>
                <button v-for="lib in libraries" :key="lib.id"
                        @click="mediaLibFilter = lib.id" 
                        :class="mediaLibFilter === lib.id ? 'btn-primary' : 'btn-secondary'" 
                        class="px-3 py-1 rounded text-sm">{{ lib.path.split('/').pop() }} ({{ media.filter(m => m.library_id === lib.id).length }})</button>
            </div>
            
            <!-- Poster Grid -->
            <div v-if="filteredMedia.length > 0" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-3">
                <div v-for="m in filteredMedia" :key="m.id" 
                     @click="handleMediaClick(m)"
                     class="poster-card cursor-pointer card"
                     :class="manageContent && isSelected(m.id) ? 'ring-2 ring-blue-400' : ''">
                    <div class="poster-wrap">
                        <img v-if="m.poster_path" 
                             :src="'https://image.tmdb.org/t/p/w300' + m.poster_path" 
                             :alt="m.title" loading="lazy">
                        <div v-else class="no-poster">
                            <div><div class="text-2xl mb-1">🎬</div>{{ m.title }}</div>
                        </div>
                    </div>
                    
                    <!-- Status badge -->
                    <div class="absolute top-1" :class="manageContent ? 'right-8' : 'right-1'">
                        <span v-if="m.file_count > 0 && m.completed_count >= m.file_count" 
                              class="bg-green-600 text-white text-xs px-1.5 py-0.5 rounded font-bold">✓</span>
                        <span v-else-if="m.processing_count > 0" 
                              class="bg-blue-600 text-white text-xs px-1.5 py-0.5 rounded animate-pulse">⟳</span>
                        <span v-else-if="m.pending_count > 0" 
                              class="bg-yellow-600 text-white text-xs px-1.5 py-0.5 rounded">{{ m.pending_count }}</span>
                        <span v-else-if="m.failed_count > 0" 
                              class="bg-red-600 text-white text-xs px-1.5 py-0.5 rounded">!</span>
                    </div>

                    <div v-if="manageContent" class="absolute top-1 right-1">
                        <input type="checkbox" :checked="isSelected(m.id)" @click.stop="toggleMediaSelection(m.id)" class="rounded">
                    </div>
                    
                    <!-- Profile badge -->
                    <div class="absolute top-1 left-1">
                        <span :class="'profile-' + m.profile" class="text-xs px-1 py-0.5 rounded font-bold uppercase" style="font-size:9px;">{{ m.profile }}</span>
                    </div>
                    
                    <!-- Title overlay -->
                    <div class="overlay">
                        <div class="text-white text-xs font-medium truncate">{{ m.title }}</div>
                        <div class="text-gray-400" style="font-size:10px;">{{ m.year || '' }} <span v-if="m.file_count">• {{ m.completed_count || 0 }}/{{ m.file_count }} files</span></div>
                    </div>
                </div>
            </div>
            
            <div v-else class="text-center text-gray-500 py-12">
                <p class="text-lg mb-2">No media found</p>
                <p class="text-sm">Go to the <span class="text-blue-400 cursor-pointer" @click="tab = 'import'">Import</span> tab to add media from your libraries.</p>
            </div>
        </div>

        <!-- ==================== IMPORT TAB ==================== -->
        <div v-if="tab === 'import'" class="space-y-6">
            <!-- Batch Scan & Import -->
            <div class="card rounded-lg p-6">
                <h3 class="text-lg font-semibold text-white mb-4">📂 Batch Scan & Import</h3>
                <div class="flex gap-3 mb-4">
                    <select v-model="selectedLibrary" class="flex-1 bg-gray-700 rounded px-3 py-2">
                        <option value="">Select library...</option>
                        <option v-for="lib in libraries" :value="lib.id">{{ lib.path }}</option>
                    </select>
                    <button @click="scanLibraryPreview" :disabled="!selectedLibrary || scanning" class="btn-primary px-4 py-2 rounded">
                        {{ scanning ? 'Scanning...' : 'Scan & Match' }}
                    </button>
                </div>
                
                <!-- Scan Preview -->
                <div v-if="scanPreview">
                    <div class="flex justify-between items-center mb-3">
                        <span>{{ scanPreview.matched }}/{{ scanPreview.total }} matched</span>
                        <div class="flex gap-2">
                            <button @click="selectAllPreviews" class="btn-secondary px-3 py-1 rounded text-sm">
                                {{ allSelected ? 'Deselect All' : 'Select All' }}
                            </button>
                            <button @click="importSelected" class="btn-primary px-3 py-1 rounded text-sm">
                                Import Selected ({{ selectedCount }})
                            </button>
                        </div>
                    </div>
                    <div class="space-y-2 max-h-96 overflow-y-auto">
                        <div v-for="p in scanPreview.previews" :key="p.path" 
                             class="card rounded p-3" :class="p.match_status === 'matched' ? 'match-good' : 'match-none'">
                            <div class="flex items-center gap-3">
                                <input type="checkbox" v-model="p.selected" class="flex-shrink-0">
                                <div class="flex-1 min-w-0">
                                    <div class="text-white truncate">{{ p.folder_name }}</div>
                                    <div class="text-sm" :class="p.tmdb_title ? 'text-green-400' : 'text-yellow-400'">
                                        {{ p.tmdb_title || 'No TMDB match' }}
                                        <span v-if="p.year" class="text-gray-400">({{ p.year }})</span>
                                    </div>
                                </div>
                                <select v-model="p.profile" class="bg-gray-700 rounded px-2 py-1 text-sm">
                                    <option v-for="(pr, key) in profiles" :value="key">{{ key }}</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>
                <div v-if="importResult" class="mt-4 p-3 bg-gray-800 rounded text-sm">
                    Imported: {{ importResult.total_imported }} | Failed: {{ importResult.total_failed }}
                </div>
            </div>
            
            <!-- Manual Search -->
            <div class="card rounded-lg p-6">
                <h3 class="text-lg font-semibold text-white mb-4">🔍 Manual TMDB Search</h3>
                <div class="flex gap-3 mb-4">
                    <input v-model="searchQuery" @keyup.enter="searchTMDB" placeholder="Enter title..." class="flex-1 bg-gray-700 rounded px-3 py-2">
                    <button @click="searchTMDB" class="btn-primary px-4 py-2 rounded">Search</button>
                </div>
                <div v-if="searchResults.length" class="space-y-2 mb-4">
                    <div v-for="r in searchResults" :key="r.tmdb_id + r.media_type" 
                         @click="selectResult(r)"
                         class="card rounded p-3 cursor-pointer hover:border-blue-500 flex items-center gap-3"
                         :class="selectedResult?.tmdb_id === r.tmdb_id ? 'border-blue-500' : ''">
                        <img v-if="r.poster_path" :src="'https://image.tmdb.org/t/p/w92' + r.poster_path" class="w-10 h-14 rounded object-cover">
                        <div class="flex-1 min-w-0">
                            <span class="text-white">{{ r.title }}</span>
                            <span class="text-gray-400 text-sm ml-2">({{ r.year || '?' }})</span>
                            <span class="text-xs px-2 py-0.5 rounded ml-2" :class="r.media_type === 'movie' ? 'bg-purple-600' : 'bg-blue-600'">{{ r.media_type }}</span>
                        </div>
                    </div>
                </div>
                <div v-if="selectedResult" class="p-4 bg-gray-800 rounded space-y-3">
                    <div class="text-white font-medium">{{ selectedResult.title }} ({{ selectedResult.year }})</div>
                    <div class="flex gap-3 items-end">
                        <div class="flex-1">
                            <label class="text-xs text-gray-400 block mb-1">Folder Path</label>
                            <div class="flex gap-2">
                                <input v-model="importPath" placeholder="/media/movies/..." class="flex-1 bg-gray-700 rounded px-3 py-2 text-sm">
                                <button @click="showBrowser = true; browseTarget = 'import'" class="btn-secondary px-3 py-2 rounded text-sm">Browse</button>
                            </div>
                        </div>
                        <div>
                            <label class="text-xs text-gray-400 block mb-1">Profile</label>
                            <select v-model="importProfile" class="bg-gray-700 rounded px-3 py-2 text-sm">
                                <option value="">Auto ({{ suggestedProfile }})</option>
                                <option v-for="(pr, key) in profiles" :value="key">{{ key }}</option>
                            </select>
                        </div>
                        <button @click="importMedia" class="btn-primary px-4 py-2 rounded text-sm">Import</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- ==================== LIBRARIES TAB ==================== -->
        <div v-if="tab === 'libraries'" class="space-y-4">
            <div class="card rounded-lg p-4 flex gap-3">
                <input v-model="newLibraryPath" placeholder="/media/movies" class="flex-1 bg-gray-700 rounded px-3 py-2">
                <button @click="showBrowser = true; browseTarget = 'library'" class="btn-secondary px-3 py-2 rounded text-sm">Browse</button>
                <button @click="addLibrary" class="btn-primary px-4 py-2 rounded">Add Library</button>
            </div>
            <div v-for="lib in libraries" :key="lib.id" class="card rounded-lg p-4">
                <div class="flex justify-between items-center">
                    <div>
                        <div class="text-white font-medium">{{ lib.path }}</div>
                        <div class="text-sm text-gray-400 mt-1">
                            <span v-if="lib.auto_import" class="text-green-400">● Auto-import ON</span>
                            <span v-else class="text-gray-500">○ Auto-import OFF</span>
                            <span class="ml-3">Profile: {{ lib.default_profile || 'default' }}</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-2 relative">
                        <button @click="toggleLibraryMenu(lib.id)" class="btn-secondary px-3 py-1 rounded text-sm">⚙️</button>
                        <button @click="removeLibrary(lib.id)" class="btn-danger px-3 py-1 rounded text-sm">Remove</button>
                        
                        <div v-if="openLibraryMenu === lib.id" class="dropdown-menu card rounded-lg p-4 space-y-3">
                            <div class="flex items-center justify-between">
                                <span class="text-sm text-gray-400">Auto-import</span>
                                <label class="toggle-switch">
                                    <input type="checkbox" :checked="lib.auto_import" @change="updateLibrarySettings(lib.id, { auto_import: $event.target.checked })">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                            <div>
                                <label class="text-sm text-gray-400 block mb-1">Default Profile</label>
                                <select :value="lib.default_profile || 'default'" 
                                        @change="updateLibrarySettings(lib.id, { default_profile: $event.target.value })"
                                        class="bg-gray-700 rounded px-3 py-1 text-sm w-full">
                                    <option v-for="(pr, key) in profiles" :value="key">{{ key }}</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ==================== FILES TAB ==================== -->
        <div v-if="tab === 'files'">
            <div class="flex justify-between items-center mb-4">
                <div class="flex space-x-2">
                    <button v-for="s in ['pending', 'processing', 'completed', 'failed', 'skipped']" :key="s"
                            @click="filesFilter = s; loadFiles()"
                            :class="filesFilter === s ? 'btn-primary' : 'btn-secondary'"
                            class="px-3 py-1 rounded text-sm capitalize">{{ s }}</button>
                </div>
                <div class="flex space-x-2">
                    <button v-if="filesFilter === 'pending' && files.length > 0" 
                            @click="clearPending" 
                            class="bg-red-600 hover:bg-red-500 px-3 py-1 rounded text-sm">
                        Clear Pending ({{ files.length }})
                    </button>
                    <button v-if="filesFilter === 'failed' && files.length > 0" 
                            @click="retryAllFailed" 
                            class="btn-success px-3 py-1 rounded text-sm">
                        Retry All Failed ({{ files.length }})
                    </button>
                    <button v-if="filesFilter === 'skipped' && files.length > 0" 
                            @click="requeueSkipped" 
                            class="bg-yellow-600 hover:bg-yellow-500 px-3 py-1 rounded text-sm">
                        Requeue Skipped ({{ files.length }})
                    </button>
                </div>
            </div>
            <div class="space-y-2">
                <div v-for="f in files" :key="f.id" class="card rounded-lg p-3">
                    <div class="flex justify-between items-start">
                        <div class="flex-1 min-w-0">
                            <div class="text-white truncate text-sm">{{ f.file_path.split('/').pop() }}</div>
                            <div class="text-xs text-gray-400">{{ f.media_title }} • {{ f.profile }}</div>
                        </div>
                        <div class="flex items-center gap-3">
                            <div class="text-right">
                                <span :class="{
                                    'text-yellow-400': f.status === 'pending',
                                    'text-blue-400': f.status === 'processing',
                                    'text-green-400': f.status === 'completed',
                                    'text-red-400': f.status === 'failed',
                                    'text-gray-400': f.status === 'skipped'
                                }" class="text-sm capitalize">{{ f.status }}</span>
                                <div v-if="f.error_message" class="text-xs text-red-400 mt-1 max-w-xs truncate">{{ f.error_message }}</div>
                            </div>
                            <div class="flex space-x-2">
                                <button v-if="f.status === 'completed'" 
                                        @click="resyncFile(f.id, f.file_path.split('/').pop())" 
                                        class="bg-purple-600 hover:bg-purple-500 px-2 py-1 rounded text-xs opacity-75 hover:opacity-100"
                                        title="Experimental: Try ffsubsync">Sync</button>
                                <button v-if="f.status === 'failed' || f.status === 'skipped'" 
                                        @click="retryFile(f.id)" 
                                        class="btn-primary px-2 py-1 rounded text-xs">Retry</button>
                            </div>
                        </div>
                    </div>
                </div>
                <div v-if="files.length === 0" class="text-center text-gray-500 py-8">No files with status: {{ filesFilter }}</div>
            </div>
        </div>

        <!-- ==================== MEDIA DETAIL MODAL ==================== -->
        <div v-if="selectedMedia" class="fixed inset-0 modal flex items-center justify-center z-50 p-4" @click.self="selectedMedia = null">
            <div class="card rounded-lg w-full max-w-4xl max-h-full overflow-hidden flex flex-col" style="max-height: 90vh;">
                <!-- Modal Header -->
                <div class="flex items-start gap-4 p-5 border-b border-gray-700">
                    <img v-if="selectedMedia.poster_path" 
                         :src="'https://image.tmdb.org/t/p/w185' + selectedMedia.poster_path" 
                         class="w-20 rounded shadow-lg flex-shrink-0">
                    <div v-else class="w-20 h-28 rounded bg-gray-700 flex items-center justify-center flex-shrink-0 text-2xl">🎬</div>
                    <div class="flex-1 min-w-0">
                        <h2 class="text-xl font-bold text-white">{{ selectedMedia.title }}</h2>
                        <p class="text-gray-400 text-sm mt-0.5">
                            {{ selectedMedia.year || '' }} • 
                            <span class="uppercase">{{ selectedMedia.media_type }}</span> • 
                            <span :class="'profile-' + selectedMedia.profile" class="text-xs px-1.5 py-0.5 rounded">{{ selectedMedia.profile }}</span>
                        </p>
                        <div class="flex gap-3 mt-2 text-xs">
                            <span class="text-green-400">✓ {{ selectedMedia.completed_count || 0 }}</span>
                            <span class="text-yellow-400">⏳ {{ selectedMedia.pending_count || 0 }}</span>
                            <span class="text-red-400">✗ {{ selectedMedia.failed_count || 0 }}</span>
                            <span class="text-gray-500">⊘ {{ selectedMedia.skipped_count || 0 }}</span>
                            <span class="text-gray-400">{{ (selectedMedia.file_count || 0) }} total</span>
                        </div>
                        <div class="flex gap-2 mt-3 flex-wrap">
                            <button @click="queueAllMediaFiles(selectedMedia.id)" class="btn-primary px-3 py-1 rounded text-sm">⏳ Queue All</button>
                            <button @click="scanMedia(selectedMedia.id)" class="btn-secondary px-3 py-1 rounded text-sm">🔄 Rescan</button>
                            <button @click="fixMatchMedia(selectedMedia.id)" class="btn-secondary px-3 py-1 rounded text-sm">🔎 Fix Match</button>
                            <select :value="selectedMedia.profile" @change="updateProfile(selectedMedia.id, $event.target.value)" class="bg-gray-700 rounded px-2 py-1 text-sm">
                                <option v-for="(pr, key) in profiles" :value="key">{{ key }}</option>
                            </select>
                            <button @click="deleteMedia(selectedMedia.id)" class="btn-danger px-3 py-1 rounded text-sm ml-auto">🗑 Delete</button>
                        </div>
                    </div>
                    <button @click="selectedMedia = null" class="text-gray-400 hover:text-white text-2xl flex-shrink-0 leading-none">&times;</button>
                </div>
                
                <!-- File List Header -->
                <div v-if="mediaFiles.length > 0" class="px-5 py-2 border-b border-gray-800 text-xs text-gray-500 flex items-center">
                    <span class="flex-1">{{ mediaFiles.length }} file{{ mediaFiles.length !== 1 ? 's' : '' }}</span>
                    <span class="w-20 text-center">Status</span>
                    <span class="w-16 text-center">Action</span>
                </div>
                
                <!-- File List -->
                <div class="flex-1 overflow-y-auto scrollbar-thin">
                    <div v-if="mediaFiles.length === 0" class="text-center text-gray-500 py-12">
                        <div class="text-3xl mb-2">📂</div>
                        No files found. Click "Rescan" to search the folder for video files.
                    </div>
                    <div v-for="(f, idx) in mediaFiles" :key="f.id" 
                         class="flex items-center gap-3 px-5 py-2.5 hover:bg-gray-800 transition-colors"
                         :class="idx < mediaFiles.length - 1 ? 'border-b border-gray-800' : ''">
                        <!-- Status dot -->
                        <div class="w-2 h-2 rounded-full flex-shrink-0" :class="{
                            'bg-yellow-400': f.status === 'pending',
                            'bg-blue-400 animate-pulse': f.status === 'processing',
                            'bg-green-400': f.status === 'completed',
                            'bg-red-400': f.status === 'failed',
                            'bg-gray-500': f.status === 'skipped'
                        }"></div>
                        
                        <!-- File info -->
                        <div class="flex-1 min-w-0">
                            <div class="text-white text-sm truncate">{{ f.file_path.split('/').pop() }}</div>
                            <div class="text-xs text-gray-500 flex gap-2">
                                <span v-if="f.file_size">{{ (f.file_size / 1073741824).toFixed(1) }} GB</span>
                                <span v-if="f.error_message" class="text-red-400 truncate">{{ f.error_message }}</span>
                            </div>
                        </div>
                        
                        <!-- Status label -->
                        <span class="w-20 text-center text-xs capitalize flex-shrink-0" :class="{
                            'text-yellow-400': f.status === 'pending',
                            'text-blue-400': f.status === 'processing',
                            'text-green-400': f.status === 'completed',
                            'text-red-400': f.status === 'failed',
                            'text-gray-400': f.status === 'skipped'
                        }">{{ f.status }}</span>
                        
                        <!-- Action button -->
                        <button @click.stop="queueFile(f.id)" 
                                class="w-16 text-center px-2 py-1 rounded text-xs flex-shrink-0"
                                :class="f.status === 'processing' ? 'bg-gray-700 text-gray-500 cursor-not-allowed' : f.status === 'pending' ? 'bg-gray-700 text-yellow-400' : 'btn-primary'"
                                :disabled="f.status === 'processing'">
                            {{ f.status === 'completed' ? 'Redo' : f.status === 'pending' ? 'Queued' : f.status === 'processing' ? '...' : 'Queue' }}
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- File Browser Modal -->
        <div v-if="showBrowser" class="fixed inset-0 modal flex items-center justify-center z-50" @click.self="showBrowser = false">
            <div class="card rounded-lg p-6 w-full max-w-2xl max-h-96 overflow-hidden flex flex-col">
                <div class="flex justify-between items-center mb-4">
                    <h3 class="font-semibold text-white">Browse Folders</h3>
                    <button @click="showBrowser = false" class="text-gray-400 hover:text-white text-2xl">&times;</button>
                </div>
                <div class="text-sm text-gray-400 mb-2">{{ browserPath }}</div>
                <div class="flex-1 overflow-y-auto space-y-1">
                    <div v-if="browserParent" @click="browseTo(browserParent)" class="p-2 rounded cursor-pointer hover:bg-gray-700 flex items-center">
                        <span class="mr-2">📁</span> ..
                    </div>
                    <div v-for="item in browserItems" :key="item.path" @click="browseTo(item.path)" 
                         class="p-2 rounded cursor-pointer hover:bg-gray-700 flex items-center">
                        <span class="mr-2">📁</span> {{ item.name }}
                    </div>
                </div>
                <div class="mt-4 flex justify-end space-x-2">
                    <button @click="showBrowser = false" class="btn-secondary px-4 py-2 rounded">Cancel</button>
                    <button @click="selectBrowserPath" class="btn-primary px-4 py-2 rounded">Select This Folder</button>
                </div>
            </div>
        </div>

        <!-- Toast -->
        <div v-if="toast" class="fixed bottom-4 right-4 bg-gray-800 border border-gray-600 rounded-lg px-4 py-3 shadow-lg z-50">{{ toast }}</div>
    </div>

    <script>
    const { createApp, ref, computed, onMounted } = Vue;
    
    createApp({
        setup() {
            const tab = ref('home');
            const stats = ref({});
            const status = ref({});
            const media = ref([]);
            const libraries = ref([]);
            const files = ref([]);
            const filesFilter = ref('');
            const profiles = ref({});
            const toast = ref('');
            
            // Scan preview
            const selectedLibrary = ref('');
            const scanning = ref(false);
            const scanPreview = ref(null);
            const importResult = ref(null);
            
            // Manual search
            const searchQuery = ref('');
            const searchResults = ref([]);
            const selectedResult = ref(null);
            const suggestedProfile = ref('');
            const importPath = ref('');
            const importProfile = ref('');
            
            // Libraries
            const newLibraryPath = ref('');
            const openLibraryMenu = ref(null);
            
            // Media homepage
            const mediaLibFilter = ref('all');
            const selectedMedia = ref(null);
            const mediaFiles = ref([]);
            const manageContent = ref(false);
            const selectedMediaIds = ref([]);
            
            // File browser
            const showBrowser = ref(false);
            const browserPath = ref('/media');
            const browserParent = ref(null);
            const browserItems = ref([]);
            const browseTarget = ref('import');
            
            // Computed
            const filteredMedia = computed(() => {
                if (mediaLibFilter.value === 'all') return media.value;
                return media.value.filter(m => m.library_id === mediaLibFilter.value);
            });
            
            const selectedCount = computed(() => {
                if (!scanPreview.value) return 0;
                return scanPreview.value.previews.filter(p => p.selected).length;
            });

            const selectedMediaCount = computed(() => selectedMediaIds.value.length);
            
            const allSelected = computed(() => {
                if (!scanPreview.value) return false;
                return scanPreview.value.previews.length > 0 && scanPreview.value.previews.every(p => p.selected);
            });
            
            const showToast = (msg) => { toast.value = msg; setTimeout(() => toast.value = '', 3000); };
            
            const api = async (method, url, body) => {
                const opts = { method, headers: { 'Content-Type': 'application/json' } };
                if (body) opts.body = JSON.stringify(body);
                const res = await fetch(url, opts);
                return res.json();
            };
            
            const refresh = async () => {
                stats.value = await api('GET', '/api/stats');
                status.value = await api('GET', '/api/status');
                media.value = await api('GET', '/api/media');
                libraries.value = await api('GET', '/api/libraries');
                if (filesFilter.value) loadFiles();
            };
            
            const loadFiles = async () => {
                if (filesFilter.value) {
                    files.value = await api('GET', `/api/files/${filesFilter.value}`);
                }
            };
            
            const showFiles = (status) => {
                filesFilter.value = status;
                tab.value = 'files';
                loadFiles();
            };
            
            // Scan library with TMDB preview
            const scanLibraryPreview = async () => {
                if (!selectedLibrary.value) return;
                scanning.value = true;
                scanPreview.value = null;
                importResult.value = null;
                showToast('Scanning library and matching with TMDB...');
                
                try {
                    const result = await api('POST', `/api/libraries/${selectedLibrary.value}/scan-preview`);
                    // Find the library to get default profile
                    const lib = libraries.value.find(l => l.id === selectedLibrary.value);
                    const defaultProfile = lib?.default_profile || 'default';
                    
                    result.previews.forEach(p => {
                        p.selected = true; // Select all by default
                        // For unmatched items, use library's default profile
                        if (p.match_status !== 'matched') {
                            p.profile = defaultProfile;
                        }
                    });
                    scanPreview.value = result;
                    showToast(`Found ${result.total} folders, ${result.matched} matched`);
                } catch (e) {
                    showToast('Scan failed');
                }
                scanning.value = false;
            };
            
            const selectAllPreviews = () => {
                if (!scanPreview.value) return;
                const newState = !allSelected.value;
                scanPreview.value.previews.forEach(p => {
                    p.selected = newState;
                });
            };
            
            const importSelected = async () => {
                if (!scanPreview.value) return;
                const items = scanPreview.value.previews
                    .filter(p => p.selected)
                    .map(p => ({
                        tmdb_id: p.tmdb_id,
                        media_type: p.media_type || 'tv',
                        path: p.path,
                        folder_name: p.folder_name,
                        title: p.title,
                        year: p.year,
                        profile: p.profile
                    }));
                
                if (items.length === 0) {
                    showToast('No items selected');
                    return;
                }
                
                showToast(`Importing ${items.length} items...`);
                importResult.value = await api('POST', '/api/libraries/batch-import', items);
                showToast(`Imported ${importResult.value.total_imported} items`);
                scanPreview.value = null;
                await refresh();
            };
            
            // Manual search
            const searchTMDB = async () => {
                if (!searchQuery.value) return;
                // Use multi-search for best results (searches movies + TV simultaneously)
                const results = await api('GET', `/api/tmdb/search/multi?query=${encodeURIComponent(searchQuery.value)}`);
                searchResults.value = results.slice(0, 10);
                selectedResult.value = null;
            };
            
            const selectResult = async (r) => {
                selectedResult.value = r;
                const endpoint = r.media_type === 'tv' ? `/api/tmdb/tv/${r.tmdb_id}` : `/api/tmdb/movie/${r.tmdb_id}`;
                const details = await api('GET', endpoint);
                suggestedProfile.value = details.suggested_profile;
            };
            
            const importMedia = async () => {
                if (!selectedResult.value || !importPath.value) { showToast('Please fill in folder path'); return; }
                const result = await api('POST', '/api/media/import', {
                    tmdb_id: selectedResult.value.tmdb_id,
                    media_type: selectedResult.value.media_type,
                    folder_path: importPath.value,
                    profile: importProfile.value || null
                });
                showToast(`Imported ${result.title} (${result.files_found.new} files)`);
                selectedResult.value = null; searchResults.value = []; importPath.value = '';
                await refresh();
            };
            
            const scanMedia = async (id) => {
                const result = await api('POST', `/api/media/${id}/scan`);
                showToast(`Found ${result.new} new files`);
                await refresh();
                // Refresh detail view if open
                if (selectedMedia.value && selectedMedia.value.id === id) {
                    await openMediaDetail(selectedMedia.value);
                }
            };

            const fixMatchMedia = async (id) => {
                showToast('Searching TMDB for best match...');
                try {
                    const result = await api('POST', `/api/media/${id}/fix-match`);
                    showToast(`Matched ${result.title}`);
                    await refresh();
                    if (selectedMedia.value && selectedMedia.value.id === id) {
                        const updated = media.value.find(m => m.id === id);
                        if (updated) {
                            selectedMedia.value = updated;
                        }
                        await openMediaDetail(selectedMedia.value);
                    }
                } catch (e) {
                    showToast('No TMDB match found');
                }
            };
            
            const deleteMedia = async (id) => {
                if (confirm('Delete this media entry and all its file records from Subarr? This will not remove any files from disk.')) {
                    await api('DELETE', `/api/media/${id}`);
                    selectedMedia.value = null;
                    showToast('Deleted');
                    await refresh();
                }
            };
            
            const openMediaDetail = async (m) => {
                selectedMedia.value = m;
                try {
                    const result = await api('GET', `/api/media/${m.id}/files`);
                    mediaFiles.value = result.files;
                } catch (e) {
                    mediaFiles.value = [];
                }
            };

            const isSelected = (id) => selectedMediaIds.value.includes(id);

            const toggleMediaSelection = (id) => {
                if (isSelected(id)) {
                    selectedMediaIds.value = selectedMediaIds.value.filter(item => item !== id);
                } else {
                    selectedMediaIds.value = [...selectedMediaIds.value, id];
                }
            };

            const handleMediaClick = async (m) => {
                if (manageContent.value) {
                    toggleMediaSelection(m.id);
                } else {
                    await openMediaDetail(m);
                }
            };

            const selectAllVisible = () => {
                selectedMediaIds.value = filteredMedia.value.map(item => item.id);
            };

            const clearSelectedMedia = () => {
                selectedMediaIds.value = [];
            };

            const resetSelectionOnToggle = () => {
                if (!manageContent.value) {
                    clearSelectedMedia();
                }
            };

            const fixMatchSelected = async () => {
                if (selectedMediaIds.value.length === 0) {
                    showToast('No media selected');
                    return;
                }
                showToast(`Fixing matches for ${selectedMediaIds.value.length} items...`);
                let matched = 0;
                for (const id of selectedMediaIds.value) {
                    try {
                        await api('POST', `/api/media/${id}/fix-match`);
                        matched += 1;
                    } catch (e) {
                        // Ignore individual failures
                    }
                }
                showToast(`Updated ${matched} matches`);
                await refresh();
            };

            const rescanSelected = async () => {
                if (selectedMediaIds.value.length === 0) {
                    showToast('No media selected');
                    return;
                }
                if (!confirm(`Rescan ${selectedMediaIds.value.length} selected item(s)?`)) return;
                for (const id of selectedMediaIds.value) {
                    await api('POST', `/api/media/${id}/scan`);
                }
                showToast('Rescan started');
                await refresh();
            };

            const deleteSelectedMedia = async () => {
                if (selectedMediaIds.value.length === 0) {
                    showToast('No media selected');
                    return;
                }
                if (!confirm(`Delete ${selectedMediaIds.value.length} selected item(s) from Subarr? This will not remove any files from disk.`)) return;
                for (const id of selectedMediaIds.value) {
                    await api('DELETE', `/api/media/${id}`);
                }
                clearSelectedMedia();
                showToast('Deleted selected media');
                await refresh();
            };
            
            const queueFile = async (fileId) => {
                await api('POST', `/api/files/${fileId}/queue`);
                showToast('File queued');
                // Refresh the detail view
                if (selectedMedia.value) {
                    await openMediaDetail(selectedMedia.value);
                }
                await refresh();
            };
            
            const queueAllMediaFiles = async (mediaId) => {
                if (!confirm('Queue all files for processing? This includes already-completed files.')) return;
                const result = await api('POST', `/api/media/${mediaId}/queue`);
                showToast(`${result.count} files queued`);
                if (selectedMedia.value) {
                    await openMediaDetail(selectedMedia.value);
                }
                await refresh();
            };
            
            const updateProfile = async (id, profile) => {
                await api('PUT', `/api/media/${id}/profile`, { profile });
                showToast(`Profile updated to ${profile}`);
                await refresh();
            };
            
            const addLibrary = async () => {
                if (!newLibraryPath.value) return;
                await api('POST', '/api/libraries', { path: newLibraryPath.value });
                newLibraryPath.value = '';
                showToast('Library added');
                await refresh();
            };
            
            const removeLibrary = async (id) => {
                if (confirm('Remove this library from Subarr? This will delete ALL media and file records associated with it, but will not remove any files from disk.')) {
                    await api('DELETE', `/api/libraries/${id}`);
                    showToast('Library and associated media removed');
                    await refresh();
                }
            };
            
            const toggleLibraryMenu = (id) => {
                openLibraryMenu.value = openLibraryMenu.value === id ? null : id;
            };
            
            const updateLibrarySettings = async (id, settings) => {
                await api('PUT', `/api/libraries/${id}`, settings);
                showToast('Settings updated');
                await refresh();
            };
            
            const browseTo = async (path) => {
                const result = await api('GET', `/api/browse?path=${encodeURIComponent(path)}`);
                browserPath.value = result.current;
                browserParent.value = result.parent;
                browserItems.value = result.items;
            };
            
            const selectBrowserPath = () => {
                if (browseTarget.value === 'library') {
                    newLibraryPath.value = browserPath.value;
                } else {
                    importPath.value = browserPath.value;
                }
                showBrowser.value = false;
            };
            
            const cancelProcessing = async () => {
                const result = await api('POST', '/api/process/cancel');
                showToast(result.status === 'cancelled' ? 'Cancellation requested' : 'No active processing');
                await refresh();
            };
            
            const retryFile = async (fileId) => {
                await api('POST', `/api/files/${fileId}/retry`);
                showToast('File queued for retry');
                await loadFiles();
            };
            
            const retryAllFailed = async () => {
                const result = await api('POST', '/api/files/retry-all-failed');
                showToast(`${result.count} files queued for retry`);
                await loadFiles();
                await refresh();
            };
            
            const requeueSkipped = async () => {
                if (!confirm('Requeue all skipped files? They will be processed again.')) return;
                const result = await api('POST', '/api/files/requeue-skipped');
                showToast(`${result.count} skipped files requeued`);
                await loadFiles();
                await refresh();
            };
            
            const clearPending = async () => {
                if (!confirm('Remove all pending files from the queue? This cannot be undone.')) return;
                const result = await api('POST', '/api/files/clear-pending');
                showToast(`${result.count} pending files cleared`);
                await loadFiles();
                await refresh();
            };
            
            const resyncFile = async (fileId, filename) => {
                // Confirm before proceeding - ffsubsync can make things worse
                if (!confirm(`⚠️ Experimental Feature\n\nFFSubsync will attempt to re-align subtitle timing for:\n${filename || 'this file'}\n\nWarning: This may make timing WORSE on some files. The original timing from Whisper is usually accurate.\n\nContinue?`)) {
                    return;
                }
                
                try {
                    const result = await api('POST', `/api/files/${fileId}/resync`);
                    showToast(`Sync started: ${result.file}`);
                } catch (e) {
                    showToast('Sync failed: ' + (e.message || 'Unknown error'));
                }
            };
            
            // Close dropdown when clicking outside
            document.addEventListener('click', (e) => {
                if (!e.target.closest('.dropdown-menu') && !e.target.closest('button')) {
                    openLibraryMenu.value = null;
                }
            });
            
            onMounted(async () => {
                profiles.value = await api('GET', '/api/profiles');
                await refresh();
                await browseTo('/media');
                setInterval(refresh, 5000);
            });
            
            return {
                tab, stats, status, media, libraries, files, filesFilter, profiles, toast,
                selectedLibrary, scanning, scanPreview, importResult,
                searchQuery, searchResults, selectedResult, suggestedProfile, importPath, importProfile,
                newLibraryPath, openLibraryMenu,
                mediaLibFilter, selectedMedia, mediaFiles, filteredMedia, manageContent, selectedMediaIds,
                showBrowser, browserPath, browserParent, browserItems, browseTarget,
                selectedCount, allSelected, selectedMediaCount,
                showFiles, loadFiles, scanLibraryPreview, selectAllPreviews, importSelected,
                searchTMDB, selectResult, importMedia, scanMedia, fixMatchMedia, deleteMedia, updateProfile,
                openMediaDetail, handleMediaClick, isSelected, toggleMediaSelection,
                selectAllVisible, clearSelectedMedia, resetSelectionOnToggle,
                fixMatchSelected, rescanSelected, deleteSelectedMedia, queueFile, queueAllMediaFiles,
                addLibrary, removeLibrary, toggleLibraryMenu, updateLibrarySettings,
                browseTo, selectBrowserPath, cancelProcessing, retryFile, retryAllFailed, requeueSkipped, clearPending, resyncFile
            };
        }
    }).mount('#app');
    </script>
</body>
</html>
'''

@app.on_event("startup")
async def startup_event():
    global db, tmdb, scanner, engine, whisper_model
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    db = DatabaseManager(DATABASE_PATH)
    db.reset_stuck_processing()  # Reset any files stuck from previous crash
    
    tmdb = TMDBClient(TMDB_API_KEY)
    
    device, backend = detect_device()
    log(f"Loading Whisper model ({MODEL_NAME}) on {device}...")
    whisper_model = WhisperModel(MODEL_NAME, device, backend)
    
    scanner = LibraryScanner(db, tmdb)
    engine = ProcessingEngine(db, whisper_model)
    
    log("AI-SUB initialized")


@app.get("/", response_class=HTMLResponse)
async def root():
    return WEB_UI_HTML

@app.get("/api/stats")
async def get_stats():
    return db.get_stats()

@app.get("/api/status")
async def get_status():
    current = db.get_current_processing() if db else None
    return {
        "current_file": current.get("file_path") if current else None,
        "current_filename": os.path.basename(current["file_path"]) if current else None,
        "progress": current.get("progress", 0) if current else 0,
        "stage": engine.current_stage if engine else "",
        "stats": db.get_stats() if db else {}
    }

@app.get("/api/libraries")
async def list_libraries():
    libs = db.get_libraries()
    # Convert auto_import from int to bool for frontend
    for lib in libs:
        lib["auto_import"] = bool(lib.get("auto_import", 0))
    return libs

@app.post("/api/libraries")
async def add_library(path: str = Body(..., embed=True)):
    if not os.path.exists(path):
        raise HTTPException(404, f"Path does not exist: {path}")
    
    library_id = db.add_library(path)
    return {"id": library_id, "path": path}

@app.put("/api/libraries/{library_id}")
async def update_library_settings(
    library_id: int,
    auto_import: Optional[bool] = Body(None),
    default_profile: Optional[str] = Body(None)
):
    """Update library settings"""
    if default_profile and default_profile not in PROFILES:
        raise HTTPException(400, f"Invalid profile. Must be one of: {list(PROFILES.keys())}")
    
    db.update_library(library_id, auto_import=auto_import, default_profile=default_profile)
    return {"status": "updated"}

@app.delete("/api/libraries/{library_id}")
async def remove_library(library_id: int):
    db.remove_library(library_id)
    return {"status": "removed"}

@app.post("/api/libraries/{library_id}/scan")
async def scan_library(library_id: int):
    libraries = db.get_libraries()
    library = next((l for l in libraries if l["id"] == library_id), None)
    if not library:
        raise HTTPException(404, "Library not found")
    
    return scanner.scan_library(library["path"])

@app.post("/api/libraries/{library_id}/scan-preview")
async def scan_library_preview(library_id: int):
    """Scan library and preview TMDB matches with suggested profiles"""
    libraries = db.get_libraries()
    library = next((l for l in libraries if l["id"] == library_id), None)
    if not library:
        raise HTTPException(404, "Library not found")
    
    scan_result = scanner.scan_library(library["path"])
    if "error" in scan_result:
        raise HTTPException(400, scan_result["error"])
    
    previews = []
    for folder in scan_result.get("new_folders", []):
        title = folder["title"]
        year = folder["year"]
        folder_name = folder["folder_name"]
        
        preview = {
            "folder_name": folder_name,
            "path": folder["path"],
            "title": title,
            "year": year,
            "tmdb_id": None,
            "tmdb_title": None,
            "media_type": None,
            "profile": "default",
            "match_status": "no_match"
        }
        
        # Clean the title for better search results
        clean_title = tmdb.clean_search_query(title)
        
        # Strategy 1: Multi-search with year (best results)
        results = tmdb.search_multi(clean_title, year)
        
        # Strategy 2: If no results, try without year
        if not results and year:
            results = tmdb.search_multi(clean_title)
        
        # Strategy 3: If still nothing, try the raw folder name cleaned up
        if not results and clean_title != title:
            results = tmdb.search_multi(title, year)
        
        # Strategy 4: Try movie-specific search (sometimes more accurate)
        if not results:
            results = tmdb.search_movie(clean_title, year)
            if not results and year:
                results = tmdb.search_movie(clean_title)
        
        if results:
            match = results[0]
            tmdb_id = match["tmdb_id"]
            media_type = match.get("media_type", "movie")
            
            # Get details for profile detection
            if media_type == "tv":
                details = tmdb.get_tv_details(tmdb_id)
            else:
                details = tmdb.get_movie_details(tmdb_id)
            
            if details:
                profile = tmdb.detect_profile(details)
                preview.update({
                    "tmdb_id": tmdb_id,
                    "tmdb_title": details["title"],
                    "media_type": media_type,
                    "profile": profile,
                    "match_status": "matched",
                    "genres": details.get("genre_names", []),
                    "origin_country": details.get("origin_country", []),
                    "poster_path": details.get("poster_path")
                })
        
        previews.append(preview)
    
    return {
        "library_path": library["path"],
        "total": len(previews),
        "matched": sum(1 for p in previews if p["match_status"] == "matched"),
        "unmatched": sum(1 for p in previews if p["match_status"] == "no_match"),
        "previews": previews
    }

@app.post("/api/libraries/batch-import")
async def batch_import(items: List[Dict] = Body(...)):
    """Import multiple items with user-specified profiles"""
    imported = []
    failed = []
    
    for item in items:
        if not item.get("path"):
            failed.append({"folder": item.get("folder_name", "unknown"), "reason": "Missing path"})
            continue

        details = None  # Initialize to avoid using stale value from previous iteration
        tmdb_id = item.get("tmdb_id")
        media_type = item.get("media_type", "tv")
        folder_path = item["path"]
        profile = item.get("profile", "default")
        folder_name = item.get("folder_name", os.path.basename(folder_path))
        title = item.get("title", folder_name)
        year = item.get("year")
        
        # If we have TMDB ID, get details
        if tmdb_id:
            if media_type == "tv":
                details = tmdb.get_tv_details(tmdb_id)
            else:
                details = tmdb.get_movie_details(tmdb_id)
            
            if details:
                title = details["title"]
                year = int(details["year"]) if details.get("year") else year
                genres = details.get("genres", [])
                origin_country = details.get("origin_country", [])
            else:
                # TMDB lookup failed, use folder info
                tmdb_id = 0
                genres = []
                origin_country = []
        else:
            # No TMDB match - use folder info with user-selected profile
            tmdb_id = 0
            genres = []
            origin_country = []
        
        try:
            # Find library_id
            lib_id = None
            for lib in db.get_libraries():
                if folder_path.startswith(lib["path"]):
                    lib_id = lib["id"]
                    break
            
            media_id = db.add_media(
                tmdb_id=tmdb_id,
                media_type=media_type,
                title=title,
                year=int(year) if year else None,
                genres=genres,
                origin_country=origin_country,
                profile=profile,
                folder_path=folder_path,
                poster_path=details.get("poster_path") if details else None,
                library_id=lib_id
            )
            
            file_scan = scanner.scan_media_files(media_id)
            
            imported.append({
                "folder": folder_name,
                "title": title,
                "profile": profile,
                "files": file_scan.get("new", 0)
            })
        except Exception as e:
            failed.append({"folder": folder_name, "reason": str(e)})
    
    return {
        "imported": imported,
        "failed": failed,
        "total_imported": len(imported),
        "total_failed": len(failed)
    }

@app.get("/api/tmdb/search/tv")
async def search_tv(query: str, year: Optional[int] = None):
    return tmdb.search_tv(query, year)

@app.get("/api/tmdb/search/movie")
async def search_movie(query: str, year: Optional[int] = None):
    return tmdb.search_movie(query, year)

@app.get("/api/tmdb/search/multi")
async def search_multi(query: str, year: Optional[int] = None):
    """Search across movies and TV shows simultaneously"""
    return tmdb.search_multi(query, year)

@app.get("/api/tmdb/tv/{tmdb_id}")
async def get_tv_details(tmdb_id: int):
    details = tmdb.get_tv_details(tmdb_id)
    if not details:
        raise HTTPException(404, "TV show not found")
    details["suggested_profile"] = tmdb.detect_profile(details)
    return details

@app.get("/api/tmdb/movie/{tmdb_id}")
async def get_movie_details(tmdb_id: int):
    details = tmdb.get_movie_details(tmdb_id)
    if not details:
        raise HTTPException(404, "Movie not found")
    details["suggested_profile"] = tmdb.detect_profile(details)
    return details

@app.get("/api/media")
async def list_media():
    media_list = db.get_all_media_with_stats()
    for m in media_list:
        m["genres"] = json.loads(m["genres"]) if m["genres"] else []
        m["origin_country"] = json.loads(m["origin_country"]) if m["origin_country"] else []
    return media_list

@app.get("/api/media/by-library/{library_id}")
async def list_media_by_library(library_id: int):
    """Get all media for a specific library with file stats"""
    media_list = db.get_media_by_library(library_id)
    for m in media_list:
        m["genres"] = json.loads(m["genres"]) if m["genres"] else []
        m["origin_country"] = json.loads(m["origin_country"]) if m["origin_country"] else []
    return media_list

@app.get("/api/media/{media_id}/files")
async def get_media_files(media_id: int):
    """Get all files for a specific media entry"""
    media = db.get_media_by_id(media_id)
    if not media:
        raise HTTPException(404, "Media not found")
    files = db.get_media_files(media_id)
    return {
        "media": {
            "id": media["id"],
            "title": media["title"],
            "year": media.get("year"),
            "profile": media["profile"],
            "poster_path": media.get("poster_path"),
            "media_type": media.get("media_type", "movie"),
            "tmdb_id": media.get("tmdb_id"),
        },
        "files": files
    }

@app.post("/api/media/{media_id}/queue")
async def queue_media(media_id: int):
    """Queue ALL files for a media entry (even completed ones)"""
    media = db.get_media_by_id(media_id)
    if not media:
        raise HTTPException(404, "Media not found")
    count = db.queue_media_files(media_id)
    return {"status": "queued", "count": count, "title": media["title"]}

@app.post("/api/files/{file_id}/queue")
async def queue_single_file(file_id: int):
    """Queue a single file for processing (works regardless of current status)"""
    success = db.queue_file(file_id)
    if not success:
        raise HTTPException(404, "File not found")
    return {"status": "queued", "file_id": file_id}

@app.post("/api/media/import")
async def import_media(
    tmdb_id: int = Body(...),
    media_type: str = Body(...),
    folder_path: str = Body(...),
    profile: Optional[str] = Body(None)
):
    if media_type == "tv":
        details = tmdb.get_tv_details(tmdb_id)
    else:
        details = tmdb.get_movie_details(tmdb_id)
    
    if not details:
        raise HTTPException(404, "TMDB entry not found")
    
    final_profile = profile or tmdb.detect_profile(details)
    
    # Find which library this folder belongs to
    library_id = None
    libraries = db.get_libraries()
    for lib in libraries:
        if folder_path.startswith(lib["path"]):
            library_id = lib["id"]
            break
    
    media_id = db.add_media(
        tmdb_id=tmdb_id,
        media_type=media_type,
        title=details["title"],
        year=int(details["year"]) if details.get("year") else None,
        genres=details.get("genres", []),
        origin_country=details.get("origin_country", []),
        profile=final_profile,
        folder_path=folder_path,
        poster_path=details.get("poster_path"),
        library_id=library_id
    )
    
    scan_result = scanner.scan_media_files(media_id)
    
    return {
        "media_id": media_id,
        "title": details["title"],
        "profile": final_profile,
        "files_found": scan_result
    }

@app.get("/api/media/{media_id}")
async def get_media(media_id: int):
    media = db.get_media_by_id(media_id)
    if not media:
        raise HTTPException(404, "Media not found")
    
    media["genres"] = json.loads(media["genres"]) if media["genres"] else []
    media["origin_country"] = json.loads(media["origin_country"]) if media["origin_country"] else []
    return media

@app.get("/api/media/{media_id}/files")
async def get_media_files(media_id: int):
    """Get all files for a specific media with status info"""
    media = db.get_media_by_id(media_id)
    if not media:
        raise HTTPException(404, "Media not found")
    
    files = db.get_media_files(media_id)
    
    # Add filename for display
    for f in files:
        f["filename"] = os.path.basename(f["file_path"])
        f["exists"] = os.path.exists(f["file_path"])
    
    return {
        "media": {
            "id": media["id"],
            "title": media["title"],
            "year": media.get("year"),
            "media_type": media.get("media_type"),
            "profile": media.get("profile"),
            "poster_path": media.get("poster_path"),
            "tmdb_id": media.get("tmdb_id")
        },
        "files": files
    }

@app.post("/api/files/{file_id}/queue")
async def queue_file(file_id: int):
    """Queue or re-queue a specific file for processing, regardless of current status"""
    conn = db._get_conn()
    try:
        row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        if not row:
            raise HTTPException(404, "File not found")
        
        conn.execute(
            "UPDATE files SET status = 'pending', error_message = NULL, progress = 0 WHERE id = ?",
            (file_id,)
        )
        conn.commit()
        return {"status": "queued", "file_id": file_id, "filename": os.path.basename(row["file_path"])}
    finally:
        conn.close()

@app.post("/api/media/{media_id}/queue-all")
async def queue_all_media_files(media_id: int):
    """Queue all files for a media entry, regardless of current status"""
    media = db.get_media_by_id(media_id)
    if not media:
        raise HTTPException(404, "Media not found")
    
    conn = db._get_conn()
    try:
        cursor = conn.execute(
            "UPDATE files SET status = 'pending', error_message = NULL, progress = 0 WHERE media_id = ?",
            (media_id,)
        )
        count = cursor.rowcount
        conn.commit()
        return {"status": "queued", "count": count}
    finally:
        conn.close()

@app.put("/api/media/{media_id}/profile")
async def update_media_profile(media_id: int, profile: str = Body(..., embed=True)):
    if profile not in PROFILES:
        raise HTTPException(400, f"Invalid profile. Must be one of: {list(PROFILES.keys())}")
    
    db.update_media_profile(media_id, profile)
    return {"status": "updated", "profile": profile}

@app.post("/api/media/{media_id}/fix-match")
async def fix_media_match(media_id: int):
    media = db.get_media_by_id(media_id)
    if not media:
        raise HTTPException(404, "Media not found")

    title = media.get("title") or os.path.basename(media.get("folder_path", ""))
    year = media.get("year")
    media_type = media.get("media_type", "movie")

    clean_title = tmdb.clean_search_query(title)
    results = tmdb.search_multi(clean_title, year)
    if not results and year:
        results = tmdb.search_multi(clean_title)
    if not results and clean_title != title:
        results = tmdb.search_multi(title, year)
    if not results:
        if media_type == "tv":
            results = tmdb.search_tv(clean_title, year) or tmdb.search_tv(clean_title)
        else:
            results = tmdb.search_movie(clean_title, year) or tmdb.search_movie(clean_title)

    if not results:
        raise HTTPException(404, "No TMDB match found")

    def match_score(item: Dict[str, Any]) -> Tuple[int, int]:
        type_match = 0 if item.get("media_type") == media_type else 1
        year_match = 0 if year and str(year) == str(item.get("year")) else 1
        return (type_match, year_match)

    results.sort(key=match_score)
    match = results[0]
    tmdb_id = match["tmdb_id"]
    matched_type = match.get("media_type", media_type)

    if matched_type == "tv":
        details = tmdb.get_tv_details(tmdb_id)
    else:
        details = tmdb.get_movie_details(tmdb_id)

    if not details:
        raise HTTPException(404, "TMDB entry not found")

    profile = tmdb.detect_profile(details)
    db.update_media_match(
        media_id=media_id,
        tmdb_id=tmdb_id,
        media_type=matched_type,
        title=details["title"],
        year=int(details["year"]) if details.get("year") else None,
        genres=details.get("genres", []),
        origin_country=details.get("origin_country", []),
        profile=profile,
        poster_path=details.get("poster_path"),
    )

    return {
        "status": "matched",
        "media_id": media_id,
        "tmdb_id": tmdb_id,
        "title": details["title"],
        "profile": profile,
        "poster_path": details.get("poster_path"),
        "media_type": matched_type,
    }

@app.post("/api/media/{media_id}/scan")
async def scan_media_files(media_id: int):
    return scanner.scan_media_files(media_id)

@app.delete("/api/media/{media_id}")
async def delete_media(media_id: int):
    db.delete_media(media_id)
    return {"status": "deleted"}

@app.post("/api/process")
async def trigger_processing(limit: int = Query(1, ge=1, le=10)):
    processed = engine.process_pending(limit)
    return {"processed": processed}

@app.post("/api/process/cancel")
async def cancel_processing():
    """Cancel current transcription"""
    if engine and engine.cancel_current():
        return {"status": "cancelled", "file": engine.current_file}
    return {"status": "no_active_processing"}

@app.get("/api/queue")
async def get_queue():
    return db.get_pending_files(50)

@app.get("/api/profiles")
async def list_profiles():
    return PROFILES

@app.get("/api/files/{status}")
async def get_files_by_status(status: str, limit: int = Query(100, ge=1, le=500)):
    """Get files filtered by status (pending, processing, completed, failed, skipped)"""
    valid_statuses = ["pending", "processing", "completed", "failed", "skipped"]
    if status not in valid_statuses:
        raise HTTPException(400, f"Invalid status. Must be one of: {valid_statuses}")
    return db.get_files_by_status(status, limit)

@app.post("/api/files/{file_id}/retry")
async def retry_file(file_id: int):
    """Reset a failed file back to pending for retry"""
    conn = db._get_conn()
    try:
        cursor = conn.execute("UPDATE files SET status = 'pending', error_message = NULL, progress = 0 WHERE id = ?", (file_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(404, "File not found")
        return {"status": "queued", "file_id": file_id}
    finally:
        conn.close()

@app.post("/api/files/retry-all-failed")
async def retry_all_failed():
    """Reset all failed files back to pending"""
    conn = db._get_conn()
    try:
        cursor = conn.execute("UPDATE files SET status = 'pending', error_message = NULL, progress = 0 WHERE status = 'failed'")
        count = cursor.rowcount
        conn.commit()
        return {"status": "queued", "count": count}
    finally:
        conn.close()

@app.post("/api/files/requeue-skipped")
async def requeue_skipped():
    """Reset all skipped files back to pending"""
    conn = db._get_conn()
    try:
        cursor = conn.execute("UPDATE files SET status = 'pending', error_message = NULL, progress = 0 WHERE status = 'skipped'")
        count = cursor.rowcount
        conn.commit()
        return {"status": "queued", "count": count}
    finally:
        conn.close()

@app.post("/api/files/requeue-all")
async def requeue_all():
    """Reset all failed AND skipped files back to pending"""
    conn = db._get_conn()
    try:
        cursor = conn.execute("UPDATE files SET status = 'pending', error_message = NULL, progress = 0 WHERE status IN ('failed', 'skipped')")
        count = cursor.rowcount
        conn.commit()
        return {"status": "queued", "count": count}
    finally:
        conn.close()

@app.post("/api/files/clear-pending")
async def clear_pending():
    """Remove all pending files from the queue"""
    conn = db._get_conn()
    try:
        cursor = conn.execute("DELETE FROM files WHERE status = 'pending'")
        count = cursor.rowcount
        conn.commit()
        return {"status": "cleared", "count": count}
    finally:
        conn.close()

@app.post("/api/files/{file_id}/resync")
async def resync_file(file_id: int, background_tasks: BackgroundTasks):
    """Resync subtitles for a completed file using ffsubsync"""
    conn = db._get_conn()
    try:
        row = conn.execute("SELECT file_path, status FROM files WHERE id = ?", (file_id,)).fetchone()
        if not row:
            raise HTTPException(404, "File not found")
        
        if row["status"] != "completed":
            raise HTTPException(400, f"File must be completed to resync (current status: {row['status']})")
        
        file_path = row["file_path"]
        if not os.path.exists(file_path):
            raise HTTPException(404, f"Video file not found on disk")
        
        # Run resync in background
        def do_resync():
            success, message = resync_completed_file(file_path)
            log(f"Resync {'completed' if success else 'failed'}: {os.path.basename(file_path)} - {message}")
        
        background_tasks.add_task(do_resync)
        
        return {"status": "resync_started", "file_id": file_id, "file": os.path.basename(file_path)}
    finally:
        conn.close()

@app.post("/api/files/resync-all-completed")
async def resync_all_completed(background_tasks: BackgroundTasks, limit: int = Query(10, ge=1, le=100)):
    """Resync subtitles for all completed files (in batches)"""
    conn = db._get_conn()
    try:
        rows = conn.execute("""
            SELECT id, file_path FROM files 
            WHERE status = 'completed' 
            ORDER BY id
            LIMIT ?
        """, (limit,)).fetchall()
        
        if not rows:
            return {"status": "no_files", "message": "No completed files to resync"}
        
        files_to_resync = []
        for row in rows:
            if os.path.exists(row["file_path"]):
                files_to_resync.append(row["file_path"])
        
        def do_resync_batch():
            for file_path in files_to_resync:
                success, message = resync_completed_file(file_path)
                log(f"Batch resync {'✓' if success else '✗'}: {os.path.basename(file_path)}")
        
        background_tasks.add_task(do_resync_batch)
        
        return {
            "status": "resync_started",
            "count": len(files_to_resync),
            "files": [os.path.basename(f) for f in files_to_resync[:5]]  # First 5 for preview
        }
    finally:
        conn.close()

@app.get("/api/browse")
async def browse_directory(path: str = Query("/media")):
    """Browse directories for the file browser"""
    if not os.path.exists(path):
        # Try some common paths
        for fallback in ["/media", "/", "/mnt"]:
            if os.path.exists(fallback):
                path = fallback
                break
    
    parent = os.path.dirname(path) if path != "/" else None
    return {
        "current": path,
        "parent": parent,
        "items": list_directories(path)
    }

@app.get("/api/debug/audio-streams")
async def debug_audio_streams(file_path: str = Query(...)):
    """Debug endpoint to check audio streams in a file"""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}", "exists": False}
    
    file_size = os.path.getsize(file_path)
    
    # Run ffprobe with full output
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", file_path],
        capture_output=True, text=True, timeout=120
    )
    
    if result.returncode != 0:
        # Get error info
        error_result = subprocess.run(
            ["ffprobe", "-v", "error", file_path],
            capture_output=True, text=True, timeout=60
        )
        return {
            "error": "ffprobe failed",
            "file_path": file_path,
            "file_size": file_size,
            "returncode": result.returncode,
            "stderr": error_result.stderr[:1000] if error_result.stderr else None
        }
    
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw_output": result.stdout[:1000]}
    
    all_streams = data.get("streams", [])
    format_info = data.get("format", {})
    
    streams_summary = []
    audio_streams = []
    
    for stream in all_streams:
        stream_info = {
            "index": stream.get("index"),
            "codec_type": stream.get("codec_type"),
            "codec_name": stream.get("codec_name"),
            "tags": stream.get("tags", {})
        }
        streams_summary.append(stream_info)
        
        if stream.get("codec_type") == "audio":
            tags = stream.get("tags", {})
            audio_streams.append({
                "index": stream.get("index"),
                "codec": stream.get("codec_name"),
                "channels": stream.get("channels"),
                "language": tags.get("language") or tags.get("LANGUAGE") or "und",
                "title": tags.get("title") or tags.get("TITLE") or ""
            })
    
    return {
        "file_path": file_path,
        "file_size": file_size,
        "format": format_info.get("format_name"),
        "duration": format_info.get("duration"),
        "total_streams": len(all_streams),
        "audio_streams_count": len(audio_streams),
        "audio_streams": audio_streams,
        "all_streams_summary": streams_summary
    }

@app.get("/api/debug/failed-files")
async def debug_failed_files():
    """Check file paths for all failed files"""
    conn = db._get_conn()
    try:
        rows = conn.execute("""
            SELECT f.id, f.file_path, f.error_message, m.folder_path, m.title
            FROM files f
            JOIN media m ON f.media_id = m.id
            WHERE f.status = 'failed'
            LIMIT 50
        """).fetchall()
        
        results = []
        for row in rows:
            file_path = row["file_path"]
            folder_path = row["folder_path"]
            
            # Check if file exists
            file_exists = os.path.exists(file_path) if file_path else False
            folder_exists = os.path.exists(folder_path) if folder_path else False
            
            # If file doesn't exist, try to find it in the folder
            actual_files = []
            if folder_exists and not file_exists:
                try:
                    for f in os.listdir(folder_path):
                        if f.endswith(('.mkv', '.mp4', '.avi', '.m4v')):
                            actual_files.append(os.path.join(folder_path, f))
                except:
                    pass
            
            results.append({
                "id": row["id"],
                "title": row["title"],
                "file_path": file_path,
                "file_exists": file_exists,
                "folder_path": folder_path,
                "folder_exists": folder_exists,
                "error": row["error_message"],
                "actual_files_in_folder": actual_files[:5]  # First 5 files found
            })
        
        return {"failed_files": results, "count": len(results)}
    finally:
        conn.close()

@app.post("/api/cleanup/muxtemp")
async def cleanup_muxtemp_files():
    """Remove _muxtemp files from database and optionally from disk"""
    conn = db._get_conn()
    try:
        # Find all _muxtemp entries in database
        rows = conn.execute("""
            SELECT id, file_path FROM files 
            WHERE file_path LIKE '%_muxtemp%' OR file_path LIKE '%_muxed%'
        """).fetchall()
        
        removed_from_db = 0
        removed_from_disk = 0
        
        for row in rows:
            file_path = row["file_path"]
            
            # Remove from database
            conn.execute("DELETE FROM files WHERE id = ?", (row["id"],))
            removed_from_db += 1
            
            # Remove from disk if exists
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    removed_from_disk += 1
                    log(f"Cleaned up temp file: {os.path.basename(file_path)}")
                except Exception as e:
                    log(f"Could not delete temp file {file_path}: {e}", "warning")
        
        conn.commit()
        
        return {
            "status": "cleaned",
            "removed_from_database": removed_from_db,
            "removed_from_disk": removed_from_disk
        }
    finally:
        conn.close()

@app.post("/api/cleanup/fix-file-paths")
async def fix_file_paths():
    """Fix file paths by rescanning actual files on disk.
    
    This will:
    1. Find all files with status 'failed' or 'pending' where file doesn't exist
    2. Look in the media's folder for actual video files
    3. Update the database with correct paths
    """
    conn = db._get_conn()
    try:
        # Get all files that might have wrong paths
        rows = conn.execute("""
            SELECT f.id, f.file_path, f.status, m.id as media_id, m.folder_path, m.title
            FROM files f
            JOIN media m ON f.media_id = m.id
            WHERE f.status IN ('failed', 'pending')
        """).fetchall()
        
        fixed = 0
        deleted = 0
        already_ok = 0
        
        for row in rows:
            file_path = row["file_path"]
            folder_path = row["folder_path"]
            file_id = row["id"]
            
            # Check if current path exists
            if file_path and os.path.exists(file_path):
                already_ok += 1
                continue
            
            # File doesn't exist at stored path - look for actual files in folder
            if not folder_path or not os.path.exists(folder_path):
                # Folder doesn't exist either - delete the file entry
                conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                deleted += 1
                log(f"Deleted orphaned file entry: {file_path}")
                continue
            
            # Find actual video files in the folder
            actual_files = []
            exclude_patterns = ['_muxtemp', '_muxed', '.sample', '.part']
            
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    if not filename.lower().endswith(VIDEO_EXTENSIONS):
                        continue
                    if any(p in filename.lower() for p in exclude_patterns):
                        continue
                    actual_files.append(os.path.join(root, filename))
            
            if not actual_files:
                # No video files found in folder
                conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                deleted += 1
                log(f"No video files in folder, deleted entry: {row['title']}")
                continue
            
            # Try to match by filename similarity
            old_basename = os.path.basename(file_path) if file_path else ""
            best_match = None
            
            # First, look for exact match
            for actual_file in actual_files:
                if os.path.basename(actual_file) == old_basename:
                    best_match = actual_file
                    break
            
            # If no exact match, use the first file found (most common case: single file)
            if not best_match:
                best_match = actual_files[0]
            
            # Check if this file is already in the database
            existing = conn.execute("SELECT id FROM files WHERE file_path = ?", (best_match,)).fetchone()
            if existing:
                # File already exists with correct path - delete the wrong one
                conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                deleted += 1
                continue
            
            # Update the path
            conn.execute("""
                UPDATE files 
                SET file_path = ?, status = 'pending', error_message = NULL, progress = 0
                WHERE id = ?
            """, (best_match, file_id))
            fixed += 1
            log(f"Fixed path: {row['title']} -> {os.path.basename(best_match)}")
        
        conn.commit()
        
        return {
            "status": "completed",
            "fixed": fixed,
            "deleted_orphans": deleted,
            "already_correct": already_ok,
            "total_checked": len(rows)
        }
    finally:
        conn.close()

@app.post("/api/cleanup/rescan-all-media")
async def rescan_all_media():
    """Delete all file entries and rescan all media folders from scratch."""
    conn = db._get_conn()
    try:
        # Delete all file entries
        conn.execute("DELETE FROM files")
        conn.commit()
        log("Deleted all file entries for rescan")
    finally:
        conn.close()
    
    # Rescan all media
    all_media = db.get_all_media()
    total_files = 0
    
    for media in all_media:
        result = scanner.scan_media_files(media["id"])
        total_files += result.get("new", 0) + result.get("skipped", 0)
    
    return {
        "status": "completed",
        "media_rescanned": len(all_media),
        "total_files_found": total_files
    }

@app.get("/api/cleanup/find-temp-files")
async def find_temp_files():
    """Find all temp files in media libraries"""
    libraries = db.get_libraries()
    
    temp_patterns = ['_muxtemp', '_muxed', '.bak']
    found_files = []
    
    for library in libraries:
        lib_path = library["path"]
        if not os.path.exists(lib_path):
            continue
        
        for root, dirs, files in os.walk(lib_path):
            for filename in files:
                if any(pattern in filename for pattern in temp_patterns):
                    full_path = os.path.join(root, filename)
                    found_files.append({
                        "path": full_path,
                        "size": os.path.getsize(full_path),
                        "library": lib_path
                    })
    
    return {
        "temp_files": found_files,
        "count": len(found_files),
        "total_size_mb": sum(f["size"] for f in found_files) / (1024*1024)
    }

@app.post("/api/libraries/{library_id}/auto-import")
async def auto_import_library(library_id: int):
    """Scan library and auto-import all folders with TMDB matching"""
    libraries = db.get_libraries()
    library = next((l for l in libraries if l["id"] == library_id), None)
    if not library:
        raise HTTPException(404, "Library not found")
    
    scan_result = scanner.scan_library(library["path"])
    if "error" in scan_result:
        raise HTTPException(400, scan_result["error"])
    
    imported = []
    failed = []
    
    for folder in scan_result.get("new_folders", []):
        title = folder["title"]
        year = folder["year"]
        folder_path = folder["path"]
        folder_name = folder["folder_name"]
        
        # Clean the title for better search results
        clean_title = tmdb.clean_search_query(title)
        
        # Strategy 1: Multi-search with year (best results)
        results = tmdb.search_multi(clean_title, year)
        
        # Strategy 2: If no results, try without year
        if not results and year:
            results = tmdb.search_multi(clean_title)
        
        # Strategy 3: If still nothing, try the raw title
        if not results and clean_title != title:
            results = tmdb.search_multi(title, year)
        
        # Strategy 4: Try movie-specific search as last resort
        if not results:
            results = tmdb.search_movie(clean_title, year)
            if not results and year:
                results = tmdb.search_movie(clean_title)
        
        if not results:
            failed.append({"folder": folder_name, "reason": "No TMDB match"})
            continue
        
        # Take first result
        match = results[0]
        tmdb_id = match["tmdb_id"]
        media_type = match.get("media_type", "movie")
        
        # Get details for profile detection
        if media_type == "tv":
            details = tmdb.get_tv_details(tmdb_id)
        else:
            details = tmdb.get_movie_details(tmdb_id)
        
        if not details:
            failed.append({"folder": folder_name, "reason": "Could not get TMDB details"})
            continue
        
        profile = tmdb.detect_profile(details)
        
        # Import
        try:
            media_id = db.add_media(
                tmdb_id=tmdb_id,
                media_type=media_type,
                title=details["title"],
                year=int(details["year"]) if details.get("year") else None,
                genres=details.get("genres", []),
                origin_country=details.get("origin_country", []),
                profile=profile,
                folder_path=folder_path,
                poster_path=details.get("poster_path"),
                library_id=library_id
            )
            
            file_scan = scanner.scan_media_files(media_id)
            
            imported.append({
                "folder": folder_name,
                "title": details["title"],
                "tmdb_id": tmdb_id,
                "media_type": media_type,
                "profile": profile,
                "files": file_scan.get("new", 0)
            })
        except Exception as e:
            failed.append({"folder": folder_name, "reason": str(e)})
    
    return {
        "imported": imported,
        "failed": failed,
        "total_imported": len(imported),
        "total_failed": len(failed)
    }


# =============================================================================
# Background Processing
# =============================================================================

def background_processor():
    log("Background processor started")
    consecutive_errors = 0
    
    while True:
        try:
            if engine and db:
                pending = db.get_pending_files(1)
                if pending:
                    engine.process_pending(1)
                    consecutive_errors = 0  # Reset on success
                else:
                    time.sleep(10)
            else:
                time.sleep(5)
        except Exception as e:
            consecutive_errors += 1
            log(f"Background processor error: {e}", "error")
            # Exponential backoff: wait longer after repeated errors
            wait_time = min(30 * consecutive_errors, 300)  # Max 5 minutes
            log(f"Waiting {wait_time}s before retry (error #{consecutive_errors})")
            time.sleep(wait_time)

def auto_import_folder(folder: Dict, library: Dict) -> Optional[int]:
    """Auto-import a single folder using TMDB matching or default profile"""
    title = folder["title"]
    year = folder["year"]
    folder_path = folder["path"]
    default_profile = library.get("default_profile", "default")
    library_id = library.get("id")
    
    # Clean title for better search results
    clean_title = tmdb.clean_search_query(title)
    
    # Search TMDB using multi-search with fallbacks
    results = tmdb.search_multi(clean_title, year)
    if not results and year:
        results = tmdb.search_multi(clean_title)
    if not results and clean_title != title:
        results = tmdb.search_multi(title, year)
    
    if results:
        match = results[0]
        tmdb_id = match["tmdb_id"]
        media_type = match.get("media_type", "movie")
        
        if media_type == "tv":
            details = tmdb.get_tv_details(tmdb_id)
        else:
            details = tmdb.get_movie_details(tmdb_id)
        
        if details:
            profile = tmdb.detect_profile(details)
            media_id = db.add_media(
                tmdb_id=tmdb_id,
                media_type=media_type,
                title=details["title"],
                year=int(details["year"]) if details.get("year") else None,
                genres=details.get("genres", []),
                origin_country=details.get("origin_country", []),
                profile=profile,
                folder_path=folder_path,
                poster_path=details.get("poster_path"),
                library_id=library_id
            )
            log(f"Auto-imported: {details['title']} (profile: {profile})")
            return media_id
    
    # No TMDB match - use default profile
    media_id = db.add_media(
        tmdb_id=0,
        media_type="tv",
        title=title,
        year=year,
        genres=[],
        origin_country=[],
        profile=default_profile,
        folder_path=folder_path,
        library_id=library_id
    )
    log(f"Auto-imported (no TMDB match): {title} (profile: {default_profile})")
    return media_id

def library_scanner_loop():
    """Background loop that scans libraries for new content"""
    log("Library scanner started")
    
    # Track last scan times for faster polling of recent changes
    last_full_scan = 0
    QUICK_SCAN_INTERVAL = 30  # Check for new episodes every 30 seconds
    
    while True:
        try:
            current_time = time.time()
            
            if not (db and scanner and tmdb):
                time.sleep(5)
                continue
            
            # Quick scan: Check existing media for new files (new episodes)
            # Only scan media whose library still exists
            media_list = db.get_monitored_media()
            for media in media_list:
                result = scanner.scan_media_files(media["id"])
                if result.get("new", 0) > 0:
                    log(f"Found {result['new']} new files in: {media['title']}")
            
            # Full library scan: Check for new folders (new shows)
            # This runs every LIBRARY_SCAN_INTERVAL
            if current_time - last_full_scan >= LIBRARY_SCAN_INTERVAL:
                last_full_scan = current_time
                
                libraries = db.get_libraries()
                for library in libraries:
                    log(f"Scanning library: {library['path']}")
                    scan_result = scanner.scan_library(library["path"])
                    
                    new_folders = scan_result.get("new_folders", [])
                    if not new_folders:
                        continue
                    
                    # Auto-import if enabled for this library
                    if library.get("auto_import"):
                        log(f"Auto-importing {len(new_folders)} new folders from {library['path']}")
                        for folder in new_folders:
                            media_id = auto_import_folder(folder, library)
                            if media_id:
                                scanner.scan_media_files(media_id)
                    
                    db.update_library_scan_time(library["id"])
            
            time.sleep(QUICK_SCAN_INTERVAL)
            
        except Exception as e:
            log(f"Library scanner error: {e}", "error")
            import traceback
            traceback.print_exc()
            time.sleep(30)


# =============================================================================
# Main
# =============================================================================

class EndpointFilter(logging.Filter):
    """Filter out noisy HTTP endpoint logs - only show errors and important requests"""
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        
        # Filter out ALL successful (200/304) polling endpoints
        noisy_patterns = [
            "/api/stats",
            "/api/status", 
            "/api/media",
            "/api/libraries",
            "/api/queue",
            "/api/files/",      # /api/files/pending, /api/files/completed, etc.
            "/api/profiles",
            "/api/browse",
            "/ HTTP",           # Root page loads
        ]
        
        # Only filter out successful responses (200, 304)
        if "200" in message or "304" in message:
            for pattern in noisy_patterns:
                if pattern in message:
                    return False
        
        return True

def main():
    log("=" * 70)
    log(f"AI-SUB v{VERSION} - Library Manager")
    log("=" * 70)
    
    device, backend = detect_device()
    log(f"Device: {device} | Backend: {backend} | Model: {MODEL_NAME}")
    log(f"API Port: {WEBHOOK_PORT}")
    log(f"Data Directory: {DATA_DIR}")
    log(f"TMDB API: {'Configured' if TMDB_API_KEY else 'NOT CONFIGURED'}")
    log("=" * 70)
    
    processor_thread = threading.Thread(target=background_processor, daemon=True)
    processor_thread.start()
    
    scanner_thread = threading.Thread(target=library_scanner_loop, daemon=True)
    scanner_thread.start()
    
    log(f"\nStarting API server on port {WEBHOOK_PORT}...")
    log(f"Web UI: http://YOUR_IP:{WEBHOOK_PORT}")
    
    # Add filter to suppress noisy health check logs
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
    
    uvicorn.run(app, host="0.0.0.0", port=WEBHOOK_PORT, log_level="warning" if not DEBUG else "info")

if __name__ == "__main__":
    main()
