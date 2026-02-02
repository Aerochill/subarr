# 🎬 AI-SUB

**Intelligent Subtitle Generator with TMDB Integration**

AI-SUB is a self-hosted subtitle generator that automatically transcribes your media library using OpenAI's Whisper. It uses TMDB metadata to detect content type (anime, cartoon, TV, movie) and applies optimized transcription profiles.

Think of it as a replacement for Bazarr, but powered by AI transcription instead of downloading subtitles.

## ✨ Features

- **TMDB Integration** - Automatically detect anime vs cartoon vs TV vs movie
- **Smart Profiles** - Optimized VAD settings for different content types
- **Library Management** - Import shows/movies, scan for new files
- **Auto-Muxing** - Subtitles embedded directly into video files
- **Web UI** - Easy management interface
- **Intel XPU Support** - Native support for Intel Arc GPUs (A380, B580, etc.)
- **Background Processing** - Queue-based processing with status tracking

## 🚀 Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/youruser/subarr.git
cd subarr

# Copy example config and edit
cp .env.example .env
nano .env
```

### 2. Configure `.env`

```env
# Required: Get free API key at themoviedb.org
TMDB_API_KEY=your_key_here

# Point to your media libraries
MEDIA_TV=/path/to/tv
MEDIA_MOVIES=/path/to/movies
MEDIA_ANIME=/path/to/anime

# GPU config (use level_zero:0 for first GPU)
GPU_SELECTOR=level_zero:0
```

### 3. Start

```bash
docker compose up -d
```

### 4. Access Web UI

Open `http://YOUR_IP:9876` in your browser.

## 📖 Usage

### Adding a Library

1. Go to **Libraries** tab
2. Enter path (e.g., `/media/tv`)
3. Select type (TV or Movie)
4. Click **Add**

### Importing Media

1. Go to **Import** tab
2. Select a library and click **Scan**
3. Click on a discovered folder
4. Search TMDB to match the show/movie
5. Review the auto-detected profile
6. Click **Import & Scan**

Files are automatically queued for processing!

### Profile Auto-Detection

| TMDB Metadata | Profile | Optimized For |
|---------------|---------|---------------|
| Animation + Japan | `anime` | Japanese voice acting, music |
| Animation + US/UK | `cartoon` | Western animation style |
| Movie | `movie` | Film dialogue |
| TV Show | `tv` | Television dialogue |

### Muxing Behavior

By default, AI-SUB will:
1. Transcribe audio to SRT
2. Mux SRT into the video file
3. Replace the original file
4. Delete the external SRT

Configure in `.env`:
```env
MUX_SUBTITLES=true        # Mux into video
REPLACE_ORIGINAL=true     # Replace original file
DELETE_SRT_AFTER_MUX=true # Clean up SRT
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TMDB_API_KEY` | (required) | Your TMDB API key |
| `WHISPER_MODEL` | `large-v2` | Whisper model size |
| `GPU_SELECTOR` | `level_zero:0` | GPU to use |
| `MUX_SUBTITLES` | `true` | Embed subs in video |
| `REPLACE_ORIGINAL` | `true` | Replace original file |
| `DELETE_SRT_AFTER_MUX` | `true` | Delete external SRT |
| `LIBRARY_SCAN_INTERVAL` | `300` | Scan interval (seconds) |

### Adding More Libraries

Edit `docker-compose.yaml` to add more volume mounts:

```yaml
volumes:
  - ${MEDIA_ANIME_2:-/media/anime2}:/media/anime2:rw
```

## 🖥️ Hardware Support

### Intel Arc GPUs (Recommended)
- A380, A580, A750, A770
- B580, B570

### NVIDIA GPUs
Change the base image in `Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
```

### CPU Only
Set `GPU_SELECTOR=cpu` and use `int8` compute type.

## 📁 File Structure

```
ai-sub/
├── docker-compose.yaml  # Container config
├── Dockerfile           # Build instructions
├── script.py           # Main application
├── .env.example        # Configuration template
├── .env                # Your configuration (create this)
├── data/               # SQLite database
├── cache/              # Model cache
│   ├── whisper/
│   ├── huggingface/
│   └── torch-hub/
└── output/             # Temporary output
```

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/stats` | GET | Processing statistics |
| `/api/status` | GET | Current status |
| `/api/libraries` | GET/POST | Manage libraries |
| `/api/media` | GET | List imported media |
| `/api/media/import` | POST | Import from TMDB |
| `/api/tmdb/search/tv` | GET | Search TV shows |
| `/api/tmdb/search/movie` | GET | Search movies |
| `/api/queue` | GET | View processing queue |
| `/api/profiles` | GET | List available profiles |
| `/docs` | GET | Swagger API docs |

## 🐛 Troubleshooting

### "No GPU detected"
- Check `ls /dev/dri/` for render devices
- Verify `RENDER_GROUP` matches your system
- Try `GPU_SELECTOR=level_zero:0`

### "TMDB API error"
- Verify your API key at themoviedb.org
- Check network connectivity

### "Muxing failed"
- Ensure media paths are mounted with `:rw`
- Check disk space
- Verify ffmpeg can write to the location

## 📜 License

MIT License - feel free to use and modify!

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Stable-TS](https://github.com/jianfch/stable-ts)
- [Intel IPEX](https://github.com/intel/intel-extension-for-pytorch)
- [TMDB](https://www.themoviedb.org/)
