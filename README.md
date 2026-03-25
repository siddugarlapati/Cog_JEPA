# COG-JEPA: Cognitive Predictive Memory Architecture

A complete vision agent system with predictive memory, built for Apple Silicon M4 MacBooks.

## Features

- **Video Pipeline** - Process video frames with JEPA (Joint Embedding Predictive Architecture)
- **Surprise Detection** - Detect unexpected events based on prediction errors
- **Semantic Memory** - Store and query events using memory gates
- **Image Search** - Semantic search across uploaded images using MiniLM embeddings
- **Unified Search** - Cross-modal search across both images and video events
- **Web Dashboard** - Gradio-based UI for real-time visualization

## Requirements

- Python 3.11+
- Apple Silicon M4 Mac (or any Mac with MPS)
- 16GB RAM recommended

## Installation

```bash
# Clone the repo
git clone https://github.com/siddugarlapati/Cog_JEPA.git
cd Cog_JEPA

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install sentence-transformers (for semantic search)
pip install sentence-transformers
```

## Usage

### Test Mode
```bash
python main.py --mode test
```

### Web Dashboard
```bash
python main.py --mode dashboard
```
Then open http://localhost:7860

### Webcam Mode
```bash
python main.py --mode webcam
```

### API Server
```bash
python main.py --mode api
```

## Project Structure

```
cog_jepa/
├── main.py              # Entry point
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── setup.sh            # Setup script
├── encoder/            # Vision encoders
├── jepa/               # JEPA components
├── memory/             # Memory system
├── reasoning/          # LLM reasoner
├── pipeline/           # Video pipeline
├── search/             # Semantic search
├── api/                # Flask API
└── ui/                 # Dashboard
```

## License

MIT