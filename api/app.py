"""
Flask API for COG-JEPA with Image Search Integration

REST API for semantic image search and video processing.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, List

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from config import config
from pipeline.video_pipeline import VideoPipeline
from memory.cognee_store import CogneeMemoryStore
from reasoning.llm_reasoner import LLMReasoner
from search.semantic_search import ImageSearchAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__, static_folder="static")
app.config["UPLOAD_FOLDER"] = config.data_dir + "/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename: str, allowed: set) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


# Global instances
image_search_api: Optional[ImageSearchAPI] = None
video_pipeline: Optional[VideoPipeline] = None
memory_store: Optional[CogneeMemoryStore] = None
llm_reasoner: Optional[LLMReasoner] = None


def get_image_search():
    """Get or initialize image search API."""
    global image_search_api
    if image_search_api is None:
        image_search_api = ImageSearchAPI()
    return image_search_api


def get_memory_store():
    """Get or initialize memory store."""
    global memory_store
    if memory_store is None:
        memory_store = CogneeMemoryStore()
    return memory_store


def get_llm_reasoner():
    """Get or initialize LLM reasoner."""
    global llm_reasoner
    if llm_reasoner is None:
        llm_reasoner = LLMReasoner(use_fallback=True)
    return llm_reasoner


# ========== Image Endpoints ==========


@app.route("/api/images/upload", methods=["POST"])
def upload_image():
    """Upload and index an image."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    caption = request.form.get("caption", "")

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({"error": "Invalid file type"}), 400

    # Save file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
    file.save(filepath)

    # Index image
    search_api = get_image_search()
    result = search_api.upload_image(filepath, caption if caption else None)

    return jsonify(
        {
            "success": True,
            "image_id": result.get("image_id"),
            "caption": result.get("caption"),
            "path": filepath,
        }
    )


@app.route("/api/images/search", methods=["GET"])
def search_images():
    """Search images by query."""
    query = request.args.get("q", "")
    top_k = int(request.args.get("top_k", 5))

    if not query:
        return jsonify({"error": "Query parameter 'q' required"}), 400

    search_api = get_image_search()
    results = search_api.search_images(query, top_k=top_k)

    return jsonify(
        {
            "query": query,
            "results": results,
            "count": len(results),
        }
    )


@app.route("/api/images", methods=["GET"])
def list_images():
    """List all indexed images."""
    search_api = get_image_search()
    images = search_api.search.get_all_images()
    return jsonify({"images": images, "count": len(images)})


@app.route("/api/images/<image_id>", methods=["DELETE"])
def delete_image(image_id: str):
    """Delete image from index."""
    search_api = get_image_search()
    success = search_api.search.delete_image(image_id)
    return jsonify({"success": success})


@app.route("/uploads/<filename>")
def serve_upload(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ========== Video/Event Endpoints ==========


@app.route("/api/events", methods=["GET"])
def list_events():
    """List stored video events."""
    store = get_memory_store()
    limit = int(request.args.get("limit", 50))
    events = store.get_recent_events_sync(n=limit)
    return jsonify({"events": events, "count": len(events)})


@app.route("/api/events/<event_id>", methods=["GET"])
def get_event(event_id: str):
    """Get specific event by ID."""
    store = get_memory_store()
    events = store.get_all_events()
    for event in events:
        if event.get("event_id") == event_id:
            return jsonify(event)
    return jsonify({"error": "Event not found"}), 404


@app.route("/api/events/query", methods=["POST"])
def query_events():
    """Query events with natural language."""
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question required"}), 400

    store = get_memory_store()
    reasoner = get_llm_reasoner()

    # Get context
    events = store.get_recent_events_sync(n=20)
    context_parts = [
        f"Frame {e.get('frame_index', '?')}: {e.get('description', 'N/A')}"
        for e in events
    ]
    context = "\n".join(context_parts)

    # Query
    answer = reasoner.query_memory(question, context)

    return jsonify(
        {
            "question": question,
            "answer": answer,
            "events_used": len(events),
        }
    )


@app.route("/api/summary", methods=["GET"])
def get_summary():
    """Get session summary."""
    store = get_memory_store()
    summary = store.get_summary_sync()
    return jsonify({"summary": summary})


# ========== Unified Search Endpoints ==========


@app.route("/api/search", methods=["GET"])
def unified_search():
    """Cross-modal search across images and video events."""
    query = request.args.get("q", "")
    top_k = int(request.args.get("top_k", 10))

    if not query:
        return jsonify({"error": "Query parameter 'q' required"}), 400

    search_api = get_image_search()
    store = get_memory_store()

    # Get video events
    events = store.get_recent_events_sync(n=50)

    # Unified search
    results = search_api.search_all(query, events=events)

    return jsonify(
        {
            "query": query,
            "images": results.get("images", []),
            "events": results.get("events", []),
            "unified": results.get("unified", []),
        }
    )


# ========== System Endpoints ==========


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get system statistics."""
    import torch

    stats = {
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "mps_available": torch.backends.mps.is_available(),
    }

    if torch.backends.mps.is_available():
        stats["mps_memory_gb"] = torch.mps.current_allocated_memory() / 1e9

    # Get image count
    search_api = get_image_search()
    stats["indexed_images"] = len(search_api.search.image_index)

    # Get event count
    store = get_memory_store()
    stats["stored_events"] = len(store.events)

    return jsonify(stats)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


# ========== Video Pipeline Endpoints ==========


@app.route("/api/pipeline/start", methods=["POST"])
def start_pipeline():
    """Start video processing pipeline."""
    global video_pipeline

    data = request.get_json() or {}
    mode = data.get("mode", "test")
    video_path = data.get("video_path")
    max_frames = data.get("max_frames")
    threshold = data.get("threshold", 0.9)

    # Determine video source
    if mode == "test":
        source = "test"
    elif mode == "webcam":
        source = 0
    elif mode == "file" and video_path:
        source = video_path
    else:
        source = "test"

    try:
        video_pipeline = VideoPipeline(
            video_source=source,
            use_llm=False,
            threshold_override=threshold,
        )

        import threading

        thread = threading.Thread(
            target=video_pipeline.run, args=(max_frames, False), daemon=True
        )
        thread.start()

        return jsonify({"status": "started", "mode": mode})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/stop", methods=["POST"])
def stop_pipeline():
    """Stop video processing pipeline."""
    global video_pipeline

    if video_pipeline is not None:
        video_pipeline.running = False
        return jsonify({"status": "stopped"})

    return jsonify({"error": "No pipeline running"}), 400


@app.route("/api/pipeline/status", methods=["GET"])
def pipeline_status():
    """Get pipeline status."""
    if video_pipeline is None:
        return jsonify({"status": "not_started"})

    state = video_pipeline.get_current_state()
    return jsonify(
        {
            "status": "running" if video_pipeline.running else "stopped",
            **state,
        }
    )


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Run Flask server."""
    logger.info(f"Starting COG-JEPA API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
