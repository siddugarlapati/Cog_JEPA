"""
Dashboard Module

Gradio-based dashboard for COG-JEPA system with integrated image search.
"""

import logging
import threading
import time
import os
from typing import Optional, List, Dict

import gradio as gr
import numpy as np
from PIL import Image

from config import config
from pipeline.video_pipeline import VideoPipeline
from memory.cognee_store import CogneeMemoryStore
from reasoning.llm_reasoner import LLMReasoner
from search.semantic_search import ImageSearchAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COGJEPDashboard:
    """
    Gradio dashboard for COG-JEPA system with integrated image search.

    Provides real-time visualization, video processing, and semantic image search.
    """

    def __init__(self, use_llm: bool = True):
        """Initialize dashboard."""
        self.pipeline: Optional[VideoPipeline] = None
        self.pipeline_thread: Optional[threading.Thread] = None
        self.use_llm = use_llm

        # Shared components
        self.memory_store = CogneeMemoryStore()
        self.llm_reasoner = LLMReasoner(use_fallback=not use_llm)

        # Image search
        self.image_search_api = ImageSearchAPI()

        # Upload directory
        self.upload_dir = "static/uploads"
        os.makedirs(self.upload_dir, exist_ok=True)

        # State
        self.running = False
        self.start_time: Optional[float] = None

        logger.info("Dashboard initialized with image search")

    def start_pipeline(
        self, mode: str, file_path: str, max_frames: int, threshold: float
    ):
        """
        Start the pipeline in a background thread.

        Args:
            mode: "test", "file", or "webcam"
            file_path: Path to video file (for file mode)
            max_frames: Maximum frames to process
            threshold: Surprise threshold
        """
        if self.running:
            return "Pipeline already running"

        # Determine video source
        if mode == "test":
            video_source = "test"
        elif mode == "webcam":
            video_source = 0
        elif mode == "file" and file_path:
            video_source = file_path
        else:
            video_source = "test"

        # Create pipeline with threshold
        self.pipeline = VideoPipeline(
            video_source=video_source,
            use_llm=self.use_llm,
            threshold_override=threshold,
        )

        # Start in background thread
        def run_pipeline():
            self.pipeline.run(
                max_frames=max_frames if max_frames > 0 else None, display=False
            )

        self.pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
        self.pipeline_thread.start()

        self.running = True
        self.start_time = time.time()

        return f"Pipeline started in {mode} mode with threshold {threshold}"

    def stop_pipeline(self):
        """Stop the running pipeline."""
        if not self.running or self.pipeline is None:
            return "No pipeline running"

        self.pipeline.running = False
        self.running = False

        # Wait for thread to finish
        if self.pipeline_thread is not None:
            self.pipeline_thread.join(timeout=5)

        return "Pipeline stopped"

    def get_state(self) -> Dict:
        """Get current pipeline state."""
        if self.pipeline is None:
            return {
                "frame_count": 0,
                "surprise_score": 0.0,
                "stored_events": 0,
                "compression_ratio": 0.0,
                "avg_surprise": 0.0,
                "surprise_history": [],
            }

        return self.pipeline.get_current_state()

    def query_memory(self, question: str) -> str:
        """
        Query memory with natural language.

        Args:
            question: User question

        Returns:
            Answer string
        """
        # Reload events from log file
        self.memory_store._load_from_log()

        if not self.memory_store.events:
            return "No events in memory yet. Start the pipeline first."

        # Get context from recent events
        recent = self.memory_store.get_recent_events_sync(n=10)
        context_parts = []
        for event in recent:
            context_parts.append(
                f"Frame {event.get('frame_index', '?')}: "
                f"surprise={event.get('surprise_score', 0):.3f}, "
                f"description={event.get('description', 'N/A')}"
            )
        context = "\n".join(context_parts)

        # Query with LLM or use video summary
        question_lower = question.lower()

        if (
            "summary" in question_lower
            or "explain" in question_lower
            or "what happened" in question_lower
            or "tell me" in question_lower
        ):
            # Use the video captioner's summary
            from reasoning.captioner import get_captioner

            captioner = get_captioner()
            events = self.memory_store.get_recent_events_sync(n=50)
            answer = captioner.generate_video_summary(events)
        else:
            answer = self.llm_reasoner.query_memory(question, context)

        return answer

    def get_recent_events(self) -> List[Dict]:
        """Get recent events for display."""
        # Reload from log first
        self.memory_store._load_from_log()
        return self.memory_store.get_recent_events_sync(n=10)

    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        state = self.get_state()

        # Calculate duration
        duration = 0.0
        if self.start_time is not None:
            duration = time.time() - self.start_time

        # Get MPS memory if available
        mps_memory = "N/A"
        try:
            import torch

            if torch.backends.mps.is_available():
                mps_memory = f"{torch.mps.current_allocated_memory() / 1e9:.2f} GB"
        except:
            pass

        return {
            "mps_memory": mps_memory,
            "frames_processed": state["frame_count"],
            "events_stored": state["stored_events"],
            "events_discarded": state["frame_count"] - state["stored_events"],
            "compression_ratio": f"{state['compression_ratio']:.1%}",
            "avg_surprise": f"{state['avg_surprise']:.3f}",
            "duration": f"{duration:.1f}s",
        }


def create_dashboard(use_llm: bool = True) -> gr.Blocks:
    """
    Create the Gradio dashboard interface.

    Args:
        use_llm: Whether to use LLM features

    Returns:
        Gradio Blocks interface
    """
    dashboard = COGJEPDashboard(use_llm=use_llm)

    with gr.Blocks(title="COG-JEPA Dashboard") as demo:
        gr.Markdown("# COG-JEPA: Cognitive Predictive Memory")
        gr.Markdown("Vision agent with predictive memory architecture")

        with gr.Tabs():
            # Tab 1: Live Feed
            with gr.Tab("Live Feed"):
                with gr.Row():
                    with gr.Column():
                        mode_select = gr.Radio(
                            ["test", "file", "webcam"],
                            label="Pipeline Mode",
                            value="test",
                        )
                        file_input = gr.File(
                            label="Upload Video File",
                            file_count="single",
                            file_types=["video"],
                        )
                        max_frames_input = gr.Number(
                            label="Max Frames (0 = unlimited)", value=100
                        )
                        threshold_input = gr.Number(
                            label="Surprise Threshold", value=0.9
                        )
                        start_btn = gr.Button("Start Pipeline", variant="primary")
                        stop_btn = gr.Button("Stop Pipeline", variant="stop")
                        status_output = gr.Textbox(label="Status")

                    with gr.Column():
                        live_plot = gr.LinePlot(
                            title="Surprise Score (Last 100 Frames)",
                            height=300,
                        )

                with gr.Row():
                    with gr.Column():
                        surprise_display = gr.Number(label="Current Surprise")
                    with gr.Column():
                        stored_display = gr.Number(label="Stored Events")
                    with gr.Column():
                        compression_display = gr.Number(label="Compression Ratio")

                # Auto-refresh
                refresh_btn = gr.Button("Refresh State")

                def update_state():
                    state = dashboard.get_state()
                    return (
                        state["surprise_score"],
                        state["stored_events"],
                        state["compression_ratio"],
                        state["surprise_history"],
                    )

                def start_pipeline(mode, file_obj, max_frames, threshold):
                    file_path = str(file_obj.name) if file_obj else None
                    return dashboard.start_pipeline(
                        mode, file_path, int(max_frames), float(threshold)
                    )

                start_btn.click(
                    start_pipeline,
                    inputs=[mode_select, file_input, max_frames_input, threshold_input],
                    outputs=status_output,
                )

                stop_btn.click(dashboard.stop_pipeline, outputs=status_output)

                refresh_btn.click(
                    update_state,
                    outputs=[
                        surprise_display,
                        stored_display,
                        compression_display,
                        live_plot,
                    ],
                )

            # Tab 2: Memory Explorer
            with gr.Tab("Memory Explorer"):
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Ask about what happened...",
                        placeholder="What were the most notable events?",
                    )
                    query_btn = gr.Button("Query Memory", variant="primary")

                query_output = gr.Textbox(label="Answer", lines=5)

                # Show events on load
                initial_events = dashboard.get_recent_events()
                initial_event_data = [
                    [
                        e.get("frame_index", 0),
                        f"{e.get('surprise_score', 0):.3f}",
                        e.get("description", "N/A")[:50],
                    ]
                    for e in initial_events
                ]

                recent_events = gr.Dataframe(
                    value=initial_event_data,
                    headers=["Frame", "Surprise", "Description"],
                    label="Recent Events (auto-loaded)",
                    max_height=300,
                )

                def handle_query(question):
                    answer = dashboard.query_memory(question)
                    events = dashboard.get_recent_events()
                    event_data = [
                        [
                            e.get("frame_index", 0),
                            f"{e.get('surprise_score', 0):.3f}",
                            e.get("description", "N/A")[:50],
                        ]
                        for e in events
                    ]
                    return answer, event_data

                query_btn.click(
                    handle_query,
                    inputs=query_input,
                    outputs=[query_output, recent_events],
                )

            # Tab 2b: Image Search
            with gr.Tab("Image Search"):
                gr.Markdown("## Upload and Search Images")
                gr.Markdown(
                    "Upload images for semantic search using natural language queries"
                )

                with gr.Row():
                    with gr.Column():
                        image_upload = gr.File(
                            label="Upload Image",
                            file_count="single",
                            file_types=["image"],
                        )
                        image_caption = gr.Textbox(
                            label="Image Caption (optional)",
                            placeholder="Auto-generated if empty",
                        )
                        upload_btn = gr.Button("Add to Index", variant="primary")
                        upload_status = gr.Textbox(label="Upload Status")

                    with gr.Column():
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g., 'a dog playing in snow'",
                        )
                        top_k = gr.Slider(1, 20, value=5, label="Top K Results")
                        search_btn = gr.Button("Search", variant="primary")

                search_results = gr.Dataframe(
                    headers=["Score", "Caption", "Type", "Path/Frame"],
                    label="Search Results",
                    max_height=300,
                )

                def handle_image_upload(file_obj, caption):
                    if file_obj is None:
                        return "No file selected"

                    try:
                        # Save uploaded file
                        file_path = file_obj.name
                        filename = os.path.basename(file_path)
                        dest_path = os.path.join(dashboard.upload_dir, filename)

                        # Copy file
                        import shutil

                        shutil.copy(file_path, dest_path)

                        # Add to search index
                        result = dashboard.image_search_api.upload_image(
                            dest_path, caption
                        )

                        if "error" in result:
                            return f"Error: {result['error']}"

                        return f"Added: {result['caption'][:60]}..."
                    except Exception as e:
                        return f"Error: {str(e)}"

                upload_btn.click(
                    handle_image_upload,
                    inputs=[image_upload, image_caption],
                    outputs=upload_status,
                )

                def handle_search(query, k):
                    if not query:
                        return []

                    results = dashboard.image_search_api.search_images(
                        query, top_k=int(k)
                    )

                    return [
                        [
                            f"{r['score']:.3f}",
                            r["caption"][:50],
                            "image",
                            r.get("path", "")[:30],
                        ]
                        for r in results
                    ]

                search_btn.click(
                    handle_search,
                    inputs=[search_query, top_k],
                    outputs=search_results,
                )

            # Tab 2c: Unified Search (Images + Video Events)
            with gr.Tab("Unified Search"):
                gr.Markdown("## Cross-Modal Semantic Search")
                gr.Markdown(
                    "Search across both uploaded images and video events simultaneously"
                )

                with gr.Row():
                    unified_query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., 'unexpected motion', 'outdoor scene'",
                        lines=2,
                    )
                    unified_top_k = gr.Slider(1, 20, value=10, label="Top K Results")

                unified_search_btn = gr.Button("Search All Sources", variant="primary")

                gr.Markdown("### Results")
                unified_results = gr.Dataframe(
                    headers=["Score", "Type", "Description", "Source"],
                    label="Unified Search Results",
                    max_height=400,
                )

                def handle_unified_search(query, k):
                    if not query:
                        return []

                    # Get video events
                    events = dashboard.get_recent_events()

                    # Get image results
                    image_results = dashboard.image_search_api.search_images(
                        query, top_k=int(k)
                    )

                    # Get video event results
                    event_results = (
                        dashboard.image_search_api.search.search_video_events(
                            query, events, top_k=int(k)
                        )
                    )

                    # Combine and rank
                    combined = []

                    for r in image_results:
                        combined.append(
                            [
                                f"{r['score']:.3f}",
                                "Image",
                                r["caption"][:40],
                                os.path.basename(r.get("path", "unknown")),
                            ]
                        )

                    for r in event_results:
                        combined.append(
                            [
                                f"{r['score']:.3f}",
                                "Video Event",
                                r["description"][:40],
                                f"Frame {r['frame_index']}",
                            ]
                        )

                    # Sort by score
                    combined.sort(key=lambda x: float(x[0]), reverse=True)

                    return combined[: int(k)]

                unified_search_btn.click(
                    handle_unified_search,
                    inputs=[unified_query, unified_top_k],
                    outputs=unified_results,
                )

            # Tab 3: System Stats
            with gr.Tab("System Stats"):
                gr.Markdown("## System Statistics")

                with gr.Row():
                    with gr.Column():
                        mps_label = gr.Label(label="MPS Memory")
                        frames_label = gr.Label(label="Frames Processed")
                        stored_label = gr.Label(label="Events Stored")

                    with gr.Column():
                        discarded_label = gr.Label(label="Events Discarded")
                        compression_label = gr.Label(label="Compression Ratio")
                        duration_label = gr.Label(label="Session Duration")

                refresh_stats_btn = gr.Button("Refresh Stats")

                def update_stats():
                    stats = dashboard.get_system_stats()
                    return (
                        {"MPS Memory": stats["mps_memory"]},
                        {"Frames": stats["frames_processed"]},
                        {"Stored": stats["events_stored"]},
                        {"Discarded": stats["events_discarded"]},
                        {"Ratio": stats["compression_ratio"]},
                        {"Duration": stats["duration"]},
                    )

                refresh_stats_btn.click(
                    update_stats,
                    outputs=[
                        mps_label,
                        frames_label,
                        stored_label,
                        discarded_label,
                        compression_label,
                        duration_label,
                    ],
                )

        # Demo loop for updating live plot
        def periodic_update():
            if dashboard.running:
                state = dashboard.get_state()
                return state["surprise_history"]
            return []

    return demo


def launch_dashboard(use_llm: bool = True, port: int = 7860):
    """
    Launch the Gradio dashboard.

    Args:
        use_llm: Whether to use LLM features
        port: Port to listen on
    """
    logger.info(f"Launching dashboard on port {port}...")

    demo = create_dashboard(use_llm=use_llm)
    demo.launch(server_port=port, server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    launch_dashboard()
