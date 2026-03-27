"""
COG-JEPA - Production-Ready Dashboard

A polished, startup-level interface for the cognitive predictive memory system.
"""

import logging
import threading
import time
from typing import Dict, List

import gradio as gr
import numpy as np

from config import config
from pipeline.video_pipeline import VideoPipeline
from memory.cognee_store import CogneeMemoryStore
from reasoning.ollama_client import get_ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COGJEPDashboard:
    """Production-ready dashboard for COG-JEPA."""

    def __init__(self, use_ollama: bool = True):
        self.pipeline = None
        self.pipeline_thread = None
        self.use_ollama = use_ollama

        self.memory_store = CogneeMemoryStore()
        self.ollama = get_ollama()

        self.running = False
        self.start_time = None

        logger.info("Dashboard initialized")

    def set_model(self, model_name: str):
        """Change the Ollama model."""
        if self.ollama and self.ollama.connected:
            self.ollama.model = model_name
            return f"✅ Switched to {model_name}"
        return "❌ Ollama not connected"

    def start_pipeline(
        self, mode: str, file_path: str, max_frames: int, threshold: float
    ):
        if self.running:
            return "⚠️ Pipeline already running"

        if mode == "test":
            video_source = "test"
        elif mode == "webcam":
            video_source = 0
        elif mode == "file" and file_path:
            video_source = file_path
        else:
            video_source = "test"

        self.pipeline = VideoPipeline(
            video_source=video_source, use_llm=False, threshold_override=threshold
        )

        def run_pipeline():
            self.pipeline.run(
                max_frames=max_frames if max_frames > 0 else None, display=False
            )

        self.pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
        self.pipeline_thread.start()

        self.running = True
        self.start_time = time.time()

        return f"✅ Started: {mode} mode | Threshold: {threshold} | Max frames: {max_frames}"

    def stop_pipeline(self):
        if not self.running or self.pipeline is None:
            return "❌ No pipeline running"

        self.pipeline.running = False
        self.running = False

        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=5)

        return "✅ Pipeline stopped"

    def get_state(self) -> Dict:
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

    def query_video(self, question: str) -> str:
        """Query video using Ollama for intelligent responses."""
        # Reload events
        self.memory_store._load_from_log()

        if not self.memory_store.events:
            return "📭 No video analyzed yet. Upload and process a video first!"

        events = self.memory_store.get_recent_events_sync(n=50)

        # Use Ollama for intelligent response
        answer = self.ollama.answer_question(question, events)

        return answer

    def get_video_summary(self) -> str:
        """Get comprehensive video summary."""
        self.memory_store._load_from_log()

        if not self.memory_store.events:
            return "📭 No video analyzed yet."

        events = self.memory_store.get_recent_events_sync(n=100)
        return self.ollama.generate_video_summary(events)

    def get_recent_events(self) -> List:
        self.memory_store._load_from_log()
        return self.memory_store.get_recent_events_sync(n=10)

    def generate_image(self, prompt: str) -> str:
        """Generate image based on prompt using Ollama (when available)."""
        if not self.ollama.connected:
            return "🔴 Ollama not connected. Start Ollama to enable image generation."

        # Use Ollama to generate an image description
        response = self.ollama.generate_text(
            f"Create a detailed prompt for an AI image generator based on this: {prompt}",
            temperature=0.8,
        )

        return f"🎨 Generated prompt:\n\n{response}\n\n(Note: Image generation requires additional setup)"


def create_dashboard() -> gr.Blocks:
    """Create the production dashboard."""
    dashboard = COGJEPDashboard()

    # Custom CSS for startup look
    custom_css = """
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    .subtitle {
        font-size: 1.1rem !important;
        color: #666 !important;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        color: white !important;
    }
    .event-card {
        border-left: 4px solid #667eea !important;
        background: #f8f9fa !important;
        padding: 12px !important;
        margin: 8px 0 !important;
        border-radius: 8px !important;
    }
    """

    with gr.Blocks(title="COG-JEPA - Cognitive Memory") as demo:
        gr.Markdown("""
        # 🧠 COG-JEPA
        ## Cognitive Predictive Memory for Vision AI
        ---
        *JEPA + Cognee + Ollama-powered video understanding*
        """)

        with gr.Tab("📷 Webcam Preview"):
            gr.Markdown("### 🔴 Live Camera Preview")
            gr.Markdown("Check your camera before starting analysis")

            # Use Video with webcam source for Gradio 6.0
            with gr.Row():
                webcam_cam = gr.Video(
                    label="Camera",
                    sources=["webcam"],
                    height=400,
                )

            gr.Markdown("""
            **To start analysis:**
            1. Make sure camera permission is granted
            2. Click the camera icon in the video above to start
            3. Go to **Video Analysis** tab and select webcam mode
            """)

        with gr.Tab("🎬 Video Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Configuration")

                    # Model selector (displays available models)
                    import requests

                    try:
                        resp = requests.get(
                            "http://localhost:11434/api/tags", timeout=2
                        )
                        if resp.status_code == 200:
                            models = resp.json().get("models", [])
                            model_choices = [m["name"] for m in models]
                        else:
                            model_choices = ["llama3.2:latest"]
                    except:
                        model_choices = ["llama3.2:latest"]

                    model_select = gr.Dropdown(
                        label="🧠 LLM Model",
                        choices=model_choices,
                        value=model_choices[0] if model_choices else "llama3.2:latest",
                    )

                    mode_select = gr.Radio(
                        ["test", "file", "webcam"], label="Input Source", value="file"
                    )

                    file_input = gr.File(
                        label="📁 Upload Video",
                        file_count="single",
                        file_types=["video"],
                    )

                    with gr.Row():
                        max_frames = gr.Number(
                            label="Max Frames", value=50, precision=0
                        )
                        threshold = gr.Slider(
                            label="Threshold",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                        )

                    with gr.Row():
                        start_btn = gr.Button(
                            "▶️ Start Analysis", variant="primary", size="lg"
                        )
                        stop_btn = gr.Button("⏹️ Stop", variant="stop")

                    status = gr.Textbox(label="Status", lines=3)

                    # Model change button
                    model_change_btn = gr.Button("🔄 Change Model")

                with gr.Column(scale=2):
                    gr.Markdown("### 📹 Live Camera Feed")

                    # Live webcam display
                    live_feed = gr.Image(
                        label="Live Feed",
                        height=300,
                        sources=["webcam"]
                        if False
                        else None,  # Will be updated via code
                    )

                    # Placeholder for live updates
                    gr.Markdown("*Camera feed will appear when processing webcam mode*")

                    gr.Markdown("### 📊 Live Statistics")

                    with gr.Row():
                        frames_stat = gr.Number(label="Frames Processed", value=0)
                        fps_stat = gr.Number(label="Processing Speed (FPS)", value=0)
                        surprise_stat = gr.Number(label="Current Surprise", value=0)

                    with gr.Row():
                        stored_stat = gr.Number(label="Events Stored", value=0)
                        compression_stat = gr.Number(label="Compression %", value=0)

                    refresh_btn = gr.Button("🔄 Refresh Stats")

                    gr.Markdown("### 📈 Surprise History")
                    plot = gr.LinePlot(
                        title="Surprise Score Over Time",
                        height=250,
                    )

        with gr.Tab("💬 Query Video"):
            gr.Markdown("### 🔍 Ask anything about your video")

            with gr.Row():
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What happened in this video? Tell me the story...",
                    lines=2,
                )

            with gr.Row():
                query_btn = gr.Button("🔎 Ask", variant="primary", size="lg")

            gr.Markdown("### 📝 Answer")
            query_output = gr.Markdown("")

            gr.Markdown("---")

            # Quick actions
            gr.Markdown("### ⚡ Quick Questions")
            with gr.Row():
                gr.Button("📖 What happened?", variant="secondary").click(
                    lambda: dashboard.query_video(
                        "What happened in this video? Tell me the complete story."
                    ),
                    outputs=query_output,
                )
                gr.Button("🎬 Key moments?", variant="secondary").click(
                    lambda: dashboard.query_video(
                        "What are the key moments in this video?"
                    ),
                    outputs=query_output,
                )
                gr.Button("📊 Summary", variant="secondary").click(
                    lambda: dashboard.get_video_summary(), outputs=query_output
                )

        with gr.Tab("🧠 Memory Explorer"):
            gr.Markdown("### 📦 Stored Events")

            events_data = gr.Dataframe(
                headers=["Frame", "Surprise", "Description"],
                label="Recent Events",
                max_height=400,
            )

            refresh_events = gr.Button("🔄 Load Events")

        with gr.Tab("🎨 Image Gen"):
            gr.Markdown("### 🎨 Image Generation")
            gr.Markdown("*Coming soon - Generate images from video moments*")

            gr.Markdown("""
            **Image generation requires:**
            1. Stable Diffusion model (local)
            2. Or an OpenAI API key
            
            For now, use the Query tab to ask about your video!
            """)

        with gr.Tab("🎨 Generate Image"):
            gr.Markdown("### 🎨 Generate Image from Video")
            gr.Markdown(
                "*Generate AI images based on video moments using LOCAL Stable Diffusion*"
            )

            # Import local image generator
            from reasoning.local_image_gen import get_image_generator

            img_gen = get_image_generator()

            gr.Markdown("✅ **Local Image Generator Ready** - Using Stable Diffusion!")

            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.Textbox(
                        label="Enter prompt",
                        placeholder="A dramatic sniper action scene...",
                        lines=2,
                    )
                with gr.Column():
                    size_select = gr.Radio(
                        ["512x512", "1024x1024"],
                        label="Image Size",
                        value="512x512",
                    )

            gen_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")

            with gr.Row():
                gen_output = gr.Image(label="Generated Image", height=400)

            # Quick generation from video
            gr.Markdown("### 🚀 Quick Generate from Video")
            quick_btn = gr.Button("🎬 Generate from Most Dramatic Moment")

            def generate_from_video():
                dashboard.memory_store._load_from_log()
                events = dashboard.memory_store.get_recent_events_sync(n=20)
                if not events:
                    return None, "No video analyzed yet!"

                result = img_gen.generate_from_video_context(events)
                if result.startswith("http"):
                    return result, "✅ Image generated successfully!"
                else:
                    return None, result

            quick_btn.click(generate_from_video, outputs=[gen_output, gen_prompt])

            def handle_generate(prompt, size):
                if not prompt:
                    return None, "Please enter a prompt"
                result = img_gen.generate_image(prompt, size)
                return result, result

            gen_btn.click(
                handle_generate,
                inputs=[gen_prompt, size_select],
                outputs=[gen_output, gen_prompt],
            )

        # Event handlers
        def start_pipeline(mode, file_obj, frames, thresh):
            file_path = str(file_obj.name) if file_obj else None
            return dashboard.start_pipeline(mode, file_path, int(frames), float(thresh))

        start_btn.click(
            start_pipeline,
            inputs=[mode_select, file_input, max_frames, threshold],
            outputs=status,
        )

        stop_btn.click(dashboard.stop_pipeline, outputs=status)

        # Model change handler
        def change_model(model_name):
            return dashboard.set_model(model_name)

        model_change_btn.click(change_model, inputs=[model_select], outputs=status)

        def update_stats():
            state = dashboard.get_state()
            history = state.get("surprise_history", [])
            return (
                state["frame_count"],
                state["frame_count"]
                / max(1, time.time() - (dashboard.start_time or time.time())),
                state["surprise_score"],
                state["stored_events"],
                state["compression_ratio"] * 100,
                [[i, s] for i, s in enumerate(history[-50:])] if history else [],
            )

        refresh_btn.click(
            update_stats,
            outputs=[
                frames_stat,
                fps_stat,
                surprise_stat,
                stored_stat,
                compression_stat,
                plot,
            ],
        )

        query_btn.click(dashboard.query_video, inputs=query_input, outputs=query_output)

        def load_events():
            events = dashboard.get_recent_events()
            data = [
                [
                    e.get("frame_index", 0),
                    f"{e.get('surprise_score', 0):.3f}",
                    e.get("description", "N/A")[:80],
                ]
                for e in events
            ]
            return data

        refresh_events.click(load_events, outputs=events_data)

    return demo


def launch_dashboard(port: int = 7860):
    """Launch the production dashboard."""
    logger.info(f"Launching COG-JEPA Dashboard on port {port}...")

    demo = create_dashboard()
    demo.launch(server_port=port, server_name="0.0.0.0", share=False, show_error=True)


if __name__ == "__main__":
    launch_dashboard()
