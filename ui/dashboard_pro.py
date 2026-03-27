"""
COG-JEPA - Production-Ready Dashboard (FIXED)
"""

import logging
import os
import threading
import time
from typing import Dict, List, Optional

import gradio as gr
import numpy as np
import pandas as pd

from config import config
from memory.cognee_store import CogneeMemoryStore
from pipeline.video_pipeline import VideoPipeline
from reasoning.ollama_client import get_ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COGJEPDashboard:
    """Production-ready dashboard for COG-JEPA."""

    def __init__(self):
        self.pipeline: Optional[VideoPipeline] = None
        self.pipeline_thread: Optional[threading.Thread] = None
        self.memory_store = CogneeMemoryStore()
        self.ollama = get_ollama()
        self.running = False
        self.start_time: Optional[float] = None
        logger.info("Dashboard initialized")

    # ── Pipeline control ────────────────────────────────────────────────────

    def start_pipeline(self, mode: str, file_obj, max_frames: int, threshold: float) -> str:
        if self.running:
            return "⚠️ Pipeline already running"

        if mode == "test":
            video_source = "test"
        elif mode == "webcam":
            video_source = 0
        elif mode == "file":
            if file_obj is None:
                return "❌ Please upload a video file first"
            video_source = file_obj if isinstance(file_obj, str) else file_obj.name
        else:
            video_source = "test"

        try:
            self.pipeline = VideoPipeline(
                video_source=video_source,
                use_llm=False,
                threshold_override=float(threshold),
            )
        except Exception as e:
            return f"❌ Failed to init pipeline: {e}"

        self.running = True
        self.start_time = time.time()

        def _run():
            try:
                self.pipeline.run(
                    max_frames=int(max_frames) if max_frames > 0 else None,
                    display=False,
                )
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
            finally:
                self.running = False

        self.pipeline_thread = threading.Thread(target=_run, daemon=True)
        self.pipeline_thread.start()

        return f"✅ Started: {mode} | threshold={threshold} | max_frames={max_frames}"

    def stop_pipeline(self) -> str:
        if not self.running or self.pipeline is None:
            return "❌ No pipeline running"
        self.pipeline.running = False
        self.running = False
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=5)
        return "✅ Pipeline stopped"

    # ── State ────────────────────────────────────────────────────────────────

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

    def _make_plot_df(self, history: list) -> pd.DataFrame:
        """Safely create a DataFrame for LinePlot. Never empty — always at least one row."""
        if history:
            recent = history[-50:]
            df = pd.DataFrame({
                "frame": list(range(len(recent))),
                "surprise": [float(s) for s in recent],
            })
        else:
            # Provide a minimal seed row so LinePlot doesn't crash
            df = pd.DataFrame({"frame": [0], "surprise": [0.0]})
        return df

    def get_stats_tuple(self):
        """Return (frames, fps, surprise, stored, compression%, df) for UI."""
        try:
            state = self.get_state()
            history = state.get("surprise_history", [])
            elapsed = time.time() - (self.start_time or time.time())
            frames = int(state.get("frame_count", 0))
            fps = round(frames / max(1.0, elapsed), 2)
            surprise = round(float(state.get("surprise_score", 0.0)), 4)
            stored = int(state.get("stored_events", 0))
            comp = round(float(state.get("compression_ratio", 0.0)) * 100, 1)
            df = self._make_plot_df(history)
            return frames, fps, surprise, stored, comp, df
        except Exception as e:
            logger.error(f"get_stats_tuple error: {e}")
            df = pd.DataFrame({"frame": [0], "surprise": [0.0]})
            return 0, 0.0, 0.0, 0, 0.0, df

    # ── Query / Summary ──────────────────────────────────────────────────────

    def query_video(self, question: str) -> str:
        self.memory_store._load_from_log()
        if not self.memory_store.events:
            return "📭 No video analyzed yet. Process a video first!"
        events = self.memory_store.get_recent_events_sync(n=50)
        if self.ollama.connected:
            return self.ollama.answer_question(question, events)
        # Fallback
        from reasoning.llm_reasoner import LLMReasoner
        reasoner = LLMReasoner(use_fallback=True)
        context = "\n".join(
            f"Frame {e.get('frame_index','?')}: surprise={e.get('surprise_score',0):.3f}, "
            f"description={e.get('description','N/A')}"
            for e in events
        )
        return reasoner.query_memory(question, context)

    def get_video_summary(self) -> str:
        self.memory_store._load_from_log()
        if not self.memory_store.events:
            return "📭 No video analyzed yet."
        events = self.memory_store.get_recent_events_sync(n=100)
        return self.ollama.generate_video_summary(events)

    # ── Events ───────────────────────────────────────────────────────────────

    def get_events_table(self):
        self.memory_store._load_from_log()
        events = self.memory_store.get_recent_events_sync(n=20)
        return [
            [
                e.get("frame_index", 0),
                f"{e.get('surprise_score', 0):.4f}",
                e.get("description", "N/A")[:80],
            ]
            for e in events
        ]

    # ── Model ────────────────────────────────────────────────────────────────

    def set_model(self, model_name: str) -> str:
        if self.ollama and self.ollama.connected:
            self.ollama.model = model_name
            return f"✅ Switched to {model_name}"
        return "❌ Ollama not connected"


# ── Image generation helpers ─────────────────────────────────────────────────

def _generate_placeholder_image(prompt: str):
    """Generate a simple colored placeholder image with text when API unavailable."""
    from PIL import Image as PILImage, ImageDraw
    import textwrap

    img = PILImage.new("RGB", (512, 512), color=(30, 30, 50))
    draw = ImageDraw.Draw(img)

    # Draw gradient-like background
    for y in range(512):
        r = int(30 + (y / 512) * 60)
        g = int(30 + (y / 512) * 20)
        b = int(50 + (y / 512) * 80)
        draw.line([(0, y), (512, y)], fill=(r, g, b))

    # Draw text
    draw.rectangle([20, 20, 492, 492], outline=(100, 100, 200), width=2)
    draw.text((256, 80), "🎨 Image Preview", fill=(200, 200, 255), anchor="mm")
    draw.text((256, 120), "(API temporarily unavailable)", fill=(150, 150, 200), anchor="mm")

    # Wrap prompt text
    lines = textwrap.wrap(prompt[:200], width=40)
    y_pos = 200
    for line in lines[:6]:
        draw.text((256, y_pos), line, fill=(220, 220, 220), anchor="mm")
        y_pos += 30

    return img


# ── Dashboard builder ─────────────────────────────────────────────────────────

def create_dashboard() -> gr.Blocks:
    dashboard = COGJEPDashboard()

    # Lazy-load image generator — avoids slow diffusers import at startup
    # This runs inside create_dashboard, not at module import time
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from reasoning.local_image_gen import get_image_generator
        img_gen = get_image_generator()

    # Fetch available Ollama models
    try:
        import requests as _req
        resp = _req.get("http://localhost:11434/api/tags", timeout=2)
        model_choices = [m["name"] for m in resp.json().get("models", [])] if resp.ok else []
    except Exception:
        model_choices = []
    if not model_choices:
        model_choices = ["llama3.2:latest"]

    with gr.Blocks(
        title="COG-JEPA - Cognitive Memory",
        theme=gr.themes.Soft(),
        css="""
        .error-box { color: red; }
        .stats-row { gap: 10px; }
        .plot-container { min-height: 260px; }
        """
    ) as demo:
        gr.Markdown("# 🧠 COG-JEPA  \n*JEPA + Cognee + Ollama-powered video understanding*")

        # ── Tab 1: Video Analysis ────────────────────────────────────────────
        with gr.Tab("🎬 Video Analysis"):
            with gr.Row():
                # Left column: controls
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Configuration")
                    model_select = gr.Dropdown(
                        label="🧠 LLM Model",
                        choices=model_choices,
                        value=model_choices[0],
                    )
                    mode_select = gr.Radio(
                        ["test", "file", "webcam"],
                        label="Input Source",
                        value="test",
                    )
                    file_input = gr.File(
                        label="📁 Upload Video",
                        file_count="single",
                        file_types=["video"],
                    )
                    with gr.Row():
                        max_frames = gr.Number(label="Max Frames (0=all)", value=50, precision=0)
                        threshold = gr.Slider(
                            label="Surprise Threshold",
                            minimum=0.1, maximum=2.0, value=0.3, step=0.05,
                        )
                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Analysis", variant="primary")
                        stop_btn = gr.Button("⏹️ Stop", variant="stop")
                    status = gr.Textbox(label="Status", lines=2, interactive=False)
                    model_change_btn = gr.Button("🔄 Apply Model")

                # Right column: live feed + stats
                with gr.Column(scale=2):
                    gr.Markdown("### 📹 Live Camera Feed")
                    live_feed = gr.Image(
                        label="Live Feed",
                        height=300,
                        sources=["webcam"],
                        streaming=True,
                    )
                    gr.Markdown("*Camera feed will appear when processing webcam mode*")

                    gr.Markdown("### 📊 Live Statistics")
                    with gr.Row(elem_classes=["stats-row"]):
                        frames_stat = gr.Number(label="Frames Processed", value=0, interactive=False)
                        fps_stat = gr.Number(label="Processing Speed (FPS)", value=0.0, interactive=False)
                        surprise_stat = gr.Number(label="Current Surprise", value=0.0, interactive=False)
                    with gr.Row(elem_classes=["stats-row"]):
                        stored_stat = gr.Number(label="Events Stored", value=0, interactive=False)
                        compression_stat = gr.Number(label="Compression %", value=0.0, interactive=False)

                    refresh_btn = gr.Button("🔄 Refresh Stats")

                    gr.Markdown("### 📈 Surprise History")
                    # Initialize with valid seed data
                    _init_df = pd.DataFrame({"frame": [0], "surprise": [0.0]})
                    plot = gr.LinePlot(
                        value=_init_df,
                        x="frame",
                        y="surprise",
                        title="Surprise Score Over Time",
                        height=250,
                        x_lim=[0, None],
                        y_lim=[0.0, 1.0],
                    )

        # ── Tab 2: Query Video ───────────────────────────────────────────────
        with gr.Tab("💬 Query Video"):
            gr.Markdown("### 🔍 Ask anything about your video")
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="What happened in this video? Tell me the story...",
                lines=2,
            )
            query_btn = gr.Button("🔎 Ask", variant="primary")
            query_output = gr.Markdown("*Answer will appear here...*")

            gr.Markdown("### ⚡ Quick Questions")
            with gr.Row():
                gr.Button("📖 What happened?").click(
                    fn=lambda: dashboard.query_video("What happened in this video? Tell me the complete story."),
                    outputs=query_output,
                )
                gr.Button("🎬 Key moments?").click(
                    fn=lambda: dashboard.query_video("What are the key moments in this video?"),
                    outputs=query_output,
                )
                gr.Button("📊 Summary").click(
                    fn=dashboard.get_video_summary,
                    outputs=query_output,
                )

        # ── Tab 3: Memory Explorer ───────────────────────────────────────────
        with gr.Tab("🧠 Memory Explorer"):
            gr.Markdown("### 📦 Stored Events")
            events_data = gr.Dataframe(
                headers=["Frame", "Surprise Score", "Description"],
                label="Recent Events (Precise Surprise Values)",
                max_height=400,
                interactive=False,
            )
            refresh_events_btn = gr.Button("🔄 Load Events")

        # ── Tab 4: Image Gen (Pollinations.ai) ────────────────────────────────
        with gr.Tab("🎨 Image Gen"):
            gr.Markdown("### 🎨 AI Image Generation")
            gr.Markdown(
                "**FREE real image generation** using **Pollinations.ai** (Flux model — no auth required). "
                "Images are real AI-generated art, not placeholders!"
            )

            with gr.Row():
                with gr.Column():
                    ig_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A dramatic cinematic scene with golden hour lighting...",
                        lines=3,
                    )
                    ig_size = gr.Radio(["512x512", "768x512", "1024x1024"], label="Size", value="512x512")
                    ig_model = gr.Dropdown(
                        label="AI Model",
                        choices=["flux", "flux-realism", "flux-anime", "flux-3d", "turbo"],
                        value="flux",
                    )
                    ig_enhance_btn = gr.Button("✨ Enhance Prompt with AI")
                    ig_gen_btn = gr.Button("🎨 Generate Image", variant="primary")
                with gr.Column():
                    ig_output = gr.Image(label="Generated Image", height=450)
                    ig_status = gr.Textbox(label="Status", interactive=False)

            def enhance_prompt(prompt):
                if not prompt:
                    return prompt, "Enter a prompt first"
                if dashboard.ollama.connected:
                    enhanced = dashboard.ollama.generate_text(
                        f"Enhance this image generation prompt to be more detailed and cinematic. "
                        f"Add lighting, mood, style details. Keep it under 200 chars: {prompt}",
                        temperature=0.7,
                    )
                    return enhanced, "✅ Prompt enhanced with Ollama"
                return prompt, "ℹ️ Ollama not connected - prompt unchanged"

            def do_generate(prompt, size, model):
                """Always use Pollinations.ai API directly for reliable real image generation."""
                if not prompt.strip():
                    return None, "❌ Please enter a prompt"
                try:
                    import requests as _req
                    import urllib.parse
                    import time as _time
                    from io import BytesIO
                    from PIL import Image as PILImage

                    # Parse size
                    parts = size.lower().replace(" ", "").split("x")
                    w, h = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (512, 512)

                    encoded = urllib.parse.quote(prompt.strip()[:500], safe="")
                    seed = abs(hash(prompt + str(_time.time()))) % 1000000
                    url = (
                        f"https://image.pollinations.ai/prompt/{encoded}"
                        f"?width={w}&height={h}&model={model}&seed={seed}&nologo=true&enhance=false"
                    )

                    logger.info(f"🎨 Calling Pollinations.ai ({model}, {w}x{h})...")

                    # Try up to 3 times
                    for attempt in range(1, 4):
                        try:
                            resp = _req.get(url, timeout=120, stream=True,
                                           headers={"User-Agent": "COG-JEPA/1.0", "Accept": "image/*"})

                            if resp.status_code == 200:
                                buf = BytesIO()
                                for chunk in resp.iter_content(8192):
                                    if chunk:
                                        buf.write(chunk)
                                buf_size = buf.tell()
                                buf.seek(0)

                                if buf_size < 1000:
                                    logger.warning(f"Response too small ({buf_size}B), retry {attempt}")
                                    _time.sleep(2)
                                    continue

                                pil_img = PILImage.open(buf).convert("RGB")
                                iw, ih = pil_img.size

                                # Reject tiny images (error responses)
                                if iw < 64 or ih < 64:
                                    logger.warning(f"Image too small ({iw}x{ih}), retry {attempt}")
                                    _time.sleep(2)
                                    continue

                                # Check not all-black (pixel sum > 0)
                                import numpy as _np
                                arr = _np.array(pil_img)
                                if arr.mean() < 1.0:
                                    logger.warning(f"Image appears black (mean={arr.mean():.2f}), retry {attempt}")
                                    seed = (seed + 1) % 1000000
                                    url = (
                                        f"https://image.pollinations.ai/prompt/{encoded}"
                                        f"?width={w}&height={h}&model={model}&seed={seed}&nologo=true&enhance=false"
                                    )
                                    _time.sleep(2)
                                    continue

                                os.makedirs("data", exist_ok=True)
                                save_path = f"data/generated_{abs(hash(prompt)) % 100000}.png"
                                pil_img.save(save_path)
                                logger.info(f"✅ Image generated {iw}x{ih} ({buf_size} bytes)")
                                return pil_img, f"✅ Generated via Pollinations.ai | {iw}x{ih}px | model={model}"

                            elif resp.status_code == 429:
                                logger.warning("Rate limited, waiting 5s...")
                                _time.sleep(5)
                            else:
                                logger.error(f"API error {resp.status_code}, retry {attempt}")
                                _time.sleep(2)

                        except Exception as req_err:
                            logger.error(f"Request attempt {attempt} failed: {req_err}")
                            _time.sleep(2)

                    return _generate_placeholder_image(prompt), "❌ Pollinations.ai unavailable after 3 attempts"

                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    return _generate_placeholder_image(prompt), f"❌ Error: {e}"

            ig_enhance_btn.click(enhance_prompt, inputs=[ig_prompt], outputs=[ig_prompt, ig_status])
            ig_gen_btn.click(do_generate, inputs=[ig_prompt, ig_size, ig_model], outputs=[ig_output, ig_status])

        # ── Tab 5: Generate Image from Video ────────────────────────────────
        with gr.Tab("🎬 Generate Image"):
            gr.Markdown("### 🎬 Generate Image from Video Moments")
            gr.Markdown("Automatically creates an image prompt from the most dramatic video moment.")

            with gr.Row():
                with gr.Column():
                    gv_prompt_display = gr.Textbox(
                        label="Auto-generated Prompt (editable)",
                        lines=3,
                        placeholder="Process a video first, then click 'Extract from Video'",
                    )
                    gv_size = gr.Radio(["512x512", "768x512", "1024x1024"], label="Size", value="512x512")
                    gv_model = gr.Dropdown(
                        label="AI Model",
                        choices=["flux", "flux-realism", "flux-anime", "turbo"],
                        value="flux-realism",
                    )
                    with gr.Row():
                        gv_extract_btn = gr.Button("🎯 Extract from Video")
                        gv_gen_btn = gr.Button("🎨 Generate", variant="primary")
                with gr.Column():
                    gv_output = gr.Image(label="Generated Image", height=450)
                    gv_status = gr.Textbox(label="Status", interactive=False)

            def extract_video_prompt():
                dashboard.memory_store._load_from_log()
                events = dashboard.memory_store.get_recent_events_sync(n=20)
                if not events:
                    return "", "❌ No video analyzed yet. Process a video first!"
                # Pick most dramatic event
                top_event = max(events, key=lambda e: e.get("surprise_score", 0))
                desc = top_event.get("description", "dramatic scene")
                score = top_event.get("surprise_score", 0)
                frame = top_event.get("frame_index", 0)
                prompt = (
                    f"Cinematic movie still: {desc}, "
                    "dramatic lighting, professional cinematography, 4k, photorealistic, film grain"
                )
                return prompt, f"✅ Extracted from frame {frame} (surprise={score:.4f})"

            def generate_from_video(prompt, size, model):
                if not prompt.strip():
                    return None, "❌ Extract a prompt from video first"
                return do_generate(prompt, size, model)

            gv_extract_btn.click(extract_video_prompt, outputs=[gv_prompt_display, gv_status])
            gv_gen_btn.click(
                generate_from_video,
                inputs=[gv_prompt_display, gv_size, gv_model],
                outputs=[gv_output, gv_status],
            )

        # ── Event wiring ─────────────────────────────────────────────────────

        def run_analysis(mode, file_obj, n_frames, thresh):
            try:
                msg = dashboard.start_pipeline(mode, file_obj, int(n_frames), float(thresh))
                time.sleep(0.5)  # Brief wait for pipeline to init
                frames, fps, surprise, stored, comp, df = dashboard.get_stats_tuple()
                return msg, frames, fps, surprise, stored, comp, df
            except Exception as e:
                logger.error(f"run_analysis error: {e}")
                df = pd.DataFrame({"frame": [0], "surprise": [0.0]})
                return f"❌ Error: {e}", 0, 0.0, 0.0, 0, 0.0, df

        start_btn.click(
            run_analysis,
            inputs=[mode_select, file_input, max_frames, threshold],
            outputs=[status, frames_stat, fps_stat, surprise_stat, stored_stat, compression_stat, plot],
        )
        stop_btn.click(dashboard.stop_pipeline, outputs=status)

        def refresh_stats():
            try:
                frames, fps, surprise, stored, comp, df = dashboard.get_stats_tuple()
                return frames, fps, surprise, stored, comp, df
            except Exception as e:
                logger.error(f"refresh_stats error: {e}")
                df = pd.DataFrame({"frame": [0], "surprise": [0.0]})
                return 0, 0.0, 0.0, 0, 0.0, df

        refresh_btn.click(
            refresh_stats,
            outputs=[frames_stat, fps_stat, surprise_stat, stored_stat, compression_stat, plot],
        )

        model_change_btn.click(dashboard.set_model, inputs=[model_select], outputs=status)
        query_btn.click(dashboard.query_video, inputs=[query_input], outputs=query_output)
        refresh_events_btn.click(dashboard.get_events_table, outputs=events_data)

        # Load initial state on page open — safe version
        demo.load(
            refresh_stats,
            outputs=[frames_stat, fps_stat, surprise_stat, stored_stat, compression_stat, plot],
        )

        # Auto-refresh every 3 seconds when running
        try:
            timer = gr.Timer(value=3, active=False)
            timer.tick(
                fn=refresh_stats,
                outputs=[frames_stat, fps_stat, surprise_stat, stored_stat, compression_stat, plot],
            )
            # Activate timer when pipeline starts
            start_btn.click(lambda: gr.Timer(active=True), outputs=[timer])
            stop_btn.click(lambda: gr.Timer(active=False), outputs=[timer])
        except Exception:
            pass  # Older Gradio without Timer support

    return demo


def launch_dashboard(port: int = 7860):
    logger.info(f"Launching COG-JEPA Dashboard on port {port}...")
    demo = create_dashboard()
    demo.launch(server_port=port, server_name="0.0.0.0", share=False, show_error=True)


if __name__ == "__main__":
    launch_dashboard()
