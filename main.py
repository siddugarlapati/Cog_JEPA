"""
COG-JEPA Main Entry Point

CLI interface for running the COG-JEPA system in different modes.
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from pipeline.video_pipeline import VideoPipeline
from ui.dashboard import launch_dashboard
from ui.dashboard_pro import launch_dashboard as launch_pro_dashboard

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_test_mode(max_frames: int, use_llm: bool, threshold: float):
    """Run pipeline with synthetic test frames."""
    logger.info("Running in TEST mode...")

    pipeline = VideoPipeline(
        video_source="test", use_llm=use_llm, threshold_override=threshold
    )

    pipeline.run(max_frames=max_frames)

    return pipeline


def run_webcam_mode(max_frames: int, use_llm: bool, threshold: float):
    """Run pipeline with live webcam feed."""
    logger.info("Running in WEBCAM mode...")

    pipeline = VideoPipeline(
        video_source=0, use_llm=use_llm, threshold_override=threshold
    )

    pipeline.run(max_frames=max_frames)


def run_file_mode(video_path: str, max_frames: int, use_llm: bool, threshold: float):
    """Run pipeline with video file."""
    logger.info(f"Running in FILE mode: {video_path}")

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return

    pipeline = VideoPipeline(
        video_source=video_path, use_llm=use_llm, threshold_override=threshold
    )

    pipeline.run(max_frames=max_frames)


def run_dashboard_mode(use_llm: bool, pro: bool = False):
    """Launch Gradio dashboard."""
    logger.info("Running in DASHBOARD mode...")

    if pro:
        logger.info("Using PRO dashboard with Ollama integration")
        launch_pro_dashboard()
    else:
        launch_dashboard(use_llm=use_llm)


def run_api_mode(port: int, debug: bool):
    """Run Flask API server."""
    logger.info(f"Running in API mode on port {port}...")
    from api.app import run_server

    run_server(port=port, debug=debug)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="COG-JEPA: Cognitive Predictive Memory for Vision Agents"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "webcam", "file", "dashboard", "api"],
        default="test",
        help="Operating mode",
    )

    parser.add_argument("--video", type=str, help="Path to video file (for file mode)")

    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Max frames to process (default: unlimited)",
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM loading (use rule-based describer)",
    )

    parser.add_argument(
        "--threshold", type=float, default=None, help="Override surprise threshold"
    )

    parser.add_argument(
        "--pro", action="store_true", help="Use PRO dashboard with Ollama integration"
    )

    parser.add_argument(
        "--port", type=int, default=5000, help="Port for API mode (default: 5000)"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for API"
    )

    args = parser.parse_args()

    use_llm = not args.no_llm

    logger.info(f"Starting COG-JEPA in {args.mode} mode")
    logger.info(f"LLM enabled: {use_llm}")
    if args.threshold:
        logger.info(f"Threshold override: {args.threshold}")

    try:
        if args.mode == "test":
            run_test_mode(args.frames or 100, use_llm, args.threshold)

        elif args.mode == "webcam":
            run_webcam_mode(args.frames, use_llm, args.threshold)

        elif args.mode == "file":
            if not args.video:
                logger.error("--video required for file mode")
                sys.exit(1)
            run_file_mode(args.video, args.frames, use_llm, args.threshold)

        elif args.mode == "dashboard":
            run_dashboard_mode(use_llm, pro=args.pro)

        elif args.mode == "api":
            run_api_mode(args.port, args.debug)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
