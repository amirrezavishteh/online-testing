#!/usr/bin/env python3
"""
BAIT Online Detection UI — Gradio interface for /probe and /scan endpoints.

Three-column layout: inputs, live progress, results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Generator, Tuple, List
from datetime import datetime

import pandas as pd
import requests
import gradio as gr
from loguru import logger


class DetectionClient:
    """HTTP client for BAIT detection server."""

    def __init__(self, server_url: str):
        """Initialize with server URL."""
        self.server_url = server_url.rstrip("/")
        logger.info(f"Detection server URL: {self.server_url}")

    def check_server(self) -> bool:
        """Check if server is running."""
        try:
            resp = requests.get(f"{self.server_url}/status", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Server not responding: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        try:
            resp = requests.get(f"{self.server_url}/status", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to get server status: {e}")

    def probe(self, model_path: str, model_zoo_id: str = "", dataset: str = "alpaca") -> Dict[str, Any]:
        """Run fast probe."""
        payload = {
            "model_path": model_path,
            "model_zoo_id": model_zoo_id,
            "dataset": dataset,
            "attack": "cba"
        }
        try:
            resp = requests.post(f"{self.server_url}/probe", json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise RuntimeError(f"Probe failed: {e}")

    def scan(
        self,
        model_path: str,
        model_zoo_id: str = "",
        dataset: str = "alpaca",
        use_slm: bool = True,
        tau: float = 0.1,
        beta: float = 0.0
    ) -> Dict[str, Any]:
        """Run full scan."""
        payload = {
            "model_path": model_path,
            "model_zoo_id": model_zoo_id,
            "dataset": dataset,
            "attack": "cba",
            "use_slm": use_slm,
            "tau": tau,
            "beta": beta
        }
        try:
            resp = requests.post(f"{self.server_url}/scan", json=payload, timeout=600)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise RuntimeError(f"Scan failed: {e}")


def _verdict_html(verdict: str, score: float) -> str:
    """
    Generate HTML for the verdict badge.

    Args:
        verdict: "BACKDOORED" or "CLEAN"
        score: Confidence score

    Returns:
        HTML string for display
    """
    if verdict == "BACKDOORED":
        color, icon = "#C0392B", "✗"
    elif verdict == "CLEAN":
        color, icon = "#1E8449", "✓"
    else:
        color, icon = "#888", "?"

    return f"""
    <div style='background:{color};color:white;padding:16px;border-radius:8px;
                font-size:1.4em;font-weight:bold;text-align:center;margin:10px 0'>
        {icon} {verdict} &nbsp; <span style='font-size:0.8em'>score={score:.3f}</span>
    </div>
    """


def run_detection_stream(
    model_path: str,
    zoo_id: str,
    mode: str,
    use_slm: bool,
    tau: float,
    client: DetectionClient
) -> Generator[Tuple[str, str, float, Dict, Dict], None, None]:
    """
    Run detection and stream progress.

    Yields tuples of:
      (log_text, verdict_html, cbss_score, top_candidates_df, component_scores_df)
    """
    if not model_path:
        yield ("Error: Model path is required", "", 0.0, {}, {})
        return

    # Log start
    log_text = f"[{datetime.now().strftime('%H:%M:%S')}] Starting {mode}...\n"
    yield (log_text, "", 0.0, {}, {})

    try:
        # Determine which endpoint to call
        is_fast = "Fast" in mode

        log_text += f"[{datetime.now().strftime('%H:%M:%S')}] Loading model: {model_path}\n"
        yield (log_text, "", 0.0, {}, {})

        if is_fast:
            result = client.probe(model_path, zoo_id)
        else:
            result = client.scan(
                model_path,
                model_zoo_id=zoo_id,
                use_slm=use_slm,
                tau=tau,
                beta=0.0
            )

        # Update log
        log_text += f"[{datetime.now().strftime('%H:%M:%S')}] Scan complete in {result['time_taken_seconds']:.1f}s\n"
        log_text += f"Verdict: {result['verdict']}\n"
        log_text += f"Q-Score: {result['q_score']:.4f}\n"

        if result["cbss_lite_score"] > 0:
            log_text += f"CBSS-Lite Score: {result['cbss_lite_score']:.4f}\n"

        # Format results
        verdict_html = _verdict_html(result["verdict"], result["cbss_lite_score"] or result["q_score"])

        # Top candidates dataframe
        top_candidates = pd.DataFrame([
            {
                "Rank": c["rank"],
                "Inverted Target": c["inverted_target"][:50] + "..." if len(c["inverted_target"]) > 50 else c["inverted_target"],
                "Q-Score": f"{c['q_score']:.4f}",
                "Length": c["token_length"]
            }
            for c in result.get("top_candidates", [])[:5]
        ])

        # Component scores (if available)
        component_scores = pd.DataFrame([
            {"Component": "Q", "Score": result["component_scores"]["Q"]},
            {"Component": "Q_bar", "Score": result["component_scores"]["Q_bar"]},
            {"Component": "E", "Score": result["component_scores"]["E"]},
            {"Component": "U", "Score": result["component_scores"]["U"]},
            {"Component": "B", "Score": result["component_scores"]["B"]},
        ]) if result["component_scores"]["Q"] > 0 else pd.DataFrame()

        yield (
            log_text,
            verdict_html,
            result["cbss_lite_score"] or result["q_score"],
            top_candidates,
            component_scores
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        log_text += f"[ERROR] {str(e)}\n"
        yield (log_text, _verdict_html("ERROR", 0.0), 0.0, {}, {})


def main():
    """Build and launch Gradio UI."""
    parser = argparse.ArgumentParser(description="BAIT Online Detection UI")
    parser.add_argument("--server-url", default="http://localhost:8787", help="Detection server URL")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")

    args = parser.parse_args()

    # Initialize client
    client = DetectionClient(args.server_url)

    # Check server
    if not client.check_server():
        print(f"ERROR: Server not running at {args.server_url}")
        print("Start the server with: python online/detection_server.py --slm-repo-id ... --cache-dir ...")
        sys.exit(1)

    # Import pandas for dataframes
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed. Run: pip install pandas")
        sys.exit(1)

    # Build UI
    with gr.Blocks(
        title="BAIT Backdoor Detection",
        theme=gr.themes.Soft(),
        css="""
        .verdict-container { margin: 20px 0; }
        .stat-row { display: flex; gap: 10px; margin: 5px 0; }
        .stat-label { font-weight: bold; width: 150px; }
        """
    ) as demo:

        gr.Markdown("# BAIT Online Backdoor Detection")
        gr.Markdown("Fast probe (~30s) or full scan (~3-5 min) for LLM backdoors")

        with gr.Row():
            # LEFT COLUMN — Inputs
            with gr.Column(scale=1):
                gr.Markdown("## Input Settings")

                model_path = gr.Textbox(
                    label="Model path or HF ID",
                    placeholder="/media/.../model_zoo/models/id-0055 or meta-llama/Llama-2-7b-hf",
                    info="Local path or HuggingFace model ID"
                )

                zoo_id = gr.Textbox(
                    label="Model zoo ID (optional)",
                    placeholder="id-0055",
                    info="Auto-resolves base model + LoRA adapter"
                )

                mode = gr.Radio(
                    label="Detection mode",
                    choices=["Fast probe (~30s)", "Full scan (3-5 min)"],
                    value="Fast probe (~30s)"
                )

                use_slm = gr.Checkbox(
                    label="Use SLM guidance (NeuroBAIT)",
                    value=True,
                    info="Enable fine-tuned Qwen guidance"
                )

                tau = gr.Slider(
                    label="τ (universality tolerance)",
                    minimum=0.01,
                    maximum=1.0,
                    value=0.1,
                    step=0.01,
                    info="CBSS-Lite universality parameter"
                )

                with gr.Row():
                    run_btn = gr.Button("Run Detection", variant="primary", size="lg")
                    status_btn = gr.Button("Check server status", size="lg")

            # MIDDLE COLUMN — Progress
            with gr.Column(scale=1):
                gr.Markdown("## Progress")

                log_output = gr.Textbox(
                    label="Live log",
                    lines=15,
                    interactive=False,
                    max_lines=20
                )

                progress_bar = gr.Progress(label="Scan progress")

            # RIGHT COLUMN — Results
            with gr.Column(scale=1):
                gr.Markdown("## Results")

                verdict_html = gr.HTML(
                    value="<div style='background:#888;color:white;padding:16px;border-radius:8px;text-align:center'>Awaiting scan...</div>"
                )

                cbss_score = gr.Number(
                    label="CBSS-Lite / Q-Score",
                    precision=4,
                    interactive=False
                )

                component_scores = gr.Dataframe(
                    headers=["Component", "Score"],
                    label="CBSS-Lite component scores",
                    interactive=False
                )

                top_candidates = gr.Dataframe(
                    headers=["Rank", "Inverted Target", "Q-Score", "Length"],
                    label="Top-5 candidates",
                    interactive=False
                )

        # Event handlers
        def on_run_click(m_path, zoo, detection_mode, use_slm_flag, tau_val):
            """Handle Run Detection button."""
            for log, verdict, score, candidates, components in run_detection_stream(
                m_path,
                zoo,
                detection_mode,
                use_slm_flag,
                tau_val,
                client
            ):
                yield log, verdict, score, candidates, components

        run_btn.click(
            on_run_click,
            inputs=[model_path, zoo_id, mode, use_slm, tau],
            outputs=[log_output, verdict_html, cbss_score, top_candidates, component_scores],
            queue=True
        )

        def on_status_click():
            """Handle Check Status button."""
            try:
                status = client.get_status()
                msg = (
                    f"✓ SLM loaded: {status['slm_model_id']}\n"
                    f"✓ GPU memory: {status['gpu_memory_used_gb']:.2f} GB\n"
                    f"✓ Diverse prompts: {status['diverse_prompts_count']}\n"
                )
                if status["cached_model"]:
                    msg += f"✓ Cached model: {status['cached_model']}\n"
                gr.Info(msg.strip())
            except Exception as e:
                gr.Warning(f"Server error: {str(e)}")

        status_btn.click(on_status_click, queue=False)

    # Launch
    logger.info(f"Launching Gradio UI on {args.server_url}")
    demo.launch(
        share=args.share,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
