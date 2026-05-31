#!/usr/bin/env python3
"""
BAIT Online Detection Server — FastAPI endpoints for backdoor scanning.

Two main endpoints:
  POST /probe   — Fast Q-score only (~30s)
  POST /scan    — Full BAIT + CBSS-Lite + optional SLM (~3-5 min)

Loads SLM and diverse prompts at startup, caches target models to avoid OOM.
"""

import argparse
import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from time import time

import torch
import torch.nn.functional as F
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# CONNECTS TO: existing BAIT modules
try:
    from src.core.detector import BAIT, ScanResult, TopKResult
    from src.core.neuro_detector import NeuroBAITDetector
    from src.config.arguments import BAITArguments, ModelArguments, DataArguments
    from src.models.model import build_model
    from src.data.dataset import build_data_module, TokenizedDataset
    from cbss_lite.cbss_scorer import CBSSLiteScorer
except ImportError as e:
    logger.error(f"Failed to import BAIT modules: {e}")
    sys.exit(1)


# ────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ────────────────────────────────────────

class ProbeRequest(BaseModel):
    """Request schema for /probe endpoint."""
    model_path: str
    model_zoo_id: str = ""
    dataset: str = "alpaca"
    attack: str = "cba"


class ScanRequest(ProbeRequest):
    """Request schema for /scan endpoint."""
    use_slm: bool = True
    tau: float = 0.1
    beta: float = 0.0


class ComponentScores(BaseModel):
    """CBSS-Lite component scores."""
    Q: float
    Q_bar: float
    E: float
    U: float
    B: float


class DetectionResult(BaseModel):
    """Response schema for detection endpoints."""
    model_path: str
    verdict: str  # "BACKDOORED" or "CLEAN"
    cbss_lite_score: float
    q_score: float
    p_value: float
    top_candidates: List[Dict[str, Any]]
    component_scores: ComponentScores
    time_taken_seconds: float
    scan_type: str  # "fast_probe" or "full_scan"


class StatusResponse(BaseModel):
    """Response schema for /status endpoint."""
    slm_loaded: bool
    slm_model_id: str
    diverse_prompts_count: int
    cached_model: Optional[str]
    gpu_memory_used_gb: float
    timestamp: str


# ────────────────────────────────────────
# STREAMING BAIT — emits SSE progress events
# ────────────────────────────────────────

class StreamingBAIT(BAIT):
    """BAIT subclass that emits progress events to a queue."""

    def __init__(self, *args, progress_queue: Optional[asyncio.Queue] = None, **kwargs):
        """Initialize with optional progress queue for SSE streaming."""
        super().__init__(*args, **kwargs)
        self.progress_queue = progress_queue

    def _update(self, *args, **kwargs):
        """Override _update to emit progress events."""
        result = super()._update(*args, **kwargs)

        # Emit progress event if queue exists
        if self.progress_queue and result[0] is not None:
            try:
                candidates_count = len(getattr(self, "_all_candidate_pairs", []))
                self.progress_queue.put_nowait({
                    "type": "progress",
                    "candidates_so_far": candidates_count,
                    "message": "Scanning vocabulary..."
                })
            except asyncio.QueueFull:
                pass

        return result


# ────────────────────────────────────────
# GLOBAL STATE & HELPERS
# ────────────────────────────────────────

class AppState:
    """Global app state holder."""
    def __init__(self):
        self.slm_model = None
        self.slm_tokenizer = None
        self.diverse_prompts: List[str] = []
        self.target_model_cache: Dict[str, tuple] = {}
        self.progress_queues: Dict[str, asyncio.Queue] = {}
        self.slm_repo_id: str = ""


def _resolve_base_model_path(cache_dir: str, model_name: str) -> str:
    """
    Resolve base model path handling Layout A (snapshots) and Layout B (flat).

    Args:
        cache_dir: Base model cache directory
        model_name: Model identifier (e.g., "Llama-2-7B-hf")

    Returns:
        Full path to the model directory
    """
    cache_path = Path(cache_dir)

    # Try Layout A: models--org--name/snapshots/hash/
    model_pattern = model_name.replace("/", "--")
    for model_dir in cache_path.glob("models--*"):
        if model_pattern.lower() in model_dir.name.lower():
            snapshots = model_dir / "snapshots"
            if snapshots.exists():
                hash_dirs = list(snapshots.glob("*/"))
                if hash_dirs:
                    return str(hash_dirs[0])
            if (model_dir / "config.json").exists():
                return str(model_dir)

    # Try Layout B: flat directory
    model_base = cache_path / model_name.split("/")[-1]
    if model_base.exists() and (model_base / "config.json").exists():
        return str(model_base)

    # Fallback
    logger.warning(f"Model path not found via patterns, using fallback: {model_name}")
    return str(cache_path / model_name.split("/")[-1])


def _load_slm(slm_repo_or_path: str) -> tuple:
    """
    Load SLM from HuggingFace or local path.

    Args:
        slm_repo_or_path: HuggingFace repo ID or local directory path

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading SLM from {slm_repo_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(slm_repo_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            slm_repo_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        logger.info("✓ SLM loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load SLM: {e}")
        raise


def _load_diverse_prompts(slm_repo_id: str) -> List[str]:
    """
    Load diverse prompts cache from HuggingFace repo.

    Args:
        slm_repo_id: HuggingFace repo ID

    Returns:
        List of diverse prompts, empty list if not found
    """
    try:
        logger.info(f"Downloading diverse prompts cache from {slm_repo_id}")
        cache_file = hf_hub_download(
            repo_id=slm_repo_id,
            filename="diverse_prompts_cache.json"
        )
        with open(cache_file) as f:
            cache_data = json.load(f)

        # Handle both list and dict formats
        if isinstance(cache_data, list):
            prompts = cache_data
        elif isinstance(cache_data, dict) and "prompts" in cache_data:
            prompts = cache_data["prompts"]
        else:
            prompts = []

        logger.info(f"✓ Loaded {len(prompts)} diverse prompts")
        return prompts
    except Exception as e:
        logger.warning(f"Could not load diverse prompts: {e}. Will select on first probe.")
        return []


def _get_target_model(req: ProbeRequest, app_state: AppState, cache_dir: str) -> tuple:
    """
    Load or retrieve target model from cache.

    Args:
        req: Probe/Scan request
        app_state: Global app state
        cache_dir: Base model cache directory

    Returns:
        Tuple of (model, tokenizer)
    """
    # Use model path as cache key
    cache_key = req.model_path

    # Check if already cached
    if cache_key in app_state.target_model_cache:
        logger.info(f"Using cached model: {cache_key}")
        return app_state.target_model_cache[cache_key]

    # Evict previous model to avoid OOM
    if app_state.target_model_cache:
        old_key = list(app_state.target_model_cache.keys())[0]
        del app_state.target_model_cache[old_key]
        torch.cuda.empty_cache()
        logger.info(f"Evicted previous model to free GPU memory")

    logger.info(f"Loading target model: {req.model_path}")

    try:
        model_args = ModelArguments(
            model_name_or_path=req.model_path,
            cache_dir=cache_dir
        )

        # If model_zoo_id provided, resolve paths
        if req.model_zoo_id:
            logger.info(f"Resolving from model zoo ID: {req.model_zoo_id}")
            # In actual usage, would resolve from config.json in model zoo
            # For now, use provided path directly
            pass

        # Resolve base model if it's a name rather than path
        if "/" in model_args.model_name_or_path and not Path(model_args.model_name_or_path).exists():
            model_args.model_name_or_path = _resolve_base_model_path(
                cache_dir,
                model_args.model_name_or_path
            )

        model, tokenizer = build_model(model_args)
        model.eval()

        # Cache the loaded model
        app_state.target_model_cache[cache_key] = (model, tokenizer)

        logger.info(f"✓ Model loaded and cached")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load target model: {e}")
        raise


def _get_gpu_memory_used() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


# ────────────────────────────────────────
# STARTUP & SHUTDOWN
# ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load SLM and diverse prompts at startup, cleanup on shutdown."""
    app.state.app_state = AppState()

    logger.info("=== BAIT Detection Server Starting ===")

    try:
        # Load SLM
        app.state.app_state.slm_model, app.state.app_state.slm_tokenizer = _load_slm(
            app.state.slm_repo_id
        )
        app.state.app_state.slm_repo_id = app.state.slm_repo_id

        # Load diverse prompts
        app.state.app_state.diverse_prompts = _load_diverse_prompts(app.state.slm_repo_id)

        logger.info(f"GPU memory used: {_get_gpu_memory_used():.2f} GB")
        logger.info("=== Server ready to accept requests ===")

    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down server...")
    if hasattr(app.state, "app_state"):
        if app.state.app_state.slm_model is not None:
            del app.state.app_state.slm_model
        if app.state.app_state.target_model_cache:
            app.state.app_state.target_model_cache.clear()
    torch.cuda.empty_cache()
    logger.info("✓ Cleanup complete")


# ────────────────────────────────────────
# FASTAPI APP
# ────────────────────────────────────────

app = FastAPI(
    title="BAIT Online Detection",
    description="Backdoor detection via BAIT + NeuroBAIT",
    lifespan=lifespan
)


def _component_scores_from_dict(data: Dict[str, float]) -> ComponentScores:
    """Convert dictionary to ComponentScores, filling missing values with 0.0."""
    return ComponentScores(
        Q=data.get("Q", 0.0),
        Q_bar=data.get("Q_bar", 0.0),
        E=data.get("E", 0.0),
        U=data.get("U", 0.0),
        B=data.get("B", 0.0)
    )


@app.post("/probe", response_model=DetectionResult)
async def probe(req: ProbeRequest) -> DetectionResult:
    """
    Fast backdoor probe using warmup inversion only (Q-score).

    Returns in ~30 seconds on an A100.
    """
    start_time = time()
    logger.info(f"[PROBE] Request: {req.model_path}")

    try:
        app_state = app.state.app_state

        # Load target model
        model, tokenizer = _get_target_model(req, app_state, app.state.cache_dir)

        # Build minimal dataloader with diverse prompts
        data_args = DataArguments(
            data_dir=app.state.data_dir,
            dataset=req.dataset,
            prompt_size=min(10, len(app_state.diverse_prompts)) if app_state.diverse_prompts else 10
        )

        # Build dataloader (will use diverse prompts if available)
        _, dataloader = build_data_module(data_args, tokenizer, logger)

        # Fast BAIT config: warmup only
        bait_args = BAITArguments()
        bait_args.warmup_steps = 5
        bait_args.full_steps = 0  # Skip full inversion
        bait_args.batch_size = data_args.batch_size
        bait_args.prompt_size = data_args.prompt_size

        # Instantiate scanner
        with torch.no_grad():
            scanner = BAIT(
                model=model,
                tokenizer=tokenizer,
                dataloader=dataloader,
                bait_args=bait_args,
                logger=logger,
                device=torch.device(app.state.device)
            )

            # Run scan
            scan_result: ScanResult = scanner.run()

        # Extract results
        verdict = "BACKDOORED" if scan_result.is_backdoor else "CLEAN"
        q_score = scan_result.top_k_results[0].q_score if scan_result.top_k_results else 0.0
        p_value = scan_result.p_value if hasattr(scan_result, "p_value") else 0.0

        # Format top candidates
        top_candidates = [
            {
                "rank": i + 1,
                "inverted_target": result.invert_target,
                "q_score": result.q_score,
                "token_length": len(result.invert_target.split())
            }
            for i, result in enumerate(scan_result.top_k_results[:5])
        ]

        elapsed = time() - start_time
        logger.info(f"[PROBE] Complete: {verdict} (Q={q_score:.3f}, time={elapsed:.1f}s)")

        return DetectionResult(
            model_path=req.model_path,
            verdict=verdict,
            cbss_lite_score=0.0,
            q_score=q_score,
            p_value=p_value,
            top_candidates=top_candidates,
            component_scores=ComponentScores(Q=0, Q_bar=0, E=0, U=0, B=0),
            time_taken_seconds=elapsed,
            scan_type="fast_probe"
        )

    except Exception as e:
        logger.error(f"[PROBE] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan", response_model=DetectionResult)
async def scan(req: ScanRequest) -> DetectionResult:
    """
    Full scan with complete BAIT inversion + CBSS-Lite + optional SLM.

    Takes 3-5 minutes on an A100.
    """
    start_time = time()
    logger.info(f"[SCAN] Request: {req.model_path}, use_slm={req.use_slm}")

    try:
        app_state = app.state.app_state

        # Load target model
        model, tokenizer = _get_target_model(req, app_state, app.state.cache_dir)

        # Build dataloader
        data_args = DataArguments(
            data_dir=app.state.data_dir,
            dataset=req.dataset,
            prompt_size=min(10, len(app_state.diverse_prompts)) if app_state.diverse_prompts else 10
        )

        _, dataloader = build_data_module(data_args, tokenizer, logger)

        # Instantiate scanner
        with torch.no_grad():
            if req.use_slm:
                # CONNECTS TO: NeuroBAITDetector
                scanner = NeuroBAITDetector(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=dataloader,
                    bait_args=BAITArguments(),
                    logger=logger,
                    device=torch.device(app.state.device),
                    slm_model=app_state.slm_model,
                    slm_tokenizer=app_state.slm_tokenizer
                )
            else:
                # CONNECTS TO: BAIT
                scanner = BAIT(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=dataloader,
                    bait_args=BAITArguments(),
                    logger=logger,
                    device=torch.device(app.state.device)
                )

            # Run full scan
            scan_result: ScanResult = scanner.run()

        # Extract results
        verdict = "BACKDOORED" if scan_result.is_backdoor else "CLEAN"
        q_score = scan_result.top_k_results[0].q_score if scan_result.top_k_results else 0.0
        p_value = scan_result.p_value if hasattr(scan_result, "p_value") else 0.0

        # Format top candidates
        top_candidates = [
            {
                "rank": i + 1,
                "inverted_target": result.invert_target,
                "q_score": result.q_score,
                "token_length": len(result.invert_target.split())
            }
            for i, result in enumerate(scan_result.top_k_results[:5])
        ]

        # Compute CBSS-Lite score on best candidate
        cbss_lite_score = 0.0
        component_scores = ComponentScores(Q=0, Q_bar=0, E=0, U=0, B=0)

        if scan_result.top_k_results and req.use_slm:
            try:
                scorer = CBSSLiteScorer(tau=req.tau, beta=req.beta, device=app.state.device)
                best = scan_result.top_k_results[0]

                # CONNECTS TO: CBSSLiteScorer.score_candidate
                scores = scorer.score_candidate(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=app_state.diverse_prompts,
                    target_tokens=best.invert_target.split()
                )

                if isinstance(scores, dict):
                    component_scores = _component_scores_from_dict(scores)
                    cbss_lite_score = sum(scores.values()) / len(scores) if scores else 0.0

            except Exception as e:
                logger.warning(f"CBSS-Lite scoring failed: {e}")

        elapsed = time() - start_time
        logger.info(f"[SCAN] Complete: {verdict} (CBSS={cbss_lite_score:.3f}, time={elapsed:.1f}s)")

        return DetectionResult(
            model_path=req.model_path,
            verdict=verdict,
            cbss_lite_score=cbss_lite_score,
            q_score=q_score,
            p_value=p_value,
            top_candidates=top_candidates,
            component_scores=component_scores,
            time_taken_seconds=elapsed,
            scan_type="full_scan"
        )

    except Exception as e:
        logger.error(f"[SCAN] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Health check endpoint with system status."""
    app_state = app.state.app_state
    cached_model = list(app_state.target_model_cache.keys())[0] if app_state.target_model_cache else None

    return StatusResponse(
        slm_loaded=app_state.slm_model is not None,
        slm_model_id=app_state.slm_repo_id,
        diverse_prompts_count=len(app_state.diverse_prompts),
        cached_model=cached_model,
        gpu_memory_used_gb=_get_gpu_memory_used(),
        timestamp=datetime.now().isoformat()
    )


async def _progress_generator(job_id: str, app_state: AppState):
    """Generate SSE events from progress queue."""
    queue = app_state.progress_queues.get(job_id)
    if not queue:
        yield "data: {\"error\": \"Job not found\"}\n\n"
        return

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=60.0)
            yield f"data: {json.dumps(event)}\n\n"

            if event.get("type") == "done":
                break

        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'timeout'})}\n\n"
            break


@app.get("/progress/{job_id}")
async def progress(job_id: str):
    """Server-sent events stream for scan progress."""
    app_state = app.state.app_state
    return StreamingResponse(
        _progress_generator(job_id, app_state),
        media_type="text/event-stream"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BAIT Online Detection Server")
    parser.add_argument("--slm-repo-id", required=True, help="SLM HuggingFace repo ID or local path")
    parser.add_argument("--cache-dir", required=True, help="Base model cache directory")
    parser.add_argument("--data-dir", required=True, help="Dataset directory")
    parser.add_argument("--results-dir", default="/tmp", help="Results directory")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8787, help="Server port")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Store in app state for lifespan access
    app.state.slm_repo_id = args.slm_repo_id
    app.state.cache_dir = args.cache_dir
    app.state.data_dir = args.data_dir
    app.state.results_dir = args.results_dir
    app.state.device = args.device

    logger.info(f"Starting BAIT Detection Server on {args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
