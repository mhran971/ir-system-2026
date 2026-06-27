import asyncio
import json
import os
import subprocess
import sys
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from api.models import EvaluateRunResponse

router = APIRouter(tags=["evaluation"])

_RESULTS_PATH = "data/evaluation/results_clinical.json"
_EVALUATE_SCRIPT = "scripts/evaluate.py"


@router.get("/evaluate")
def get_evaluation_results() -> Dict[str, Any]:
    """
    Return pre-computed evaluation metrics from data/evaluation/results_clinical.json.
    Run POST /evaluate/run first if the file does not exist.
    """
    if not os.path.exists(_RESULTS_PATH):
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run POST /evaluate/run first.",
        )
    with open(_RESULTS_PATH, "r") as f:
        return json.load(f)


@router.post("/evaluate/run", response_model=EvaluateRunResponse)
async def run_evaluation():
    """
    Execute scripts/evaluate.py (same subprocess call as the Streamlit GUI).
    Runs in a thread pool so the async event loop is not blocked.
    The evaluation pipeline may take several minutes.
    """
    loop = asyncio.get_event_loop()

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, _EVALUATE_SCRIPT],
            capture_output=True,
            text=True,
        )

    try:
        proc = await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=600.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Evaluation timed out after 600 seconds.",
        )

    status = "completed" if proc.returncode == 0 else "error"
    return EvaluateRunResponse(
        status=status,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
    )
