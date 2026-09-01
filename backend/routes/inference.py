"""POST /infer -> runs backbone + threshold + (maybe) explanation, via the orchestrator."""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.orchestrator import run_inference
from backend.schemas import InferenceResponse

router = APIRouter()


@router.post("/infer", response_model=InferenceResponse)
async def infer(category: str = Form(...), image: UploadFile = File(...)):
    suffix = Path(image.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_inference(tmp_path, category, original_filename=image.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))

    heatmap_path = result.pop("heatmap_path", None)
    heatmap_url = None
    if heatmap_path:
        # served via the /heatmaps static mount in backend/main.py
        heatmap_url = f"/heatmaps/{Path(heatmap_path).name}"

    return InferenceResponse(**result, heatmap_url=heatmap_url)
