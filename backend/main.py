"""FastAPI app entrypoint."""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()

from backend.routes import inference, logs  # noqa: E402

app = FastAPI(title="StitchAI", version="0.1.0")

app.include_router(inference.router)
app.include_router(logs.router)

_heatmap_dir = Path("storage/heatmaps")
_heatmap_dir.mkdir(parents=True, exist_ok=True)
app.mount("/heatmaps", StaticFiles(directory=str(_heatmap_dir)), name="heatmaps")


@app.get("/health")
async def health():
    """Phase 0 'hello world' endpoint - prove the deployment loop works before adding
    model complexity."""
    return {"status": "ok"}
