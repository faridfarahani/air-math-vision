import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import BASE_DIR
from .session import SessionController
from .vision import VisionEngine


session = SessionController()
vision = VisionEngine(session)


@asynccontextmanager
async def lifespan(app: FastAPI):
    vision.start()
    try:
        yield
    finally:
        vision.stop()


app = FastAPI(title="Air Math Vision", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/api/status")
async def api_status():
    snap = session.snapshot
    return JSONResponse({
        "camera_ok": snap.camera_ok,
        "people_count": snap.people_count,
        "identity_enrolled": snap.identity_enrolled,
        "identity_verified": snap.identity_verified,
        "face_similarity": snap.face_similarity,
        "gesture": snap.gesture,
        "state": snap.state,
        "current_character": snap.current_character,
        "buffer": snap.buffer,
        "tokens": snap.tokens,
        "expression": snap.expression,
        "result": snap.result,
        "events": list(session.events),
    })


@app.post("/api/start")
async def api_start():
    vision.cancel_pending_input()
    session.start()
    return {"ok": True}


@app.post("/api/stop")
async def api_stop():
    vision.cancel_pending_input()
    session.stop()
    return {"ok": True}


@app.post("/api/clear")
async def api_clear():
    vision.cancel_pending_input()
    session.clear()
    return {"ok": True}


@app.post("/api/reset-identity")
async def api_reset_identity():
    vision.cancel_pending_input()
    vision.reset_identity()
    session.start()
    return {"ok": True}


@app.get("/video")
async def video():
    async def generate():
        while True:
            jpg = vision.get_jpeg()
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpg +
                    b"\r\n"
                )
            await asyncio.sleep(0.035)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
