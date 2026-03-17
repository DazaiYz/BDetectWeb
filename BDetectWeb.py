from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont, ImageOps
import io
import logging
from ultralytics import YOLO
import numpy as np
import base64

# Keep only necessary logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI: Completely disable all docs/OpenAPI (eliminate redundant pages)
app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# Minimal CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ====================== [Modify Model Path] ======================
GRASS_MODEL_PATH = "D:/53180/yolov12-main/yolov12-main/byc_best.pt"  # Grass spike model
SEED_MODEL_PATH = "D:/53180/yolov12-main/yolov12-main/seed_best.pt"  # Seed model
# ==============================================================

# Load models
grass_model = None
seed_model = None
try:
    grass_model = YOLO(GRASS_MODEL_PATH)
    seed_model = YOLO(SEED_MODEL_PATH)
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")


def detect_image(image: Image.Image, model_type: str):
    """Detection + Drawing + Return Base64 + Statistics (Correct grass spike mature/immature mapping)"""
    model = grass_model if model_type == "grass" else seed_model
    if not model:
        raise Exception(f"{model_type} model not loaded")

    # Correct mobile image orientation (compatible with gallery/camera photos)
    image = ImageOps.exif_transpose(image)

    # Detection
    results = model(image, conf=0.3)
    boxes = results[0].boxes

    # Statistical data
    stats = {"total": 0, "mature": 0, "immature": 0, "seed": 0}
    # Draw detection boxes
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)

    # [Core Correction] Grass spike class mapping reversed: cls=0→immature, cls=1→mature
    class_map = {"grass": {0: "immature", 1: "mature"}, "seed": {0: "seed"}}
    colors = {"grass": {0: (255, 0, 0), 1: (0, 255, 0)}, "seed": {0: (0, 0, 255)}}

    for box in boxes:
        stats["total"] += 1
        cls = int(box.cls[0])
        conf = round(float(box.conf[0]), 2)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Statistics (synchronize grass spike mature/immature logic correction)
        if model_type == "grass":
            if cls == 1:  # cls=1 corresponds to mature
                stats["mature"] += 1
                label = f"mature {conf}"
            else:  # cls=0 corresponds to immature
                stats["immature"] += 1
                label = f"immature {conf}"
        else:
            stats["seed"] += 1
            label = f"seed {conf}"

        # Draw box + label: Keep border width at 5 pixels (thick box)
        draw.rectangle([x1, y1, x2, y2], outline=colors[model_type][cls], width=5)
        draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font, background=colors[model_type][cls])

    # Compress image (mobile adaptation)
    if image.width > 800:
        ratio = 800 / image.width
        image = image.resize((800, int(image.height * ratio)), Image.Resampling.LANCZOS)

    # Convert to Base64
    img_byte = io.BytesIO()
    image.save(img_byte, format="JPEG", quality=80)
    img_base64 = base64.b64encode(img_byte.getvalue()).decode("utf-8")

    return img_base64, stats


# Root path: Only display target page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Grass Spike/Seed Identification</title>
        <style>
            * {margin:0; padding:0; box-sizing:border-box;}
            body {padding:20px; font-family:Microsoft YaHei; background:#fff;}
            .box {max-width:100%;}
            h1 {text-align:center; margin-bottom:20px; font-size:22px;}
            .item {margin-bottom:15px;}
            label {display:block; margin-bottom:8px; font-size:18px;}
            select, input[type=file] {width:100%; padding:12px; border:1px solid #ddd; border-radius:8px; font-size:16px;}
            button {width:100%; padding:15px; background:#007bff; color:#fff; border:none; border-radius:8px; font-size:18px; cursor:pointer;}
            #result {margin-top:25px; display:none;}
            #result-img {width:100%; margin:10px 0; border-radius:8px;}
            .stats {font-size:18px; line-height:2; background:#f8f9fa; padding:15px; border-radius:8px;}
            .loading {text-align:center; margin-top:15px; display:none;}
            .error {color:red; text-align:center; margin-top:15px; display:none;}
        </style>
    </head>
    <body>
        <div class="box">
            <h1>Grass Spike/Seed Identification</h1>

            <!-- Core function area: Support gallery + camera -->
            <div class="item">
                <label>Select recognition type：</label>
                <select id="model">
                    <option value="grass">Grass Spikelet Recognition</option>
                    <option value="seed">Grass Seed Recognition</option>
                </select>
            </div>
            <div class="item">
                <label>Choose Image (Take Photo / Gallery)：</label>
                <input type="file" id="img" accept="image/jpeg,image/png,image/jpg" multiple="false">
            </div>
            <button id="btn">Start Recognition</button>

            <!-- Status prompts -->
            <div class="loading" id="loading">Recognition in progress...</div>
            <div class="error" id="error"></div>

            <!-- Result area -->
            <div id="result">
                <h3>Test Results：</h3>
                <img id="result-img" alt="Detection Diagram">
                <div class="stats" id="stats"></div>
            </div>
        </div>

        <script>
            // Only handle core logic
            const btn = document.getElementById('btn');
            const model = document.getElementById('model');
            const img = document.getElementById('img');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const result = document.getElementById('result');
            const resultImg = document.getElementById('result-img');
            const stats = document.getElementById('stats');

            btn.onclick = async () => {
                loading.style.display = 'block';
                error.style.display = 'none';
                result.style.display = 'none';

                if (!img.files.length) {
                    loading.style.display = 'none';
                    error.textContent = 'Please select an image';
                    error.style.display = 'block';
                    return;
                }

                const formData = new FormData();
                formData.append('file', img.files[0]);
                formData.append('model', model.value);

                try {
                    const res = await fetch('/detect', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();

                    loading.style.display = 'none';
                    result.style.display = 'block';
                    resultImg.src = `data:image/jpeg;base64,${data.img}`;

                    // Display statistics (grass spike results corrected)
                    if (model.value === 'grass') {
                        stats.innerHTML = `Total Tests：${data.stats.total}<br>Mature：${data.stats.mature}<br>Immature：${data.stats.immature}`;
                    } else {
                        // Seed prediction formula: 0.83*seed count +30.75 rounded
                        const seedCount = data.stats.seed;
                        const predictedResult = Math.round(0.83 * seedCount + 30.75);
                        stats.innerHTML = `Total detected spikes：${data.stats.total}<br>Predicted result：${predictedResult}`;
                    }
                } catch (e) {
                    loading.style.display = 'none';
                    error.textContent = 'Recognition failed';
                    error.style.display = 'block';
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# Detection interface
@app.post("/detect")
async def detect(file: UploadFile = File(...), model: str = Form(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_base64, stats = detect_image(image, model)
        return JSONResponse({"img": img_base64, "stats": stats})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Start server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
# http://192.168.31.23:8001