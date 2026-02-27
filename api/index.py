from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
import cv2

app = FastAPI()

# allow browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# IMAGE PROCESS
# =========================

def preprocess(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,2
    )

    return thresh


def compute_density(thresh, box):

    x,y,w,h = box
    roi = thresh[y:y+h, x:x+w]

    total = w*h
    filled = cv2.countNonZero(roi)

    return filled / total


def detect_answers(thresh):

    template = {
        1: {"A":[100,200,40,40],"B":[160,200,40,40],"C":[220,200,40,40],"D":[280,200,40,40]},
        2: {"A":[100,260,40,40],"B":[160,260,40,40],"C":[220,260,40,40],"D":[280,260,40,40]}
    }

    results = {}

    for q, opts in template.items():

        densities = {}

        for opt, box in opts.items():
            densities[opt] = compute_density(thresh, box)

        best = max(densities, key=densities.get)

        if densities[best] < 0.15:
            results[q] = None
        else:
            results[q] = best

    return results


# =========================
# ROUTES
# =========================

@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error":"invalid image"}

    thresh = preprocess(image)

    answers = detect_answers(thresh)

    return {"answers": answers}


@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Exam Scanner</title>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
    </head>
    <body>

    <h2>Exam Scanner</h2>

    <input type="file" id="file" accept="image/*" capture="environment"/>
    <button onclick="upload()">Scan</button>

    <pre id="result"></pre>

    <script>

    async function upload(){

      const fileInput = document.getElementById("file")

      if(!fileInput.files.length) return

      const form = new FormData()
      form.append("file", fileInput.files[0])

      const res = await fetch("/api/scan",{
        method:"POST",
        body:form
      })

      const data = await res.json()

      document.getElementById("result").textContent =
          JSON.stringify(data,null,2)
    }

    </script>

    </body>
    </html>
    """
