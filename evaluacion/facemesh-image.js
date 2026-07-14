// facemesh-image.js — Variante de facemesh.js para EVALUACIÓN OFFLINE con
// imágenes estáticas (Nivel 1 / Sección 5.1 del reporte, dataset RAF-DB).
// NO modifica facemesh.js — ese sigue intacto con runningMode: "VIDEO" para
// el pipeline en vivo de app.js. Misma librería (@mediapipe/tasks-vision),
// pero:
//   - runningMode: "IMAGE" (no "VIDEO")
//   - landmarker.detect(imagen) en vez de landmarker.detectForVideo(...)
//   - numFaces: 1 (las imágenes de RAF-DB ya vienen alineadas a un rostro)
//   - sin facialTransformationMatrixes (no necesitamos head pose aquí)

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

class FaceMeshImageEngine {
  constructor() {
    this.landmarker = null;
    this.ready = false;
  }

  async load(onStatus) {
    if (onStatus) onStatus("Cargando MediaPipe FaceMesh (modo imagen)…");
    const filesetResolver = await FilesetResolver.forVisionTasks(WASM_URL);
    this.landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU",
      },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
      runningMode: "IMAGE",
      numFaces: 1,
    });
    this.ready = true;
  }

  // imgEl: HTMLImageElement ya cargado (con naturalWidth/naturalHeight listos).
  // A diferencia de facemesh.js, detect() aquí es una llamada síncrona sobre
  // una sola imagen — no hay throttling de timestamp de video que manejar.
  detect(imgEl) {
    if (!this.ready) return { faces: [], count: 0 };
    const result = this.landmarker.detect(imgEl);
    const w = imgEl.naturalWidth || imgEl.width;
    const h = imgEl.naturalHeight || imgEl.height;

    const faces = (result.faceLandmarks || []).map((landmarks, i) => {
      const box = boxFromLandmarks(landmarks, w, h);
      const blendshapes =
        result.faceBlendshapes && result.faceBlendshapes[i]
          ? result.faceBlendshapes[i].categories
          : null;
      return { landmarks, box, blendshapes };
    });

    return { faces, count: faces.length };
  }
}

function boxFromLandmarks(landmarks, w, h) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const x = minX * w, y = minY * h;
  const width = (maxX - minX) * w, height = (maxY - minY) * h;
  return { x, y, width, height, area: width * height };
}

window.FaceMeshImageEngine = new FaceMeshImageEngine();
