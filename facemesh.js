// facemesh.js — Integración de MediaPipe FaceLandmarker.
// Sustituye a tinyFaceDetector + faceLandmark68Net de face-api.js para:
//   - Detección facial (478 puntos 3D en vez de 68 en 2D)
//   - Head pose real (matriz de transformación facial, no aproximación geométrica)
//   - Base para Blendshapes (52 Action Units) usados en Atención y, en Fase 2-4 (I2),
//     en la fusión tardía con audio para clasificar emociones.
//
// Se carga como módulo ES (por eso index.html lo referencia con type="module").
// Expone window.FaceMeshEngine con una API mínima y asíncrona para que app.js
// (script clásico) la consuma sin convertirse él mismo en módulo.
//
// NOTA IMPORTANTE (para el equipo): la extracción de yaw/pitch/roll desde la
// matriz de transformación (eulerFromMatrix) usa una convención estándar de
// descomposición de rotación, pero el signo/orden de ejes debe VALIDARSE contra
// el dataset de prueba interno (Sección 5.1 del reporte, MAE < 5° umbral) antes
// de considerarla definitiva. Es un punto de la validación de Nivel 1 pendiente.

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
// Modelo oficial de Google (float16, corre bien en CPU/GPU de laptop).
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

class FaceMeshEngine {
  constructor() {
    this.landmarker = null;
    this.ready = false;
    this.lastVideoTime = -1;
  }

  async load(onStatus) {
    if (onStatus) onStatus("Cargando MediaPipe FaceMesh…");
    const filesetResolver = await FilesetResolver.forVisionTasks(WASM_URL);
    this.landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU", // cae a CPU automáticamente si no hay GPU disponible
      },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      runningMode: "VIDEO",
      numFaces: 4, // detecta hasta 4 rostros; el "principal" se elige después por área
    });
    this.ready = true;
  }

  // Ejecuta la detección sobre el frame de <video> actual.
  // videoEl: elemento <video>. Debe llamarse una vez por frame en detectLoop().
  // Devuelve { faces: [...], count } — cada face trae:
  //   - landmarks: 478 puntos normalizados [0,1] con x,y,z
  //   - box: bounding box en píxeles derivado de los landmarks (min/max)
  //   - blendshapes: 52 categorías con score [0,1] (o null si no disponible)
  //   - matrix: Float32Array(16) de transformación facial (column-major)
  //   - pose: { yaw, pitch, roll } en grados, derivado de matrix
  detect(videoEl) {
    if (!this.ready) return { faces: [], count: 0 };

    // detectForVideo requiere timestamps estrictamente crecientes.
    const now = performance.now();
    if (now === this.lastVideoTime) return this._lastResult || { faces: [], count: 0 };
    this.lastVideoTime = now;

    const result = this.landmarker.detectForVideo(videoEl, now);
    const w = videoEl.videoWidth || 640;
    const h = videoEl.videoHeight || 480;

    const faces = (result.faceLandmarks || []).map((landmarks, i) => {
      const box = boxFromLandmarks(landmarks, w, h);
      const matrix =
        result.facialTransformationMatrixes && result.facialTransformationMatrixes[i]
          ? result.facialTransformationMatrixes[i].data
          : null;
      const blendshapes =
        result.faceBlendshapes && result.faceBlendshapes[i]
          ? result.faceBlendshapes[i].categories
          : null;
      return {
        landmarks,
        box,
        blendshapes,
        matrix,
        pose: matrix ? eulerFromMatrix(matrix) : { yaw: 0, pitch: 0, roll: 0 },
      };
    });

    this._lastResult = { faces, count: faces.length };
    return this._lastResult;
  }
}

// Bounding box en píxeles a partir de los 478 puntos normalizados (para elegir
// el rostro "principal" por área y para reutilizar el recuadro en overlays).
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

// Extrae yaw/pitch/roll (grados) de la matriz de transformación facial 4x4
// (column-major, formato MediaPipe: traslación + rotación en espacio cámara).
function eulerFromMatrix(m) {
  const r00 = m[0], r10 = m[1], r20 = m[2];
  const r01 = m[4], r11 = m[5];
  const r21 = m[6], r22 = m[10];

  const yaw = Math.atan2(-r20, Math.sqrt(r00 * r00 + r10 * r10));
  const pitch = Math.atan2(r10, r00);
  const roll = Math.atan2(r21, r22);

  return {
    yaw: (yaw * 180) / Math.PI,
    pitch: (pitch * 180) / Math.PI,
    roll: (roll * 180) / Math.PI,
  };
}

// Busca el score de un blendshape por nombre (p.ej. "eyeBlinkLeft").
function blendshapeScore(blendshapes, name) {
  if (!blendshapes) return 0;
  const b = blendshapes.find((x) => x.categoryName === name);
  return b ? b.score : 0;
}

window.FaceMeshEngine = new FaceMeshEngine();
window.FaceMeshUtils = { blendshapeScore };
