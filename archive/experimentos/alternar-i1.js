// alternar-i1.js — Un solo <video> + un solo <canvas id="overlay">, MISMA
// estructura que app-solo-facemesh.js (ya confirmada funcional). Un botón
// alterna cuál modelo corre y dibuja: face-api.js (68 pts) o MediaPipe
// FaceMesh (478 pts). Incluye detection_rate (ventana 5s) y roll/yaw/pitch.

const video    = document.getElementById("video");
const overlay  = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn  = document.getElementById("stopBtn");
const toggleBtn = document.getElementById("toggleBtn");
const mModel   = document.getElementById("mModel");
const mFace    = document.getElementById("mFace");
const mPts     = document.getElementById("mPts");
const mDetRate = document.getElementById("mDetRate");
const mRoll    = document.getElementById("mRoll");
const mYaw     = document.getElementById("mYaw");
const mPitch   = document.getElementById("mPitch");
const mDebugBox = document.getElementById("mDebugBox");

let stream = null;
let loopId = null;
let activeModel = "mediapipe"; // "faceapi" | "mediapipe"

const detectorOpts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

// Ventana deslizante de 5s, UNA POR MODELO, para que al alternar cada uno
// mantenga su propia estadística de detection_rate.
const WINDOW_MS = 5000;
const frames = { faceapi: [], mediapipe: [] };
function pushFrame(model, hasFace) {
  const now = performance.now();
  const arr = frames[model];
  arr.push({ t: now, hasFace });
  while (arr.length && arr[0].t < now - WINDOW_MS) arr.shift();
}
function detectionRate(model) {
  const arr = frames[model];
  if (!arr.length) return null;
  return arr.filter((f) => f.hasFace).length / arr.length;
}

function setStatus(msg) {
  statusEl.textContent = msg;
  statusEl.style.display = msg ? "block" : "none";
}

// Misma fórmula que el fallback de app.js (computeHeadPoseFallback).
function computeHeadPose68(landmarks) {
  const p = landmarks.positions;
  const centroid = (a, b) => {
    let sx = 0, sy = 0;
    for (let i = a; i <= b; i++) { sx += p[i].x; sy += p[i].y; }
    const n = b - a + 1;
    return { x: sx / n, y: sy / n };
  };
  const eyeR = centroid(36, 41);
  const eyeL = centroid(42, 47);
  const noseTip = p[30];
  const mouth = { x: (p[48].x + p[54].x) / 2, y: (p[48].y + p[54].y) / 2 };
  const eyesMid = { x: (eyeR.x + eyeL.x) / 2, y: (eyeR.y + eyeL.y) / 2 };
  const interocular = Math.hypot(eyeL.x - eyeR.x, eyeL.y - eyeR.y) || 1;
  const roll = (Math.atan2(eyeL.y - eyeR.y, eyeL.x - eyeR.x) * 180) / Math.PI;
  const yaw = Math.max(-90, Math.min(90, ((noseTip.x - eyesMid.x) / interocular) * 90));
  const faceH = (mouth.y - eyesMid.y) || 1;
  const pitch = Math.max(-90, Math.min(90, (((noseTip.y - eyesMid.y) / faceH) - 0.55) * 120));
  return { roll, yaw, pitch };
}

async function loadModels() {
  setStatus("Cargando modelos de IA…");
  const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
  ]);
  await window.FaceMeshEngine.load(setStatus);
  setStatus("Modelos listos. Presiona «Iniciar cámara».");
  startBtn.disabled = false;
  toggleBtn.disabled = false;
  updateToggleLabel();
}

function updateToggleLabel() {
  toggleBtn.textContent = activeModel === "faceapi"
    ? "Modelo: 🟠 face-api.js (68 pts) — clic para cambiar"
    : "Modelo: 🟢 MediaPipe FaceMesh (478 pts) — clic para cambiar";
  mModel.textContent = activeModel === "faceapi" ? "face-api.js" : "MediaPipe FaceMesh";
}

function toggleModel() {
  activeModel = activeModel === "faceapi" ? "mediapipe" : "faceapi";
  updateToggleLabel();
  // Limpia el canvas al cambiar para no dejar puntos del modelo anterior.
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
    });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus("");
    detectLoop();
  } catch (err) {
    console.error(err);
    setStatus("No se pudo acceder a la cámara: " + err.message);
  }
}

function stopCamera() {
  if (loopId) cancelAnimationFrame(loopId);
  loopId = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Cámara detenida.");
  mFace.textContent = "—";
  mPts.textContent = "—";
  mDetRate.textContent = "—";
  mRoll.textContent = mYaw.textContent = mPitch.textContent = "—";
}

// Detección de face-api EN SEGUNDO PLANO (fire-and-forget): nunca se hace
// "await" dentro del bucle de dibujo. Se guarda el último resultado y el
// dibujo lo consume de forma síncrona en cada frame — igual patrón que
// MediaPipe (que es síncrono por naturaleza). Esto evita que el dibujo de
// face-api ocurra como continuación de una promesa (después de un "await"),
// que es la hipótesis confirmada de por qué no se pintaba en pantalla aunque
// el buffer del canvas tuviera los píxeles correctos (verificado con
// getImageData).
let lastFaceApiResult = null;
let faceApiBusy = false;
function kickFaceApiDetection() {
  if (faceApiBusy) return;
  faceApiBusy = true;
  faceapi.detectSingleFace(video, detectorOpts).withFaceLandmarks()
    .then((r) => { lastFaceApiResult = r || null; })
    .catch((e) => console.error("Error en detección face-api:", e))
    .finally(() => { faceApiBusy = false; });
}

function detectLoop() {
  if (!stream) return;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (activeModel === "faceapi") {
    kickFaceApiDetection(); // dispara detección en segundo plano, no bloquea
    const r = lastFaceApiResult; // dibuja con el último resultado disponible
    pushFrame("faceapi", !!r);
    if (r) {
      ctx.strokeStyle = "#ffb020";
      ctx.lineWidth = 3;
      ctx.strokeRect(r.detection.box.x, r.detection.box.y, r.detection.box.width, r.detection.box.height);
      ctx.fillStyle = "#ffb020";
      for (const pt of r.landmarks.positions) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
      const pose = computeHeadPose68(r.landmarks);
      mRoll.textContent = `${pose.roll.toFixed(0)}°`;
      mYaw.textContent = `${pose.yaw.toFixed(0)}°`;
      mPitch.textContent = `${pose.pitch.toFixed(0)}°`;
      mFace.textContent = "Sí";
      mPts.textContent = "68";
      const px0 = Math.round(r.landmarks.positions[0].x);
      const py0 = Math.round(r.landmarks.positions[0].y);
      let pixelReal = "?";
      try {
        const d = ctx.getImageData(px0, py0, 1, 1).data;
        pixelReal = `rgba(${d[0]},${d[1]},${d[2]},${d[3]})`;
      } catch (e) {
        pixelReal = "ERROR: " + e.message;
      }
      mDebugBox.textContent =
        `debug: box=(${r.detection.box.x.toFixed(0)},${r.detection.box.y.toFixed(0)}) ` +
        `${r.detection.box.width.toFixed(0)}x${r.detection.box.height.toFixed(0)} · ` +
        `landmarks=${r.landmarks.positions.length} · overlay=${overlay.width}x${overlay.height} · ` +
        `primerPunto=(${px0},${py0}) · pixelEnBuffer=${pixelReal} (naranja≈rgba(255,176,32,255))`;
    } else {
      mFace.textContent = "No";
      mPts.textContent = "—";
      mRoll.textContent = mYaw.textContent = mPitch.textContent = "—";
      mDebugBox.textContent = "debug: sin rostro";
    }
    mDetRate.textContent = detectionRate("faceapi") != null ? `${Math.round(detectionRate("faceapi") * 100)}%` : "—";

  } else {
    const fm = window.FaceMeshEngine.detect(video);
    const mesh = fm.faces[0] || null;
    pushFrame("mediapipe", !!mesh);
    if (mesh) {
      ctx.strokeStyle = "#4ade80";
      ctx.lineWidth = 3;
      ctx.strokeRect(mesh.box.x, mesh.box.y, mesh.box.width, mesh.box.height);
      ctx.fillStyle = "#4ade80";
      for (let i = 0; i < 468; i++) {
        const pt = mesh.landmarks[i];
        ctx.beginPath();
        ctx.arc(pt.x * overlay.width, pt.y * overlay.height, 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = "#22d3ee";
      for (let i = 468; i < mesh.landmarks.length; i++) {
        const pt = mesh.landmarks[i];
        ctx.beginPath();
        ctx.arc(pt.x * overlay.width, pt.y * overlay.height, 2, 0, Math.PI * 2);
        ctx.fill();
      }
      mRoll.textContent = `${mesh.pose.roll.toFixed(0)}°`;
      mYaw.textContent = `${mesh.pose.yaw.toFixed(0)}°`;
      mPitch.textContent = `${mesh.pose.pitch.toFixed(0)}°`;
      mFace.textContent = "Sí";
      mDebugBox.textContent =
        `debug: box=(${mesh.box.x.toFixed(0)},${mesh.box.y.toFixed(0)}) ` +
        `${mesh.box.width.toFixed(0)}x${mesh.box.height.toFixed(0)} · ` +
        `landmarks=${mesh.landmarks.length} · overlay=${overlay.width}x${overlay.height} · ` +
        `primerPunto=(${(mesh.landmarks[0].x * overlay.width).toFixed(0)},${(mesh.landmarks[0].y * overlay.height).toFixed(0)})`;
      mPts.textContent = String(mesh.landmarks.length);
    } else {
      mFace.textContent = "No";
      mPts.textContent = "—";
      mRoll.textContent = mYaw.textContent = mPitch.textContent = "—";
    }
    mDetRate.textContent = detectionRate("mediapipe") != null ? `${Math.round(detectionRate("mediapipe") * 100)}%` : "—";
  }

  loopId = requestAnimationFrame(detectLoop);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
toggleBtn.addEventListener("click", toggleModel);

window.addEventListener("load", loadModels);