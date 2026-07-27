// comparacion-i1.js — Comparación I1, construida como duplicado EXACTO del
// patrón de app-solo-facemesh.js (ya confirmado funcional): <video> nativo +
// <canvas> overlay dimensionado UNA vez, sin drawImage, sin resetear
// canvas.width en el bucle. Cada columna tiene su PROPIA llamada a
// getUserMedia() (no se comparte ni se clona un solo stream, que era una
// variable no probada en el intento anterior que sí falló).

const videoL = document.getElementById("videoL");
const videoR = document.getElementById("videoR");
const overlayL = document.getElementById("overlayL");
const overlayR = document.getElementById("overlayR");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const mFaceL = document.getElementById("mFaceL");
const mFaceR = document.getElementById("mFaceR");
const mPoseL = document.getElementById("mPoseL");
const mPoseR = document.getElementById("mPoseR");
const mPtsR = document.getElementById("mPtsR");

const detectorOpts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

let streamL = null;
let streamR = null;
let loopId = null;

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

async function loadAll() {
  const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
  ]);
  await window.FaceMeshEngine.load();
  startBtn.disabled = false;
  startBtn.textContent = "Iniciar cámaras";
}
startBtn.disabled = true;
startBtn.textContent = "Cargando modelos…";

async function startCameras() {
  try {
    // UNA sola llamada a getUserMedia — muchas laptops con una sola cámara
    // física (sobre todo en Linux/V4L2) solo permiten un acceso EXCLUSIVO al
    // dispositivo. Pedirla dos veces por separado puede colgarse para
    // siempre en la segunda llamada, que es justo lo que parece haber pasado.
    streamL = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
    streamR = streamL; // mismo stream, sin clonar, para ambos <video>
    videoL.srcObject = streamL;
    videoR.srcObject = streamR;
    await Promise.all([videoL.play(), videoR.play()]);

    // Overlay dimensionado UNA vez, igual que app.js — nunca se vuelve a tocar.
    overlayL.width = videoL.videoWidth || 640;
    overlayL.height = videoL.videoHeight || 480;
    overlayR.width = videoR.videoWidth || 640;
    overlayR.height = videoR.videoHeight || 480;

    startBtn.disabled = true;
    stopBtn.disabled = false;
    detectLoop();
  } catch (err) {
    console.error(err);
    alert("No se pudo acceder a la cámara: " + err.message);
  }
}

function stopCameras() {
  if (loopId) cancelAnimationFrame(loopId);
  loopId = null;
  if (streamL) streamL.getTracks().forEach((t) => t.stop());
  streamL = null;
  streamR = null;
  overlayL.getContext("2d").clearRect(0, 0, overlayL.width, overlayL.height);
  overlayR.getContext("2d").clearRect(0, 0, overlayR.width, overlayR.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  mFaceL.textContent = mFaceR.textContent = "—";
  mPoseL.textContent = mPoseR.textContent = "—";
  mPtsR.textContent = "—";
}

async function detectLoop() {
  if (!streamL || !streamR) return;

  // --- Izquierda: face-api.js (línea base) ---
  const ctxL = overlayL.getContext("2d");
  ctxL.clearRect(0, 0, overlayL.width, overlayL.height);
  const r = await faceapi.detectSingleFace(videoL, detectorOpts).withFaceLandmarks();
  if (r) {
    ctxL.strokeStyle = "#ffb020";
    ctxL.lineWidth = 3;
    ctxL.strokeRect(r.detection.box.x, r.detection.box.y, r.detection.box.width, r.detection.box.height);
    ctxL.fillStyle = "#ffb020";
    for (const pt of r.landmarks.positions) {
      ctxL.beginPath();
      ctxL.arc(pt.x, pt.y, 1.6, 0, Math.PI * 2);
      ctxL.fill();
    }
    const pose = computeHeadPose68(r.landmarks);
    mPoseL.textContent = `${pose.roll.toFixed(0)}° / ${pose.yaw.toFixed(0)}° / ${pose.pitch.toFixed(0)}°`;
    mFaceL.textContent = "Sí";
  } else {
    mPoseL.textContent = "—";
    mFaceL.textContent = "No";
  }

  // --- Derecha: MediaPipe FaceMesh (propuesto) ---
  const ctxR = overlayR.getContext("2d");
  ctxR.clearRect(0, 0, overlayR.width, overlayR.height);
  const fm = window.FaceMeshEngine.detect(videoR);
  const mesh = fm.faces[0] || null;
  if (mesh) {
    ctxR.strokeStyle = "#4ade80";
    ctxR.lineWidth = 3;
    ctxR.strokeRect(mesh.box.x, mesh.box.y, mesh.box.width, mesh.box.height);
    ctxR.fillStyle = "#4ade80";
    for (let i = 0; i < 468; i++) {
      const pt = mesh.landmarks[i];
      ctxR.beginPath();
      ctxR.arc(pt.x * overlayR.width, pt.y * overlayR.height, 1.2, 0, Math.PI * 2);
      ctxR.fill();
    }
    ctxR.fillStyle = "#22d3ee";
    for (let i = 468; i < mesh.landmarks.length; i++) {
      const pt = mesh.landmarks[i];
      ctxR.beginPath();
      ctxR.arc(pt.x * overlayR.width, pt.y * overlayR.height, 2, 0, Math.PI * 2);
      ctxR.fill();
    }
    mPtsR.textContent = String(mesh.landmarks.length);
    mPoseR.textContent = `${mesh.pose.roll.toFixed(0)}° / ${mesh.pose.yaw.toFixed(0)}° / ${mesh.pose.pitch.toFixed(0)}°`;
    mFaceR.textContent = "Sí";
  } else {
    mPtsR.textContent = "—";
    mPoseR.textContent = "—";
    mFaceR.textContent = "No";
  }

  loopId = requestAnimationFrame(detectLoop);
}

startBtn.addEventListener("click", startCameras);
stopBtn.addEventListener("click", stopCameras);

window.addEventListener("load", () => {
  loadAll().catch((e) => {
    console.error("Error cargando modelos:", e);
    startBtn.textContent = "Error al cargar (ver consola)";
  });
});