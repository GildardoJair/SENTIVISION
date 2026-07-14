// app-solo-facemesh.js — Versión MÍNIMA: solo MediaPipe FaceMesh.
// Calca la estructura de app.js (que sí funciona en el sistema del equipo):
// <video> visible reproduciendo nativo + <canvas id="overlay"> encima que
// solo dibuja anotaciones. Sin comparación, sin face-api, sin sparklines.

const video    = document.getElementById("video");
const overlay  = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn  = document.getElementById("stopBtn");
const mFace    = document.getElementById("mFace");
const mPts     = document.getElementById("mPts");
const mPose    = document.getElementById("mPose");

let stream = null;
let loopId = null;

function setStatus(msg) {
  statusEl.textContent = msg;
  statusEl.style.display = msg ? "block" : "none";
}

// --- Carga del modelo (igual patrón que loadModels en app.js) ---
async function loadModel() {
  setStatus("Cargando MediaPipe FaceMesh…");
  try {
    await window.FaceMeshEngine.load(setStatus);
    setStatus("Modelo listo. Presiona «Iniciar cámara».");
    startBtn.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus("Error cargando FaceMesh: " + e.message);
  }
}

// --- Cámara (CALCADO de startCamera en app.js) ---
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
    });
    video.srcObject = stream;
    await video.play();
    // Igual que app.js: overlay dimensionado UNA vez, tras play().
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
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Cámara detenida.");
  mFace.textContent = "—";
  mPts.textContent = "—";
  mPose.textContent = "—";
}

// --- Bucle de detección (CALCADO de detectLoop en app.js) ---
function detectLoop() {
  if (!stream) return;

  const fm = window.FaceMeshEngine.detect(video);
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // Rostro principal = mayor área.
  let mesh = null;
  if (fm.faces.length) {
    mesh = fm.faces.reduce((m, f) => (f.box.area > m.box.area ? f : m), fm.faces[0]);
  }

  if (mesh) {
    // Recuadro (igual que drawBox en app.js).
    ctx.strokeStyle = "#4ade80";
    ctx.lineWidth = 3;
    ctx.strokeRect(mesh.box.x, mesh.box.y, mesh.box.width, mesh.box.height);

    // 468 puntos de malla (verde) + 10 de iris (cian) que face-api no tiene.
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

    mFace.textContent = "Sí";
    mPts.textContent = String(mesh.landmarks.length);
    mPose.textContent = `${mesh.pose.roll.toFixed(0)}° / ${mesh.pose.yaw.toFixed(0)}° / ${mesh.pose.pitch.toFixed(0)}°`;
  } else {
    mFace.textContent = "No";
    mPts.textContent = "—";
    mPose.textContent = "—";
  }

  loopId = requestAnimationFrame(detectLoop);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

window.addEventListener("load", loadModel);
