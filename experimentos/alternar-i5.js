// alternar-i5.js — I5: ageGenderNet (face-api.js, línea base) vs
// @vladmandic/human · módulo faceres (propuesto). Un solo <video> + un solo
// <canvas id="overlay">, MISMA estructura que alternar-i1.js (ya confirmada
// funcional). Un botón alterna cuál modelo estima edad/género.

const video    = document.getElementById("video");
const overlay  = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn  = document.getElementById("stopBtn");
const toggleBtn = document.getElementById("toggleBtn");
const mModel   = document.getElementById("mModel");
const mFace    = document.getElementById("mFace");
const mDetRate = document.getElementById("mDetRate");
const mAge     = document.getElementById("mAge");
const mGender  = document.getElementById("mGender");
const mGenderConf = document.getElementById("mGenderConf");

let stream = null;
let loopId = null;
let activeModel = "human"; // "faceapi" | "human"
let human = null;

const detectorOpts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

// Ventana deslizante de 5s, UNA POR MODELO (igual patrón que alternar-i1.js).
const WINDOW_MS = 5000;
const frames = { faceapi: [], human: [] };
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

async function loadModels() {
  setStatus("Cargando modelos de IA…");

  // Línea base: face-api (tinyFaceDetector + ageGenderNet).
  const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
  ]);

  // Propuesto I5: @vladmandic/human, módulo faceres (edad/género).
  // Se desactivan los módulos que no usamos (body, hand, object, gesture,
  // segmentation, emotion) para que cargue más rápido y no compita por GPU.
  // Propuesto I5: @vladmandic/human, módulo faceres (edad/género).
  // La build IIFE puede exponer window.Human como la clase directa O como
  // un namespace { default: ClaseHuman } según la versión/bundler — se
  // resuelve de forma defensiva para no depender de adivinar cuál es.
  const HumanCtor = (typeof Human === "function")
    ? Human
    : (Human && typeof Human.default === "function" ? Human.default : null);
  if (!HumanCtor) {
    throw new Error(
      "No se encontró el constructor de Human. window.Human = " + JSON.stringify(Object.keys(Human || {}))
    );
  }

  human = new HumanCtor({
    modelBasePath: "https://cdn.jsdelivr.net/npm/@vladmandic/human/models/",
    backend: "webgl",
    // cacheSensitivity: 0 desactiva el caché por similitud de frame. Por
    // default, Human reutiliza el último resultado de edad/género si el
    // frame se ve "suficientemente parecido" al anterior (para ahorrar
    // cómputo) — pero sentado frente a la cámara casi todo frame se ve
    // parecido, así que en la práctica solo corre el modelo una vez y
    // repite el mismo número. Para esta comparación queremos que SIEMPRE
    // re-evalúe, aunque cueste más CPU/GPU.
    cacheSensitivity: 0,
    face: { enabled: true, description: { enabled: true }, emotion: { enabled: false }, iris: { enabled: false } },
    body: { enabled: false },
    hand: { enabled: false },
    object: { enabled: false },
    gesture: { enabled: false },
    segmentation: { enabled: false },
  });
  await human.load();

  setStatus("Modelos listos. Presiona «Iniciar cámara».");
  startBtn.disabled = false;
  toggleBtn.disabled = false;
  updateToggleLabel();
}

function updateToggleLabel() {
  toggleBtn.textContent = activeModel === "faceapi"
    ? "Modelo: 🟠 ageGenderNet (face-api.js) — clic para cambiar"
    : "Modelo: 🟢 Human · faceres — clic para cambiar";
  mModel.textContent = activeModel === "faceapi" ? "ageGenderNet (face-api.js)" : "Human (faceres)";
}

function toggleModel() {
  activeModel = activeModel === "faceapi" ? "human" : "faceapi";
  updateToggleLabel();
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
  mAge.textContent = "—";
  mGender.textContent = "—";
  mGenderConf.textContent = "—";
  mDetRate.textContent = "—";
}

function drawBox(ctx, x, y, w, h, color, label) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = color;
  ctx.font = "16px sans-serif";
  ctx.fillText(label, x, y > 20 ? y - 8 : y + h + 18);
}

async function detectLoop() {
  if (!stream) return;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (activeModel === "faceapi") {
    const r = await faceapi.detectSingleFace(video, detectorOpts).withAgeAndGender();
    pushFrame("faceapi", !!r);
    if (r) {
      const b = r.detection.box;
      const genero = r.gender === "male" ? "Hombre" : "Mujer";
      drawBox(ctx, b.x, b.y, b.width, b.height, "#ffb020", `${Math.round(r.age)} años · ${genero}`);
      mAge.textContent = `${Math.round(r.age)} años`;
      mGender.textContent = genero;
      mGenderConf.textContent = `${Math.round(r.genderProbability * 100)}%`;
      mFace.textContent = "Sí";
    } else {
      mAge.textContent = mGender.textContent = mGenderConf.textContent = "—";
      mFace.textContent = "No";
    }
    mDetRate.textContent = detectionRate("faceapi") != null ? `${Math.round(detectionRate("faceapi") * 100)}%` : "—";

  } else {
    const result = await human.detect(video);
    const face = (result.face && result.face[0]) || null;
    pushFrame("human", !!face);
    if (face) {
      // Human entrega box como [x, y, width, height] en píxeles.
      const [x, y, w, h] = face.box;
      const genero = face.gender === "male" ? "Hombre" : face.gender === "female" ? "Mujer" : (face.gender || "—");
      drawBox(ctx, x, y, w, h, "#4ade80", `${Math.round(face.age)} años · ${genero}`);
      mAge.textContent = face.age != null ? `${Math.round(face.age)} años` : "—";
      mGender.textContent = genero;
      mGenderConf.textContent = face.genderScore != null ? `${Math.round(face.genderScore * 100)}%` : "—";
      mFace.textContent = "Sí";
    } else {
      mAge.textContent = mGender.textContent = mGenderConf.textContent = "—";
      mFace.textContent = "No";
    }
    mDetRate.textContent = detectionRate("human") != null ? `${Math.round(detectionRate("human") * 100)}%` : "—";
  }

  loopId = requestAnimationFrame(detectLoop);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
toggleBtn.addEventListener("click", toggleModel);

window.addEventListener("load", () => {
  loadModels().catch((e) => {
    console.error("Error cargando modelos:", e);
    setStatus("Error al cargar (ver consola, F12): " + e.message);
  });
});