// explorar-human-emotion.js — Paso 2 (previo a la comparación de 3 vías).
// Único propósito: ver la forma REAL de result.face[0].emotion en Human.js,
// sin adivinar. No compara nada todavía, no toca app.js ni facemesh.js.
//
// cacheSensitivity: 0 aplicado desde el inicio — ya aprendimos en I5 que sin
// esto Human reutiliza el resultado del frame anterior si "se ve parecido",
// y para inspección queremos que SIEMPRE re-evalúe.

const video    = document.getElementById("video");
const overlay  = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn  = document.getElementById("stopBtn");
const mFace    = document.getElementById("mFace");
const mKeys    = document.getElementById("mKeys");
const rawEmotion = document.getElementById("rawEmotion");
const rawRef      = document.getElementById("rawRef");

let stream = null;
let loopId = null;
let human = null;
let lastConsoleLogAt = 0;

function setStatus(msg) {
  statusEl.textContent = msg;
  statusEl.style.display = msg ? "block" : "none";
}

async function loadHuman() {
  setStatus("Cargando Human…");
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
    cacheSensitivity: 0, // fix ya conocido de I5: sin esto se congela el resultado
    face: {
      enabled: true,
      description: { enabled: true }, // trae edad/género, útil de referencia
      emotion: { enabled: true },     // el módulo que queremos inspeccionar
      iris: { enabled: false },
    },
    body: { enabled: false },
    hand: { enabled: false },
    object: { enabled: false },
    gesture: { enabled: false },
    segmentation: { enabled: false },
  });
  await human.load();

  setStatus("Human listo. Presiona «Iniciar cámara».");
  startBtn.disabled = false;
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
  mKeys.textContent = "—";
  rawEmotion.textContent = "—";
  rawRef.textContent = "—";
}

function drawBox(ctx, box, color) {
  const [x, y, w, h] = box;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);
}

async function detectLoop() {
  if (!stream) return;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const result = await human.detect(video);
  const face = (result.face && result.face[0]) || null;

  if (face) {
    if (face.box) drawBox(ctx, face.box, "#4ade80");
    mFace.textContent = "Sí";
    mKeys.textContent = Object.keys(face).join(", ");

    // Objeto crudo, sin transformar, tal cual lo entrega Human.
    rawEmotion.textContent = JSON.stringify(face.emotion, null, 2);
    rawRef.textContent = JSON.stringify(
      { age: face.age, gender: face.gender, genderScore: face.genderScore },
      null, 2
    );

    // Log en consola, throttled a ~1/seg para no saturarla.
    const now = performance.now();
    if (now - lastConsoleLogAt > 1000) {
      lastConsoleLogAt = now;
      console.log("face.emotion (crudo):", face.emotion);
      console.log("typeof face.emotion:", typeof face.emotion, Array.isArray(face.emotion) ? "(es array)" : "(no es array)");
    }
  } else {
    mFace.textContent = "No";
    mKeys.textContent = "—";
    rawEmotion.textContent = "—";
    rawRef.textContent = "—";
  }

  loopId = requestAnimationFrame(detectLoop);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

window.addEventListener("load", () => {
  loadHuman().catch((e) => {
    console.error("Error cargando Human:", e);
    setStatus("Error al cargar (ver consola, F12): " + e.message);
  });
});
