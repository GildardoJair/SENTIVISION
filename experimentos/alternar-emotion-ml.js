// alternar-emotion-ml.js — Compara en vivo el modelo propio (emotion-ml.js,
// MLP entrenado en el notebook) contra emotion-fusion.js (heurístico manual),
// ambos alimentados con los mismos 52 blendshapes de un único frame de
// facemesh.js. Aislado de app.js — mismo patrón que alternar-i1.js / alternar-i5.js.
//
// Notas de diseño (lecciones ya documentadas en el proyecto):
//   - facemesh.js.detect() es SÍNCRONO (detectForVideo internamente), así que
//     no hace falta decouplear detección/dibujo con promesas — se llama
//     directo dentro de requestAnimationFrame.
//   - overlay.width/height se fija una sola vez al iniciar la cámara, nunca
//     dentro del loop (evita reflow/reset de contexto en cada frame).
//   - Nunca display:none sobre <video>; aquí ni se oculta.

const EMOTIONS = {
  neutral:   { es: "Neutral",  emoji: "😐" },
  happy:     { es: "Feliz",    emoji: "😄" },
  sad:       { es: "Triste",   emoji: "😢" },
  angry:     { es: "Enojo",    emoji: "😠" },
  fearful:   { es: "Miedo",    emoji: "😨" },
  disgusted: { es: "Disgusto", emoji: "🤢" },
  surprised: { es: "Sorpresa", emoji: "😲" },
};

const video   = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn  = document.getElementById("stopBtn");
const landmarksToggle = document.getElementById("landmarksToggle");
const detRateEl = document.getElementById("detRate");
const agreeEl = document.getElementById("agree");

const mlEmoji = document.getElementById("mlEmoji");
const mlLabel = document.getElementById("mlLabel");
const mlConf  = document.getElementById("mlConf");
const mlBars  = document.getElementById("mlBars");

const fusionEmoji = document.getElementById("fusionEmoji");
const fusionLabel = document.getElementById("fusionLabel");
const fusionConf  = document.getElementById("fusionConf");
const fusionBars  = document.getElementById("fusionBars");

const faceapiEmoji = document.getElementById("faceapiEmoji");
const faceapiLabel = document.getElementById("faceapiLabel");
const faceapiConf  = document.getElementById("faceapiConf");
const faceapiBars  = document.getElementById("faceapiBars");
const faceapiDetRateEl = document.getElementById("faceapiDetRate");

let stream = null;
let loopId = null;
let faceapiLoopActive = false;

// face-api.js es ASÍNCRONO (a diferencia de facemesh.js) — se corre en un loop
// aparte, desacoplado del rAF de dibujo, y solo se lee su último resultado en
// caché dentro de detectLoop(). Así nunca bloquea el frame de MediaPipe.
const FACEAPI_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
const faceapiDetectorOpts = {
  inputSize: 320,
  scoreThreshold: 0.5,
};
let lastFaceApiExpressions = null; // null = sin rostro detectado en la última corrida
let faceapiDetSamples = []; // { t, hasFace } — ventana propia de 5s

// Detection rate en ventana de 5s — mismo criterio que Nivel 2 (metrics.js),
// implementado aquí de forma local y simple para no depender de ese módulo.
const DET_WINDOW_MS = 5000;
let detSamples = []; // { t, hasFace }

function setStatus(msg) {
  statusEl.textContent = msg;
  statusEl.style.display = msg ? "block" : "none";
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Timeout cargando ${label} (${ms}ms)`)), ms)
    ),
  ]);
}

async function loadModels() {
  setStatus("Cargando MediaPipe FaceMesh + modelo propio + face-api.js…");
  const tasks = [
    withTimeout(window.FaceMeshEngine.load(), 20000, "FaceMeshEngine"),
    withTimeout(window.EmotionML.load("../emotion_ml_weights.json"), 20000, "EmotionML"),
  ];
  if (typeof faceapi !== "undefined") {
    tasks.push(
      withTimeout(
        Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(FACEAPI_MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(FACEAPI_MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(FACEAPI_MODEL_URL),
        ]),
        20000,
        "face-api.js"
      )
    );
  } else {
    faceapiConf.textContent = "face-api.js no cargó (¿sin internet?)";
  }
  await Promise.all(tasks);
  setStatus("Modelos listos. Presiona «Iniciar cámara».");
  startBtn.disabled = false;
}

// Loop de face-api.js, independiente del rAF de dibujo — se relanza a sí mismo
// apenas termina una detección (fire-and-forget), nunca espera al frame visual.
async function faceapiLoop() {
  if (!faceapiLoopActive || typeof faceapi === "undefined") return;
  try {
    const results = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions(faceapiDetectorOpts))
      .withFaceLandmarks()
      .withFaceExpressions();

    const now = performance.now();
    const hasFace = results.length > 0;
    faceapiDetSamples.push({ t: now, hasFace });
    faceapiDetSamples = faceapiDetSamples.filter((s) => now - s.t <= DET_WINDOW_MS);
    const rate = faceapiDetSamples.length
      ? faceapiDetSamples.filter((s) => s.hasFace).length / faceapiDetSamples.length
      : 0;
    faceapiDetRateEl.textContent = `${Math.round(rate * 100)}%`;

    if (hasFace) {
      const principal = results.reduce(
        (m, r) => (r.detection.box.area > m.detection.box.area ? r : m),
        results[0]
      );
      lastFaceApiExpressions = { ...principal.expressions };
    } else {
      lastFaceApiExpressions = null;
    }
  } catch (e) {
    console.error("Error face-api.js:", e);
    lastFaceApiExpressions = null;
  }
  if (faceapiLoopActive) faceapiLoop(); // se relanza sola, sin esperar al rAF
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
    detSamples = [];
    faceapiDetSamples = [];
    faceapiLoopActive = true;
    faceapiLoop();
    detectLoop();
  } catch (err) {
    console.error(err);
    setStatus("No se pudo acceder a la cámara: " + err.message);
  }
}

function stopCamera() {
  if (loopId) cancelAnimationFrame(loopId);
  loopId = null;
  faceapiLoopActive = false;
  lastFaceApiExpressions = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Cámara detenida.");
  resetPanels();
}

function resetPanels() {
  mlEmoji.textContent = "🙂";
  fusionEmoji.textContent = "🙂";
  faceapiEmoji.textContent = "🙂";
  mlLabel.textContent = "—";
  fusionLabel.textContent = "—";
  faceapiLabel.textContent = "—";
  mlConf.textContent = "Esperando…";
  fusionConf.textContent = "Esperando…";
  faceapiConf.textContent = "Esperando…";
  mlBars.innerHTML = "";
  fusionBars.innerHTML = "";
  faceapiBars.innerHTML = "";
  faceapiDetRateEl.textContent = "—";
  agreeEl.textContent = "—";
  detRateEl.textContent = "—";
}

function dominantOf(probs) {
  let best = null, bestP = -1;
  for (const [k, v] of Object.entries(probs)) {
    if (v > bestP) { bestP = v; best = k; }
  }
  return best;
}

function updatePanel(probs, predicted, emojiEl, labelEl, confEl, barsEl) {
  if (!probs || !predicted) return;
  const info = EMOTIONS[predicted] || { es: predicted, emoji: "🙂" };
  emojiEl.textContent = info.emoji;
  labelEl.textContent = info.es;
  confEl.textContent = `Confianza: ${Math.round(probs[predicted] * 100)}%`;

  barsEl.innerHTML = Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .map(([key, val]) => {
      const info2 = EMOTIONS[key] || { es: key };
      const pct = Math.round(val * 100);
      return `
        <li>
          <div class="bar-row"><span>${info2.es}</span><span class="val">${pct}%</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        </li>`;
    })
    .join("");
}

function drawLandmarks(ctx, landmarks) {
  ctx.fillStyle = "#4ade80";
  for (const pt of landmarks) {
    ctx.beginPath();
    ctx.arc(pt.x * overlay.width, pt.y * overlay.height, 1.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function detectLoop() {
  if (!stream) return;

  const result = window.FaceMeshEngine.detect(video);
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const now = performance.now();
  const hasFace = result.count > 0;
  detSamples.push({ t: now, hasFace });
  detSamples = detSamples.filter((s) => now - s.t <= DET_WINDOW_MS);
  const detRate = detSamples.length
    ? detSamples.filter((s) => s.hasFace).length / detSamples.length
    : 0;
  detRateEl.textContent = `${Math.round(detRate * 100)}%`;

  if (hasFace) {
    // Rostro principal = mayor área, mismo criterio que app.js.
    const face = result.faces.reduce(
      (m, f) => (f.box.area > m.box.area ? f : m),
      result.faces[0]
    );

    if (landmarksToggle.checked) drawLandmarks(ctx, face.landmarks);

    const mlResult = window.EmotionML.classify(face.blendshapes);
    const fusionProbs = window.EmotionFusion.classify(face.blendshapes, null, "cliente");
    const fusionPred = fusionProbs ? dominantOf(fusionProbs) : null;

    if (mlResult) updatePanel(mlResult.probs, mlResult.predicted, mlEmoji, mlLabel, mlConf, mlBars);
    if (fusionProbs) updatePanel(fusionProbs, fusionPred, fusionEmoji, fusionLabel, fusionConf, fusionBars);

    if (mlResult && fusionPred) {
      agreeEl.textContent =
        mlResult.predicted === fusionPred
          ? `✅ Coinciden — ambos dicen "${EMOTIONS[mlResult.predicted]?.es || mlResult.predicted}"`
          : `❌ Difieren — MLP: "${EMOTIONS[mlResult.predicted]?.es}" · fusion: "${EMOTIONS[fusionPred]?.es}"`;
    }
  } else {
    mlConf.textContent = "Sin rostro…";
    fusionConf.textContent = "Sin rostro…";
    agreeEl.textContent = "—";
  }

  // face-api.js se lee de su propia caché (loop desacoplado) — su detección es
  // independiente de la de MediaPipe, por eso se actualiza aparte, siempre.
  if (lastFaceApiExpressions) {
    const faceapiPred = dominantOf(lastFaceApiExpressions);
    updatePanel(lastFaceApiExpressions, faceapiPred, faceapiEmoji, faceapiLabel, faceapiConf, faceapiBars);
  } else {
    faceapiConf.textContent = "Sin rostro (face-api)…";
    faceapiBars.innerHTML = "";
  }

  loopId = requestAnimationFrame(detectLoop);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

window.addEventListener("load", async () => {
  resetPanels();
  if (typeof window.FaceMeshEngine === "undefined" || typeof window.EmotionML === "undefined") {
    setStatus("No se pudo cargar facemesh.js o emotion-ml.js (¿rutas correctas? ¿sin conexión?).");
    return;
  }
  try {
    await loadModels();
  } catch (err) {
    console.error(err);
    setStatus("Error al cargar modelos: " + err.message);
  }
});
