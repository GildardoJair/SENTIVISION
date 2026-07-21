// alternar-emociones.js — Comparación de clasificadores de emoción (I2).
// Un solo <video> + un solo <canvas id="overlay">, MISMO patrón que
// alternar-i1.js / alternar-i5.js (ya confirmados funcionales). Un botón
// cicla entre tres motores:
//   - "faceapi"  → faceExpressionNet (face-api.js), línea base de app.js
//   - "fusion"   → emotion-fusion.js sobre blendshapes de MediaPipe + voz
//   - "human"    → Human.face.emotion (@vladmandic/human)
//
// NO toca app.js, NO toca facemesh.js. Reusa ambos (y voice.js,
// emotion-fusion.js) tal cual, importados como scripts globales.
//
// Objetivo: comparación VISUAL/cualitativa uno-a-la-vez, más métricas
// Nivel 2 (detection_rate, expression_entropy) para no depender solo de
// "se ve razonable". La métrica de acuerdo entre las 3 fuentes es Paso 3.

const EMOTIONS = {
  neutral:   { es: "Neutral",  emoji: "😐" },
  happy:     { es: "Feliz",    emoji: "😄" },
  sad:       { es: "Triste",   emoji: "😢" },
  angry:     { es: "Enojo",    emoji: "😠" },
  fearful:   { es: "Miedo",    emoji: "😨" },
  disgusted: { es: "Disgusto", emoji: "🤢" },
  surprised: { es: "Sorpresa", emoji: "😲" },
};

// Mapeo verificado a mano (console.log, no adivinado) el 8 de julio de 2026
// contra Human 3.3.6: fingiendo Feliz/Sorpresa/Disgusto/Miedo/Triste/Enojo
// se confirmaron los 7 nombres exactos que usa Human para face.emotion.
const HUMAN_TO_KEY = {
  neutral:  "neutral",
  happy:    "happy",
  sad:      "sad",
  angry:    "angry",
  fear:     "fearful",
  disgust:  "disgusted",
  surprise: "surprised",
};

// Human NO siempre devuelve las 7 categorías (parece filtrar por score bajo;
// en pruebas el array trajo entre 1 y 5 elementos, y la suma de scores casi
// nunca llegó a 1.0). Rellenamos las ausentes con 0 y renormalizamos a que
// sume 1, para que expression_entropy sea comparable contra los otros dos
// motores (que sí devuelven distribución completa). Si se prefiere ver los
// scores crudos de Human sin este ajuste, es el punto a cambiar.
function humanEmotionToProbs(emotionArr) {
  const probs = {};
  for (const key of Object.keys(EMOTIONS)) probs[key] = 0;
  if (!emotionArr) return probs;
  for (const e of emotionArr) {
    const key = HUMAN_TO_KEY[e.emotion];
    if (key) probs[key] = e.score;
  }
  const total = Object.values(probs).reduce((a, b) => a + b, 0);
  if (total > 0) for (const key of Object.keys(probs)) probs[key] /= total;
  return probs;
}

const video     = document.getElementById("video");
const overlay   = document.getElementById("overlay");
const statusEl  = document.getElementById("status");
const startBtn  = document.getElementById("startBtn");
const stopBtn   = document.getElementById("stopBtn");
const toggleBtn = document.getElementById("toggleBtn");

const roleCliente = document.getElementById("roleCliente");
const roleAsesor  = document.getElementById("roleAsesor");
const micStatus   = document.getElementById("micStatus");
const vPitch      = document.getElementById("vPitch");
const vEnergy     = document.getElementById("vEnergy");
const vArousal    = document.getElementById("vArousal");

const mModel     = document.getElementById("mModel");
const mFace      = document.getElementById("mFace");
const mDetRate   = document.getElementById("mDetRate");
const mEntropy   = document.getElementById("mEntropy");
const mAudioUsed = document.getElementById("mAudioUsed");

const domEmoji = document.getElementById("domEmoji");
const domLabel = document.getElementById("domLabel");
const domConf  = document.getElementById("domConf");
const barsEl   = document.getElementById("bars");

const logToggle   = document.getElementById("logToggle");
const intervalSel = document.getElementById("intervalSel");
const logCount    = document.getElementById("logCount");
const csvBtn      = document.getElementById("csvBtn");
const clearBtn    = document.getElementById("clearBtn");
const logBody     = document.getElementById("logBody");

let stream = null;
let loopId = null;
let activeModel = "faceapi"; // "faceapi" | "fusion" | "human"
let activeRole = "cliente";  // "cliente" | "asesor" — mismo criterio que app.js

const voice = new VoiceAnalyzer();
let human = null;

// Registro de comparación (mismo patrón que emotionLog/maybeLog en app.js).
const comparisonLog = [];
let lastLogAt = 0;
const MAX_TABLE_ROWS = 15;

const detectorOpts = new faceapi.TinyFaceDetectorOptions({
  inputSize: 320,
  scoreThreshold: 0.5,
});

// Ventana deslizante de 5s, UNA POR MOTOR (mismo patrón que alternar-i1.js /
// alternar-i5.js), para que detection_rate no se contamine al alternar.
const WINDOW_MS = 5000;
const frames = { faceapi: [], fusion: [], human: [] };
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

const barFills = {};
function buildBars() {
  barsEl.innerHTML = "";
  for (const key of Object.keys(EMOTIONS)) {
    const { es } = EMOTIONS[key];
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="bar-row"><span>${es}</span><span class="val" data-val="${key}">0%</span></div>
      <div class="bar-track"><div class="bar-fill" data-fill="${key}"></div></div>`;
    barsEl.appendChild(li);
    barFills[key] = {
      fill: li.querySelector(`[data-fill="${key}"]`),
      val:  li.querySelector(`[data-val="${key}"]`),
    };
  }
}

async function loadModels() {
  setStatus("Cargando modelos de IA…");
  const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
  ]);
  await window.FaceMeshEngine.load(setStatus);

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
    // Fix ya conocido de I5: sin esto Human reutiliza el resultado del frame
    // anterior si "se ve parecido" y las probabilidades se quedan congeladas.
    cacheSensitivity: 0,
    face: {
      enabled: true,
      description: { enabled: false }, // no necesitamos edad/género aquí
      emotion: { enabled: true },
      iris: { enabled: false },
    },
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

const MODEL_LABELS = {
  faceapi: { short: "faceExpressionNet (face-api.js)", toggle: "🟠 faceExpressionNet (face-api.js)" },
  fusion:  { short: "emotion-fusion.js (Blendshapes + voz)", toggle: "🟢 emotion-fusion.js (Blendshapes + voz)" },
  human:   { short: "Human.face.emotion (@vladmandic/human)", toggle: "🔵 Human.face.emotion (@vladmandic/human)" },
};
const MODEL_ORDER = ["faceapi", "fusion", "human"];

function updateToggleLabel() {
  toggleBtn.textContent = `Motor: ${MODEL_LABELS[activeModel].toggle} — clic para cambiar`;
  mModel.textContent = MODEL_LABELS[activeModel].short;
}

function toggleModel() {
  const idx = MODEL_ORDER.indexOf(activeModel);
  activeModel = MODEL_ORDER[(idx + 1) % MODEL_ORDER.length];
  updateToggleLabel();
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
}

function setRole(role) {
  activeRole = role;
  roleCliente.classList.toggle("active", role === "cliente");
  roleAsesor.classList.toggle("active", role === "asesor");
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: true,
    });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus("");

    if (stream.getAudioTracks().length > 0) {
      try {
        await voice.start(stream);
        micStatus.textContent = "🎤 micrófono activo";
        micStatus.classList.add("on");
      } catch (e) {
        console.error(e);
        micStatus.textContent = "🎤 voz no disponible";
      }
    } else {
      micStatus.textContent = "🎤 sin micrófono (solo rostro)";
    }

    detectLoop();
  } catch (err) {
    console.error(err);
    setStatus("No se pudo acceder a la cámara/micrófono: " + err.message);
  }
}

function stopCamera() {
  if (loopId) cancelAnimationFrame(loopId);
  loopId = null;
  voice.stop();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Cámara detenida.");
  micStatus.textContent = "🎤 micrófono inactivo";
  micStatus.classList.remove("on");
  resetMetrics();
}

function resetMetrics() {
  mFace.textContent = "—";
  mDetRate.textContent = "—";
  mEntropy.textContent = "—";
  mAudioUsed.textContent = "—";
  domEmoji.textContent = "🙂";
  domLabel.textContent = "—";
  domConf.textContent = "Esperando…";
  vPitch.textContent = vEnergy.textContent = vArousal.textContent = "—";
  for (const key of Object.keys(EMOTIONS)) {
    barFills[key].fill.style.width = "0%";
    barFills[key].val.textContent = "0%";
  }
}

function updateVoiceUI(vf) {
  if (!vf || !vf.voiced) {
    vPitch.textContent = "—";
    vEnergy.textContent = vf ? "silencio" : "—";
    vArousal.textContent = vf ? "0%" : "—";
    return;
  }
  vPitch.textContent = vf.pitch ? `${Math.round(vf.pitch)} Hz` : "—";
  vEnergy.textContent = `${Math.round(Math.min(1, vf.rms / 0.15) * 100)}%`;
  vArousal.textContent = `${Math.round(vf.arousal * 100)}%`;
}

// Entropía de Shannon sobre la distribución de 7 emociones. Máximo ≈ ln(7) ≈
// 1.9459 con distribución uniforme (motor "confundido"); 0 = certeza total.
function entropyOf(probs) {
  let h = 0;
  for (const key of Object.keys(EMOTIONS)) {
    const p = probs[key] || 0;
    if (p > 0) h += -p * Math.log(p);
  }
  return h;
}

// Actualiza barras + dominante a partir de una distribución {emocion: prob}.
// Funciona igual para expressions de face-api (no siempre suma exactamente 1,
// pero se muestra tal cual, como en app.js) y para la salida de EmotionFusion
// (normalizada a 1 por diseño).
function renderDistribution(probs) {
  let topKey = null, topVal = -1;
  for (const key of Object.keys(EMOTIONS)) {
    const p = probs[key] || 0;
    const pct = Math.round(p * 100);
    barFills[key].fill.style.width = pct + "%";
    barFills[key].val.textContent = pct + "%";
    if (p > topVal) { topVal = p; topKey = key; }
  }
  const info = EMOTIONS[topKey];
  domEmoji.textContent = info.emoji;
  domLabel.textContent = info.es;
  domConf.textContent = `Confianza: ${Math.round(topVal * 100)}%`;
  const entropy = entropyOf(probs);
  mEntropy.textContent = entropy.toFixed(3);
  return { dominant: topKey, confidence: topVal, entropy };
}

// --- Registro / CSV ---------------------------------------------------------

function maybeLog(motor, hasFace, dist, detRate, vf, audioUsedText, probs) {
  if (!logToggle.checked) return;
  const now = performance.now();
  const intervalMs = Number(intervalSel.value);
  if (now - lastLogAt < intervalMs) return;
  lastLogAt = now;

  const expr = {};
  for (const key of Object.keys(EMOTIONS)) expr[key] = (probs && probs[key]) || 0;

  comparisonLog.push({
    time: new Date(),
    motor,
    hasFace,
    dominant: hasFace && dist ? EMOTIONS[dist.dominant].es : "",
    dominantRaw: hasFace && dist ? dist.dominant : "",
    confidence: hasFace && dist ? dist.confidence : null,
    entropy: hasFace && dist ? dist.entropy : null,
    expressions: expr,
    detRate,
    role: activeRole === "cliente" ? "Cliente" : "Asesor",
    audioUsedText,
    vPitch: vf && vf.voiced && vf.pitch ? vf.pitch : null,
    vEnergy: vf && vf.voiced ? vf.rms : null,
    vArousal: vf && vf.voiced ? vf.arousal : null,
  });
  renderLog();
}

function renderLog() {
  const n = comparisonLog.length;
  logCount.textContent = n === 1 ? "1 registro" : `${n} registros`;
  csvBtn.disabled = n === 0;
  clearBtn.disabled = n === 0;

  if (n === 0) {
    logBody.innerHTML =
      '<tr class="log-empty"><td colspan="7">Aún no hay registros. Inicia la cámara para comenzar a muestrear.</td></tr>';
    return;
  }
  const recent = comparisonLog.slice(-MAX_TABLE_ROWS).reverse();
  logBody.innerHTML = recent
    .map((r) => {
      const hora = r.time.toLocaleTimeString("es-MX");
      const motorEs = MODEL_LABELS[r.motor] ? MODEL_LABELS[r.motor].short : r.motor;
      const conf = r.confidence != null ? `${Math.round(r.confidence * 100)}%` : "—";
      const ent = r.entropy != null ? r.entropy.toFixed(2) : "—";
      return `<tr><td>${hora}</td><td>${motorEs}</td><td>${r.hasFace ? "Sí" : "No"}</td>` +
        `<td>${r.dominant || "—"}</td><td>${conf}</td><td>${ent}</td><td>${r.role}</td></tr>`;
    })
    .join("");
}

function clearLog() {
  comparisonLog.length = 0;
  lastLogAt = 0;
  renderLog();
}

function downloadCSV() {
  if (comparisonLog.length === 0) return;
  const keys = Object.keys(EMOTIONS);
  const header = [
    "timestamp_iso", "hora_local", "motor", "rostro_detectado",
    "emocion_dominante", "confianza", "entropia",
    ...keys,
    "detection_rate_5s", "rol_activo", "audio_usado",
    "voz_pitch_hz", "voz_energia", "voz_arousal",
  ];
  const rows = comparisonLog.map((r) => {
    const cells = [
      r.time.toISOString(),
      r.time.toLocaleString("es-MX"),
      MODEL_LABELS[r.motor] ? MODEL_LABELS[r.motor].short : r.motor,
      r.hasFace ? "1" : "0",
      r.dominant,
      r.confidence != null ? r.confidence.toFixed(4) : "",
      r.entropy != null ? r.entropy.toFixed(4) : "",
      ...keys.map((k) => (r.expressions[k] || 0).toFixed(4)),
      r.detRate != null ? r.detRate.toFixed(4) : "",
      r.role,
      r.audioUsedText,
      r.vPitch != null ? Math.round(r.vPitch) : "",
      r.vEnergy != null ? r.vEnergy.toFixed(4) : "",
      r.vArousal != null ? r.vArousal.toFixed(4) : "",
    ];
    return cells.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",");
  });
  const csv = [header.join(","), ...rows].join("\r\n");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  saveCSV(csv, `comparacion_emociones_${stamp}.csv`);
}

function saveCSV(csv, filename) {
  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function drawBox(ctx, box, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(box.x, box.y, box.width, box.height);
}

// --- face-api: detección EN SEGUNDO PLANO (fire-and-forget), mismo patrón
// que alternar-i1.js para evitar el bug de compositor en Linux/Intel HD
// (dibujar como continuación de una promesa dentro de requestAnimationFrame).
let lastFaceApiResult = null;
let faceApiBusy = false;
function kickFaceApiDetection() {
  if (faceApiBusy) return;
  faceApiBusy = true;
  faceapi.detectSingleFace(video, detectorOpts).withFaceLandmarks().withFaceExpressions()
    .then((r) => { lastFaceApiResult = r || null; })
    .catch((e) => console.error("Error en detección face-api:", e))
    .finally(() => { faceApiBusy = false; });
}

async function detectLoop() {
  if (!stream) return;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const vf = voice.running ? voice.features : null;
  updateVoiceUI(vf);

  if (activeModel === "faceapi") {
    kickFaceApiDetection();
    const r = lastFaceApiResult;
    pushFrame("faceapi", !!r);
    mAudioUsed.textContent = "no aplica (línea base solo visión)";
    let dist = null, probs = null;
    if (r) {
      drawBox(ctx, r.detection.box, "#ffb020");
      mFace.textContent = "Sí";
      probs = r.expressions;
      dist = renderDistribution(probs);
    } else {
      mFace.textContent = "No";
    }
    const detRate = detectionRate("faceapi");
    mDetRate.textContent = detRate != null ? `${Math.round(detRate * 100)}%` : "—";
    maybeLog("faceapi", !!r, dist, detRate, vf, "no aplica (línea base solo visión)", probs);

  } else if (activeModel === "fusion") {
    const fm = window.FaceMeshEngine.detect(video);
    const mesh = fm.faces[0] || null;
    pushFrame("fusion", !!mesh);
    const audioWeight = activeRole === "cliente" ? 0.3 : 0;
    const audioUsedText = (vf && vf.voiced && audioWeight > 0)
      ? `sí (peso ${audioWeight}, rol Cliente)`
      : `no (${activeRole === "cliente" ? "cliente en silencio" : "rol Asesor"})`;
    mAudioUsed.textContent = audioUsedText;
    let dist = null, probs = null;
    if (mesh) {
      drawBox(ctx, mesh.box, "#4ade80");
      mFace.textContent = "Sí";
      probs = window.EmotionFusion.classify(mesh.blendshapes, vf, activeRole);
      dist = renderDistribution(probs);
    } else {
      mFace.textContent = "No";
    }
    const detRate = detectionRate("fusion");
    mDetRate.textContent = detRate != null ? `${Math.round(detRate * 100)}%` : "—";
    maybeLog("fusion", !!mesh, dist, detRate, vf, audioUsedText, probs);

  } else {
    // Human: mismo patrón de await directo que alternar-i5.js (ya confirmado
    // funcional en esta máquina), no el fire-and-forget usado para face-api.
    const result = await human.detect(video);
    const face = (result.face && result.face[0]) || null;
    pushFrame("human", !!face);
    mAudioUsed.textContent = "no aplica (Human.face.emotion es solo visión)";
    let dist = null, probs = null;
    if (face) {
      if (face.box) {
        const [x, y, w, h] = face.box;
        drawBox(ctx, { x, y, width: w, height: h }, "#22d3ee");
      }
      mFace.textContent = "Sí";
      probs = humanEmotionToProbs(face.emotion);
      dist = renderDistribution(probs);
    } else {
      mFace.textContent = "No";
    }
    const detRate = detectionRate("human");
    mDetRate.textContent = detRate != null ? `${Math.round(detRate * 100)}%` : "—";
    maybeLog("human", !!face, dist, detRate, vf, "no aplica (Human.face.emotion es solo visión)", probs);
  }

  loopId = requestAnimationFrame(detectLoop);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
toggleBtn.addEventListener("click", toggleModel);
roleCliente.addEventListener("click", () => setRole("cliente"));
roleAsesor.addEventListener("click", () => setRole("asesor"));
csvBtn.addEventListener("click", downloadCSV);
clearBtn.addEventListener("click", clearLog);

window.addEventListener("load", () => {
  buildBars();
  loadModels().catch((e) => {
    console.error("Error cargando modelos:", e);
    setStatus("Error al cargar (ver consola, F12): " + e.message);
  });
});