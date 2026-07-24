// Detector de emociones — versión integrada (I1 FaceMesh + I2 fusión + I2 MLP
// propio + I4 métricas + edad/género vía recorte de MediaPipe).
//
// Motores de clasificación de emoción disponibles vía selector (classifierSel):
//   - faceExpressionNet (face-api.js) — baseline original, con HAPPY_BIAS.
//   - emotion-fusion.js (I2, heurístico) — Blendshapes + audio.
//   - Modelo propio (I2, MLP) — entrenado sobre 52 blendshapes (ver notebook),
//     mostrado/logueado en su versión ACTIVA de producción (parche de "sad"
//     + suavizado EMA + histéresis, ver smoothEmotionProbs/stableDominantKey),
//     no la salida cruda — igual que lo que decide qué mostrarle a la gente
//     cuando este motor está seleccionado.
// Las 3 fuentes (face-api crudo, fusión, MLP activo) se calculan y loguean
// SIEMPRE al CSV, sin importar cuál esté activa pilotando satisfacción, para
// poder comparar accuracy/F1 offline en cualquier momento (ver protocolo de
// elicitación con clips validados, 21 jul 2026). El MLP CRUDO (sin parche ni
// suavizado) también se loguea aparte (columnas mlp_raw_*) — no es uno de
// los "3 motores" mostrados en el panel, es un extra de diagnóstico interno,
// porque perderlo hubiera hecho imposible repetir análisis como el de la
// sesión de elicitación (que reveló el problema de histéresis) sin volver a
// grabar sesión.
//
// Detección: MediaPipe FaceMesh (I1) es el detector principal — 478 landmarks
// 3D, head pose real, blendshapes. face-api.js (tinyFaceDetector) sigue
// corriendo en paralelo SOLO para alimentar faceExpressionNet (opción del
// selector), la comparación visual 68 vs 478 puntos, y como fallback si
// MediaPipe no está listo. Edad/género: ageGenderNet, pero alimentado con el
// recorte del box de MediaPipe (I1) en vez de tinyFaceDetector — validado
// offline contra UTKFace (n=881 pareado): cobertura de ~33% a ~99%, MAE y
// accuracy de género equivalentes o mejores.
//
// Todo el procesamiento es local en el navegador.

const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";

// Traducción y emoji para cada expresión que devuelve cualquiera de los 3 motores.
const EMOTIONS = {
  neutral:   { es: "Neutral",  emoji: "😐" },
  happy:     { es: "Feliz",    emoji: "😄" },
  sad:       { es: "Triste",   emoji: "😢" },
  angry:     { es: "Enojo",    emoji: "😠" },
  fearful:   { es: "Miedo",    emoji: "😨" },
  disgusted: { es: "Disgusto", emoji: "🤢" },
  surprised: { es: "Sorpresa", emoji: "😲" },
};

// Coordenadas en el modelo circumplejo (Russell): valencia y activación en [-1, 1].
const AFFECT = {
  neutral:   { v:  0.0, a: -0.1 },
  happy:     { v:  0.8, a:  0.5 },
  sad:       { v: -0.7, a: -0.4 },
  angry:     { v: -0.6, a:  0.7 },
  fearful:   { v: -0.6, a:  0.8 },
  disgusted: { v: -0.6, a:  0.3 },
  surprised: { v:  0.3, a:  0.7 },
};

const video     = document.getElementById("video");
const overlay   = document.getElementById("overlay");
const statusEl  = document.getElementById("status");
const startBtn  = document.getElementById("startBtn");
const stopBtn   = document.getElementById("stopBtn");
const boxToggle = document.getElementById("boxToggle");
const compareToggle = document.getElementById("compareToggle");
const compareCounts = document.getElementById("compareCounts");
const classifierSel = document.getElementById("classifierSel");
const cmpFaceapi = document.getElementById("cmpFaceapi");
const cmpFusion  = document.getElementById("cmpFusion");
const cmpMLP     = document.getElementById("cmpMLP");
const domEmoji  = document.getElementById("domEmoji");
const domLabel  = document.getElementById("domLabel");
const domConf   = document.getElementById("domConf");
const barsEl    = document.getElementById("bars");

// Métricas tipo MorphCast.
const mAge    = document.getElementById("mAge");
const mGender = document.getElementById("mGender");
const mRoll   = document.getElementById("mRoll");
const mYaw    = document.getElementById("mYaw");
const mPitch  = document.getElementById("mPitch");
const mPos    = document.getElementById("mPos");
const mVal    = document.getElementById("mVal");
const mAro    = document.getElementById("mAro");
const circ    = document.getElementById("circumplex");

const logToggle   = document.getElementById("logToggle");
const intervalSel = document.getElementById("intervalSel");
const logCount    = document.getElementById("logCount");
const csvBtn      = document.getElementById("csvBtn");
const clearBtn    = document.getElementById("clearBtn");
const logBody     = document.getElementById("logBody");

// Interacción cliente–servicio (voz, fusión, momentos clave).
const roleCliente   = document.getElementById("roleCliente");
const roleAsesor    = document.getElementById("roleAsesor");
const micStatus     = document.getElementById("micStatus");
const vPitch        = document.getElementById("vPitch");
const vEnergy       = document.getElementById("vEnergy");
const vArousal      = document.getElementById("vArousal");
const satValue      = document.getElementById("satValue");
const satLabel      = document.getElementById("satLabel");
const satTimeline   = document.getElementById("satTimeline");
const momentsList   = document.getElementById("momentsList");
const momentsCount  = document.getElementById("momentsCount");

// Salidas de afecto (estilo MorphCast).
const mQuadrant   = document.getElementById("mQuadrant");
const mAffect98   = document.getElementById("mAffect98");
const mAffect38   = document.getElementById("mAffect38");
const mPositivity = document.getElementById("mPositivity");
const mAttention  = document.getElementById("mAttention");
const mFaces      = document.getElementById("mFaces");
const mDetRate    = document.getElementById("mDetRate");
const mEntropy    = document.getElementById("mEntropy");
const mFlipRate   = document.getElementById("mFlipRate");

// Umbrales de satisfacción. HAPPY_BIAS solo se aplica cuando el motor activo
// es faceExpressionNet (ver updateAffect) — emotion-fusion.js y el MLP no lo
// necesitan (el MLP ya corrige el sesgo durante el entrenamiento).
const SAT_THRESHOLD_POS = 0.5;
const SAT_THRESHOLD_NEG = -0.3;
const HAPPY_BIAS = 0.15;

let stream = null;
let loopId = null;

// Registro de emociones para análisis posterior.
const emotionLog = [];
let lastLogAt = 0;
const MAX_TABLE_ROWS = 12;

// --- Estado de interacción y satisfacción ---
const voice = new VoiceAnalyzer();
let activeRole = "cliente";
let lastFaceCount = 0;
let satEMA = 0;
let hasSat = false;
const satHist = [];
const SAT_HIST_MAX = 900;

// Detección de momentos clave.
const momentsLog = [];
let momentState = "neutral";
let lastMomentAt = 0;
const MOMENT_COOLDOWN_MS = 3500;
const MAX_MOMENT_ROWS = 30;
let sessionStart = 0;

// Construye las barras una sola vez para luego solo actualizar anchos.
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
  drawCircumplex(0, 0);
}

function setStatus(msg) {
  statusEl.textContent = msg;
  statusEl.style.display = msg ? "block" : "none";
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Timeout (${ms}ms) cargando ${label}`)), ms)
    ),
  ]);
}

async function loadModels() {
  setStatus("Cargando modelos de IA…");

  // face-api: se mantiene completo (detector + landmarks + expresiones +
  // edad/género) — el detector y los landmarks alimentan el fallback y la
  // comparación visual; expresiones alimenta la opción "faceExpressionNet"
  // del selector; ageGenderNet se usa aparte, sobre el recorte de MediaPipe
  // (ver cropForAgeGender), no encadenado a detectAllFaces().
  const faceApiLoad = Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
  ]);

  // I1 — MediaPipe FaceMesh, en paralelo, con timeout de 20s.
  const faceMeshLoad = withTimeout(
    window.FaceMeshEngine.load(setStatus),
    20000,
    "MediaPipe FaceMesh"
  ).catch((e) => {
    console.error("No se pudo cargar MediaPipe FaceMesh:", e);
    return "fallback";
  });

  // I2 — modelo propio (MLP), en paralelo también. Si falla, el selector cae
  // de vuelta a faceExpressionNet/fusión automáticamente (ver detectLoop).
  const emotionMLLoad = withTimeout(
    window.EmotionML.load("emotion_ml_weights.json"),
    20000,
    "Modelo propio (MLP)"
  ).catch((e) => {
    console.error("No se pudo cargar el modelo propio (MLP):", e);
    return "fallback";
  });

  const [, faceMeshResult] = await Promise.all([faceApiLoad, faceMeshLoad, emotionMLLoad]);
  if (faceMeshResult === "fallback") {
    setStatus("FaceMesh no disponible; se usará la aproximación anterior (fallback). Revisa la consola (F12).");
    await new Promise((r) => setTimeout(r, 2500));
  }

  setStatus("Modelos listos. Presiona «Iniciar cámara».");
  startBtn.disabled = false;
}

// --- Programación del loop, con respaldo para pestañas en segundo plano ----
// Los navegadores pausan o limitan MUCHO requestAnimationFrame cuando la
// pestaña no está visible (ahorro de batería) — por eso "se congela" el
// registro al cambiar a otra pestaña (p.ej. un video de YouTube), aunque la
// cámara siga técnicamente activa. No es un bug del sistema ni del
// navegador específico (pasa en Chrome/Firefox/Safari por igual): es
// comportamiento estándar de la Page Visibility API. El respaldo: cuando la
// pestaña está oculta, se usa setTimeout (menos limitado) a ~5 fps — no hace
// falta más, ya que el registro al CSV muestrea cada 1s por defecto de
// cualquier forma, y no hay nadie viendo el overlay mientras la pestaña está
// en segundo plano.
function scheduleNextFrame(fn) {
  if (document.visibilityState === "hidden") {
    return setTimeout(fn, 200);
  }
  return requestAnimationFrame(fn);
}

function cancelLoop(id) {
  // Un id de rAF y uno de setTimeout no se confunden entre sí — llamar a
  // ambos "por si acaso" es seguro y evita tener que rastrear cuál se usó.
  cancelAnimationFrame(id);
  clearTimeout(id);
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
    sessionStart = performance.now();

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
  if (loopId) cancelLoop(loopId);
  loopId = null;
  voice.stop();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Cámara detenida.");
  micStatus.textContent = "🎤 micrófono inactivo";
  micStatus.classList.remove("on");
  window.QualityMetrics.reset();
  resetMetrics();
}

function resetMetrics() {
  domLabel.textContent = "—";
  domConf.textContent = "Esperando…";
  domEmoji.textContent = "🙂";
  [mAge, mGender, mRoll, mYaw, mPitch, mPos, mVal, mAro,
   vPitch, vEnergy, vArousal,
   mQuadrant, mAffect98, mAffect38, mPositivity, mAttention, mFaces,
   mDetRate, mEntropy, mFlipRate].forEach((el) => (el.textContent = "—"));
  [mDetRate, mEntropy, mFlipRate].forEach((el) => (el.style.color = "var(--text)"));
  satValue.textContent = "—";
  satLabel.textContent = "esperando…";
  drawCircumplex(0, 0);
  resetEmotionSmoothing();
  resetAgeBuffer();
  updateComparisonPanel(null, null, null);
}

const detectorOpts = new faceapi.TinyFaceDetectorOptions({
  inputSize: 320,
  scoreThreshold: 0.5,
});

// Margen de recorte para alimentar ageGenderNet con el box de MediaPipe. 60%
// validado offline contra UTKFace (n=881 pareado) — porcentaje del tamaño
// del box en CADA frame, no píxeles fijos, así escala con la distancia real
// de la persona a la cámara.
const AGE_GENDER_CROP_MARGIN = 0.6;
const ageGenderCropCanvas = document.createElement("canvas");
const ageGenderCropCtx = ageGenderCropCanvas.getContext("2d");

function cropForAgeGender(box) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const mw = box.width * AGE_GENDER_CROP_MARGIN;
  const mh = box.height * AGE_GENDER_CROP_MARGIN;
  const x = Math.max(0, box.x - mw / 2);
  const y = Math.max(0, box.y - mh / 2);
  const w = Math.min(vw - x, box.width + mw);
  const h = Math.min(vh - y, box.height + mh);
  if (w <= 10 || h <= 10) return null;

  ageGenderCropCanvas.width = Math.round(w);
  ageGenderCropCanvas.height = Math.round(h);
  ageGenderCropCtx.drawImage(video, x, y, w, h, 0, 0, w, h);
  return ageGenderCropCanvas;
}

// --- Suavizado del modelo propio (EMA + histéresis) -------------------------
// Frame a frame, dos emociones cercanas en score (ej. happy vs sad) pueden
// intercambiar el primer lugar por ruido de un solo frame. Se suavizan las 7
// probabilidades con una media móvil exponencial y la ETIQUETA MOSTRADA solo
// cambia si la nueva dominante supera un umbral de confianza Y se sostiene
// un tiempo mínimo. Se aplica SOLO al modelo propio (MLP) — es donde se
// observó el problema; faceExpressionNet y emotion-fusion.js usan argmax
// directo, como siempre.
//
// AJUSTADO (21 jul 2026) de 0.70/600ms a 0.45/300ms — PROVISIONAL, sin
// confirmar todavía con datos. Motivo: la sesión de elicitación con clips
// validados (26 min, 5 emociones + neutral, ground truth real) mostró que
// con 0.70/600ms el sistema quedaba "congelado" en Neutral el 92% del
// tiempo — ninguna otra emoción del MLP alcanzaba 70% de confianza
// sostenida 600ms, sin importar qué clip se estuviera viendo. No se sabe
// todavía si el problema real era SOLO la histéresis muy estricta o si el
// MLP en sí tiene poca señal para happy/fearful/angry en esta sesión/sujeto
// (ver hallazgo de esa misma sesión: probabilidad promedio de la clase
// verdadera ~2-3% para happy/fearful, rank 5-7 de 7 — la histéresis no
// puede arreglar eso). Este cambio es el primer experimento para
// distinguir ambas causas: se repite el protocolo de elicitación con estos
// valores más permisivos y se compara accuracy/recall contra la sesión
// anterior. Si mejora sustancialmente → la histéresis era el problema
// principal. Si sigue igual de mal en happy/fearful/angry → el problema
// está en el MLP mismo (features, entrenamiento, o domain shift), no en el
// suavizado, y bajar más estos números no va a ayudar.
const EMOTION_EMA_ALPHA = 0.25;
const EMOTION_SWITCH_CONFIDENCE = 0.45;
const EMOTION_SWITCH_HOLD_MS = 300;

let emotionEMA = null;
let displayedEmotion = null;
let pendingEmotion = null;
let pendingSince = 0;

function smoothEmotionProbs(rawProbs) {
  const adjustedProbs = { ...rawProbs };

  // PARCHE TEMPORAL, sin calibrar todavía — pendiente de reemplazar por una
  // corrección medida (capturar ejemplos reales de webcam, confirmar si
  // "sad" de verdad se sobre-predice ahí, y por cuánto). En el test set de
  // RAF-DB "sad" NO se sobre-predice (precisión 0.459, recall 0.347) — el
  // problema reportado aparece en uso real con webcam, no en el modelo en
  // sí, lo que apunta a domain shift (ángulo/luz de webcam vs. fotos de
  // entrenamiento) más que a un defecto del MLP. Mientras se mide bien, se
  // recorta "sad" y se renormaliza para que las 7 probabilidades sigan
  // sumando 1 (la versión anterior de este parche NO renormalizaba).
  if (adjustedProbs.sad !== undefined) {
    adjustedProbs.sad *= 0.20;
    const total = Object.values(adjustedProbs).reduce((a, b) => a + b, 0);
    if (total > 0) {
      for (const k of Object.keys(adjustedProbs)) adjustedProbs[k] /= total;
    }
  }

  if (!emotionEMA) {
    emotionEMA = { ...adjustedProbs };
  } else {
    for (const key of Object.keys(adjustedProbs)) {
      emotionEMA[key] = EMOTION_EMA_ALPHA * adjustedProbs[key] + (1 - EMOTION_EMA_ALPHA) * (emotionEMA[key] || 0);
    }
  }
  return emotionEMA;
}

function stableDominantKey(probs) {
  let bestKey = null, bestP = -1;
  for (const [k, v] of Object.entries(probs)) if (v > bestP) { bestP = v; bestKey = k; }

  if (displayedEmotion === null) {
    displayedEmotion = bestKey;
    pendingEmotion = null;
    return displayedEmotion;
  }
  if (bestKey === displayedEmotion) {
    pendingEmotion = null;
    return displayedEmotion;
  }
  if (bestP < EMOTION_SWITCH_CONFIDENCE) {
    return displayedEmotion;
  }
  const now = performance.now();
  if (pendingEmotion !== bestKey) {
    pendingEmotion = bestKey;
    pendingSince = now;
    return displayedEmotion;
  }
  if (now - pendingSince >= EMOTION_SWITCH_HOLD_MS) {
    displayedEmotion = bestKey;
    pendingEmotion = null;
  }
  return displayedEmotion;
}

function resetEmotionSmoothing() {
  emotionEMA = null;
  displayedEmotion = null;
  pendingEmotion = null;
}

// Encuentra la clave dominante de un objeto {emocion: prob}. Se usa solo
// para el panel de comparación en vivo — no toca la lógica de satisfacción
// ni momentos clave, que siguen dependiendo de classifierSel como siempre.
function dominantOf(probs) {
  let bestKey = null, bestP = -1;
  for (const [k, v] of Object.entries(probs)) if (v > bestP) { bestP = v; bestKey = k; }
  return bestKey ? { key: bestKey, conf: bestP } : null;
}

// Panel de comparación en vivo — muestra los 3 motores SIMULTÁNEAMENTE,
// aunque solo uno esté "activo" (classifierSel) para satisfacción/momentos.
// No hace ningún cálculo nuevo: reusa faceapiExpr/fusionProbs/mlProbs que
// detectLoop() ya calcula cada frame de todas formas (mismos datos que ya
// se registran siempre en el CSV, columnas fusion_*/mlp_*).
function updateComparisonPanel(faceapiExpr, fusionProbs, mlProbs) {
  const fmt = (probs, el) => {
    if (!probs) { el.textContent = "—"; return; }
    const d = dominantOf(probs);
    if (!d) { el.textContent = "—"; return; }
    const info = EMOTIONS[d.key] || { es: d.key, emoji: "" };
    el.textContent = `${info.emoji} ${info.es} (${Math.round(d.conf * 100)}%)`;
  };
  fmt(faceapiExpr, cmpFaceapi);
  fmt(fusionProbs, cmpFusion);
  fmt(mlProbs, cmpMLP);
}

async function detectLoop() {
  if (!stream) return;

  // I1 — MediaPipe FaceMesh: detección principal, 478 landmarks 3D, head
  // pose real, blendshapes.
  const fm = window.FaceMeshEngine.detect(video);

  // face-api.js: detección + landmarks 68 + expresiones (fallback,
  // comparación visual, y opción "faceExpressionNet" del selector). Ya NO
  // se le pide edad/género aquí — eso viene del recorte de MediaPipe abajo.
  const results = await faceapi
    .detectAllFaces(video, detectorOpts)
    .withFaceLandmarks()
    .withFaceExpressions();

  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const vf = voice.running ? voice.features : null;
  updateVoiceUI(vf);

  lastFaceCount = window.FaceMeshEngine.ready ? fm.count : results.length;
  mFaces.textContent = String(lastFaceCount);

  let result = null;
  if (results.length) {
    result = results.reduce((m, r) =>
      r.detection.box.area > m.detection.box.area ? r : m, results[0]);
  }
  let mesh = null;
  if (fm.faces.length) {
    mesh = fm.faces.reduce((m, f) => (f.box.area > m.box.area ? f : m), fm.faces[0]);
  }

  if (mesh) {
    // Camino normal: MediaPipe detectó — I1 completo + selector de 3 vías.

    // I2: las 3 fuentes se calculan SIEMPRE que haya blendshapes, para poder
    // loguearlas todas y comparar offline (F1/ECE, Nivel 1), sin importar
    // cuál esté "activa" pilotando el sistema en este frame.
    const fusionProbs = (window.EmotionFusion && mesh.blendshapes)
      ? window.EmotionFusion.classify(mesh.blendshapes, vf, activeRole)
      : null;
    const mlResult = (window.EmotionML && window.EmotionML.ready && mesh.blendshapes)
      ? window.EmotionML.classify(mesh.blendshapes)
      : null;

    // MLP ACTIVO (parche de "sad" + EMA + histéresis) — se calcula SIEMPRE
    // que haya mlResult, sin importar si classifierSel="mlp" está elegido o
    // no. Antes esto solo corría dentro del if(classifierChoice==="mlp"),
    // lo que dejaba el panel de comparación y el CSV mostrando el MLP CRUDO
    // (sin parche/suavizado) — que es justo lo que confundía "qué está
    // pasando en producción" con "qué predice el modelo en bruto". Ahora el
    // estado de EMA/histéresis avanza cada frame igual, independientemente
    // de cuál motor esté pilotando satisfacción/momentos.
    let mlpActive = null, mlpForcedDom = null;
    if (mlResult) {
      mlpActive = smoothEmotionProbs(mlResult.probs);
      mlpForcedDom = stableDominantKey(mlpActive);
    }

    // Panel de comparación en vivo — los 3 motores pedidos (face-api crudo,
    // MLP activo/final, fusión), sin importar cuál esté "activo" abajo para
    // satisfacción/momentos.
    updateComparisonPanel(result ? result.expressions : null, fusionProbs, mlpActive);

    // Edad/género: ageGenderNet sobre el recorte del box de MediaPipe. Puede
    // fallar (recorte inválido, red no lista) sin tumbar el resto del
    // pipeline — ageGenderResult queda null y updateMetrics()/maybeLog() ya
    // saben mostrar "—" en ese caso.
    let ageGenderResult = null;
    try {
      const crop = cropForAgeGender(mesh.box);
      if (crop) {
        const pred = await faceapi.nets.ageGenderNet.predictAgeAndGender(crop);
        ageGenderResult = Array.isArray(pred) ? pred[0] : pred;
      }
    } catch (e) {
      console.error("Error en predicción de edad/género:", e);
    }

    // Selector de 3 vías, con fallback seguro: si la fuente elegida no está
    // disponible este frame, cae a faceExpressionNet — nunca se queda sin
    // predicción (si result tampoco existe, activeExpr queda null y se
    // maneja como "sin rostro" más abajo).
    const classifierChoice = classifierSel.value; // "faceapi" | "fusion" | "mlp"
    let activeExpr = null, forcedDom = null, applyHappyBias = false, usedFusion = false, usedML = false;

    if (classifierChoice === "mlp" && mlpActive) {
      activeExpr = mlpActive;
      forcedDom = mlpForcedDom;
      usedML = true;
    } else if (classifierChoice === "fusion" && fusionProbs) {
      activeExpr = fusionProbs;
      usedFusion = true;
    } else if (result) {
      activeExpr = result.expressions;
      applyHappyBias = true;
    }

    if (activeExpr) {
      const dom    = updateEmotions(activeExpr, forcedDom);
      const affect = updateAffect(activeExpr, applyHappyBias);
      const pose   = mesh.pose;
      const pos    = computeFacePosition(mesh.box);
      updateMetrics(ageGenderResult, pose, pos);
      const af     = updateAffectModels(affect.valence, affect.arousal, pose, mesh.blendshapes);

      const sat = updateSatisfaction(affect, vf);
      detectKeyMoment(sat, activeExpr, dom, vf);

      if (boxToggle.checked) {
        drawBox(ctx, mesh.box, true);
        drawFaceMeshLandmarks(ctx, mesh.landmarks, overlay.width, overlay.height);
        if (compareToggle.checked && result) {
          drawLandmarks68(ctx, result.landmarks, "#ffb020");
          compareCounts.style.display = "block";
          compareCounts.textContent =
            `🟠 face-api.js: 68 puntos   ·   🟢 MediaPipe FaceMesh: ${mesh.landmarks.length} puntos`;
        } else {
          compareCounts.style.display = "none";
        }
      } else {
        compareCounts.style.display = "none";
      }

      window.QualityMetrics.update(true, activeExpr, dom.raw);
      maybeLog(dom, activeExpr, ageGenderResult, affect, pose, pos, vf, sat, af,
        result ? result.expressions : null, fusionProbs, usedFusion,
        mlpActive, usedML, mlResult ? mlResult.probs : null);
    } else {
      domConf.textContent = "Ninguna fuente de emoción disponible…";
      window.QualityMetrics.update(false, null, null);
      pushSatSample(satEMA * 0.97);
    }
  } else if (result) {
    // Fallback: FaceMesh no detectó a tiempo (aún cargando, sin GPU, etc.).
    // Sin mesh no hay blendshapes, así que I2 (fusión y MLP) no puede correr
    // aquí — se usa la aproximación geométrica anterior con faceExpressionNet.
    const dom    = updateEmotions(result.expressions);
    const affect = updateAffect(result.expressions, /* applyHappyBias */ true);
    const pose   = computeHeadPoseFallback(result.landmarks);
    const pos    = computeFacePosition(result.detection.box);
    updateMetrics(null, pose, pos); // sin recorte de MediaPipe, no hay edad/género este frame
    const af     = updateAffectModels(affect.valence, affect.arousal, pose, null);
    updateComparisonPanel(result.expressions, null, null);

    const sat = updateSatisfaction(affect, vf);
    detectKeyMoment(sat, result.expressions, dom, vf);

    if (boxToggle.checked) {
      drawBox(ctx, result.detection.box, true);
      drawLandmarks68(ctx, result.landmarks);
    }
    window.QualityMetrics.update(true, result.expressions, dom.raw);
    maybeLog(dom, result.expressions, null, affect, pose, pos, vf, sat, af,
      result.expressions, null, false, null, false, null);
  } else {
    domConf.textContent = "No se detecta rostro…";
    window.QualityMetrics.update(false, null, null);
    updateComparisonPanel(null, null, null);
    pushSatSample(satEMA * 0.97);
  }

  updateQualityMetricsUI();
  drawSatTimeline();
  loopId = scheduleNextFrame(detectLoop);
}

// I4: pinta el snapshot de QualityMetrics (Nivel 2, Sección 5.2) y marca en
// rojo cuando se cruza el umbral de alerta definido en el reporte.
function updateQualityMetricsUI() {
  const { detectionRate, entropy, flipRate, alerts } = window.QualityMetrics.snapshot();
  const ALERT = "#ff6b6b";
  const NORMAL = "var(--text)";

  mDetRate.textContent = detectionRate != null ? `${Math.round(detectionRate * 100)}%` : "—";
  mDetRate.style.color = alerts.detectionRate ? ALERT : NORMAL;

  mEntropy.textContent = entropy != null ? entropy.toFixed(2) : "—";
  mEntropy.style.color = alerts.entropy ? ALERT : NORMAL;

  mFlipRate.textContent = Number.isFinite(flipRate) ? `${flipRate.toFixed(1)}/s` : "—";
  mFlipRate.style.color = alerts.flipRate ? ALERT : NORMAL;
}

// Salidas de afecto estilo MorphCast a partir de (valencia, activación).
function updateAffectModels(v, a, pose, blendshapes) {
  const A = window.AffectModels;
  const af = {
    quadrant:   A.quadrantOf(v, a),
    affect98:   A.nearestAffect(A.AFFECTS_98, v, a),
    affect38:   A.nearestAffect(A.AFFECTS_38, v, a),
    positivity: A.positivityOf(v),
    attention:  A.attentionScore(pose, blendshapes),
  };
  mQuadrant.textContent   = af.quadrant;
  mAffect98.textContent   = af.affect98;
  mAffect38.textContent   = af.affect38;
  mPositivity.textContent = Math.round(af.positivity * 100) + "%";
  mAttention.textContent  = Math.round(af.attention * 100) + "%";
  return af;
}

// --- Voz, fusión y momentos clave ------------------------------------------

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

function computeSatisfaction(affect, vf) {
  let s = affect.valence;
  if (vf && vf.voiced && activeRole === "cliente") {
    const va = vf.arousal;
    if (affect.valence < 0) {
      s -= 0.4 * va * (-affect.valence);
    } else {
      s += 0.1 * va * affect.valence;
    }
  }
  return Math.max(-1, Math.min(1, s));
}

function updateSatisfaction(affect, vf) {
  const s = computeSatisfaction(affect, vf);
  satEMA = hasSat ? 0.85 * satEMA + 0.15 * s : s;
  hasSat = true;
  pushSatSample(satEMA);

  satValue.textContent = (satEMA >= 0 ? "+" : "") + satEMA.toFixed(2);
  let label, color;
  if (satEMA > SAT_THRESHOLD_POS)      { label = "Satisfecho"; color = "var(--good)"; }
  else if (satEMA < SAT_THRESHOLD_NEG) { label = "Insatisfecho"; color = "#ff6b6b"; }
  else                                 { label = "Neutral"; color = "var(--muted)"; }
  satLabel.textContent = label;
  satValue.style.color = color;
  return satEMA;
}

function pushSatSample(s, event) {
  satHist.push({ t: performance.now(), s, event: event || null });
  if (satHist.length > SAT_HIST_MAX) satHist.shift();
}

function detectKeyMoment(sat, expressions, dom, vf) {
  const now = performance.now();
  const neg = (expressions.angry || 0) + (expressions.disgusted || 0) +
              (expressions.sad || 0) + (expressions.fearful || 0);

  if (sat > -0.15 && sat < 0.25 && neg < 0.4) momentState = "neutral";

  if (now - lastMomentAt < MOMENT_COOLDOWN_MS) return;

  let type = null, trigger = "";
  if ((sat < SAT_THRESHOLD_NEG || neg > 0.6) && momentState !== "neg") {
    type = "neg";
    trigger = neg > 0.6 ? "expresión negativa fuerte" :
              (vf && vf.voiced && activeRole === "cliente" && vf.arousal > 0.5)
                ? "voz tensa + valencia baja" : "valencia facial baja";
  } else if (sat > SAT_THRESHOLD_POS && (expressions.happy || 0) > 0.45 && momentState !== "pos") {
    type = "pos";
    trigger = (vf && vf.voiced && activeRole === "cliente")
                ? "sonrisa + voz positiva" : "sonrisa sostenida";
  }

  if (type) {
    momentState = type;
    lastMomentAt = now;
    const m = {
      time: new Date(),
      elapsed: (now - sessionStart) / 1000,
      type,
      trigger,
      sat,
      emotion: dom.es,
      role: activeRole === "cliente" ? "Cliente" : "Asesor",
      pitch: vf && vf.pitch ? Math.round(vf.pitch) : null,
      arousalVoz: vf && vf.voiced ? vf.arousal : null,
    };
    momentsLog.push(m);
    if (satHist.length) satHist[satHist.length - 1].event = type;
    renderMoments();
  }
}

function renderMoments() {
  const n = momentsLog.length;
  momentsCount.textContent = n === 1 ? "1 momento" : `${n} momentos`;
  if (n === 0) {
    momentsList.innerHTML =
      '<li class="moments-empty">Aún no se detectan momentos clave. Inicia la cámara y el micrófono.</li>';
    return;
  }
  const recent = momentsLog.slice(-MAX_MOMENT_ROWS).reverse();
  momentsList.innerHTML = recent
    .map((m) => {
      const tag = m.type === "pos" ? "● Satisfacción" : "● Insatisfacción";
      const mmss = fmtElapsed(m.elapsed);
      return `<li class="${m.type}">
        <span class="m-time">${mmss}</span>
        <span class="m-tag">${tag}</span>
        <span class="m-detail">${m.trigger} · ${m.emotion} · ${m.role}</span>
      </li>`;
    })
    .join("");
}

function fmtElapsed(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function drawSatTimeline() {
  if (!satTimeline) return;
  const ctx = satTimeline.getContext("2d");
  const W = satTimeline.width, H = satTimeline.height;
  ctx.clearRect(0, 0, W, H);

  const yFor = (s) => H / 2 - (s * (H / 2 - 8));
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

  if (satHist.length < 2) return;
  const n = satHist.length;
  const xFor = (i) => (i / (SAT_HIST_MAX - 1)) * W;

  ctx.strokeStyle = "#6c8cff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xFor(i), y = yFor(satHist[i].s);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  for (let i = 0; i < n; i++) {
    const ev = satHist[i].event;
    if (!ev) continue;
    ctx.fillStyle = ev === "pos" ? "#4ade80" : "#ff6b6b";
    ctx.beginPath();
    ctx.arc(xFor(i), yFor(satHist[i].s), 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function setRole(role) {
  activeRole = role;
  roleCliente.classList.toggle("active", role === "cliente");
  roleAsesor.classList.toggle("active", role === "asesor");
}

// --- Métricas derivadas -----------------------------------------------------

// Valencia y activación como promedio ponderado por las probabilidades de la
// fuente activa. HAPPY_BIAS solo se resta cuando applyHappyBias=true (motor
// faceExpressionNet) — emotion-fusion.js y el MLP no lo necesitan.
function updateAffect(expressions, applyHappyBias) {
  const happyAdj = applyHappyBias
    ? Math.max(0, (expressions.happy || 0) - HAPPY_BIAS)
    : (expressions.happy || 0);
  let v = 0, a = 0;
  for (const key of Object.keys(AFFECT)) {
    const p = key === "happy" ? happyAdj : (expressions[key] || 0);
    v += p * AFFECT[key].v;
    a += p * AFFECT[key].a;
  }
  mVal.textContent = (v >= 0 ? "+" : "") + v.toFixed(2);
  mAro.textContent = (a >= 0 ? "+" : "") + a.toFixed(2);
  drawCircumplex(v, a);
  return { valence: v, arousal: a };
}

// Fallback: aproximación geométrica 2D sobre 68 puntos de face-api.js — solo
// se usa si MediaPipe no detectó este frame (ver rama "else if (result)" en
// detectLoop). roll es bastante fiable; yaw y pitch son aproximaciones.
function computeHeadPoseFallback(landmarks) {
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

function computeFacePosition(box) {
  const w = video.videoWidth || overlay.width;
  const h = video.videoHeight || overlay.height;
  const cxRaw = (box.x + box.width / 2) / w;
  const cy = (box.y + box.height / 2) / h;
  return {
    x: 1 - cxRaw,
    y: cy,
    size: box.width / w,
    xRaw: cxRaw,
  };
}

// --- Estabilización de edad (mediana + rango) --------------------------------
// La edad NO cambia dentro de una sesión — cada lectura de ageGenderNet es
// una medición ruidosa de un valor constante. Buffer muestreado a 1 Hz (no
// cada frame, para reducir correlación serial entre muestras) y se reporta
// la MEDIANA junto con el rango intercuartílico [Q1,Q3].
const AGE_BUFFER_WINDOW_MS = 15000;
const AGE_SAMPLE_INTERVAL_MS = 1000;
let ageBuffer = [];
let lastAgeSampleAt = 0;

function pushAgeSample(age) {
  const now = performance.now();
  if (now - lastAgeSampleAt < AGE_SAMPLE_INTERVAL_MS) return;
  lastAgeSampleAt = now;
  ageBuffer.push({ t: now, age });
  ageBuffer = ageBuffer.filter((s) => now - s.t <= AGE_BUFFER_WINDOW_MS);
}

function quantile(sortedArr, q) {
  const pos = (sortedArr.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sortedArr[base + 1] !== undefined
    ? sortedArr[base] + rest * (sortedArr[base + 1] - sortedArr[base])
    : sortedArr[base];
}

function stableAge() {
  if (ageBuffer.length === 0) return null;
  const ages = ageBuffer.map((s) => s.age).sort((a, b) => a - b);
  return {
    median: quantile(ages, 0.5),
    q1: quantile(ages, 0.25),
    q3: quantile(ages, 0.75),
    n: ages.length,
  };
}

function resetAgeBuffer() {
  ageBuffer = [];
  lastAgeSampleAt = 0;
}

function updateMetrics(ageGenderResult, pose, pos) {
  if (ageGenderResult) {
    pushAgeSample(ageGenderResult.age);
    const stable = stableAge();
    if (stable && stable.n >= 3) {
      mAge.textContent = `${Math.round(stable.median)} años (${Math.round(stable.q1)}–${Math.round(stable.q3)})`;
    } else {
      mAge.textContent = `${Math.round(ageGenderResult.age)} años (calculando…)`;
    }
    const esGenero = ageGenderResult.gender === "male" ? "Hombre" : "Mujer";
    mGender.textContent = `${esGenero} (${Math.round(ageGenderResult.genderProbability * 100)}%)`;
  } else {
    mAge.textContent = "—";
    mGender.textContent = "—";
  }
  mRoll.textContent  = `${pose.roll.toFixed(0)}°`;
  mYaw.textContent   = `${pose.yaw.toFixed(0)}°`;
  mPitch.textContent = `${pose.pitch.toFixed(0)}°`;
  mPos.textContent   = `X ${(pos.x * 100).toFixed(0)}% · Y ${(pos.y * 100).toFixed(0)}% · ⌀ ${(pos.size * 100).toFixed(0)}%`;
}

// --- Dibujo en el canvas ----------------------------------------------------

function drawBox(ctx, box, primary = true) {
  ctx.strokeStyle = primary ? "#6c8cff" : "rgba(108,140,255,0.45)";
  ctx.lineWidth = primary ? 3 : 2;
  ctx.strokeRect(box.x, box.y, box.width, box.height);
}

// 478 puntos normalizados [0,1] de MediaPipe.
function drawFaceMeshLandmarks(ctx, landmarks, w, h) {
  ctx.fillStyle = "#4ade80";
  for (const pt of landmarks) {
    ctx.beginPath();
    ctx.arc(pt.x * w, pt.y * h, 1.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

// 68 puntos en píxeles de face-api.js (.positions).
function drawLandmarks68(ctx, landmarks, color = "#4ade80") {
  ctx.fillStyle = color;
  for (const pt of landmarks.positions) {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawCircumplex(v, a) {
  if (!circ) return;
  const ctx = circ.getContext("2d");
  const W = circ.width, H = circ.height;
  const cx = W / 2, cy = H / 2, R = W / 2 - 14;
  ctx.clearRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
  ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("activación +", cx, 10);
  ctx.fillText("activación −", cx, H - 3);
  ctx.textAlign = "left";
  ctx.fillText("valencia +", cx + 6, cy - 4);
  ctx.textAlign = "right";
  ctx.fillText("valencia −", cx - 6, cy - 4);

  const px = cx + v * R;
  const py = cy - a * R;
  ctx.fillStyle = "#6c8cff";
  ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

// --- Registro / CSV ---------------------------------------------------------

function maybeLog(dom, expressions, ageGenderResult, affect, pose, pos, vf, sat, af,
  faceapiExpr, fusionProbs, usedFusion, mlProbs, usedML, mlRawProbs) {
  if (!logToggle.checked) return;
  const now = performance.now();
  const intervalMs = Number(intervalSel.value);
  if (now - lastLogAt < intervalMs) return;
  lastLogAt = now;

  const expr = {};
  for (const key of Object.keys(EMOTIONS)) expr[key] = expressions[key] || 0;

  // Los 3 motores pedidos: face-api crudo, fusión, y MLP ACTIVO (parche +
  // suavizado) — lo que de verdad refleja "qué le mostramos a la gente"
  // cuando cada uno está seleccionado, no una aproximación.
  let faceapiExprLog = null;
  if (faceapiExpr) {
    faceapiExprLog = {};
    for (const key of Object.keys(EMOTIONS)) faceapiExprLog[key] = faceapiExpr[key] || 0;
  }

  let fusionExpr = null;
  if (fusionProbs) {
    fusionExpr = {};
    for (const key of Object.keys(EMOTIONS)) fusionExpr[key] = fusionProbs[key] || 0;
  }

  let mlExpr = null;
  if (mlProbs) {
    mlExpr = {};
    for (const key of Object.keys(EMOTIONS)) mlExpr[key] = mlProbs[key] || 0;
  }

  // Respaldo de diagnóstico, NO es uno de los "3 motores" del panel — MLP
  // crudo, sin parche de "sad" ni suavizado EMA/histéresis. Se guarda aparte
  // para poder seguir haciendo análisis tipo elicitación (comparar contra
  // ground truth real) sin depender de que el parche/suavizado actual sean
  // los definitivos — si se recalibran más adelante, este dato sigue siendo
  // válido para volver a evaluar desde cero.
  let mlRawExpr = null;
  if (mlRawProbs) {
    mlRawExpr = {};
    for (const key of Object.keys(EMOTIONS)) mlRawExpr[key] = mlRawProbs[key] || 0;
  }

  emotionLog.push({
    time: new Date(),
    dominant: dom.es,
    dominantRaw: dom.raw,
    confidence: dom.conf,
    expressions: expr,
    faceapiExpressions: faceapiExprLog,
    fusionExpressions: fusionExpr,
    usedFusion: !!usedFusion,
    mlExpressions: mlExpr,
    usedML: !!usedML,
    mlRawExpressions: mlRawExpr,
    age: ageGenderResult ? ageGenderResult.age : null,
    gender: ageGenderResult ? (ageGenderResult.gender === "male" ? "Hombre" : "Mujer") : "—",
    genderProb: ageGenderResult ? ageGenderResult.genderProbability : null,
    valence: affect.valence,
    arousal: affect.arousal,
    roll: pose.roll,
    yaw: pose.yaw,
    pitch: pose.pitch,
    posX: pos.x,
    posY: pos.y,
    size: pos.size,
    vPitch: vf && vf.voiced && vf.pitch ? vf.pitch : null,
    vEnergy: vf && vf.voiced ? vf.rms : null,
    vArousal: vf && vf.voiced ? vf.arousal : null,
    role: activeRole === "cliente" ? "Cliente" : "Asesor",
    satisfaction: sat,
    quadrant: af ? af.quadrant : "",
    affect98: af ? af.affect98 : "",
    affect38: af ? af.affect38 : "",
    positivity: af ? af.positivity : null,
    attention: af ? af.attention : null,
    faces: lastFaceCount,
    quality: window.QualityMetrics.snapshot(),
  });
  renderLog();
}

function renderLog() {
  const n = emotionLog.length;
  logCount.textContent = n === 1 ? "1 registro" : `${n} registros`;
  csvBtn.disabled = (n === 0 && momentsLog.length === 0);
  clearBtn.disabled = n === 0;

  if (n === 0) {
    logBody.innerHTML =
      '<tr class="log-empty"><td colspan="3">Aún no hay registros. Inicia la cámara para comenzar a muestrear.</td></tr>';
    return;
  }
  const recent = emotionLog.slice(-MAX_TABLE_ROWS).reverse();
  logBody.innerHTML = recent
    .map((r) => {
      const hora = r.time.toLocaleTimeString("es-MX");
      const emoji = EMOTIONS[r.dominantRaw].emoji;
      return `<tr><td>${hora}</td><td>${emoji} ${r.dominant}</td><td>${Math.round(
        r.confidence * 100
      )}%</td></tr>`;
    })
    .join("");
}

function clearLog() {
  emotionLog.length = 0;
  lastLogAt = 0;
  renderLog();
}

// --- Descarga combinada (un solo botón, un solo archivo) --------------------
// Antes había dos botones/archivos separados (emociones + momentos clave) —
// eso fue justo la causa de las sesiones "huérfanas" en la sesión de
// pruebas UDLAP (se descargaba uno y se olvidaba el otro). Ahora un solo
// clic descarga ambos registros en un solo CSV, en dos secciones separadas
// por una línea marcadora — para leerlo con pandas, se parte el archivo en
// esas líneas antes de pd.read_csv() de cada sección (te puedo dar el
// snippet de lectura cuando lo necesites).
function downloadSessionCSV() {
  if (emotionLog.length === 0 && momentsLog.length === 0) return;
  const keys = Object.keys(EMOTIONS);
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");

  const emoHeader = [
    "timestamp_iso", "hora_local", "emocion_dominante", "confianza",
    ...keys,
    "edad", "genero", "genero_prob",
    "valencia", "arousal",
    "roll_deg", "yaw_deg", "pitch_deg",
    "pos_x", "pos_y", "tamano",
    "voz_pitch_hz", "voz_energia", "voz_arousal", "rol_activo", "satisfaccion",
    "cuadrante", "afecto_98", "afecto_38", "positividad", "atencion", "n_rostros",
    "detection_rate", "expression_entropy", "flip_rate",
    ...keys.map((k) => `faceapi_${k}`),
    "usa_fusion_i2", ...keys.map((k) => `fusion_${k}`),
    "usa_mlp_i2", ...keys.map((k) => `mlp_${k}`),
    ...keys.map((k) => `mlp_raw_${k}`),
  ];
  const emoRows = emotionLog.map((r) => {
    const cells = [
      r.time.toISOString(),
      r.time.toLocaleString("es-MX"),
      r.dominant,
      r.confidence.toFixed(4),
      ...keys.map((k) => (r.expressions[k] || 0).toFixed(4)),
      r.age != null ? Math.round(r.age) : "",
      r.gender,
      r.genderProb != null ? r.genderProb.toFixed(4) : "",
      r.valence.toFixed(4),
      r.arousal.toFixed(4),
      r.roll.toFixed(1),
      r.yaw.toFixed(1),
      r.pitch.toFixed(1),
      r.posX.toFixed(4),
      r.posY.toFixed(4),
      r.size.toFixed(4),
      r.vPitch != null ? Math.round(r.vPitch) : "",
      r.vEnergy != null ? r.vEnergy.toFixed(4) : "",
      r.vArousal != null ? r.vArousal.toFixed(4) : "",
      r.role,
      r.satisfaction != null ? r.satisfaction.toFixed(4) : "",
      r.quadrant,
      r.affect98,
      r.affect38,
      r.positivity != null ? r.positivity.toFixed(4) : "",
      r.attention != null ? r.attention.toFixed(4) : "",
      r.faces,
      r.quality && r.quality.detectionRate != null ? r.quality.detectionRate.toFixed(4) : "",
      r.quality && r.quality.entropy != null ? r.quality.entropy.toFixed(4) : "",
      r.quality && Number.isFinite(r.quality.flipRate) ? r.quality.flipRate.toFixed(2) : "",
      ...keys.map((k) => (r.faceapiExpressions ? (r.faceapiExpressions[k] || 0).toFixed(4) : "")),
      r.usedFusion ? "1" : "0",
      ...keys.map((k) => (r.fusionExpressions ? (r.fusionExpressions[k] || 0).toFixed(4) : "")),
      r.usedML ? "1" : "0",
      ...keys.map((k) => (r.mlExpressions ? (r.mlExpressions[k] || 0).toFixed(4) : "")),
      ...keys.map((k) => (r.mlRawExpressions ? (r.mlRawExpressions[k] || 0).toFixed(4) : "")),
    ];
    return cells.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",");
  });

  const momHeader = ["timestamp_iso", "tiempo_mmss", "tipo", "disparador",
                      "satisfaccion", "emocion", "rol", "pitch_hz", "arousal_voz"];
  const momRows = momentsLog.map((m) => {
    const cells = [
      m.time.toISOString(),
      fmtElapsed(m.elapsed),
      m.type === "pos" ? "satisfaccion" : "insatisfaccion",
      m.trigger,
      m.sat.toFixed(3),
      m.emotion,
      m.role,
      m.pitch ?? "",
      m.arousalVoz != null ? m.arousalVoz.toFixed(3) : "",
    ];
    return cells.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",");
  });

  const csv = [
    "### SECCION: emociones",
    emoHeader.join(","),
    ...emoRows,
    "",
    "### SECCION: momentos_clave",
    momHeader.join(","),
    ...momRows,
  ].join("\r\n");

  saveCSV(csv, `sesion_${stamp}.csv`);
}

function saveCSV(csv, filename) {
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// --- Panel de emociones -----------------------------------------------------

function updateEmotions(expressions, forcedDominant) {
  let topKey = forcedDominant || null;
  let topVal = forcedDominant ? (expressions[forcedDominant] || 0) : -1;
  for (const key of Object.keys(EMOTIONS)) {
    const v = expressions[key] || 0;
    const pct = Math.round(v * 100);
    barFills[key].fill.style.width = pct + "%";
    barFills[key].val.textContent = pct + "%";
    if (!forcedDominant && v > topVal) { topVal = v; topKey = key; }
  }
  const info = EMOTIONS[topKey];
  domEmoji.textContent = info.emoji;
  domLabel.textContent = info.es;
  domConf.textContent = `Confianza: ${Math.round(topVal * 100)}%`;
  return { raw: topKey, es: info.es, conf: topVal };
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
csvBtn.addEventListener("click", downloadSessionCSV);
clearBtn.addEventListener("click", clearLog);
roleCliente.addEventListener("click", () => setRole("cliente"));
roleAsesor.addEventListener("click", () => setRole("asesor"));

window.addEventListener("load", async () => {
  buildBars();
  if (typeof faceapi === "undefined") {
    setStatus("No se pudo cargar face-api.js (¿sin conexión a internet?).");
    return;
  }
  try {
    await loadModels();
  } catch (err) {
    console.error(err);
    setStatus("Error al cargar modelos: " + err.message);
  }
});
