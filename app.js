// Detector de emociones con webcam usando face-api.js
// Todo el procesamiento es local en el navegador.

const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";

// Traducción y emoji para cada expresión que devuelve el modelo.
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
// Sirven para estimar valence/arousal como promedio ponderado de las expresiones.
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
const momentsCsvBtn = document.getElementById("momentsCsvBtn");

// Salidas de afecto (estilo MorphCast).
const mQuadrant   = document.getElementById("mQuadrant");
const mAffect98   = document.getElementById("mAffect98");
const mAffect38   = document.getElementById("mAffect38");
const mPositivity = document.getElementById("mPositivity");
const mAttention  = document.getElementById("mAttention");
const mFaces      = document.getElementById("mFaces");

// Umbrales de satisfacción (ajustables). Antes el sistema marcaba "satisfecho"
// casi siempre porque face-api sobre-estima "happy"; subimos el listón y
// ampliamos la banda neutra para corregirlo.
const SAT_THRESHOLD_POS = 0.5;    // satisfecho por encima de este valor
const SAT_THRESHOLD_NEG = -0.3;   // insatisfecho por debajo
const HAPPY_BIAS = 0.15;          // resta el sesgo de "happy" antes de medir valencia

let stream = null;
let loopId = null;

// Registro de emociones para análisis posterior.
const emotionLog = [];
let lastLogAt = 0;
const MAX_TABLE_ROWS = 12;

// --- Estado de interacción y satisfacción ---
const voice = new VoiceAnalyzer();
let activeRole = "cliente";        // quién habla: "cliente" | "asesor"
let lastFaceCount = 0;
let satEMA = 0;                    // satisfacción suavizada [-1, 1]
let hasSat = false;                // ¿ya hay señal válida?
const satHist = [];               // { t, s, event } para la línea de tiempo
const SAT_HIST_MAX = 900;

// Detección de momentos clave.
const momentsLog = [];            // { time, type, trigger, sat, emotion, role, pitch, rms }
let momentState = "neutral";      // "neutral" | "pos" | "neg"
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
  drawCircumplex(0, 0); // ejes vacíos
}

function setStatus(msg) {
  statusEl.textContent = msg;
  statusEl.style.display = msg ? "block" : "none";
}

async function loadModels() {
  setStatus("Cargando modelos de IA…");

  // Transitorio: tinyFaceDetector + faceLandmark68Net se mantienen SOLO para
  // alimentar faceExpressionNet/ageGenderNet. La detección, los landmarks, el
  // head pose y la atención ya NO dependen de estos dos modelos (ver I1 más
  // abajo). Esta redundancia se elimina en I2 (Fase 2-4, Blendshapes → clasi-
  // ficador de emociones propio) e I5 (Fase 2, reemplazo de ageGenderNet por
  // @vladmandic/human), momento en el que face-api.js podrá retirarse del todo.
  const faceApiLoad = Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
  ]);

  // I1 — MediaPipe FaceMesh: detección + 478 landmarks 3D + matriz de
  // transformación facial (head pose real) + blendshapes (52 AUs).
  // Se corre EN PARALELO con face-api (antes iba en serie, duplicando el
  // tiempo de espera) y con un timeout de 20s para no quedarse colgado si
  // el WASM/modelo no descarga (red bloqueada, CDN inaccesible, etc.).
  const withTimeout = (promise, ms, label) =>
    Promise.race([
      promise,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`Timeout (${ms}ms) cargando ${label}`)), ms)
      ),
    ]);

  const faceMeshLoad = withTimeout(
    window.FaceMeshEngine.load(setStatus),
    20000,
    "MediaPipe FaceMesh"
  ).catch((e) => {
    console.error("No se pudo cargar MediaPipe FaceMesh:", e);
    return "fallback";
  });

  const [, faceMeshResult] = await Promise.all([faceApiLoad, faceMeshLoad]);
  if (faceMeshResult === "fallback") {
    setStatus("FaceMesh no disponible; se usará la aproximación anterior (fallback). Revisa la consola (F12).");
    await new Promise((r) => setTimeout(r, 2500)); // deja el aviso visible un momento
  }

  setStatus("Modelos listos. Presiona «Iniciar cámara».");
  startBtn.disabled = false;
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

    // Inicia el análisis de voz si hay pista de audio.
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
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Cámara detenida.");
  micStatus.textContent = "🎤 micrófono inactivo";
  micStatus.classList.remove("on");
  resetMetrics();
}

function resetMetrics() {
  domLabel.textContent = "—";
  domConf.textContent = "Esperando…";
  domEmoji.textContent = "🙂";
  [mAge, mGender, mRoll, mYaw, mPitch, mPos, mVal, mAro,
   vPitch, vEnergy, vArousal,
   mQuadrant, mAffect98, mAffect38, mPositivity, mAttention, mFaces].forEach((el) => (el.textContent = "—"));
  satValue.textContent = "—";
  satLabel.textContent = "esperando…";
  drawCircumplex(0, 0);
}

const detectorOpts = new faceapi.TinyFaceDetectorOptions({
  inputSize: 320,
  scoreThreshold: 0.5,
});

async function detectLoop() {
  if (!stream) return;

  // I1 — MediaPipe FaceMesh: detección principal, 478 landmarks 3D, head pose
  // real (matriz de transformación) y blendshapes (base de Atención e I2).
  const fm = window.FaceMeshEngine.detect(video);

  // Transitorio: face-api.js sigue corriendo SOLO para expresiones/edad/género.
  // Ver nota en loadModels().
  const results = await faceapi
    .detectAllFaces(video, detectorOpts)
    .withFaceLandmarks()
    .withFaceExpressions()
    .withAgeAndGender();

  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // Voz: leemos las características más recientes (se calculan en su propio timer).
  const vf = voice.running ? voice.features : null;
  updateVoiceUI(vf);

  // Conteo de rostros: preferimos FaceMesh cuando está listo (más robusto ante
  // iluminación/accesorios que TinyFaceDetector, ver Hallazgos #1 y #5).
  lastFaceCount = window.FaceMeshEngine.ready ? fm.count : results.length;
  mFaces.textContent = String(lastFaceCount);

  // Rostro principal de face-api (para expresiones/edad/género) por área.
  let result = null;
  if (results.length) {
    result = results.reduce((m, r) =>
      r.detection.box.area > m.detection.box.area ? r : m, results[0]);
  }

  // Rostro principal de FaceMesh (landmarks/pose/blendshapes) por área.
  let mesh = null;
  if (fm.faces.length) {
    mesh = fm.faces.reduce((m, f) => (f.box.area > m.box.area ? f : m), fm.faces[0]);
  }

  if (result && mesh) {
    // Camino normal: ambos pipelines detectaron al mismo rostro principal.
    const dom    = updateEmotions(result.expressions);
    const affect = updateAffect(result.expressions);
    const pose   = mesh.pose; // I1: head pose real desde la matriz de transformación
    const pos    = computeFacePosition(mesh.box);
    updateMetrics(result, pose, pos);
    const af     = updateAffectModels(affect.valence, affect.arousal, pose, mesh.blendshapes);

    // Fusión multimodal → satisfacción del cliente.
    const sat = updateSatisfaction(affect, vf);
    detectKeyMoment(sat, result.expressions, dom, vf);

    if (boxToggle.checked) {
      drawBox(ctx, mesh.box, true);
      drawFaceMeshLandmarks(ctx, mesh.landmarks, overlay.width, overlay.height);
    }
    maybeLog(dom, result.expressions, result, affect, pose, pos, vf, sat, af);
  } else if (result) {
    // Fallback: FaceMesh no detectó a tiempo (p.ej. aún cargando o sin GPU
    // disponible). Se usa la aproximación anterior para no perder la sesión.
    const dom    = updateEmotions(result.expressions);
    const affect = updateAffect(result.expressions);
    const pose   = computeHeadPoseFallback(result.landmarks);
    const pos    = computeFacePosition(result.detection.box);
    updateMetrics(result, pose, pos);
    const af     = updateAffectModels(affect.valence, affect.arousal, pose, null);

    const sat = updateSatisfaction(affect, vf);
    detectKeyMoment(sat, result.expressions, dom, vf);

    if (boxToggle.checked) {
      drawBox(ctx, result.detection.box, true);
      drawLandmarks68(ctx, result.landmarks);
    }
    maybeLog(dom, result.expressions, result, affect, pose, pos, vf, sat, af);
  } else {
    domConf.textContent = "No se detecta rostro…";
    // Sin rostro: la señal se desvanece lentamente hacia 0.
    pushSatSample(satEMA * 0.97);
  }

  drawSatTimeline();
  loopId = requestAnimationFrame(detectLoop);
}

// Salidas de afecto estilo MorphCast a partir de (valencia, activación).
// I1: attentionScore ahora recibe (pose, blendshapes) en vez de (landmarks, pose);
// ver affect.js.
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

// Combina valencia facial del cliente con su activación vocal (solo si habla el cliente).
function computeSatisfaction(affect, vf) {
  let s = affect.valence; // base [-1, 1]
  if (vf && vf.voiced && activeRole === "cliente") {
    const va = vf.arousal; // 0..1
    if (affect.valence < 0) {
      s -= 0.4 * va * (-affect.valence); // voz alta/agitada acentúa lo negativo
    } else {
      s += 0.1 * va * affect.valence;    // entusiasmo positivo leve
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

// Detector de momentos clave de (in)satisfacción, con histéresis y enfriamiento.
function detectKeyMoment(sat, expressions, dom, vf) {
  const now = performance.now();
  const neg = (expressions.angry || 0) + (expressions.disgusted || 0) +
              (expressions.sad || 0) + (expressions.fearful || 0);

  // Reset de estado en zona neutral para permitir nuevos eventos.
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
  momentsCsvBtn.disabled = n === 0;
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

// Línea de tiempo de la satisfacción con marcadores en los momentos clave.
function drawSatTimeline() {
  if (!satTimeline) return;
  const ctx = satTimeline.getContext("2d");
  const W = satTimeline.width, H = satTimeline.height;
  ctx.clearRect(0, 0, W, H);

  // Línea base (satisfacción = 0) y bandas.
  const yFor = (s) => H / 2 - (s * (H / 2 - 8));
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

  if (satHist.length < 2) return;
  const n = satHist.length;
  const xFor = (i) => (i / (SAT_HIST_MAX - 1)) * W;

  // Curva.
  ctx.strokeStyle = "#6c8cff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xFor(i), y = yFor(satHist[i].s);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Marcadores de eventos.
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

function downloadMomentsCSV() {
  if (momentsLog.length === 0) return;
  const header = ["timestamp_iso", "tiempo_mmss", "tipo", "disparador",
                  "satisfaccion", "emocion", "rol", "pitch_hz", "arousal_voz"];
  const rows = momentsLog.map((m) => {
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
  const csv = [header.join(","), ...rows].join("\r\n");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  saveCSV(csv, `momentos_clave_${stamp}.csv`);
}

// --- Métricas derivadas -----------------------------------------------------

// Valencia y activación como promedio ponderado por las probabilidades.
// Restamos HAPPY_BIAS a "happy" porque face-api lo sobre-estima en caras neutras.
function updateAffect(expressions) {
  const happyAdj = Math.max(0, (expressions.happy || 0) - HAPPY_BIAS);
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

// FALLBACK (solo si MediaPipe FaceMesh no está disponible): orientación de la
// cabeza estimada a mano a partir de los 68 puntos de face-api.js. roll es
// bastante fiable; yaw y pitch son aproximaciones (no usan solvePnP 3D) y
// acumulan error >30° (Hallazgo #4 del reporte de diagnóstico). Desde I1, el
// camino normal usa mesh.pose (matriz de transformación real de FaceMesh).
function computeHeadPoseFallback(landmarks) {
  const p = landmarks.positions;
  const centroid = (a, b) => {
    let sx = 0, sy = 0;
    for (let i = a; i <= b; i++) { sx += p[i].x; sy += p[i].y; }
    const n = b - a + 1;
    return { x: sx / n, y: sy / n };
  };
  const eyeR = centroid(36, 41); // ojo derecho del sujeto (izquierda de la imagen)
  const eyeL = centroid(42, 47);
  const noseTip = p[30];
  const mouth = { x: (p[48].x + p[54].x) / 2, y: (p[48].y + p[54].y) / 2 };
  const eyesMid = { x: (eyeR.x + eyeL.x) / 2, y: (eyeR.y + eyeL.y) / 2 };
  const interocular = Math.hypot(eyeL.x - eyeR.x, eyeL.y - eyeR.y) || 1;

  // Roll: ángulo de la línea entre los ojos.
  const roll = (Math.atan2(eyeL.y - eyeR.y, eyeL.x - eyeR.x) * 180) / Math.PI;

  // Yaw: desplazamiento horizontal de la nariz respecto al centro de los ojos.
  const yaw = Math.max(-90, Math.min(90, ((noseTip.x - eyesMid.x) / interocular) * 90));

  // Pitch: posición vertical de la nariz entre ojos y boca (≈0.5 de frente).
  const faceH = (mouth.y - eyesMid.y) || 1;
  const pitch = Math.max(-90, Math.min(90, (((noseTip.y - eyesMid.y) / faceH) - 0.55) * 120));

  return { roll, yaw, pitch };
}

// Posición y tamaño del rostro como porcentaje del cuadro (en la vista espejada).
function computeFacePosition(box) {
  const w = video.videoWidth || overlay.width;
  const h = video.videoHeight || overlay.height;
  const cxRaw = (box.x + box.width / 2) / w;
  const cy = (box.y + box.height / 2) / h;
  return {
    x: 1 - cxRaw,          // espejado: lo que el usuario ve
    y: cy,
    size: box.width / w,
    xRaw: cxRaw,
  };
}

function updateMetrics(result, pose, pos) {
  mAge.textContent = `${Math.round(result.age)} años`;
  const esGenero = result.gender === "male" ? "Hombre" : "Mujer";
  mGender.textContent = `${esGenero} (${Math.round(result.genderProbability * 100)}%)`;
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

// FALLBACK: dibuja los 68 puntos de face-api.js (solo si FaceMesh no cargó).
function drawLandmarks68(ctx, landmarks) {
  ctx.fillStyle = "#4ade80";
  for (const pt of landmarks.positions) {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

// I1: dibuja los 478 puntos de MediaPipe FaceMesh (normalizados [0,1] → píxeles).
// Se muestrea cada 2 puntos para no saturar el canvas a 60fps sin perder densidad visual.
function drawFaceMeshLandmarks(ctx, landmarks, w, h) {
  ctx.fillStyle = "#4ade80";
  for (let i = 0; i < landmarks.length; i += 2) {
    const pt = landmarks[i];
    ctx.beginPath();
    ctx.arc(pt.x * w, pt.y * h, 1.1, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Mini-gráfico circumplejo: valencia (X) vs activación (Y), un punto móvil.
function drawCircumplex(v, a) {
  if (!circ) return;
  const ctx = circ.getContext("2d");
  const W = circ.width, H = circ.height;
  const cx = W / 2, cy = H / 2, R = W / 2 - 14;
  ctx.clearRect(0, 0, W, H);

  // Círculo y ejes.
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
  ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
  ctx.stroke();

  // Etiquetas de cuadrantes.
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("activación +", cx, 10);
  ctx.fillText("activación −", cx, H - 3);
  ctx.textAlign = "left";
  ctx.fillText("valencia +", cx + 6, cy - 4);
  ctx.textAlign = "right";
  ctx.fillText("valencia −", cx - 6, cy - 4);

  // Punto (valencia → X, activación → Y invertida porque el canvas crece hacia abajo).
  const px = cx + v * R;
  const py = cy - a * R;
  ctx.fillStyle = "#6c8cff";
  ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

// --- Registro / CSV ---------------------------------------------------------

function maybeLog(dom, expressions, result, affect, pose, pos, vf, sat, af) {
  if (!logToggle.checked) return;
  const now = performance.now();
  const intervalMs = Number(intervalSel.value);
  if (now - lastLogAt < intervalMs) return;
  lastLogAt = now;

  const expr = {};
  for (const key of Object.keys(EMOTIONS)) expr[key] = expressions[key] || 0;
  emotionLog.push({
    time: new Date(),
    dominant: dom.es,
    dominantRaw: dom.raw,
    confidence: dom.conf,
    expressions: expr,
    age: result.age,
    gender: result.gender === "male" ? "Hombre" : "Mujer",
    genderProb: result.genderProbability,
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
  });
  renderLog();
}

function renderLog() {
  const n = emotionLog.length;
  logCount.textContent = n === 1 ? "1 registro" : `${n} registros`;
  csvBtn.disabled = n === 0;
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

function downloadCSV() {
  if (emotionLog.length === 0) return;
  const keys = Object.keys(EMOTIONS);
  const header = [
    "timestamp_iso", "hora_local", "emocion_dominante", "confianza",
    ...keys,
    "edad", "genero", "genero_prob",
    "valencia", "arousal",
    "roll_deg", "yaw_deg", "pitch_deg",
    "pos_x", "pos_y", "tamano",
    "voz_pitch_hz", "voz_energia", "voz_arousal", "rol_activo", "satisfaccion",
    "cuadrante", "afecto_98", "afecto_38", "positividad", "atencion", "n_rostros",
  ];
  const rows = emotionLog.map((r) => {
    const cells = [
      r.time.toISOString(),
      r.time.toLocaleString("es-MX"),
      r.dominant,
      r.confidence.toFixed(4),
      ...keys.map((k) => (r.expressions[k] || 0).toFixed(4)),
      Math.round(r.age),
      r.gender,
      r.genderProb.toFixed(4),
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
    ];
    return cells.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",");
  });
  const csv = [header.join(","), ...rows].join("\r\n");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  saveCSV(csv, `emociones_${stamp}.csv`);
}

// Helper compartido: descarga un string CSV como archivo (con BOM UTF-8).
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

function updateEmotions(expressions) {
  let topKey = null;
  let topVal = -1;
  for (const key of Object.keys(EMOTIONS)) {
    const v = expressions[key] || 0;
    const pct = Math.round(v * 100);
    barFills[key].fill.style.width = pct + "%";
    barFills[key].val.textContent = pct + "%";
    if (v > topVal) { topVal = v; topKey = key; }
  }
  const info = EMOTIONS[topKey];
  domEmoji.textContent = info.emoji;
  domLabel.textContent = info.es;
  domConf.textContent = `Confianza: ${Math.round(topVal * 100)}%`;
  return { raw: topKey, es: info.es, conf: topVal };
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
csvBtn.addEventListener("click", downloadCSV);
clearBtn.addEventListener("click", clearLog);
roleCliente.addEventListener("click", () => setRole("cliente"));
roleAsesor.addEventListener("click", () => setRole("asesor"));
momentsCsvBtn.addEventListener("click", downloadMomentsCSV);

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