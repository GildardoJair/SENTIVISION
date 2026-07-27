// evaluar-dataset.js — Evaluación offline (Nivel 1, Sección 5.1 del reporte)
// de los 3 clasificadores de emoción contra imágenes ETIQUETADAS de RAF-DB.
// Reemplaza la varianza de "actuar frente a la cámara" (que vimos que era
// alta entre sesiones) por un dataset fijo y grande.
//
// NO toca app.js, facemesh.js, emotion-fusion.js. Reusa emotion-fusion.js
// tal cual (sin audio: se le pasa vf = null, que internamente ya maneja
// "sin voz" sin cambios de comportamiento). Usa facemesh-image.js (nuevo,
// modo IMAGE) en vez de facemesh.js (modo VIDEO).
//
// Todo el procesamiento ocurre en el navegador del usuario a partir de
// archivos locales seleccionados con <input type="file" webkitdirectory>;
// nada se sube a ningún servidor.
//
// Export adicional (I2, handoff diseño modelo ML): checkbox "Exportar 52
// blendshapes (ML)" — genera un CSV separado (archivo, etiqueta_real, + 52
// columnas de blendshapes con su categoryName real) para entrenar un
// clasificador propio en Python/sklearn. Pensado para correr sobre
// DATASET/train con faceExpressionNet y Human.face.emotion desmarcados
// (no se necesitan para esto y son el cuello de botella de velocidad).

const EMOTIONS = {
  neutral:   { es: "Neutral",  emoji: "😐" },
  happy:     { es: "Feliz",    emoji: "😄" },
  sad:       { es: "Triste",   emoji: "😢" },
  angry:     { es: "Enojo",    emoji: "😠" },
  fearful:   { es: "Miedo",    emoji: "😨" },
  disgusted: { es: "Disgusto", emoji: "🤢" },
  surprised: { es: "Sorpresa", emoji: "😲" },
};
const EMOTION_KEYS = Object.keys(EMOTIONS);

// Mapeo carpeta (1-7) → clave interna, verificado contra la distribución de
// clases publicada del paper original de RAF-DB (Li et al., arXiv 1911.05188),
// que coincide número por número con train_labels.csv / test_labels.csv.
const LABEL_TO_KEY = {
  "1": "surprised",
  "2": "fearful",
  "3": "disgusted",
  "4": "happy",
  "5": "sad",
  "6": "angry",
  "7": "neutral",
};

// Mismo mapeo verificado a mano (ver conversación) para Human.face.emotion.
const HUMAN_TO_KEY = {
  neutral: "neutral", happy: "happy", sad: "sad", angry: "angry",
  fear: "fearful", disgust: "disgusted", surprise: "surprised",
};
function humanEmotionToProbs(emotionArr) {
  const probs = {};
  for (const k of EMOTION_KEYS) probs[k] = 0;
  if (!emotionArr) return probs;
  for (const e of emotionArr) {
    const key = HUMAN_TO_KEY[e.emotion];
    if (key) probs[key] = e.score;
  }
  const total = Object.values(probs).reduce((a, b) => a + b, 0);
  if (total > 0) for (const k of EMOTION_KEYS) probs[k] /= total;
  return probs;
}

function dominantOf(probs) {
  let topKey = null, topVal = -1;
  for (const k of EMOTION_KEYS) {
    if ((probs[k] || 0) > topVal) { topVal = probs[k] || 0; topKey = k; }
  }
  return { key: topKey, confidence: topVal };
}

function entropyOf(probs) {
  let h = 0;
  for (const k of EMOTION_KEYS) {
    const p = probs[k] || 0;
    if (p > 0) h += -p * Math.log(p);
  }
  return h;
}

// --- Elementos de UI ---------------------------------------------------

const folderInput   = document.getElementById("folderInput");
const folderStatus  = document.getElementById("folderStatus");
const classSummary  = document.getElementById("classSummary");
const perClassInput = document.getElementById("perClassInput");
const chkFaceapi    = document.getElementById("chkFaceapi");
const chkFusion     = document.getElementById("chkFusion");
const chkHuman      = document.getElementById("chkHuman");
const chkBlendshapes = document.getElementById("chkBlendshapes");
const startBtn      = document.getElementById("startBtn");
const cancelBtn     = document.getElementById("cancelBtn");
const csvBtn        = document.getElementById("csvBtn");
const blendshapesCsvBtn = document.getElementById("blendshapesCsvBtn");
const progressFill  = document.getElementById("progressFill");
const progressText  = document.getElementById("progressText");
const resultsSection = document.getElementById("resultsSection");

let filesByLabel = {}; // { "1": [File,...], ..., "7": [...] }
let modelsReady = false;
let cancelRequested = false;
let rawResults = []; // filas para el CSV crudo

// Export de blendshapes crudos (I2, handoff diseño modelo ML): un CSV con
// las 52 categorías de MediaPipe tal cual, sin pasar por emotion-fusion.js,
// para entrenar un clasificador propio (p.ej. regresión logística en sklearn).
// blendshapeColumns se fija con los categoryName reales de la primera cara
// detectada en la corrida (son siempre las mismas 52, en el mismo orden).
let blendshapeRows = [];
let blendshapeColumns = null;

const detectorOpts = new faceapi.TinyFaceDetectorOptions({
  inputSize: 320,
  scoreThreshold: 0.5,
});

let human = null;

// --- Carga de modelos (los 3 motores, igual que alternar-emociones.js) --

async function loadModels() {
  progressText.textContent = "Cargando modelos de IA…";
  const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
  ]);
  await window.FaceMeshImageEngine.load((msg) => { progressText.textContent = msg; });

  const HumanCtor = (typeof Human === "function")
    ? Human
    : (Human && typeof Human.default === "function" ? Human.default : null);
  if (!HumanCtor) throw new Error("No se encontró el constructor de Human.");
  human = new HumanCtor({
    modelBasePath: "https://cdn.jsdelivr.net/npm/@vladmandic/human/models/",
    backend: "webgl",
    cacheSensitivity: 0,
    face: {
      enabled: true,
      description: { enabled: false },
      emotion: { enabled: true },
      iris: { enabled: false },
      // Experimento: RAF-DB "aligned" son recortes muy ajustados (100x100,
      // cara ocupando casi todo el cuadro, sin margen de fondo) — un patrón
      // atípico para un detector entrenado con fotos "in-the-wild". Se relaja
      // minConfidence y se habilita rotation por si el detector por defecto
      // estaba descartando estos casos de más. Valores por defecto de Human
      // no confirmados; esto es exploratorio, se ajusta según resultado.
      detector: { minConfidence: 0.1, rotation: true, maxDetected: 1 },
    },
    body: { enabled: false }, hand: { enabled: false }, object: { enabled: false },
    gesture: { enabled: false }, segmentation: { enabled: false },
  });
  await human.load();

  modelsReady = true;
  progressText.textContent = "Modelos listos. Selecciona la carpeta DATASET/test si aún no lo has hecho.";
  maybeEnableStart();
}

// --- Selección de carpeta ------------------------------------------------

folderInput.addEventListener("change", () => {
  const files = Array.from(folderInput.files || []);
  filesByLabel = {};
  for (const key of Object.keys(LABEL_TO_KEY)) filesByLabel[key] = [];

  let ignored = 0;
  for (const file of files) {
    const rel = file.webkitRelativePath || file.name;
    const parts = rel.split("/");
    const label = parts[parts.length - 2]; // carpeta inmediata contenedora
    const isImage = /\.(jpe?g|png)$/i.test(file.name);
    if (isImage && filesByLabel[label]) {
      filesByLabel[label].push(file);
    } else {
      ignored++;
    }
  }

  const totalFound = Object.values(filesByLabel).reduce((a, arr) => a + arr.length, 0);
  if (totalFound === 0) {
    folderStatus.textContent = "No se encontraron subcarpetas 1–7 con imágenes en la carpeta elegida. " +
      "Verifica que seleccionaste DATASET/test (o DATASET/train).";
    classSummary.innerHTML = "";
    maybeEnableStart();
    return;
  }

  folderStatus.textContent = `${totalFound} imágenes encontradas en ${Object.keys(filesByLabel).filter(k => filesByLabel[k].length).length} clases` +
    (ignored ? ` (${ignored} archivos ignorados, no son .jpg/.png o no están en una subcarpeta 1-7).` : ".");

  classSummary.innerHTML = Object.keys(LABEL_TO_KEY).map((label) => {
    const key = LABEL_TO_KEY[label];
    const n = filesByLabel[label].length;
    return `<div class="class-chip"><span class="n">${n}</span>${EMOTIONS[key].emoji} ${EMOTIONS[key].es} (carpeta ${label})</div>`;
  }).join("");

  maybeEnableStart();
});

function maybeEnableStart() {
  const hasFiles = Object.values(filesByLabel).some((arr) => arr && arr.length);
  startBtn.disabled = !(modelsReady && hasFiles);
}

// --- Utilidades ------------------------------------------------------------

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    // Debe seguir "renderizado" (no display:none) por si alguna librería lee
    // tamaño de layout en vez de naturalWidth/naturalHeight; se añade al DOM
    // fuera de pantalla y se remueve después de procesar cada imagen.
    img.style.position = "absolute";
    img.style.left = "-99999px";
    img.style.top = "0";
    document.body.appendChild(img);
    img.onload = () => resolve({ img, url });
    img.onerror = (e) => reject(e);
    img.src = url;
  });
}

let humanDebugLogsLeft = 5; // solo las primeras N para no saturar la consola

// --- Evaluación --------------------------------------------------------

// Estructura de resultados por motor: matriz de confusión (incluye columna
// "miss" para "no se detectó rostro"), contadores y tiempos de inferencia.
function newMotorAccumulator() {
  const confusion = {};
  for (const t of EMOTION_KEYS) {
    confusion[t] = {};
    for (const p of [...EMOTION_KEYS, "miss"]) confusion[t][p] = 0;
  }
  return { confusion, total: 0, detected: 0, correct: 0, totalMs: 0 };
}

function recordResult(acc, trueKey, predKey /* null si no detectó */, ms) {
  acc.total++;
  acc.totalMs += ms;
  if (predKey == null) {
    acc.confusion[trueKey].miss++;
  } else {
    acc.detected++;
    acc.confusion[trueKey][predKey]++;
    if (predKey === trueKey) acc.correct++;
  }
}

// Precision/recall/F1 por clase a partir de la matriz de confusión. Los
// "miss" cuentan como falso negativo para recall (nunca se predijeron esa
// clase), pero no entran en el denominador de precisión (nunca se contaron
// como una predicción de ninguna clase).
function classMetrics(acc) {
  const rows = [];
  for (const c of EMOTION_KEYS) {
    let tp = acc.confusion[c][c];
    let rowSum = 0; // instancias reales de c (para recall)
    for (const p of [...EMOTION_KEYS, "miss"]) rowSum += acc.confusion[c][p];
    let colSum = 0; // veces que el motor predijo c (para precisión)
    for (const t of EMOTION_KEYS) colSum += acc.confusion[t][c];
    const precision = colSum > 0 ? tp / colSum : 0;
    const recall = rowSum > 0 ? tp / rowSum : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    rows.push({ key: c, n: rowSum, precision, recall, f1 });
  }
  return rows;
}

async function runEvaluation() {
  cancelRequested = false;
  rawResults = [];
  startBtn.disabled = true;
  cancelBtn.disabled = false;
  csvBtn.disabled = true;
  resultsSection.innerHTML = "";

  const perClass = Math.max(0, parseInt(perClassInput.value, 10) || 0);
  const runFaceapi = chkFaceapi.checked;
  const runFusion = chkFusion.checked;
  const runHuman = chkHuman.checked;
  const exportBlendshapes = chkBlendshapes.checked;
  blendshapeRows = [];
  blendshapeColumns = null;
  blendshapesCsvBtn.disabled = true;

  // Construye la lista de imágenes a procesar: shuffle dentro de cada clase
  // para no sesgar por el orden en que el sistema de archivos las entregó,
  // luego recorta a perClass (0 = todas).
  let imageList = [];
  for (const label of Object.keys(LABEL_TO_KEY)) {
    let files = shuffle(filesByLabel[label] || []);
    if (perClass > 0) files = files.slice(0, perClass);
    for (const file of files) imageList.push({ file, trueKey: LABEL_TO_KEY[label] });
  }
  imageList = shuffle(imageList); // orden de procesamiento también aleatorio

  const total = imageList.length;
  const acc = {
    faceapi: runFaceapi ? newMotorAccumulator() : null,
    fusion: runFusion ? newMotorAccumulator() : null,
    human: runHuman ? newMotorAccumulator() : null,
  };

  const t0 = performance.now();
  for (let i = 0; i < total; i++) {
    if (cancelRequested) break;
    const { file, trueKey } = imageList[i];

    let url = null, imgEl = null;
    try {
      const loaded = await loadImageFile(file);
      imgEl = loaded.img;
      url = loaded.url;
    } catch (e) {
      console.error("No se pudo cargar imagen:", file.name, e);
      continue;
    }

    if (runFaceapi) {
      const tStart = performance.now();
      let predKey = null, probs = null;
      try {
        const r = await faceapi.detectSingleFace(imgEl, detectorOpts).withFaceLandmarks().withFaceExpressions();
        if (r) { probs = r.expressions; predKey = dominantOf(probs).key; }
      } catch (e) { console.error("Error face-api:", file.name, e); }
      const ms = performance.now() - tStart;
      recordResult(acc.faceapi, trueKey, predKey, ms);
      rawResults.push(logRow(file.name, trueKey, "faceExpressionNet", predKey, probs, ms));
    }

    // FaceMesh se calcula una sola vez por imagen y se reutiliza tanto para
    // la clasificación de emotion-fusion.js como para el export de
    // blendshapes crudos — evita correr MediaPipe dos veces por imagen.
    let mesh = null;
    if (runFusion || exportBlendshapes) {
      try {
        const fm = window.FaceMeshImageEngine.detect(imgEl);
        mesh = fm.faces[0] || null;
      } catch (e) { console.error("Error FaceMesh:", file.name, e); }
    }

    if (runFusion) {
      const tStart = performance.now();
      let predKey = null, probs = null;
      try {
        if (mesh) { probs = window.EmotionFusion.classify(mesh.blendshapes, null, "cliente"); predKey = dominantOf(probs).key; }
      } catch (e) { console.error("Error emotion-fusion:", file.name, e); }
      const ms = performance.now() - tStart;
      recordResult(acc.fusion, trueKey, predKey, ms);
      rawResults.push(logRow(file.name, trueKey, "emotion-fusion.js", predKey, probs, ms));
    }

    if (exportBlendshapes && mesh && mesh.blendshapes && mesh.blendshapes.length) {
      if (!blendshapeColumns) {
        blendshapeColumns = mesh.blendshapes.map((c) => c.categoryName);
      }
      const row = { file: file.name, trueKey, trueEs: EMOTIONS[trueKey].es };
      for (const c of mesh.blendshapes) row[c.categoryName] = c.score;
      blendshapeRows.push(row);
    }

    if (runHuman) {
      const tStart = performance.now();
      let predKey = null, probs = null;
      try {
        const result = await human.detect(imgEl);
        if (humanDebugLogsLeft > 0) {
          humanDebugLogsLeft--;
          console.log(`[DEBUG Human] ${file.name} — naturalSize=${imgEl.naturalWidth}x${imgEl.naturalHeight} — result.face:`, result.face);
        }
        const face = (result.face && result.face[0]) || null;
        if (face) { probs = humanEmotionToProbs(face.emotion); predKey = dominantOf(probs).key; }
      } catch (e) {
        console.error("Error Human:", file.name, e);
        if (humanDebugLogsLeft > 0) { humanDebugLogsLeft--; console.error("[DEBUG Human] excepción completa:", e); }
      }
      const ms = performance.now() - tStart;
      recordResult(acc.human, trueKey, predKey, ms);
      rawResults.push(logRow(file.name, trueKey, "Human.face.emotion", predKey, probs, ms));
    }

    URL.revokeObjectURL(url);
    if (imgEl && imgEl.parentNode) imgEl.parentNode.removeChild(imgEl);

    if (i % 3 === 0 || i === total - 1) {
      const pct = Math.round(((i + 1) / total) * 100);
      progressFill.style.width = pct + "%";
      const elapsedS = (performance.now() - t0) / 1000;
      const rate = (i + 1) / elapsedS;
      const etaS = rate > 0 ? (total - i - 1) / rate : 0;
      progressText.textContent =
        `${i + 1}/${total} imágenes (${pct}%) · ${elapsedS.toFixed(0)}s transcurridos · ` +
        `ETA ~${etaS.toFixed(0)}s`;
    }
  }

  cancelBtn.disabled = true;
  startBtn.disabled = false;
  csvBtn.disabled = rawResults.length === 0;
  blendshapesCsvBtn.disabled = blendshapeRows.length === 0;
  progressText.textContent = cancelRequested
    ? `Cancelado. Se procesaron ${rawResults.length ? new Set(rawResults.map(r=>r.file)).size : 0} imágenes antes de detener — métricas calculadas sobre lo procesado.`
    : `Listo. ${total} imágenes procesadas.`;

  renderResults(acc, { faceapi: runFaceapi, fusion: runFusion, human: runHuman });
}

function logRow(fileName, trueKey, motor, predKey, probs, ms) {
  const row = {
    file: fileName,
    trueKey,
    trueEs: EMOTIONS[trueKey].es,
    motor,
    detected: predKey != null,
    predKey: predKey || "",
    predEs: predKey ? EMOTIONS[predKey].es : "",
    confidence: predKey && probs ? dominantOf(probs).confidence : null,
    entropy: probs ? entropyOf(probs) : null,
    ms,
  };
  return row;
}

// --- Render de resultados ------------------------------------------------

const MOTOR_LABELS = {
  faceapi: "faceExpressionNet (face-api.js)",
  fusion: "emotion-fusion.js (Blendshapes + voz, sin audio en dataset)",
  human: "Human.face.emotion (@vladmandic/human)",
};

function renderResults(acc, enabled) {
  let html = "";
  for (const motor of ["faceapi", "fusion", "human"]) {
    if (!enabled[motor] || !acc[motor]) continue;
    const a = acc[motor];
    const detRate = a.total > 0 ? a.detected / a.total : 0;
    const accOnDetected = a.detected > 0 ? a.correct / a.detected : 0;
    const accOverall = a.total > 0 ? a.correct / a.total : 0;
    const avgMs = a.total > 0 ? a.totalMs / a.total : 0;
    const cls = classMetrics(a);
    const macroF1 = cls.reduce((s, r) => s + r.f1, 0) / cls.length;

    html += `<section class="panel motor-result">
      <h3>${MOTOR_LABELS[motor]}</h3>
      <div class="summary-metrics">
        <div class="metric"><span class="m-label">Imágenes evaluadas</span><span class="m-val">${a.total}</span></div>
        <div class="metric"><span class="m-label">Detection rate</span><span class="m-val">${(detRate*100).toFixed(1)}%</span></div>
        <div class="metric"><span class="m-label">Accuracy (sobre detectados)</span><span class="m-val">${(accOnDetected*100).toFixed(1)}%</span></div>
        <div class="metric"><span class="m-label">Accuracy (total, miss=error)</span><span class="m-val">${(accOverall*100).toFixed(1)}%</span></div>
        <div class="metric"><span class="m-label">Macro F1</span><span class="m-val">${macroF1.toFixed(3)}</span></div>
        <div class="metric"><span class="m-label">Tiempo medio / imagen</span><span class="m-val">${avgMs.toFixed(0)} ms</span></div>
      </div>

      <table class="cls-table">
        <thead><tr><th>Emoción</th><th>n</th><th>Precisión</th><th>Recall</th><th>F1</th></tr></thead>
        <tbody>
          ${cls.map(r => `<tr><td>${EMOTIONS[r.key].emoji} ${EMOTIONS[r.key].es}</td><td>${r.n}</td>` +
            `<td>${(r.precision*100).toFixed(1)}%</td><td>${(r.recall*100).toFixed(1)}%</td><td>${r.f1.toFixed(3)}</td></tr>`).join("")}
        </tbody>
      </table>

      <p class="vc-note" style="margin-top:10px;">Matriz de confusión (filas = etiqueta real de RAF-DB, columnas = predicción del motor):</p>
      <div style="overflow-x:auto;">
        <table class="cm-table">
          <thead><tr><th>Real \\ Predicho</th>${[...EMOTION_KEYS, "miss"].map(k => `<th>${k === "miss" ? "Sin rostro" : EMOTIONS[k].emoji}</th>`).join("")}</tr></thead>
          <tbody>
            ${EMOTION_KEYS.map(t => `<tr><th>${EMOTIONS[t].emoji} ${EMOTIONS[t].es}</th>` +
              [...EMOTION_KEYS, "miss"].map(p => {
                const v = acc[motor].confusion[t][p];
                const cls2 = p === t ? "diag" : (p === "miss" ? "miss" : "");
                return `<td class="${cls2}">${v}</td>`;
              }).join("") + `</tr>`).join("")}
          </tbody>
        </table>
      </div>
    </section>`;
  }
  resultsSection.innerHTML = html;
}

// --- CSV crudo -------------------------------------------------------------

function downloadCSV() {
  if (rawResults.length === 0) return;
  const header = ["archivo", "etiqueta_real", "emocion_real", "motor", "detectado", "prediccion", "confianza", "entropia", "ms_inferencia"];
  const rows = rawResults.map((r) => [
    r.file, r.trueKey, r.trueEs, r.motor, r.detected ? "1" : "0", r.predEs,
    r.confidence != null ? r.confidence.toFixed(4) : "",
    r.entropy != null ? r.entropy.toFixed(4) : "",
    r.ms.toFixed(1),
  ].map((c) => `"${String(c).replace(/"/g, '""')}"`).join(","));
  const csv = [header.join(","), ...rows].join("\r\n");
  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  a.download = `evaluacion_rafdb_${stamp}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// CSV de entrenamiento: archivo, etiqueta real y las 52 columnas de
// blendshapes con su categoryName real de MediaPipe como encabezado (no un
// índice ciego), listo para leer con pandas y entrenar en sklearn.
function downloadBlendshapesCSV() {
  if (blendshapeRows.length === 0 || !blendshapeColumns) return;
  const header = ["archivo", "etiqueta_real", "emocion_real", ...blendshapeColumns];
  const rows = blendshapeRows.map((r) => {
    const cells = [
      r.file, r.trueKey, r.trueEs,
      ...blendshapeColumns.map((c) => (r[c] != null ? r[c].toFixed(6) : "")),
    ];
    return cells.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",");
  });
  const csv = [header.join(","), ...rows].join("\r\n");
  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  a.download = `blendshapes_rafdb_${stamp}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

startBtn.addEventListener("click", runEvaluation);
cancelBtn.addEventListener("click", () => { cancelRequested = true; });
csvBtn.addEventListener("click", downloadCSV);
blendshapesCsvBtn.addEventListener("click", downloadBlendshapesCSV);

window.addEventListener("load", () => {
  loadModels().catch((e) => {
    console.error("Error cargando modelos:", e);
    progressText.textContent = "Error al cargar modelos (ver consola, F12): " + e.message;
  });
});