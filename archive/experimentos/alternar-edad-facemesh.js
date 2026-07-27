window.addEventListener("load", () => {

  // ---------- Config ----------
  const FACEAPI_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
  const BUFFER_HZ_MS = 1000;
  const BUFFER_MAX = 30;
  const IQR_FREEZE_WIDTH = 6;
  const FREEZE_MIN_SAMPLES = 20;
  const UNFREEZE_DEVIATION = 10;
  const UNFREEZE_STREAK = 3;
  const MODEL_LOAD_TIMEOUT_MS = 20000;

  // ---------- DOM ----------
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const statusEl = document.getElementById("status");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const engTfdBtn = document.getElementById("engTfd");
  const engMeshBtn = document.getElementById("engMesh");
  const marginSlider = document.getElementById("marginSlider");
  const marginVal = document.getElementById("marginVal");
  const conditionInput = document.getElementById("conditionInput");
  const sweepValuesInput = document.getElementById("sweepValues");
  const sweepSecondsInput = document.getElementById("sweepSeconds");
  const sweepStartBtn = document.getElementById("sweepStartBtn");
  const sweepStopBtn = document.getElementById("sweepStopBtn");
  const sweepStatus = document.getElementById("sweepStatus");
  const csvBtn = document.getElementById("csvBtn");
  const clearBtn = document.getElementById("clearBtn");
  const logCount = document.getElementById("logCount");

  const ui = {
    tfd: {
      det: document.getElementById("tfdDet"),
      instant: document.getElementById("tfdInstant"),
      median: document.getElementById("tfdMedian"),
      iqr: document.getElementById("tfdIqr"),
      state: document.getElementById("tfdState"),
      rate: document.getElementById("tfdRate"),
    },
    mesh: {
      det: document.getElementById("meshDet"),
      instant: document.getElementById("meshInstant"),
      median: document.getElementById("meshMedian"),
      iqr: document.getElementById("meshIqr"),
      state: document.getElementById("meshState"),
      rate: document.getElementById("meshRate"),
    },
  };

  let stream = null;
  let drawLoopId = null;
  let sampleTimerId = null;
  let activeEngine = "tfd";
  let cropMargin = 0.25;
  let sweepStopRequested = false;
  let sweepRunning = false; // % de padding alrededor del box de FaceMesh

  marginSlider.addEventListener("input", () => {
    cropMargin = Number(marginSlider.value) / 100;
    marginVal.textContent = `${marginSlider.value}%`;
  });

  function setMargin(pct) {
    cropMargin = pct / 100;
    marginSlider.value = String(pct);
    marginVal.textContent = `${pct}%`;
  }

  async function runSweep() {
    if (!stream) {
      sweepStatus.textContent = "Inicia la cámara antes de arrancar el barrido.";
      return;
    }
    const values = sweepValuesInput.value
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => Number.isFinite(n) && n >= 0 && n <= 100);
    if (values.length === 0) {
      sweepStatus.textContent = "Lista de valores inválida. Usa algo como: 0,25,50,80,100";
      return;
    }
    const secondsPerStep = Math.max(5, Math.min(60, Number(sweepSecondsInput.value) || 15));

    if (!conditionInput.value.trim()) conditionInput.value = "barrido_margen";

    sweepRunning = true;
    sweepStopRequested = false;
    sweepStartBtn.disabled = true;
    sweepStopBtn.disabled = false;
    marginSlider.disabled = true;

    for (let i = 0; i < values.length; i++) {
      if (sweepStopRequested) break;
      setMargin(values[i]);
      for (let s = secondsPerStep; s > 0; s--) {
        if (sweepStopRequested) break;
        sweepStatus.textContent =
          `Margen ${values[i]}% — paso ${i + 1}/${values.length} — ${s}s restantes…`;
        await new Promise((r) => setTimeout(r, 1000));
      }
    }

    sweepRunning = false;
    sweepStartBtn.disabled = !stream;
    sweepStopBtn.disabled = true;
    marginSlider.disabled = false;

    if (sweepStopRequested) {
      sweepStatus.textContent = "Barrido detenido manualmente. Puedes descargar el CSV con lo que llevas.";
    } else {
      sweepStatus.textContent = "Barrido completo. Descargando CSV…";
      downloadCSV();
      sweepStatus.textContent = "Barrido completo. CSV descargado.";
    }
  }

  // Últimos resultados cacheados (fire-and-forget: detección en background,
  // el dibujo siempre usa el último resultado ya resuelto).
  const lastResult = { tfd: null, mesh: null }; // { box, age, gender }

  function newEngineState() {
    return {
      buffer: [], frozen: false, frozenValue: null,
      lastRawAge: null, deviationStreak: 0,
      samplesTotal: 0, samplesDetected: 0,
    };
  }
  const engineState = { tfd: newEngineState(), mesh: newEngineState() };
  window.__debug = { lastResult, engineState };

  const csvRows = [];

  // ---------- Utilidades ----------
  function setStatus(msg) {
    statusEl.textContent = msg;
    statusEl.style.display = msg ? "block" : "none";
  }
  function withTimeout(promise, ms, label) {
    return Promise.race([
      promise,
      new Promise((_, rej) => setTimeout(() => rej(new Error(`Timeout cargando ${label}`)), ms)),
    ]);
  }
  function median(arr) {
    const s = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  }
  function quartile(arr, q) {
    const s = [...arr].sort((a, b) => a - b);
    const pos = (s.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    return s[base + 1] !== undefined ? s[base] + rest * (s[base + 1] - s[base]) : s[base];
  }

  // ---------- Carga de modelos ----------
  async function loadModels() {
    setStatus("Cargando modelos (face-api.js + FaceMesh)…");

    const faceapiLoad = (async () => {
      await faceapi.nets.tinyFaceDetector.loadFromUri(FACEAPI_MODEL_URL);
      await faceapi.nets.ageGenderNet.loadFromUri(FACEAPI_MODEL_URL);
    })();

    // window.FaceMeshEngine viene del módulo ES facemesh.js (I1), ya cargado
    // como script type="module" antes de este archivo. Para el evento "load"
    // ya debería existir, pero se verifica por si acaso.
    if (!window.FaceMeshEngine) {
      throw new Error("window.FaceMeshEngine no está disponible — ¿se cargó facemesh.js?");
    }
    const faceMeshLoad = window.FaceMeshEngine.load(setStatus);

    await withTimeout(Promise.all([faceapiLoad, faceMeshLoad]), MODEL_LOAD_TIMEOUT_MS, "modelos");
    setStatus("Modelos listos. Presiona «Iniciar cámara».");
    startBtn.disabled = false;
  }

  // ---------- Cámara ----------
  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      overlay.width = video.videoWidth || 640;
      overlay.height = video.videoHeight || 480;
      startBtn.disabled = true;
      stopBtn.disabled = false;
      sweepStartBtn.disabled = false;
      setStatus("");

      runDetectionLoops();
      drawLoop();
      sampleTimerId = setInterval(sampleBuffers, BUFFER_HZ_MS);
    } catch (err) {
      console.error(err);
      setStatus("No se pudo acceder a la cámara: " + err.message);
    }
  }

  function stopCamera() {
    sweepStopRequested = true;
    if (drawLoopId) cancelAnimationFrame(drawLoopId);
    drawLoopId = null;
    if (sampleTimerId) clearInterval(sampleTimerId);
    sampleTimerId = null;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    sweepStartBtn.disabled = true;
    sweepStopBtn.disabled = true;
    setStatus("Cámara detenida.");
  }

  // ---------- Detección: pipeline A (TinyFaceDetector → ageGenderNet) ----------
  const faceapiDetectorOpts = new faceapi.TinyFaceDetectorOptions({
    inputSize: 320, scoreThreshold: 0.5,
  });

  async function tfdLoopStep() {
    if (!stream) return;
    try {
      const results = await faceapi
        .detectAllFaces(video, faceapiDetectorOpts)
        .withAgeAndGender();
      if (results.length) {
        const r = results.reduce((m, x) =>
          x.detection.box.area > m.detection.box.area ? x : m, results[0]);
        lastResult.tfd = { box: r.detection.box, age: r.age, gender: r.gender };
      } else {
        lastResult.tfd = null;
      }
    } catch (e) {
      console.error("tfd loop error", e);
      lastResult.tfd = null;
    }
    if (stream) setTimeout(tfdLoopStep, 0);
  }

  // ---------- Detección: pipeline B (FaceMesh → recorte → ageGenderNet) ----------
  const cropCanvas = document.createElement("canvas");
  const cropCtx = cropCanvas.getContext("2d");

  async function meshLoopStep() {
    if (!stream) return;
    try {
      const result = window.FaceMeshEngine.detect(video);
      if (result.faces && result.faces.length) {
        const f = result.faces.reduce((m, x) => (x.box.area > m.box.area ? x : m), result.faces[0]);
        const vw = video.videoWidth, vh = video.videoHeight;

        // Aplica margen y recorta al frame de video (sin display:none, se
        // dibuja directo del <video> a un canvas offscreen).
        const mw = f.box.width * cropMargin;
        const mh = f.box.height * cropMargin;
        const x = Math.max(0, f.box.x - mw / 2);
        const y = Math.max(0, f.box.y - mh / 2);
        const w = Math.min(vw - x, f.box.width + mw);
        const h = Math.min(vh - y, f.box.height + mh);

        if (w > 10 && h > 10) {
          cropCanvas.width = Math.round(w);
          cropCanvas.height = Math.round(h);
          cropCtx.drawImage(video, x, y, w, h, 0, 0, w, h);
          const pred = await faceapi.nets.ageGenderNet.predictAgeAndGender(cropCanvas);
          const p = Array.isArray(pred) ? pred[0] : pred;
          lastResult.mesh = p
            ? { box: f.box, age: p.age, gender: p.gender }
            : null;
        } else {
          lastResult.mesh = null;
        }
      } else {
        lastResult.mesh = null;
      }
    } catch (e) {
      console.error("mesh loop error", e);
      lastResult.mesh = null;
    }
    if (stream) setTimeout(meshLoopStep, 0);
  }

  function runDetectionLoops() {
    tfdLoopStep();
    meshLoopStep();
  }

  // ---------- Dibujo ----------
  function drawLoop() {
    if (!stream) return;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const r = lastResult[activeEngine];
    if (r) {
      ctx.strokeStyle = activeEngine === "tfd" ? "#6c8cff" : "#4ade80";
      ctx.lineWidth = 3;
      ctx.strokeRect(r.box.x, r.box.y, r.box.width, r.box.height);
    }
    drawLoopId = requestAnimationFrame(drawLoop);
  }

  // ---------- Buffer 1Hz + mediana/IQR ----------
  function sampleBuffers() {
    sampleOneEngine("tfd");
    sampleOneEngine("mesh");
    logSample();
  }

  function sampleOneEngine(engine) {
    const st = engineState[engine];
    const r = lastResult[engine];
    const age = r ? Math.round(r.age) : null;

    st.samplesTotal++;
    if (age != null) st.samplesDetected++;
    st.lastRawAge = age;

    if (st.frozen && age != null) {
      if (Math.abs(age - st.frozenValue.median) > UNFREEZE_DEVIATION) {
        st.deviationStreak++;
      } else {
        st.deviationStreak = 0;
      }
      if (st.deviationStreak >= UNFREEZE_STREAK) {
        st.frozen = false; st.frozenValue = null; st.buffer = []; st.deviationStreak = 0;
      }
    } else {
      st.deviationStreak = 0;
    }

    if (age != null && !st.frozen) {
      st.buffer.push(age);
      if (st.buffer.length > BUFFER_MAX) st.buffer.shift();
    }

    let med = null, q1 = null, q3 = null, iqrW = null;
    if (st.buffer.length >= 3) {
      med = median(st.buffer);
      q1 = quartile(st.buffer, 0.25);
      q3 = quartile(st.buffer, 0.75);
      iqrW = q3 - q1;
      if (!st.frozen && iqrW <= IQR_FREEZE_WIDTH && st.buffer.length >= FREEZE_MIN_SAMPLES) {
        st.frozen = true;
        st.frozenValue = { median: med, q1, q3 };
      }
    }

    renderEngine(engine, { age, med, q1, q3, st });
  }

  function renderEngine(engine, { age, med, q1, q3, st }) {
    const u = ui[engine];
    u.det.textContent = age != null ? "sí" : "no";
    u.det.className = "v" + (age != null ? "" : " miss");
    u.instant.textContent = age != null ? `${age} años` : "—";

    const shown = st.frozen ? st.frozenValue : { median: med, q1, q3 };
    u.median.textContent = shown.median != null ? `${Math.round(shown.median)} años` : "—";
    u.iqr.textContent = (shown.q1 != null && shown.q3 != null)
      ? `[${Math.round(shown.q1)}–${Math.round(shown.q3)}]` : "—";
    u.state.textContent = st.frozen
      ? (st.deviationStreak > 0 ? `⏸ congelado (desv. ${st.deviationStreak}/${UNFREEZE_STREAK})` : "⏸ congelado")
      : "acumulando…";
    u.state.className = "v" + (st.frozen ? " frozen" : "");
    const rate = st.samplesTotal ? (st.samplesDetected / st.samplesTotal * 100) : 0;
    u.rate.textContent = `${rate.toFixed(1)}% (${st.samplesDetected}/${st.samplesTotal})`;
    u.rate.className = "v" + (rate < 92 ? " miss" : "");
  }

  // ---------- CSV ----------
  function logSample() {
    const t = engineState.tfd, m = engineState.mesh;
    csvRows.push({
      time: new Date(),
      tfd_instant: t.lastRawAge,
      tfd_median: t.frozen ? t.frozenValue.median : (t.buffer.length >= 3 ? median(t.buffer) : null),
      tfd_frozen: t.frozen,
      mesh_instant: m.lastRawAge,
      mesh_median: m.frozen ? m.frozenValue.median : (m.buffer.length >= 3 ? median(m.buffer) : null),
      mesh_frozen: m.frozen,
      crop_margin_pct: Math.round(cropMargin * 100),
      condicion: (conditionInput.value || "sin_etiquetar").trim(),
    });
    logCount.textContent = csvRows.length === 1 ? "1 registro" : `${csvRows.length} registros`;
    csvBtn.disabled = csvRows.length === 0;
    clearBtn.disabled = csvRows.length === 0;
  }

  function downloadCSV() {
    if (csvRows.length === 0) return;
    const header = [
      "timestamp_iso", "tfd_edad_instantanea", "tfd_mediana_buffer", "tfd_congelado",
      "mesh_edad_instantanea", "mesh_mediana_buffer", "mesh_congelado", "margen_recorte_pct", "condicion",
    ];
    const rows = csvRows.map((r) => [
      r.time.toISOString(),
      r.tfd_instant ?? "",
      r.tfd_median != null ? r.tfd_median.toFixed(1) : "",
      r.tfd_frozen ? "1" : "0",
      r.mesh_instant ?? "",
      r.mesh_median != null ? r.mesh_median.toFixed(1) : "",
      r.mesh_frozen ? "1" : "0",
      r.crop_margin_pct,
      r.condicion,
    ].map((c) => `"${String(c).replace(/"/g, '""')}"`).join(","));
    const csv = [header.join(","), ...rows].join("\r\n");
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    const condSlug = (conditionInput.value || "sin_etiquetar").trim().toLowerCase().replace(/[^a-z0-9]+/g, "_");
    a.href = url;
    a.download = `comparacion_edad_facemesh_${condSlug}_${stamp}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function clearLog() {
    csvRows.length = 0;
    logCount.textContent = "0 registros";
    csvBtn.disabled = true;
    clearBtn.disabled = true;
  }

  // ---------- Eventos ----------
  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", stopCamera);
  csvBtn.addEventListener("click", downloadCSV);
  clearBtn.addEventListener("click", clearLog);
  engTfdBtn.addEventListener("click", () => {
    activeEngine = "tfd"; engTfdBtn.classList.add("active"); engMeshBtn.classList.remove("active");
  });
  engMeshBtn.addEventListener("click", () => {
    activeEngine = "mesh"; engMeshBtn.classList.add("active"); engTfdBtn.classList.remove("active");
  });
  sweepStartBtn.addEventListener("click", runSweep);
  sweepStopBtn.addEventListener("click", () => { sweepStopRequested = true; });

  // ---------- Arranque ----------
  if (typeof faceapi === "undefined") {
    setStatus("No se pudo cargar face-api.js (¿sin conexión?).");
    return;
  }
  loadModels().catch((err) => {
    console.error(err);
    setStatus("Error al cargar modelos: " + err.message);
  });
});
