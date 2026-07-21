window.addEventListener("load", () => {

  // ---------- Config ----------
  const FACEAPI_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
  const BUFFER_HZ_MS = 1000;       // muestreo del buffer de edad a 1 Hz
  const BUFFER_MAX = 30;           // ventana deslizante (30 s)
  const IQR_FREEZE_WIDTH = 6;      // años; congelar cuando el IQR cae bajo esto
  const FREEZE_MIN_SAMPLES = 20;   // antes: 8. Muy pocas muestras dieron falsa
                                    // estabilidad (ver sesión de prueba real:
                                    // 8 muestras de Human coincidieron por un
                                    // bug de caché, no por convergencia genuina)
  const UNFREEZE_DEVIATION = 10;   // años; si tras congelar llega una muestra
                                    // que se desvía más de esto, es candidata
                                    // a descongelar (ver UNFREEZE_STREAK)
  const UNFREEZE_STREAK = 3;       // muestras CONSECUTIVAS desviadas requeridas
                                    // para descongelar de verdad. Con 1 sola
                                    // muestra el buffer quedaba "nervioso"
                                    // (congela/descongela en cada outlier
                                    // puntual); exigir una racha filtra ruido
                                    // de un solo frame vs. un cambio genuino
                                    // (p.ej. la persona se fue de cámara).
  const REPEAT_WARN_COUNT = 5;     // alerta si Human repite el mismo valor N veces seguidas
  const MODEL_LOAD_TIMEOUT_MS = 20000;

  // ---------- DOM ----------
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const statusEl = document.getElementById("status");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const engFaceapiBtn = document.getElementById("engFaceapi");
  const engHumanBtn = document.getElementById("engHuman");
  const csvBtn = document.getElementById("csvBtn");
  const clearBtn = document.getElementById("clearBtn");
  const logCount = document.getElementById("logCount");
  const huRepeatWarn = document.getElementById("huRepeatWarn");

  const ui = {
    faceapi: {
      instant: document.getElementById("faInstant"),
      gender: document.getElementById("faGender"),
      median: document.getElementById("faMedian"),
      iqr: document.getElementById("faIqr"),
      iqrW: document.getElementById("faIqrW"),
      state: document.getElementById("faState"),
      n: document.getElementById("faN"),
    },
    human: {
      instant: document.getElementById("huInstant"),
      gender: document.getElementById("huGender"),
      median: document.getElementById("huMedian"),
      iqr: document.getElementById("huIqr"),
      iqrW: document.getElementById("huIqrW"),
      state: document.getElementById("huState"),
      n: document.getElementById("huN"),
    },
  };

  let stream = null;
  let drawLoopId = null;
  let sampleTimerId = null;
  let activeEngine = "faceapi"; // solo afecta qué se dibuja en el overlay
  let humanInstance = null;
  let humanConfig = null; // se reusa también como override explícito en cada .detect()

  // Últimos resultados cacheados (fire-and-forget: la detección corre en
  // background, el dibujo siempre usa el último resultado ya calculado).
  const lastResult = {
    faceapi: null, // { box, age, gender, genderProbability }
    human: null,
  };

  // Estado de buffers por motor.
  function newEngineState() {
    return {
      buffer: [],          // años, muestreados a 1 Hz
      frozen: false,
      frozenValue: null,   // { median, q1, q3 }
      lastRawAge: null,
      repeatCount: 0,
      deviationStreak: 0,  // muestras consecutivas desviadas del valor congelado
    };
  }
  const engineState = { faceapi: newEngineState(), human: newEngineState() };

  // Para depuración manual desde la consola del navegador, p.ej.:
  //   copy(JSON.stringify(window.__debug.lastResult.human, null, 2))
  // (los objetos se mutan por referencia, así que exponerlos una vez basta;
  // siempre reflejan el estado más reciente)
  window.__debug = { lastResult, engineState, get humanInstance() { return humanInstance; } };

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
    setStatus("Cargando modelos (face-api.js + Human)…");

    const faceapiLoad = (async () => {
      await faceapi.nets.tinyFaceDetector.loadFromUri(FACEAPI_MODEL_URL);
      await faceapi.nets.ageGenderNet.loadFromUri(FACEAPI_MODEL_URL);
    })();

    // Human requiere chequeo defensivo del constructor (bundle IIFE).
    const HumanCtor = typeof Human === "function" ? Human : (Human && Human.default);
    if (!HumanCtor) throw new Error("No se pudo resolver el constructor de Human");

    const humanLoad = (async () => {
      humanConfig = {
        modelBasePath: "https://cdn.jsdelivr.net/npm/@vladmandic/human/models/",
        face: {
          enabled: true,
          detector: {
            rotation: false,
            // CLAVE: aunque cacheSensitivity=0 evita reusar el análisis de
            // edad/género, el detector de la CAJA facial tiene su propio
            // mecanismo de reuso independiente (por defecto reusa la misma
            // caja hasta 99 frames o 2500ms). Si la caja no se recalcula,
            // el análisis de edad tampoco se vuelve a correr de verdad.
            // Hay que apagar los dos por separado.
            skipFrames: 0,
            skipTime: 0,
          },
          mesh: { enabled: false },
          iris: { enabled: false },
          description: { enabled: true },  // trae age/gender
          emotion: { enabled: false },
        },
        body: { enabled: false },
        hand: { enabled: false },
        gesture: { enabled: false },
        filter: { enabled: false },
        cacheSensitivity: 0, // evita reusar el análisis de edad/género en sí
      };
      humanInstance = new HumanCtor(humanConfig);
      await humanInstance.load();
      // Diagnóstico: volcar la config REAL ya mergeada por Human, para
      // confirmar sin adivinar en qué ruta terminó cacheSensitivity.
      console.log("Human config real tras load():", JSON.parse(JSON.stringify(humanInstance.config)));
    })();

    await withTimeout(Promise.all([faceapiLoad, humanLoad]), MODEL_LOAD_TIMEOUT_MS, "modelos");

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
      setStatus("");

      runDetectionLoops();   // fire-and-forget, independientes del dibujo
      drawLoop();             // dibujo síncrono, usa lastResult cacheado
      sampleTimerId = setInterval(sampleBuffers, BUFFER_HZ_MS);
    } catch (err) {
      console.error(err);
      setStatus("No se pudo acceder a la cámara: " + err.message);
    }
  }

  function stopCamera() {
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
    setStatus("Cámara detenida.");
  }

  // ---------- Detección (background, fire-and-forget) ----------
  // IMPORTANTE: nunca usar await dentro de requestAnimationFrame junto con
  // dibujo de canvas. Cada motor corre su propio ciclo asíncrono
  // desacoplado; el dibujo siempre lee el último resultado ya resuelto.
  const faceapiDetectorOpts = new faceapi.TinyFaceDetectorOptions({
    inputSize: 320,
    scoreThreshold: 0.5,
  });

  async function faceapiLoopStep() {
    if (!stream) return;
    try {
      const results = await faceapi
        .detectAllFaces(video, faceapiDetectorOpts)
        .withAgeAndGender();
      if (results.length) {
        const r = results.reduce((m, x) =>
          x.detection.box.area > m.detection.box.area ? x : m, results[0]);
        lastResult.faceapi = {
          box: r.detection.box,
          age: r.age,
          gender: r.gender,
          genderProbability: r.genderProbability,
        };
      } else {
        lastResult.faceapi = null;
      }
    } catch (e) {
      console.error("faceapi loop error", e);
    }
    if (stream) setTimeout(faceapiLoopStep, 0);
  }

  async function humanLoopStep() {
    if (!stream || !humanInstance) return;
    try {
      // Se pasa humanConfig explícitamente en cada llamada (no solo en el
      // constructor) para eliminar cualquier duda de que cacheSensitivity/
      // skipFrames/skipTime no se hayan aplicado bien en la config interna.
      const res = await humanInstance.detect(video, humanConfig);
      if (res.face && res.face.length) {
        const f = res.face.reduce((m, x) =>
          (x.boxRaw ? x.boxRaw[2] * x.boxRaw[3] : 0) > (m.boxRaw ? m.boxRaw[2] * m.boxRaw[3] : 0) ? x : m,
          res.face[0]);
        lastResult.human = {
          box: f.box, // [x, y, w, h] en px
          age: f.age,
          gender: f.gender,
          genderScore: f.genderScore,
        };
      } else {
        lastResult.human = null;
      }
    } catch (e) {
      console.error("human loop error", e);
    }
    if (stream) setTimeout(humanLoopStep, 0);
  }

  function runDetectionLoops() {
    faceapiLoopStep();
    humanLoopStep();
  }

  // ---------- Dibujo (siempre síncrono, usa cache) ----------
  function drawLoop() {
    if (!stream) return;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const r = lastResult[activeEngine];
    if (r) {
      let box;
      if (activeEngine === "faceapi") {
        box = r.box; // {x,y,width,height}
      } else {
        box = { x: r.box[0], y: r.box[1], width: r.box[2], height: r.box[3] };
      }
      ctx.strokeStyle = activeEngine === "faceapi" ? "#6c8cff" : "#4ade80";
      ctx.lineWidth = 3;
      ctx.strokeRect(box.x, box.y, box.width, box.height);
    }
    drawLoopId = requestAnimationFrame(drawLoop);
  }

  // ---------- Buffer 1Hz + mediana/IQR ----------
  function sampleBuffers() {
    sampleOneEngine("faceapi");
    sampleOneEngine("human");
    logSample();
  }

  function sampleOneEngine(engine) {
    const st = engineState[engine];
    const r = lastResult[engine];
    const age = r ? Math.round(r.age) : null;

    // Detección de repetición exacta (relevante sobre todo para Human / cacheSensitivity).
    if (age != null && age === st.lastRawAge) {
      st.repeatCount++;
    } else {
      st.repeatCount = 0;
    }
    st.lastRawAge = age;

    // Descongelado: si el buffer ya estaba "congelado" pero llegan varias
    // lecturas SEGUIDAS que se alejan del valor congelado, el congelado fue
    // prematuro (o la situación cambió de verdad) y se reabre el buffer.
    // Exigir una racha (UNFREEZE_STREAK) en vez de una sola muestra evita que
    // un outlier puntual (parpadeo, frame borroso) tire todo el progreso.
    if (st.frozen && age != null) {
      if (Math.abs(age - st.frozenValue.median) > UNFREEZE_DEVIATION) {
        st.deviationStreak++;
      } else {
        st.deviationStreak = 0;
      }
      if (st.deviationStreak >= UNFREEZE_STREAK) {
        st.frozen = false;
        st.frozenValue = null;
        st.buffer = [];
        st.deviationStreak = 0;
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

    renderEngine(engine, { age, med, q1, q3, iqrW, st });
  }

  function renderEngine(engine, { age, med, q1, q3, iqrW, st }) {
    const u = ui[engine];
    u.instant.textContent = age != null ? `${age} años` : "—";
    if (engine === "faceapi" && lastResult.faceapi) {
      u.gender.textContent = lastResult.faceapi.gender === "male" ? "Hombre" : "Mujer";
    } else if (engine === "human" && lastResult.human) {
      const g = (lastResult.human.gender || "").toLowerCase();
      u.gender.textContent = g === "male" ? "Hombre" : g === "female" ? "Mujer" : (lastResult.human.gender || "—");
    } else {
      u.gender.textContent = "—";
    }

    const shown = st.frozen ? st.frozenValue : { median: med, q1, q3 };
    u.median.textContent = shown.median != null ? `${Math.round(shown.median)} años` : "—";
    u.iqr.textContent = (shown.q1 != null && shown.q3 != null)
      ? `[${Math.round(shown.q1)}–${Math.round(shown.q3)}]` : "—";
    const width = (shown.q1 != null && shown.q3 != null) ? shown.q3 - shown.q1 : null;
    u.iqrW.textContent = width != null ? `${width.toFixed(1)} años` : "—";
    u.state.textContent = st.frozen
      ? (st.deviationStreak > 0
          ? `⏸ congelado (desviación ${st.deviationStreak}/${UNFREEZE_STREAK})`
          : "⏸ congelado (precisión suficiente)")
      : "acumulando…";
    u.state.className = "v" + (st.frozen ? " frozen" : "");
    u.n.textContent = String(st.buffer.length);

    if (engine === "human") {
      huRepeatWarn.style.display = st.repeatCount >= REPEAT_WARN_COUNT ? "block" : "none";
    }
  }

  // ---------- CSV ----------
  function logSample() {
    const fa = engineState.faceapi;
    const hu = engineState.human;
    csvRows.push({
      t: new Date(),
      fa_instant: fa.lastRawAge,
      fa_median: fa.frozen ? fa.frozenValue.median : (fa.buffer.length >= 3 ? median(fa.buffer) : null),
      fa_frozen: fa.frozen,
      fa_n: fa.buffer.length,
      fa_deviation_streak: fa.deviationStreak,
      hu_instant: hu.lastRawAge,
      hu_median: hu.frozen ? hu.frozenValue.median : (hu.buffer.length >= 3 ? median(hu.buffer) : null),
      hu_frozen: hu.frozen,
      hu_n: hu.buffer.length,
      hu_repeat_count: hu.repeatCount,
      hu_deviation_streak: hu.deviationStreak,
    });
    logCount.textContent = csvRows.length === 1 ? "1 registro" : `${csvRows.length} registros`;
    csvBtn.disabled = csvRows.length === 0;
    clearBtn.disabled = csvRows.length === 0;
  }

  function downloadCSV() {
    if (csvRows.length === 0) return;
    const header = [
      "timestamp_iso", "fa_edad_instantanea", "fa_mediana_buffer", "fa_congelado", "fa_n_muestras", "fa_racha_desviacion",
      "hu_edad_instantanea", "hu_mediana_buffer", "hu_congelado", "hu_n_muestras", "hu_repeticiones_consecutivas", "hu_racha_desviacion",
    ];
    const rows = csvRows.map((r) => [
      r.t.toISOString(),
      r.fa_instant ?? "",
      r.fa_median != null ? r.fa_median.toFixed(1) : "",
      r.fa_frozen ? "1" : "0",
      r.fa_n,
      r.fa_deviation_streak,
      r.hu_instant ?? "",
      r.hu_median != null ? r.hu_median.toFixed(1) : "",
      r.hu_frozen ? "1" : "0",
      r.hu_n,
      r.hu_repeat_count,
      r.hu_deviation_streak,
    ].map((c) => `"${String(c).replace(/"/g, '""')}"`).join(","));
    const csv = [header.join(","), ...rows].join("\r\n");
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    a.href = url;
    a.download = `comparacion_edad_${stamp}.csv`;
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
  engFaceapiBtn.addEventListener("click", () => {
    activeEngine = "faceapi";
    engFaceapiBtn.classList.add("active");
    engHumanBtn.classList.remove("active");
  });
  engHumanBtn.addEventListener("click", () => {
    activeEngine = "human";
    engHumanBtn.classList.add("active");
    engFaceapiBtn.classList.remove("active");
  });

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
