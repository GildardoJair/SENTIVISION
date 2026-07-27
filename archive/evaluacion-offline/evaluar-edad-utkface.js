// Evaluación offline: TinyFaceDetector→ageGenderNet (actual) vs
// @vladmandic/human vs FaceMesh→recorte→ageGenderNet (propuesta) contra
// UTKFace. Todo corre en el navegador sobre archivos locales — ninguna
// imagen sale de la máquina.
//
// Formato esperado de archivo UTKFace: [edad]_[genero]_[raza]_[timestamp].jpg
// (carpeta "Aligned & Cropped" de Kaggle/jangedoo, plana, sin subcarpetas).
//
// El pipeline "mesh" usa facemesh-image.js (MediaPipe en modo IMAGE), NO
// facemesh.js (modo VIDEO, requiere <video> + timestamps crecientes — no
// sirve para procesar miles de imágenes sueltas).

window.addEventListener("load", () => {

  const FACEAPI_MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
  const YIELD_EVERY = 1;
  const ENGINES = ["fa", "hu", "mesh"];
  const ENGINE_LABEL = { fa: "face-api", hu: "Human", mesh: "FaceMesh→recorte" };

  const AGE_BINS = [
    { label: "0–12", min: 0, max: 12 },
    { label: "13–19", min: 13, max: 19 },
    { label: "20–29", min: 20, max: 29 },
    { label: "30–39", min: 30, max: 39 },
    { label: "40–49", min: 40, max: 49 },
    { label: "50–59", min: 50, max: 59 },
    { label: "60+", min: 60, max: Infinity },
  ];
  function binFor(age) {
    return AGE_BINS.find((b) => age >= b.min && age <= b.max) || null;
  }

  // ---------- DOM ----------
  const folderInput = document.getElementById("folderInput");
  const folderStatus = document.getElementById("folderStatus");
  const sampleSizeSel = document.getElementById("sampleSize");
  const shuffleCheck = document.getElementById("shuffleCheck");
  const meshMarginInput = document.getElementById("meshMargin");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const csvBtn = document.getElementById("csvBtn");
  const progressBar = document.getElementById("progressBar");
  const runStatus = document.getElementById("runStatus");
  const preview = document.getElementById("preview");
  const previewImg = document.getElementById("previewImg");
  const previewLabel = document.getElementById("previewLabel");

  const statProcessed = document.getElementById("statProcessed");
  const statTime = document.getElementById("statTime");
  const statMaeFa = document.getElementById("statMaeFa");
  const statMaeHu = document.getElementById("statMaeHu");
  const statMaeMesh = document.getElementById("statMaeMesh");
  const statDetRates = document.getElementById("statDetRates");
  const binTableBody = document.getElementById("binTableBody");
  const genderTableBody = document.getElementById("genderTableBody");

  // ---------- Estado ----------
  let allFiles = [];
  let humanInstance = null, humanConfig = null;
  let faceMeshImageReady = false;
  let modelsReady = false;
  let running = false, stopRequested = false;
  let startTime = 0, timeTickId = null;

  const results = [];
  const totals = {
    processed: 0,
    det: { fa: 0, hu: 0, mesh: 0 },
    ageN: { fa: 0, hu: 0, mesh: 0 },
    ageSumAbs: { fa: 0, hu: 0, mesh: 0 },
    ageSumSq: { fa: 0, hu: 0, mesh: 0 },
    genderN: { fa: 0, hu: 0, mesh: 0 },
    genderCorrect: { fa: 0, hu: 0, mesh: 0 },
  };
  const binTotals = {};
  for (const b of AGE_BINS) {
    binTotals[b.label] = {
      n: 0,
      ageSumErr: { fa: 0, hu: 0, mesh: 0 },
      ageSumAbs: { fa: 0, hu: 0, mesh: 0 },
      genderN: { fa: 0, hu: 0, mesh: 0 },
      genderCorrect: { fa: 0, hu: 0, mesh: 0 },
    };
  }

  // ---------- Carga de modelos ----------
  function withTimeout(promise, ms, label) {
    return Promise.race([
      promise,
      new Promise((_, rej) => setTimeout(() => rej(new Error(`Timeout cargando ${label}`)), ms)),
    ]);
  }

  async function loadModels() {
    runStatus.textContent = "Cargando modelos (face-api.js + Human + FaceMesh imagen)…";

    const faceapiLoad = Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(FACEAPI_MODEL_URL),
      faceapi.nets.ageGenderNet.loadFromUri(FACEAPI_MODEL_URL),
    ]);

    const HumanCtor = typeof Human === "function" ? Human : (Human && Human.default);
    if (!HumanCtor) throw new Error("No se pudo resolver el constructor de Human");
    humanConfig = {
      modelBasePath: "https://cdn.jsdelivr.net/npm/@vladmandic/human/models/",
      face: {
        enabled: true,
        detector: { rotation: false, skipFrames: 0, skipTime: 0 },
        mesh: { enabled: false },
        iris: { enabled: false },
        description: { enabled: true },
        emotion: { enabled: false },
      },
      body: { enabled: false }, hand: { enabled: false }, gesture: { enabled: false },
      filter: { enabled: false }, cacheSensitivity: 0,
    };
    const humanLoad = (async () => {
      humanInstance = new HumanCtor(humanConfig);
      await humanInstance.load();
    })();

    if (!window.FaceMeshImageEngine) {
      throw new Error("window.FaceMeshImageEngine no está disponible — ¿se cargó facemesh-image.js?");
    }
    const faceMeshLoad = window.FaceMeshImageEngine.load(() => {}).then(() => { faceMeshImageReady = true; });

    await withTimeout(Promise.all([faceapiLoad, humanLoad, faceMeshLoad]), 20000, "modelos");
    modelsReady = true;
    runStatus.textContent = "Modelos listos. Selecciona la carpeta de UTKFace para habilitar la evaluación.";
    maybeEnableStart();
  }

  // ---------- Selección de carpeta ----------
  const FILENAME_RE = /^(\d{1,3})_(\d)_(\d)_.+\.(jpe?g|png)$/i;

  folderInput.addEventListener("change", () => {
    const files = Array.from(folderInput.files || []);
    allFiles = [];
    let malformed = 0;
    for (const file of files) {
      const m = file.name.match(FILENAME_RE);
      if (!m) { malformed++; continue; }
      const age = parseInt(m[1], 10);
      const gender = m[2] === "0" ? "male" : "female";
      if (age < 0 || age > 116) { malformed++; continue; }
      allFiles.push({ file, age, gender });
    }
    folderStatus.textContent = malformed > 0
      ? `${allFiles.length} imágenes válidas encontradas (${malformed} archivos con nombre inesperado, ignorados).`
      : `${allFiles.length} imágenes válidas encontradas.`;
    maybeEnableStart();
  });

  function maybeEnableStart() {
    startBtn.disabled = !(modelsReady && allFiles.length > 0 && !running);
  }

  // ---------- Utilidades ----------
  function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  function fmt(n, decimals = 1) {
    return Number.isFinite(n) ? n.toFixed(decimals) : "—";
  }
  function loadImageElement(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => resolve({ img, url });
      img.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
      img.src = url;
    });
  }

  // Normaliza el género que devuelve cada motor a "male"/"female".
  function normGender(g) {
    if (!g) return null;
    const s = String(g).toLowerCase();
    if (s.startsWith("m")) return "male";
    if (s.startsWith("f")) return "female";
    return null;
  }

  // ---------- Pipeline "mesh": FaceMesh (imagen) → recorte con margen → ageGenderNet ----------
  const cropCanvas = document.createElement("canvas");
  const cropCtx = cropCanvas.getContext("2d");

  async function runMeshPipeline(img) {
    if (!faceMeshImageReady) return null;
    const result = window.FaceMeshImageEngine.detect(img);
    if (!result.faces || !result.faces.length) return null;

    const f = result.faces.reduce((m, x) => (x.box.area > m.box.area ? x : m), result.faces[0]);
    const marginPct = Math.max(0, Math.min(100, Number(meshMarginInput.value) || 0)) / 100;
    const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;

    const mw = f.box.width * marginPct, mh = f.box.height * marginPct;
    const x = Math.max(0, f.box.x - mw / 2);
    const y = Math.max(0, f.box.y - mh / 2);
    const w = Math.min(iw - x, f.box.width + mw);
    const h = Math.min(ih - y, f.box.height + mh);
    if (w <= 10 || h <= 10) return null;

    cropCanvas.width = Math.round(w);
    cropCanvas.height = Math.round(h);
    cropCtx.drawImage(img, x, y, w, h, 0, 0, w, h);

    const pred = await faceapi.nets.ageGenderNet.predictAgeAndGender(cropCanvas);
    const p = Array.isArray(pred) ? pred[0] : pred;
    return p ? { age: p.age, gender: normGender(p.gender) } : null;
  }

  // ---------- Loop principal ----------
  async function runEvaluation() {
    running = true; stopRequested = false;
    startBtn.disabled = true; stopBtn.disabled = false; csvBtn.disabled = true;
    startTime = performance.now();
    timeTickId = setInterval(() => {
      statTime.textContent = `${Math.round((performance.now() - startTime) / 1000)}s`;
    }, 500);

    const sizeSel = sampleSizeSel.value;
    let pool = shuffleCheck.checked ? shuffle(allFiles) : allFiles.slice();
    if (sizeSel !== "all") pool = pool.slice(0, parseInt(sizeSel, 10));
    const total = pool.length;

    for (let i = 0; i < total; i++) {
      if (stopRequested) break;
      const item = pool[i];
      let img, url;
      try {
        ({ img, url } = await loadImageElement(item.file));
      } catch (e) { continue; }

      previewImg.src = url;
      previewLabel.textContent = `${item.file.name} — edad real: ${item.age}, género real: ${item.gender}`;
      preview.style.display = "flex";

      const pred = { fa: null, hu: null, mesh: null };

      try {
        const faResult = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 })).withAgeAndGender();
        if (faResult) pred.fa = { age: faResult.age, gender: normGender(faResult.gender) };
      } catch (e) { /* miss */ }

      try {
        const huResult = await humanInstance.detect(img, humanConfig);
        if (huResult.face && huResult.face.length) {
          const f = huResult.face.reduce((m, x) =>
            (x.boxRaw ? x.boxRaw[2] * x.boxRaw[3] : 0) > (m.boxRaw ? m.boxRaw[2] * m.boxRaw[3] : 0) ? x : m,
            huResult.face[0]);
          pred.hu = { age: f.age, gender: normGender(f.gender) };
        }
      } catch (e) { /* miss */ }

      try {
        pred.mesh = await runMeshPipeline(img);
      } catch (e) { /* miss */ }

      URL.revokeObjectURL(url);
      totals.processed++;

      const err = { fa: null, hu: null, mesh: null };
      for (const eng of ENGINES) {
        if (pred[eng] && pred[eng].age != null) {
          totals.det[eng]++;
          const e = pred[eng].age - item.age;
          err[eng] = e;
          totals.ageN[eng]++;
          totals.ageSumAbs[eng] += Math.abs(e);
          totals.ageSumSq[eng] += e * e;
        }
        if (pred[eng] && pred[eng].gender) {
          totals.genderN[eng]++;
          if (pred[eng].gender === item.gender) totals.genderCorrect[eng]++;
        }
      }

      const bin = binFor(item.age);
      if (bin) {
        const bt = binTotals[bin.label];
        bt.n++;
        for (const eng of ENGINES) {
          if (err[eng] != null) {
            bt.ageSumErr[eng] += err[eng];
            bt.ageSumAbs[eng] += Math.abs(err[eng]);
          }
          if (pred[eng] && pred[eng].gender) {
            bt.genderN[eng]++;
            if (pred[eng].gender === item.gender) bt.genderCorrect[eng]++;
          }
        }
      }

      results.push({
        filename: item.file.name,
        real_age: item.age,
        real_gender: item.gender,
        fa_age: pred.fa ? pred.fa.age : null, fa_gender: pred.fa ? pred.fa.gender : null, fa_error: err.fa,
        hu_age: pred.hu ? pred.hu.age : null, hu_gender: pred.hu ? pred.hu.gender : null, hu_error: err.hu,
        mesh_age: pred.mesh ? pred.mesh.age : null, mesh_gender: pred.mesh ? pred.mesh.gender : null, mesh_error: err.mesh,
      });

      if (i % YIELD_EVERY === 0 || i === total - 1) {
        updateLiveUI(i + 1, total);
        await new Promise((r) => setTimeout(r, 0));
      }
    }

    clearInterval(timeTickId);
    running = false;
    startBtn.disabled = false; stopBtn.disabled = true;
    csvBtn.disabled = results.length === 0;
    maybeEnableStart();
    runStatus.textContent = stopRequested
      ? `Detenido manualmente tras ${totals.processed} imágenes.`
      : `Evaluación completa: ${totals.processed} imágenes procesadas.`;
  }

  function updateLiveUI(done, total) {
    progressBar.style.width = `${Math.round((done / total) * 100)}%`;
    runStatus.textContent = `Procesando ${done} / ${total}…`;
    statProcessed.textContent = String(totals.processed);

    statMaeFa.textContent = totals.ageN.fa ? `${fmt(totals.ageSumAbs.fa / totals.ageN.fa)} años (n=${totals.ageN.fa})` : "—";
    statMaeHu.textContent = totals.ageN.hu ? `${fmt(totals.ageSumAbs.hu / totals.ageN.hu)} años (n=${totals.ageN.hu})` : "—";
    statMaeMesh.textContent = totals.ageN.mesh ? `${fmt(totals.ageSumAbs.mesh / totals.ageN.mesh)} años (n=${totals.ageN.mesh})` : "—";

    const detRate = (eng) => totals.processed ? (totals.det[eng] / totals.processed * 100) : 0;
    statDetRates.textContent = `${fmt(detRate("fa"), 1)}% / ${fmt(detRate("hu"), 1)}% / ${fmt(detRate("mesh"), 1)}%`;

    binTableBody.innerHTML = AGE_BINS.map((b) => {
      const bt = binTotals[b.label];
      if (bt.n === 0) {
        return `<tr><td>${b.label}</td><td class="num">0</td>${"<td class=\"num\">—</td>".repeat(6)}</tr>`;
      }
      const mae = {}, bias = {};
      for (const eng of ENGINES) {
        const n = bt.ageSumAbs[eng] > 0 || bt.n > 0 ? bt.n : 0;
        mae[eng] = bt.n ? bt.ageSumAbs[eng] / bt.n : null;
        bias[eng] = bt.n ? bt.ageSumErr[eng] / bt.n : null;
      }
      const biasCell = (v) => v == null ? "—" :
        `<span style="color:${v > 0 ? 'var(--bad)' : 'var(--accent)'}">${v >= 0 ? "+" : ""}${fmt(v)}</span>`;
      return `<tr>
        <td>${b.label}</td>
        <td class="num">${bt.n}</td>
        <td class="num">${fmt(mae.fa)}</td>
        <td class="num">${fmt(mae.hu)}</td>
        <td class="num">${fmt(mae.mesh)}</td>
        <td class="num">${biasCell(bias.fa)}</td>
        <td class="num">${biasCell(bias.hu)}</td>
        <td class="num">${biasCell(bias.mesh)}</td>
      </tr>`;
    }).join("");

    genderTableBody.innerHTML = AGE_BINS.map((b) => {
      const bt = binTotals[b.label];
      if (bt.n === 0) {
        return `<tr><td>${b.label}</td><td class="num">0</td>${"<td class=\"num\">—</td>".repeat(3)}</tr>`;
      }
      const acc = (eng) => bt.genderN[eng] ? (bt.genderCorrect[eng] / bt.genderN[eng] * 100) : null;
      return `<tr>
        <td>${b.label}</td>
        <td class="num">${bt.n}</td>
        <td class="num">${fmt(acc("fa"))}%</td>
        <td class="num">${fmt(acc("hu"))}%</td>
        <td class="num">${fmt(acc("mesh"))}%</td>
      </tr>`;
    }).join("");
  }

  function downloadCSV() {
    if (results.length === 0) return;
    const header = [
      "archivo", "edad_real", "genero_real",
      "edad_faceapi", "genero_faceapi", "error_faceapi",
      "edad_human", "genero_human", "error_human",
      "edad_mesh", "genero_mesh", "error_mesh",
    ];
    const rows = results.map((r) => [
      r.filename, r.real_age, r.real_gender,
      r.fa_age ?? "", r.fa_gender ?? "", r.fa_error != null ? r.fa_error.toFixed(2) : "",
      r.hu_age ?? "", r.hu_gender ?? "", r.hu_error != null ? r.hu_error.toFixed(2) : "",
      r.mesh_age ?? "", r.mesh_gender ?? "", r.mesh_error != null ? r.mesh_error.toFixed(2) : "",
    ].map((c) => `"${String(c).replace(/"/g, '""')}"`).join(","));
    const csv = [header.join(","), ...rows].join("\r\n");
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    a.href = url;
    a.download = `evaluacion_edad_genero_utkface_${stamp}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ---------- Eventos ----------
  startBtn.addEventListener("click", runEvaluation);
  stopBtn.addEventListener("click", () => { stopRequested = true; });
  csvBtn.addEventListener("click", downloadCSV);

  // ---------- Arranque ----------
  if (typeof faceapi === "undefined") {
    runStatus.textContent = "No se pudo cargar face-api.js (¿sin conexión?).";
    return;
  }
  loadModels().catch((err) => {
    console.error(err);
    runStatus.textContent = "Error al cargar modelos: " + err.message;
  });
});
