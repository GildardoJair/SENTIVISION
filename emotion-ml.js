// emotion-ml.js — Motor de clasificación de emociones propio (I2, AURA/IRIS).
// Toma los 52 blendshapes de MediaPipe y calcula la emoción con los pesos
// entrenados en Python/sklearn (emotion_ml_weights.json), en JS puro, sin
// TensorFlow.js ni ninguna otra dependencia.
//
// Soporta los dos tipos de modelo que puede exportar el notebook (campo
// model_type del JSON):
//   - "logreg": softmax(x·Wᵀ + b)                    — coef (n_clases × 52), 1 paso
//   - "mlp":    softmax(relu(x·W1 + b1)·W2 + b2)      — 2 pasos (1 capa oculta)
//
// No toca facemesh.js, emotion-fusion.js, ni app.js — módulo aislado, para
// usarse desde una página experimental (patrón ya establecido en el proyecto:
// alternar-i1.js, alternar-i5.js, etc.).

class EmotionML {
  constructor() {
    this.ready = false;
    this.weights = null;
  }

  // url: ruta al JSON exportado por el notebook (emotion_ml_weights.json).
  async load(url = "emotion_ml_weights.json") {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`No se pudo cargar ${url}: HTTP ${res.status}`);
    this.weights = await res.json();

    const { model_type, feature_names, classes } = this.weights;
    if (!["logreg", "mlp"].includes(model_type)) {
      throw new Error(`model_type desconocido en el JSON: "${model_type}"`);
    }
    if (!Array.isArray(feature_names) || feature_names.length !== 52) {
      throw new Error(`feature_names inválido (se esperaban 52, llegaron ${feature_names?.length}).`);
    }
    if (!Array.isArray(classes) || classes.length < 2) {
      throw new Error("classes inválido en el JSON de pesos.");
    }
    this.ready = true;
  }

  // blendshapes: array de {categoryName, score} — el mismo formato que entrega
  // mesh.blendshapes en facemesh.js / facemesh-image.js. El orden de entrada
  // no importa aquí: se reordena según feature_names del JSON antes de calcular.
  //
  // Devuelve { probs: {emocion: prob, ...}, predicted: "emocion_dominante", confidence: 0..1 }
  // o null si el modelo todavía no cargó o no hay blendshapes que procesar.
  classify(blendshapes) {
    if (!this.ready || !blendshapes || !blendshapes.length) return null;

    const { feature_names, classes, model_type } = this.weights;

    // Vector de entrada en el orden exacto de entrenamiento — si esto se
    // desalinea, el modelo predice basura silenciosamente (sin error visible).
    const byName = new Map(blendshapes.map((b) => [b.categoryName, b.score]));
    const x = feature_names.map((name) => byName.get(name) ?? 0);

    let logits;
    if (model_type === "logreg") {
      // coef: (n_clases, 52) · intercept: (n_clases,)  — convención sklearn LogisticRegression.
      logits = matRowsDotVecPlusBias(this.weights.coef, x, this.weights.intercept);
    } else {
      // coefs: [W1 (52, hidden), W2 (hidden, n_clases)] · intercepts: [b1 (hidden,), b2 (n_clases,)]
      // — convención sklearn MLPClassifier.coefs_ (x @ W, no W @ x).
      const [W1, W2] = this.weights.coefs;
      const [b1, b2] = this.weights.intercepts;
      const hidden = relu(vecDotColsPlusBias(x, W1, b1));
      logits = vecDotColsPlusBias(hidden, W2, b2);
    }

    const probsArr = softmax(logits);
    const probs = {};
    classes.forEach((c, i) => { probs[c] = probsArr[i]; });

    let predicted = classes[0];
    let confidence = probsArr[0];
    for (let i = 1; i < classes.length; i++) {
      if (probsArr[i] > confidence) { confidence = probsArr[i]; predicted = classes[i]; }
    }

    return { probs, predicted, confidence };
  }
}

// --- Álgebra lineal mínima, sin dependencias -------------------------------

// Para LogReg: W es (n_clases filas × n_features), cada fila es un vector de
// pesos de una clase. result[c] = dot(W[c], x) + b[c].
function matRowsDotVecPlusBias(W, x, b) {
  return W.map((row, c) => {
    let s = b[c];
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    return s;
  });
}

// Para MLP: W es (n_features_entrada filas × n_unidades_salida), es decir
// x @ W (no W @ x) — misma orientación que numpy en coefs_ de sklearn.
// result[j] = sum_i x[i]*W[i][j] + b[j].
function vecDotColsPlusBias(x, W, b) {
  const nOut = b.length;
  const out = new Array(nOut).fill(0);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = W[i];
    for (let j = 0; j < nOut; j++) out[j] += xi * row[j];
  }
  for (let j = 0; j < nOut; j++) out[j] += b[j];
  return out;
}

function relu(v) {
  return v.map((n) => (n > 0 ? n : 0));
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// Mismo patrón que el resto de módulos del proyecto (window.FaceMeshImageEngine,
// window.VoiceAnalyzer, window.EmotionFusion): instancia única global.
window.EmotionML = new EmotionML();

// Export adicional solo para poder probar este archivo con Node (validación
// numérica contra sklearn) — no afecta el uso normal en el navegador.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { EmotionML, softmax, relu, matRowsDotVecPlusBias, vecDotColsPlusBias };
}
