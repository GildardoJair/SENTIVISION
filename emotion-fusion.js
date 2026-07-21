// emotion-fusion.js — I2: Fusión tardía Blendshapes (MediaPipe) + audio para
// clasificar las 7 emociones primarias (Ekman + neutral).
//
// Resuelve (en teoría; PENDIENTE DE CALIBRACIÓN contra el dataset de prueba
// interno, Nivel 1 / Sección 5.1 del reporte) dos hallazgos del diagnóstico:
//   - Hallazgo #1: sesgo sistemático de "happy" (el parche manual HAPPY_BIAS
//     ya no aplica aquí, porque este clasificador no hereda el sesgo de
//     faceExpressionNet).
//   - Hallazgo #7: "fearful"/"disgusted" con detección casi nula (el dataset
//     de faceExpressionNet, FER2013/AffectNet, tiene pocos ejemplos
//     conversacionales de esas clases; los Blendshapes al menos permiten una
//     señal geométrica directa aunque sea débil).
//
// DISEÑO (late fusion, NO es una red neuronal nueva):
//   1) Blendshapes (52 AUs) → distribución {emocion: prob} vía suma ponderada
//      de Action Units (mapeo FACS aproximado) + normalización.
//   2) La señal de audio (arousal vocal de VoiceAnalyzer) SOLO redistribuye
//      masa de probabilidad hacia clases de alta/baja activación; nunca
//      inventa una emoción que la cara no sugirió. El peso de esta
//      redistribución (audioWeight) es la "modalidad dominante": 0 si no
//      habla el cliente (mismo criterio que computeSatisfaction en app.js).
//
// ADVERTENCIA: los pesos AU→emoción de abajo son una aproximación curada,
// igual que las tablas de 98/38 afectos en affect.js — no vienen de un
// clasificador entrenado. Úsese como punto de partida para calibración, no
// como verdad de referencia.

const AU_WEIGHTS = {
  happy: {
    mouthSmileLeft: 1.0, mouthSmileRight: 1.0,
    cheekSquintLeft: 0.4, cheekSquintRight: 0.4,
  },
  sad: {
    mouthFrownLeft: 1.0, mouthFrownRight: 1.0,
    browInnerUp: 0.5,
    mouthLowerDownLeft: 0.3, mouthLowerDownRight: 0.3,
  },
  angry: {
    browDownLeft: 1.0, browDownRight: 1.0,
    noseSneerLeft: 0.4, noseSneerRight: 0.4,
    mouthPressLeft: 0.4, mouthPressRight: 0.4,
  },
  fearful: {
    browInnerUp: 0.6,
    eyeWideLeft: 0.7, eyeWideRight: 0.7,
    mouthStretchLeft: 0.5, mouthStretchRight: 0.5,
    jawOpen: 0.3,
  },
  disgusted: {
    noseSneerLeft: 1.0, noseSneerRight: 1.0,
    mouthUpperUpLeft: 0.5, mouthUpperUpRight: 0.5,
    browDownLeft: 0.2, browDownRight: 0.2,
  },
  surprised: {
    browInnerUp: 0.6, browOuterUpLeft: 0.6, browOuterUpRight: 0.6,
    eyeWideLeft: 0.6, eyeWideRight: 0.6,
    jawOpen: 0.7,
  },
};

function toMap(blendshapes) {
  const m = {};
  if (!blendshapes) return m;
  for (const b of blendshapes) m[b.categoryName] = b.score;
  return m;
}

// Blendshapes (52 AUs) → distribución {emocion: prob 0..1, suma 1}.
function faceProbsFromBlendshapes(blendshapes) {
  const m = toMap(blendshapes);
  const raw = {};
  let activitySum = 0;

  for (const emo of Object.keys(AU_WEIGHTS)) {
    let score = 0;
    for (const [au, w] of Object.entries(AU_WEIGHTS[emo])) {
      score += (m[au] || 0) * w;
    }
    raw[emo] = score;
    activitySum += score;
  }

  // Neutral = "lo que sobra": mientras más bajas las demás señales, más neutral.
  raw.neutral = Math.max(0, 1 - Math.min(1, activitySum));

  const total = Object.values(raw).reduce((a, b) => a + b, 0);
  const probs = {};
  for (const emo of Object.keys(raw)) probs[emo] = total > 0 ? raw[emo] / total : 0;
  return probs;
}

// Redistribuye masa de probabilidad según arousal vocal (fusión tardía).
// audioWeight ∈ [0,1]: 0 = ignora el audio (usa solo cara).
function fuseWithAudio(faceProbs, vf, audioWeight) {
  if (!vf || !vf.voiced || audioWeight <= 0) return faceProbs;

  const HIGH = ["angry", "fearful", "surprised"];
  const LOW = ["sad", "neutral"];
  const shift = audioWeight * (vf.arousal - 0.5) * 2; // -audioWeight..+audioWeight

  const fused = { ...faceProbs };
  if (shift > 0) {
    // Voz tensa → empuja masa hacia clases de alta activación.
    for (const emo of HIGH) fused[emo] = Math.min(1, fused[emo] + shift / HIGH.length);
    for (const emo of LOW) fused[emo] = Math.max(0, fused[emo] - shift / LOW.length);
  } else if (shift < 0) {
    // Voz calmada → empuja masa hacia clases de baja activación.
    for (const emo of LOW) fused[emo] = Math.min(1, fused[emo] - shift / LOW.length);
    for (const emo of HIGH) fused[emo] = Math.max(0, fused[emo] + shift / HIGH.length);
  }

  const total = Object.values(fused).reduce((a, b) => a + b, 0);
  if (total > 0) for (const emo of Object.keys(fused)) fused[emo] /= total;
  return fused;
}

// Punto de entrada único: blendshapes + voz + rol activo → distribución final.
// activeRole: "cliente" | "asesor" (mismo criterio que computeSatisfaction).
function classify(blendshapes, vf, activeRole) {
  const faceProbs = faceProbsFromBlendshapes(blendshapes);
  const audioWeight = activeRole === "cliente" ? 0.3 : 0;
  return fuseWithAudio(faceProbs, vf, audioWeight);
}

window.EmotionFusion = {
  classify,
  faceProbsFromBlendshapes,
  fuseWithAudio,
  AU_WEIGHTS,
};
