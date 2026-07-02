// Modelos de afecto sobre el círculo valencia–arousal (estilo MorphCast).
// Mapea un punto (valencia, activación) en [-1,1] a:
//   - Cuadrante (Russell, Scherer & Klaus)
//   - Afecto más cercano de 98 (Russell, Scherer & Klaus)
//   - Afecto más cercano de 38 (Paltoglou & Thelwall)
// Además calcula Atención (apertura ocular + frontalidad) y Positividad.
//
// NOTA: las coordenadas son una aproximación curada de esos modelos sobre el
// circumplejo afectivo; no son las tablas propietarias de MorphCast.

// --- Cuadrantes (Russell, Scherer & Klaus) ---
function quadrantOf(v, a) {
  if (Math.abs(v) < 0.08 && Math.abs(a) < 0.08) return "Neutro · sin activación marcada";
  if (a >= 0 && v >= 0) return "Activación positiva · alegría/entusiasmo";
  if (a >= 0 && v < 0)  return "Activación negativa · tensión/enojo";
  if (a < 0 && v < 0)   return "Desactivación negativa · tristeza/abatimiento";
  return "Desactivación positiva · calma/serenidad";
}

// --- 38 afectos (Paltoglou & Thelwall) ---  [palabra, valencia, activación]
const AFFECTS_38 = [
  ["Eufórico", 0.6, 0.9], ["Entusiasmado", 0.7, 0.75], ["Emocionado", 0.6, 0.8],
  ["Asombrado", 0.35, 0.8], ["Alerta", 0.3, 0.7], ["Alegre", 0.9, 0.45],
  ["Feliz", 0.9, 0.4], ["Divertido", 0.8, 0.5], ["Contento", 0.8, 0.3],
  ["Esperanzado", 0.55, 0.3], ["Satisfecho", 0.7, 0.1], ["Sereno", 0.6, -0.55],
  ["Tranquilo", 0.6, -0.45], ["Relajado", 0.65, -0.55], ["En calma", 0.55, -0.5],
  ["Cómodo", 0.5, -0.35], ["Soñoliento", 0.1, -0.8], ["Neutral", 0.0, 0.0],
  ["Aburrido", -0.4, -0.7], ["Cansado", -0.2, -0.65], ["Fatigado", -0.3, -0.6],
  ["Decaído", -0.55, -0.45], ["Triste", -0.75, -0.4], ["Melancólico", -0.6, -0.5],
  ["Deprimido", -0.8, -0.5], ["Abatido", -0.7, -0.55], ["Solo", -0.6, -0.35],
  ["Tenso", -0.55, 0.6], ["Nervioso", -0.5, 0.65], ["Ansioso", -0.6, 0.7],
  ["Frustrado", -0.6, 0.55], ["Molesto", -0.5, 0.5], ["Enojado", -0.65, 0.7],
  ["Furioso", -0.8, 0.85], ["Asustado", -0.7, 0.8], ["Alarmado", -0.6, 0.75],
  ["Estresado", -0.65, 0.6], ["Disgustado", -0.55, 0.4],
];

// --- 98 afectos (Russell, Scherer & Klaus) ---
const AFFECTS_98 = [
  // Activación positiva alta
  ["Eufórico", 0.6, 0.9], ["Extasiado", 0.7, 0.85], ["Emocionado", 0.6, 0.8],
  ["Entusiasmado", 0.7, 0.75], ["Exaltado", 0.5, 0.85], ["Jubiloso", 0.8, 0.7],
  ["Radiante", 0.85, 0.6], ["Animado", 0.7, 0.55],
  ["Vigoroso", 0.5, 0.7], ["Enérgico", 0.45, 0.75], ["Triunfante", 0.8, 0.6],
  ["Orgulloso", 0.7, 0.45], ["Ilusionado", 0.65, 0.5], ["Asombrado", 0.35, 0.8],
  ["Maravillado", 0.6, 0.65], ["Fascinado", 0.65, 0.55], ["Divertido", 0.8, 0.5],
  ["Juguetón", 0.7, 0.5],
  // Positivo arousal medio/bajo
  ["Alegre", 0.9, 0.45], ["Feliz", 0.9, 0.4], ["Contento", 0.8, 0.3],
  ["Dichoso", 0.85, 0.35], ["Encantado", 0.8, 0.4], ["Complacido", 0.7, 0.2],
  ["Satisfecho", 0.7, 0.1], ["Optimista", 0.6, 0.3], ["Esperanzado", 0.55, 0.3],
  ["Confiado", 0.55, 0.2], ["Agradecido", 0.6, 0.15], ["Interesado", 0.4, 0.4],
  ["Atento", 0.3, 0.45], ["Curioso", 0.4, 0.5],
  // Calma positiva (desactivación positiva)
  ["Sereno", 0.6, -0.55], ["Tranquilo", 0.6, -0.45], ["Relajado", 0.65, -0.55],
  ["En calma", 0.55, -0.5], ["Apacible", 0.6, -0.6],
  ["Cómodo", 0.5, -0.35], ["Seguro", 0.45, -0.3], ["Aliviado", 0.5, -0.2],
  ["Conforme", 0.45, -0.25], ["Soñoliento", 0.1, -0.8],
  ["Adormilado", 0.05, -0.75], ["Perezoso", -0.1, -0.7],
  // Centro / neutro
  ["Neutral", 0.0, 0.0], ["Indiferente", -0.05, -0.1], ["Pensativo", 0.0, -0.1],
  ["Contemplativo", 0.1, -0.2],
  // Desactivación negativa
  ["Aburrido", -0.4, -0.7], ["Cansado", -0.2, -0.65],
  ["Fatigado", -0.3, -0.6], ["Agotado", -0.35, -0.55], ["Apático", -0.4, -0.65],
  ["Decaído", -0.55, -0.45], ["Triste", -0.75, -0.4], ["Melancólico", -0.6, -0.5],
  ["Abatido", -0.7, -0.55], ["Deprimido", -0.8, -0.5], ["Desanimado", -0.65, -0.45],
  ["Desdichado", -0.8, -0.4], ["Afligido", -0.7, -0.4], ["Solo", -0.6, -0.35],
  ["Nostálgico", -0.4, -0.4], ["Pesimista", -0.6, -0.4], ["Desilusionado", -0.65, -0.35],
  ["Culpable", -0.6, -0.2], ["Avergonzado", -0.55, -0.1], ["Arrepentido", -0.5, -0.3],
  ["Desesperanzado", -0.75, -0.35],
  // Activación negativa
  ["Tenso", -0.55, 0.6], ["Nervioso", -0.5, 0.65], ["Ansioso", -0.6, 0.7],
  ["Inquieto", -0.45, 0.6], ["Preocupado", -0.55, 0.5], ["Angustiado", -0.7, 0.6],
  ["Estresado", -0.65, 0.6], ["Agobiado", -0.6, 0.55], ["Frustrado", -0.6, 0.55],
  ["Irritado", -0.55, 0.6], ["Molesto", -0.5, 0.5], ["Enfadado", -0.6, 0.65],
  ["Enojado", -0.65, 0.7], ["Furioso", -0.8, 0.85], ["Iracundo", -0.85, 0.8],
  ["Hostil", -0.7, 0.7], ["Indignado", -0.6, 0.65], ["Celoso", -0.55, 0.5],
  ["Asqueado", -0.6, 0.4], ["Disgustado", -0.55, 0.45],
  ["Asustado", -0.7, 0.8], ["Atemorizado", -0.65, 0.75], ["Alarmado", -0.6, 0.75],
  ["Aterrado", -0.85, 0.85], ["Sobresaltado", -0.3, 0.7], ["Horrorizado", -0.75, 0.8],
  ["Desconcertado", -0.2, 0.5], ["Confundido", -0.25, 0.4], ["Abrumado", -0.6, 0.6],
];

// Afecto más cercano (distancia euclídea en el plano valencia–arousal).
function nearestAffect(table, v, a) {
  let best = null, bestD = Infinity;
  for (const [word, wv, wa] of table) {
    const d = (v - wv) ** 2 + (a - wa) ** 2;
    if (d < bestD) { bestD = d; best = word; }
  }
  return best;
}

// Positividad 0..1 a partir de la valencia.
function positivityOf(v) {
  return Math.max(0, Math.min(1, (v + 1) / 2));
}

// Atención 0..1: combina apertura ocular y frontalidad de la cabeza.
//
// MIGRACIÓN I1 (MediaPipe FaceMesh): antes el EAR se calculaba a mano con 6 de
// los 68 puntos de face-api.js. Ahora usamos los blendshapes "eyeBlinkLeft" /
// "eyeBlinkRight" que entrega FaceLandmarker (0 = ojo abierto, 1 = cerrado),
// que son más estables porque vienen de un clasificador entrenado, no de una
// razón geométrica sensible al ángulo de la cámara. El pose (yaw/pitch) ahora
// viene de la matriz de transformación real en vez de la aproximación 2D.
//
// Se mantiene compatibilidad hacia atrás: si no hay blendshapes disponibles
// (p.ej. fallback a face-api.js), cae a un valor neutro en vez de romper.
function attentionScore(pose, blendshapes) {
  let eyeScore = 0.7; // neutro si no hay señal de blendshapes
  if (blendshapes) {
    const find = (name) => {
      const b = blendshapes.find((x) => x.categoryName === name);
      return b ? b.score : 0;
    };
    const blinkL = find("eyeBlinkLeft");
    const blinkR = find("eyeBlinkRight");
    eyeScore = Math.max(0, Math.min(1, 1 - (blinkL + blinkR) / 2));
  }
  const frontal = Math.max(0, Math.min(1, 1 - (Math.abs(pose.yaw) / 35 + Math.abs(pose.pitch) / 30) / 2));
  return Math.max(0, Math.min(1, 0.6 * eyeScore + 0.4 * frontal));
}

window.AffectModels = {
  quadrantOf, nearestAffect, positivityOf, attentionScore,
  AFFECTS_98, AFFECTS_38,
};
