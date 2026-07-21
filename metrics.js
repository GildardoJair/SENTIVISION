// metrics.js — I4: Métricas de calidad del sistema en tiempo real.
// Implementa el "Nivel 2 — Calidad de Sistema en Producción" de la Sección 5.2
// del Reporte de Diagnóstico (25 jun 2026): detection_rate, expression_entropy
// y flip_rate. Permiten detectar degradación (mala luz, cara lateral, etc.)
// SIN necesitar el dataset de prueba offline del Nivel 1.
//
// Diseño modular (ver auditoría de modularización): este archivo no conoce
// face-api.js ni MediaPipe. Solo recibe, una vez por frame, señales ya
// calculadas por detectLoop() (¿hay rostro?, expresiones, emoción dominante) y
// mantiene ventanas deslizantes. Así, cuando cambien los motores de visión
// (I1, I5) o se agregue audio (I2/I3), este módulo no se toca.

// Umbrales de alerta tomados directo de la tabla de la Sección 5.2.
const THRESHOLDS = {
  detectionRateMin: 0.70,   // alerta si detection_rate < 0.70
  entropyMax: 1.9,          // alerta si expression_entropy > 1.9 sostenido
  flipRateMax: 4,           // alerta si flip_rate > 4 / seg
};

class QualityMetrics {
  constructor({ windowMs = 5000 } = {}) {
    this.windowMs = windowMs;
    this.frames = [];   // { t, hasFace, expressions, dominant }
    this.flips = [];    // timestamps (ms) de cambios de emoción dominante
    this.lastDominant = null;
  }

  reset() {
    this.frames = [];
    this.flips = [];
    this.lastDominant = null;
  }

  // Llamar UNA vez por frame desde detectLoop(), haya o no rostro detectado.
  //   hasFace:     boolean
  //   expressions: objeto {emocion: prob 0..1} o null si no hay rostro
  //   dominant:    string (clave de la emoción dominante) o null
  update(hasFace, expressions, dominant) {
    const now = performance.now();
    this.frames.push({ t: now, hasFace, expressions, dominant });

    if (hasFace && dominant && this.lastDominant && dominant !== this.lastDominant) {
      this.flips.push(now);
    }
    if (hasFace && dominant) this.lastDominant = dominant;

    this._trim(now);
  }

  _trim(now) {
    const cutoff = now - this.windowMs;
    while (this.frames.length && this.frames[0].t < cutoff) this.frames.shift();
    while (this.flips.length && this.flips[0] < cutoff) this.flips.shift();
  }

  // Snapshot de las 3 métricas de Nivel 2, más flags de alerta según Sección 5.2.
  snapshot() {
    const n = this.frames.length;
    if (n === 0) {
      return { detectionRate: null, entropy: null, flipRate: null, alerts: {} };
    }

    // detection_rate = frames_con_rostro / frames_totales (ventana deslizante).
    const withFace = this.frames.filter((f) => f.hasFace).length;
    const detectionRate = withFace / n;

    // expression_entropy = -Σ p(e)·log2 p(e), promediada sobre frames con rostro.
    const entropies = this.frames
      .filter((f) => f.hasFace && f.expressions)
      .map((f) => {
        let h = 0;
        for (const key of Object.keys(f.expressions)) {
          const p = f.expressions[key];
          if (p > 0) h -= p * Math.log2(p);
        }
        return h;
      });
    const entropy = entropies.length
      ? entropies.reduce((a, b) => a + b, 0) / entropies.length
      : null;

    // flip_rate = cambios de emoción dominante / segundo, en la ventana.
    const windowSec = this.windowMs / 1000;
    const flipRate = this.flips.length / windowSec;

    return {
      detectionRate,
      entropy,
      flipRate,
      alerts: {
        detectionRate: detectionRate < THRESHOLDS.detectionRateMin,
        entropy: entropy != null && entropy > THRESHOLDS.entropyMax,
        flipRate: flipRate > THRESHOLDS.flipRateMax,
      },
    };
  }
}

window.QualityMetrics = new QualityMetrics({ windowMs: 5000 });
window.QualityMetricsThresholds = THRESHOLDS;
