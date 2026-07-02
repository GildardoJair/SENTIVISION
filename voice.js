// Análisis de tono de voz (prosodia) con la Web Audio API.
// Extrae pitch (f0), energía y una activación vocal estimada. Todo local.
// NOTA: es un proxy prosódico, no un modelo entrenado de emoción de voz.

class VoiceAnalyzer {
  constructor() {
    this.ctx = null;
    this.analyser = null;
    this.source = null;
    this.timer = null;
    this.buf = null;
    this.running = false;
    this.pitchHist = [];          // historial reciente de pitch para variabilidad
    this.features = { voiced: false, rms: 0, pitch: null, arousal: 0 };
  }

  async start(stream) {
    // Reutiliza el stream (con pista de audio) que ya pidió la cámara.
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AudioCtx();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    this.source = this.ctx.createMediaStreamSource(stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.source.connect(this.analyser);
    this.buf = new Float32Array(this.analyser.fftSize);
    this.running = true;
    // Corre a ~20 Hz, independiente del bucle de video.
    this.timer = setInterval(() => this._tick(), 50);
  }

  stop() {
    this.running = false;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    if (this.ctx) this.ctx.close();
    this.ctx = null;
    this.pitchHist = [];
    this.features = { voiced: false, rms: 0, pitch: null, arousal: 0 };
  }

  _tick() {
    if (!this.running || !this.analyser) return;
    this.analyser.getFloatTimeDomainData(this.buf);
    const rms = VoiceAnalyzer.rms(this.buf);
    const voiced = rms > 0.012;                       // umbral simple de actividad vocal
    const pitch = voiced ? VoiceAnalyzer.autoCorrelate(this.buf, this.ctx.sampleRate) : null;

    if (pitch && pitch > 60 && pitch < 400) {
      this.pitchHist.push(pitch);
      if (this.pitchHist.length > 24) this.pitchHist.shift();
    }

    this.features = {
      voiced,
      rms,
      pitch: pitch && pitch > 60 && pitch < 400 ? pitch : null,
      arousal: this._arousal(rms, pitch),
    };
  }

  // Activación vocal [0,1] a partir de energía, nivel de pitch y su variabilidad.
  _arousal(rms, pitch) {
    const energyNorm = Math.min(1, rms / 0.15);
    let pitchNorm = 0;
    if (pitch && pitch > 60 && pitch < 400) {
      pitchNorm = Math.min(1, Math.max(0, (pitch - 90) / (300 - 90)));
    }
    let varNorm = 0;
    if (this.pitchHist.length > 4) {
      const m = this.pitchHist.reduce((a, b) => a + b, 0) / this.pitchHist.length;
      const v = this.pitchHist.reduce((a, b) => a + (b - m) ** 2, 0) / this.pitchHist.length;
      varNorm = Math.min(1, Math.sqrt(v) / 60);
    }
    return Math.min(1, 0.5 * energyNorm + 0.3 * pitchNorm + 0.2 * varNorm);
  }

  static rms(buf) {
    let s = 0;
    for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    return Math.sqrt(s / buf.length);
  }

  // Detección de pitch por autocorrelación (método estándar con recorte e interpolación).
  static autoCorrelate(buf, sampleRate) {
    const size = buf.length;
    let rms = 0;
    for (let i = 0; i < size; i++) rms += buf[i] * buf[i];
    rms = Math.sqrt(rms / size);
    if (rms < 0.01) return -1;

    let r1 = 0, r2 = size - 1;
    const thres = 0.2;
    for (let i = 0; i < size / 2; i++) if (Math.abs(buf[i]) < thres) { r1 = i; break; }
    for (let i = 1; i < size / 2; i++) if (Math.abs(buf[size - i]) < thres) { r2 = size - i; break; }

    const b = buf.slice(r1, r2);
    const n = b.length;
    const c = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n - i; j++) c[i] += b[j] * b[j + i];

    let d = 0;
    while (d < n - 1 && c[d] > c[d + 1]) d++;
    let maxval = -1, maxpos = -1;
    for (let i = d; i < n; i++) if (c[i] > maxval) { maxval = c[i]; maxpos = i; }
    let T0 = maxpos;
    if (T0 <= 0) return -1;

    // Interpolación parabólica para afinar el período.
    const x1 = c[T0 - 1] || 0, x2 = c[T0], x3 = c[T0 + 1] || 0;
    const a = (x1 + x3 - 2 * x2) / 2;
    const bb = (x3 - x1) / 2;
    if (a) T0 = T0 - bb / (2 * a);

    return sampleRate / T0;
  }
}

window.VoiceAnalyzer = VoiceAnalyzer;
