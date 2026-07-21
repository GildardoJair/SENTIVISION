# Novedades desde el Reporte de Diagnóstico (25 jun → hoy)

Resumen rápido de qué cambió en el código desde la auditoría de Fase 1, para
que la revisión con UDLAP no tenga que ir línea por línea.

## I1 — Migración a MediaPipe FaceMesh ✅ integrado

- **Antes:** `tinyFaceDetector` + `faceLandmark68Net` (68 puntos 2D), head
  pose calculado con una aproximación geométrica manual (Hallazgo #4: error
  >30° en ángulos extremos).
- **Ahora:** MediaPipe Face Landmarker — 478 puntos 3D (incluye iris, que el
  modelo anterior no tenía) + matriz de transformación facial real para el
  head pose.
- **Módulo:** `facemesh.js`. Corre en paralelo a face-api (que se mantiene
  *solo* para alimentar expresiones/edad/género, ver I2/I5 abajo).
- **Pendiente:** validar el MAE de head pose (<5°, Sección 5.1) contra un
  dataset con ángulos conocidos — el análisis hecho hasta ahora es cualitativo
  (comparación visual en vivo), no cuantitativo.

## I4 — Métricas de calidad en tiempo real ✅ integrado

- **Módulo:** `metrics.js`. Implementa las 3 métricas de Nivel 2 (Sección 5.2):
  `detection_rate`, `expression_entropy`, `flip_rate`, con alertas visuales si
  cruzan el umbral del reporte.
- Se exportan como columnas nuevas en el CSV de sesión.

## I2 — Fusión tardía Blendshapes + audio ⚠️ v1, sin calibrar

- **Módulo:** `emotion-fusion.js`. Clasificador de emociones basado en mapeo
  FACS de los 52 Blendshapes de MediaPipe, con redistribución de probabilidad
  según el arousal vocal (solo cuando habla el cliente).
- **Importante:** los pesos AU→emoción son una aproximación curada, no un
  modelo entrenado — necesita calibrarse contra el dataset interno antes de
  reemplazar la línea base en producción.
- Se activa con el toggle "I2: usar Blendshapes+audio"; el CSV registra
  siempre ambas distribuciones (línea base y fusión) para comparar F1/ECE
  offline sin importar cuál esté activa.

## I5 — Reemplazo de ageGenderNet ❌ no iniciado

Sigue pendiente. Próximo modelo a evaluar: `@vladmandic/human` (submódulo
`faceres`), para robustez a accesorios (lentes, gorras — Hallazgo #5).

## I3 — Diarización por MFCCs ❌ no iniciado

Sigue pendiente, programado para Fase 4 (semana del 13 jul) según cronograma.

## Dataset de evaluación Nivel 1 ❌ no iniciado

Identificamos las fuentes públicas a usar (no se recopilarán datos propios):
**AFLW2000-3D** y **BIWI** para head pose; **FER2013/RAF-DB/AffectNet** para
emociones; **UTKFace/IMDB-WIKI** para edad. Falta armar el arnés de
evaluación offline (cargar dataset + comparar contra ground truth).

## Nota sobre modularidad

Los módulos nuevos (`facemesh.js`, `metrics.js`, `emotion-fusion.js`) se
diseñaron como archivos separados con una interfaz mínima, siguiendo la
recomendación de refactorización incremental del reporte — `app.js` sigue
siendo el orquestador, pero cada iniciativa nueva vive en su propio archivo en
vez de crecer el monolito.
