# AURA BUAP × IRIS UDLAP — Sistema Multimodal de Emociones y Satisfacción

Sistema de análisis de emociones, afecto y satisfacción del cliente en tiempo
real a partir de webcam y micrófono. Todo corre **100&nbsp;% en el navegador**:
no hay servidor de procesamiento y ningún video o audio sale del equipo.

Este repositorio es el resultado del Plan de Trabajo AURA BUAP × IRIS UDLAP
(22 jun – 24 jul 2026), a partir de la auditoría técnica documentada en el
Reporte de Diagnóstico (Fase 1, 25 jun 2026) y las iniciativas I1–I5 de la
hoja de ruta ahí definida.

## Los dos entregables

| Archivo | Qué es |
|---|---|
| **`index.html`** | App en vivo: captura cámara + micrófono, analiza en tiempo real y exporta la sesión a CSV. |
| **`analizar-sesion.html`** | Dashboard standalone: carga los CSV de una sesión ya grabada y genera el reporte visual (timeline, cuadrantes, momentos clave, veredicto). No requiere servidor ni subida de archivos — todo se procesa localmente en el navegador desde los CSV que el usuario selecciona. |

Ambos son autocontenidos: `index.html` depende de los módulos JS de este
mismo repo (ver tabla de archivos), y `analizar-sesion.html` no depende de
ningún otro archivo del proyecto.

## Qué detecta

Por cada rostro, en tiempo real:

- **7 emociones** (Ekman + neutral): 😄 Feliz · 😐 Neutral · 😢 Triste · 😠 Enojo · 😨 Miedo · 🤢 Disgusto · 😲 Sorpresa, con tres motores de clasificación disponibles (ver abajo) y comparación en vivo entre ellos.
- **Edad y género** (estimados).
- **Orientación de la cabeza**: roll, yaw y pitch a partir de una matriz de transformación 3D real (MediaPipe), no de una aproximación geométrica 2D.
- **Posición, tamaño y número de rostros** en el cuadro.
- **Calidad del análisis en tiempo real** (Nivel 2 del marco de métricas): `detection_rate`, `expression_entropy` y `flip_rate`, con alertas visuales cuando el sistema opera en condiciones degradadas.

### Motores de clasificación de emociones (I1 + I2)

El sistema permite comparar tres fuentes en vivo, todas se registran en el
CSV para poder auditarlas offline:

1. **`faceExpressionNet`** (face-api.js) — motor original, se mantiene como referencia/baseline.
2. **Fusión tardía Blendshapes + audio** (`emotion-fusion.js`) — heurística que combina los 52 blendshapes de MediaPipe con la señal prosódica de voz.
3. **Modelo propio (MLP)** (`emotion-ml.js`) — red neuronal (52→128→7, ReLU) entrenada sobre features de blendshape del dataset RAF-DB. Alcanzó **F1 macro 0.531** contra 0.228 de `faceExpressionNet` en el mismo benchmark. Incluye suavizado EMA + histéresis para corregir sobre-predicción de tristeza en video en vivo. **Es el motor activo por defecto.**

> **Sobre `HAPPY_BIAS`:** el parche manual que restaba un sesgo fijo a "happy"
> (documentado como hallazgo crítico en el diagnóstico de Fase 1) solo se
> sigue aplicando cuando el motor activo es `faceExpressionNet` — el sesgo ya
> no existe en el modelo propio, corregido internamente por el balance del
> set de entrenamiento.

### Detección facial y landmarks (I1)

Migrado de `tinyFaceDetector` + `faceLandmark68Net` (face-api.js, 2019) a
**MediaPipe Face Landmarker**: 478 puntos 3D, matriz de transformación real
y 52 blendshapes. `face-api.js` se conserva únicamente para `ageGenderNet`.

### Edad y género (I5)

`ageGenderNet` (face-api.js) ahora recibe como entrada el recorte del
bounding box de MediaPipe (margen 60%) en vez de depender del detector
antiguo. Esto **triplicó la cobertura de detección** (~33% → ~99%) con
precisión estadísticamente equivalente (Wilcoxon p=0.114). Se evaluó
`@vladmandic/human` como alternativa y se descartó: MAE de edad
significativamente peor, en especial en el grupo 60+, además de
incompatible con imágenes recortadas y pre-alineadas.

### Salidas de afecto (modelo circumplejo, estilo MorphCast)

- **Valencia** y **Activación (arousal)** en [-1, 1], con mini-gráfico circumplejo.
- **Cuadrante** (Russell, Scherer & Klaus).
- **Afecto más cercano** de dos catálogos: 98 términos (Russell, Scherer & Klaus) y 38 (Paltoglou & Thelwall).
- **Positividad** (0–100%) derivada de la valencia.
- **Atención** (0–100%): combina apertura ocular (EAR) y frontalidad de la cabeza, usando la pose 3D real de MediaPipe.

> Los lexicones de 98 y 38 afectos son una **aproximación curada** sobre el
> círculo valencia–arousal (vecino más cercano); las tablas exactas de
> MorphCast son propietarias.

### Interacción cliente–servicio (voz + fusión + momentos clave)

- **Tono de voz (prosodia)**: pitch (f0) por autocorrelación, energía RMS y activación vocal, vía Web Audio API.
- **Botón de rol** (`Cliente` / `Asesor`): marca manual de quién habla — no hay diarización automática (limitación documentada, pendiente de Fase 4/I3).
- **Satisfacción estimada del cliente**: fusiona valencia facial con activación vocal (solo cuando habla el cliente) en una señal EMA [-1, 1], con línea de tiempo de toda la sesión.
- **Momentos clave**: detecta automáticamente instantes de posible insatisfacción o satisfacción (valencia baja / expresión negativa fuerte / voz tensa, o sonrisa sostenida + voz positiva).
- **Autoevaluación en vivo**: al descargar la sesión, un modal pregunta directamente a la persona cómo se sintió (insatisfecho / neutro / satisfecho), capturando esa respuesta mientras sigue presente en vez de inferirla después. Usa el mismo criterio que el veredicto del dashboard.

> **Ética y privacidad.** Esto procesa rostro y voz de personas. Úsalo solo
> con **consentimiento informado**; cumple la LFPDPPP (México). Todo el
> procesamiento es local, pero la satisfacción/afecto inferidos son una
> **señal exploratoria, no validada clínicamente**, y no deben usarse para
> decisiones que afecten a las personas evaluadas.

### Calidad del sistema en tiempo real (I4)

Métricas de Nivel 2 del marco de evaluación, calculadas en ventanas
deslizantes de 5 segundos e integradas tanto en la UI como en el CSV:

| Métrica | Qué mide | Alerta si… |
|---|---|---|
| `detection_rate` | Cobertura del análisis en la sesión | < 0.70 |
| `expression_entropy` | Confusión/sobreconfianza del clasificador | > 1.9 constante |
| `flip_rate` | Inestabilidad de la emoción dominante | > 4 cambios/seg |

### ⚠️ Limitación conocida: sesión = una persona

El veredicto del dashboard (`analizar-sesion.html`) asume que la grabación
corresponde a **una sola persona**. Si por el cuadro pasa más de una persona
durante una misma grabación continua, el veredicto agregado no es
significativo. La re-identificación entre frames y el soporte multi-persona
están documentados como trabajo pendiente (Fase 3 del cronograma).

## Cómo ejecutarlo

La webcam (`getUserMedia`) solo funciona en `https://` o en `localhost`, así
que no basta con abrir el archivo directamente: hay que servirlo con un
servidor local.

```bash
# Desde la raíz del proyecto
python3 -m http.server 8000
# o bien
npx serve .
```

Luego abre `http://localhost:8000` en Chrome o Safari:

1. Espera a que carguen los modelos (mensaje «Modelos listos»).
2. Pulsa **Iniciar cámara** y autoriza cámara/micrófono cuando el navegador lo pida.
3. Elige el clasificador activo si quieres comparar motores (por defecto: MLP propio).
4. Al terminar, **Descargar sesión completa (CSV)** — te pedirá la autoevaluación de la persona antes de generar el archivo.
5. Abre `analizar-sesion.html` (mismo servidor local, o directo desde el sistema de archivos) y carga los CSV descargados para ver el reporte de la sesión.

Ver `docs/INSTRUCCIONES.md` para el paso a paso en Mac con doble clic
(`iniciar.command`), y `docs/GLOSARIO.md` para el significado de cada
término técnico.

## Registro de emociones y CSV

Mientras la cámara está activa, la app muestrea la emoción dominante a
intervalos regulares (0.5–5 s, configurable) y arma un registro en memoria.
Al descargar, la sesión completa (emociones + momentos clave + calidad del
sistema, en secciones dentro del mismo archivo) se exporta con BOM UTF-8, y
se conserva únicamente si se descarga: recargar la página sin descargar
pierde el registro.

## Estructura del repositorio

```
├── index.html                  ← entregable: app en vivo
├── analizar-sesion.html        ← entregable: dashboard de sesión
├── app.js                      ← lógica principal: cámara, modelos, fusión, CSV
├── affect.js                   ← cuadrantes, afectos 98/38, positividad, atención
├── voice.js                    ← análisis de tono de voz (Web Audio API)
├── facemesh.js                 ← wrapper de MediaPipe Face Landmarker (I1)
├── emotion-ml.js                ← inferencia del MLP propio en JS puro (I2)
├── emotion_ml_weights.json     ← pesos entrenados del MLP
├── emotion-fusion.js            ← fusión heurística blendshapes + audio (I2)
├── metrics.js                   ← métricas de calidad en tiempo real (I4)
├── style.css
├── docs/                        ← glosario, instrucciones, contexto del proyecto
├── notebooks/                   ← entrenamiento y evaluación offline (Python)
├── elicitacion/                  ← estudios de elicitación de emociones
└── archive/                      ← versiones previas y herramientas de desarrollo
    ├── app_v2.js, index_v2.html      (snapshot intermedio, superado por app.js/index.html)
    ├── evaluacion-offline/           (benchmarking F1/MAE contra RAF-DB y UTKFace)
    ├── experimentos/                 (comparaciones aisladas por iniciativa)
    └── pruebas-diagnostico-interno/
```

## Notas

- **Requiere internet la primera vez** para descargar face-api.js, los
  modelos de MediaPipe y sus pesos desde CDN. Después el navegador los
  cachea.
- Los modelos infieren expresiones y afecto a partir de rasgos faciales y
  prosodia; **no leen el estado emocional real** de la persona. Es una señal
  exploratoria para investigación de experiencia de servicio y para fines
  docentes, no una medición clínica.
- Privacidad: no se sube ni se guarda ninguna imagen o audio en ningún
  servidor. Procesamiento 100&nbsp;% local.

## Demos

El proyecto puede publicarse como sitio estático (por ejemplo, vía GitHub
Pages) para hacer pruebas sin necesidad de levantar un servidor local. Pide
al equipo del proyecto el enlace vigente si necesitas una demo en línea.
