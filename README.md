# Detector de Emociones con Webcam

App web que usa tu cámara para medir emociones en tiempo real. Todo corre
**localmente en el navegador** con [face-api.js](https://github.com/vladmandic/face-api)
(TensorFlow.js): el video nunca sale de tu equipo.

## Qué detecta

Por cada rostro estima, en tiempo real:

- **6 emociones (Ekman)** + neutral, con la dominante: 😄 Feliz · 😐 Neutral · 😢 Triste · 😠 Enojo · 😨 Miedo · 🤢 Disgusto · 😲 Sorpresa
- **Edad y género** (estimados).
- **Orientación de la cabeza**: inclinación (roll), giro (yaw) y cabeceo (pitch).
- **Posición y tamaño** del rostro dentro del cuadro.
- **Número de rostros** detectados en el cuadro.

### Salidas de afecto (modelo circumplejo, estilo MorphCast)

- **Valencia** y **Activación (arousal)** en [-1, 1], con mini-gráfico circumplejo.
- **Cuadrante** (Russell, Scherer & Klaus): activación/desactivación × valencia ±.
- **Afecto más cercano de 98** (Russell, Scherer & Klaus).
- **Afecto más cercano de 38** (Paltoglou & Thelwall).
- **Positividad** (0–100%) derivada de la valencia.
- **Atención** (0–100%): combina apertura ocular (EAR) y frontalidad de la cabeza.

> Los lexicones de 98 y 38 afectos son una **aproximación curada** de esos
> modelos sobre el círculo valencia–arousal (vecino más cercano); las tablas
> exactas de MorphCast son propietarias. Valencia/activación se derivan de las
> expresiones, no de un modelo de afecto dedicado.
>
> **Sesgo corregido:** face-api sobre-estima "happy" en caras neutras, lo que hacía
> que la satisfacción saliera casi siempre "satisfecho". Se resta un sesgo
> (`HAPPY_BIAS`) a la valencia y los umbrales de satisfacción
> (`SAT_THRESHOLD_POS/NEG`) son constantes ajustables al inicio de `app.js`.

### Interacción cliente–servicio (voz + fusión + momentos clave)

Mide la **calidad percibida del servicio** combinando rostro y voz del cliente:

- **Tono de voz (prosodia)**: con la Web Audio API estima en tiempo real el
  **pitch (f0)**, la **energía/volumen** y una **activación vocal** (a partir de
  energía, nivel de pitch y su variabilidad).
- **Botón de rol** (`Cliente` / `Asesor`): el analista marca **quién habla**, para
  no atribuirle al cliente la voz del asesor. La cámara enfoca al cliente.
- **Satisfacción estimada del cliente**: fusiona la **valencia facial** del cliente
  con su **activación vocal** (solo cuando habla el cliente) en una señal [-1, 1]
  suavizada, con una **línea de tiempo** de toda la sesión.
- **Momentos clave**: marca automáticamente instantes de **posible insatisfacción**
  (valencia baja, expresión negativa fuerte, voz tensa) y de **posible satisfacción**
  (sonrisa sostenida + voz positiva), con marca de tiempo, disparador y rol. Se
  exportan en su propio CSV (`momentos_clave_*.csv`).

> **Ética y privacidad.** Esto procesa rostro y voz de personas. Úsalo solo con
> **consentimiento informado**; cumple la LFPDPPP (México). Todo el procesamiento
> es local (nada se sube), pero la inferencia de "satisfacción" es una **señal
> exploratoria, no validada clínicamente**, y no debe usarse para decisiones que
> afecten a las personas evaluadas.

### Qué es medición y qué es estimación

| Métrica | Origen |
|---|---|
| Expresiones, edad, género | Modelos entrenados de face-api.js |
| Posición y tamaño del rostro | Caja de detección (medición directa) |
| Roll | Ángulo entre los ojos (fiable) |
| Yaw, pitch | Aproximación geométrica con los 68 puntos (no usa pose 3D / solvePnP) |
| Valencia y activación | **Derivadas** de las expresiones con el modelo circumplejo de Russell |

> A diferencia de un SDK comercial como MorphCast, aquí valencia/activación y el
> head pose son aproximaciones a partir de los modelos abiertos de face-api.js,
> no salidas de un modelo dedicado de afecto/pose. Útiles como demo y para
> análisis exploratorio, no como medición clínica.

## Cómo ejecutarlo

La webcam (`getUserMedia`) solo funciona en `https://` o en `localhost`, así que
no basta con abrir el archivo: hay que servirlo con un servidor local.

```bash
cd /Users/fgrodriguez/ai_dialogue/emociones_webcam

# Opción 1: Python (ya viene en macOS)
python3 -m http.server 8000

# Opción 2: Node
npx serve .
```

Luego abre <http://localhost:8000> en Chrome o Safari y:

1. Espera a que carguen los modelos (mensaje «Modelos listos»).
2. Pulsa **Iniciar cámara** y autoriza el acceso cuando el navegador lo pida.
3. Verás tu emoción dominante y las barras de cada expresión.

> En macOS, la primera vez el navegador pedirá permiso de cámara.
> Si lo bloqueaste antes: Ajustes del Sistema → Privacidad y seguridad → Cámara.

## Registro de emociones (log)

Mientras la cámara está activa, la app **muestrea la emoción dominante** a
intervalos regulares y la guarda en un registro para análisis posterior.

- **Muestrear cada**: elige el intervalo (0.5 s, 1 s, 2 s o 5 s).
- **Registrar**: activa/desactiva el muestreo sin detener la cámara.
- La tabla muestra las filas más recientes y un contador del total.
- **Descargar CSV**: exporta la sesión completa.
- **Limpiar**: vacía el registro para empezar una sesión nueva.

El CSV (`emociones_AAAA-MM-DD-HH-MM-SS.csv`) trae una fila por muestra con:

| Columna | Contenido |
|---|---|
| `timestamp_iso` | Marca de tiempo ISO 8601 (UTC), ideal para ordenar/graficar |
| `hora_local` | Fecha y hora legible en zona local |
| `emocion_dominante` | Emoción con mayor probabilidad |
| `confianza` | Probabilidad de la dominante (0–1) |
| `neutral, happy, sad, angry, fearful, disgusted, surprised` | Probabilidad de cada expresión (0–1) |
| `edad`, `genero`, `genero_prob` | Edad estimada, género y su probabilidad |
| `valencia`, `arousal` | Afecto estimado en [-1, 1] |
| `roll_deg`, `yaw_deg`, `pitch_deg` | Orientación de la cabeza en grados |
| `pos_x`, `pos_y`, `tamano` | Centro y tamaño del rostro (fracción del cuadro) |

> Lleva BOM UTF-8, así que los acentos se ven bien al abrirlo en Excel.
> El registro vive solo en memoria: si recargas la página sin descargar, se pierde.

### Analizarlo después

```python
import pandas as pd
df = pd.read_csv("emociones_2026-06-08-12-30-00.csv", parse_dates=["timestamp_iso"])
df["emocion_dominante"].value_counts()                 # conteo por emoción
df.set_index("timestamp_iso")[["happy","sad"]].plot()  # evolución temporal
```

## Archivos

| Archivo       | Rol                                                        |
|---------------|------------------------------------------------------------|
| `index.html`  | Estructura: video, overlay, panel y sección de interacción. |
| `style.css`   | Estilos (tema oscuro, barras, layout responsivo).          |
| `app.js`      | Lógica: modelos, cámara, fusión y momentos clave.          |
| `voice.js`    | Análisis de tono de voz (pitch, energía, activación vocal).|
| `affect.js`   | Cuadrantes, afectos 98/38, positividad y atención.         |

## Notas

- **Requiere internet la primera vez** para descargar face-api.js y los pesos
  de los modelos desde el CDN (jsdelivr). Después el navegador los cachea.
- Los modelos infieren expresiones a partir de rasgos faciales; **no leen la
  mente** ni el estado emocional real. Útil como demo/experimento, no como
  medición clínica.
- Privacidad: no se sube ni se guarda ninguna imagen. Procesamiento 100 % local.

## Ideas para extender

- Registrar la emoción dominante cada segundo y graficar la evolución.
- Exportar una sesión a CSV.
- Soportar varios rostros con `detectAllFaces`.
- Versión offline: descargar los pesos a una carpeta `models/` local.
