# Sistema de Medición de Emociones por Webcam

Explicación de qué hace el sistema y para qué sirve, para quien lo recibe por
primera vez.

## Qué es

Una aplicación web que usa la **cámara (y opcionalmente el micrófono)** para
estimar, en tiempo real, las **emociones y el estado afectivo** de la persona
frente a la pantalla. Corre **dentro del navegador**: no necesita instalación ni
servidores externos para analizar.

## Qué hace

A partir del rostro y la voz, el sistema estima y muestra en vivo:

- **6 emociones básicas (modelo de Ekman)** más “neutral”: alegría, tristeza,
  enojo, miedo, asco y sorpresa, con la emoción dominante.
- **Edad y género** estimados.
- **Orientación de la cabeza**: inclinación, giro y cabeceo (roll, yaw, pitch).
- **Posición y número de rostros** en el cuadro.
- **Afecto en el modelo circumplejo**: valencia (agradable–desagradable) y
  activación (calmado–intenso), con su **cuadrante**, el **afecto más cercano**
  (catálogos de 98 y 38 términos), **positividad** y **atención**.
- **Tono de voz** (si se activa el micrófono): pitch, energía y activación vocal.
- **Satisfacción del cliente (estimada)**: combina rostro y voz para una señal de
  satisfacción a lo largo de una interacción, marcando **momentos clave** de
  posible satisfacción o insatisfacción.
- **Registro y exportación a CSV** para analizar la sesión después.

## Para qué sirve (finalidad)

El sistema tiene un doble propósito:

1. **Docente / divulgativo:** mostrar de forma tangible cómo funciona la
   **inteligencia artificial aplicada a la visión y al afecto**, directamente en
   el navegador y sin conocimientos técnicos previos. Sirve para que estudiantes
   experimenten con conceptos de IA, emociones y análisis de datos.
2. **Investigación de la experiencia de servicio:** explorar, con apoyo de IA, la
   **satisfacción y las emociones** durante una interacción (por ejemplo,
   cliente–asesor), e identificar momentos clave. Es un prototipo para estudiar
   la calidad percibida del servicio.

## Cómo funciona (en breve)

Usa modelos de IA **ya entrenados** (la biblioteca de código abierto
`face-api.js`, basada en TensorFlow.js) para reconocer el rostro y las
expresiones. Sobre esas salidas, el sistema aplica **reglas y cálculos** para
derivar el afecto, el tono de voz y la satisfacción. Los modelos no aprenden con
el uso: vienen entrenados y solo se usan para analizar.

## Privacidad y límites (importante)

- **Todo es local:** el video y el audio se procesan en el navegador del equipo;
  **no se suben ni se guardan** automáticamente. Solo se exporta lo que la
  persona decida (el CSV).
- **Es una señal exploratoria, no una medición clínica.** La expresión facial no
  equivale al estado interno real; los resultados son aproximaciones útiles para
  aprender y explorar, no diagnósticos.
- Úsese siempre con **consentimiento** de la persona analizada.

## Cómo ejecutarlo

Necesita abrirse desde un servidor local (la cámara no funciona abriendo el
archivo directo). En Mac, doble clic en `iniciar.command`. En general:

```bash
python3 -m http.server 8000
```

y abrir <http://localhost:8000>. Ver `INSTRUCCIONES.md` para el paso a paso y
`GLOSARIO.md` para el significado de cada término.

## Archivos del sistema

| Archivo | Rol |
|---|---|
| `index.html` | Estructura de la interfaz. |
| `style.css` | Estilos (apariencia). |
| `app.js` | Lógica principal: cámara, modelos, afecto y satisfacción. |
| `voice.js` | Análisis de tono de voz. |
| `affect.js` | Cuadrantes, afectos (98/38), positividad y atención. |
| `README.md` | Resumen técnico y de uso. |
| `INSTRUCCIONES.md` | Cómo ejecutarlo paso a paso (Mac). |
| `GLOSARIO.md` | Diccionario de términos. |
| `QUE_HACE_Y_FINALIDAD.md` | Este documento. |
