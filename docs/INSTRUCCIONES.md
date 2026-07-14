# Cómo correr el sistema AURA-IRIS (Fase 2)

Instrucciones para probar el sistema en cualquier computadora (Mac, Windows o
Linux). A diferencia de la versión de Fase 1, ahora hay dos formas de
arrancarlo: automática (Mac) o manual (cualquier sistema).

---

## Requisito importante: no abrir el archivo directo

La cámara y los modelos de IA **no funcionan** si abres `index.html` con doble
clic (verás la URL empezar con `file:///...`). Es una restricción de
seguridad de todos los navegadores. Siempre hay que servirlo desde un
servidor local, aunque sea en tu propia máquina — los pasos de abajo lo hacen
por ti.

## Opción A — Mac (automática)

Doble clic en **`iniciar.command`** (la primera vez: clic derecho → Abrir →
confirmar). Abre una Terminal y tu navegador solo. Ver `INSTRUCCIONES.md`
original para el detalle paso a paso.

## Opción B — Windows / Linux / Mac (manual, recomendada para probar Fase 2)

1. Necesitas **Python 3** instalado (ya viene en Mac/Linux; en Windows
   descárgalo de <https://www.python.org/downloads/> y marca "Add to PATH"
   durante la instalación).
2. Abre una terminal (o `cmd`/PowerShell en Windows) **dentro de la carpeta
   del proyecto**:
   ```bash
   cd ruta/a/sistema_emociones_IRIS-AURA
   python3 -m http.server 8000
   ```
   (En Windows a veces el comando es `python` en vez de `python3`.)
3. Abre tu navegador en:
   ```
   http://localhost:8000
   ```
4. Espera a que diga "Modelos listos" (unos segundos la primera vez, descarga
   los modelos de IA desde internet).
5. Presiona **Iniciar cámara** y acepta los permisos de cámara/micrófono.
6. Para detener: `Ctrl + C` en la terminal.

---

## Navegador recomendado

**Google Chrome** es el más probado. Firefox también funciona, pero durante
las pruebas de Fase 2 encontramos un problema de repintado de pantalla
específico de **una combinación Linux + Firefox + ciertos drivers Intel**
(el video se congela visualmente aunque el análisis siga corriendo por
dentro) — es un problema de esa máquina en particular, no del código. Si a
alguien le pasa algo similar (la interfaz se ve "pegada"), probar en Chrome
primero antes de reportarlo como bug.

## Qué probar de nuevo en Fase 2

En el panel de la cámara vas a ver controles nuevos que no estaban en la
versión de Fase 1:

- **"Comparar 68 vs 478 pts"** — superpone los landmarks del modelo anterior
  (face-api, naranja) sobre los nuevos de MediaPipe FaceMesh (verde), con
  contador de puntos en vivo.
- **"I2: usar Blendshapes+audio"** — cambia la fuente de la emoción dominante
  entre la línea base (`faceExpressionNet`) y el clasificador nuevo de fusión
  tardía. El CSV exportado siempre registra **ambas** distribuciones
  completas, esté o no activado, para poder comparar F1/ECE después.
- Panel **"Calidad de sistema (Nivel 2)"** — `detection_rate`, entropía de
  expresión y `flip_rate` en vivo, con alerta en rojo si cruzan el umbral de
  la Sección 5.2 del reporte de diagnóstico.

## Si algo falla

- **Se queda en "Cargando modelos…" para siempre:** revisa la consola del
  navegador (F12 → pestaña Console) y confirma que no falte ningún archivo
  `.js` en la carpeta (compáralo contra el árbol de archivos del proyecto).
- **"No se pudo cargar face-api.js":** revisa tu conexión a internet (los
  modelos se descargan de un CDN la primera vez).
- **La cámara no enciende:** revisa permisos del navegador y recarga con
  `Ctrl+Shift+R` (fuerza recarga sin caché).
