# Glosario del proyecto · Detector de Emociones (rostro + voz)

Guía rápida de términos para apoyarse durante la presentación. Cada término trae
una explicación sencilla y, cuando ayuda, dónde se ve en la app.

---

## 1. Conceptos generales

**Inteligencia artificial (IA).** Programas que aprenden patrones a partir de
datos para hacer una tarea, en este caso reconocer rostros, expresiones y voz.

**Modelo.** El "cerebro" ya entrenado que recibe una entrada (imagen o audio) y
devuelve una predicción. La app usa varios modelos a la vez.

**Inferencia.** El momento en que el modelo analiza la entrada y entrega un
resultado. Aquí ocurre en tiempo real, varias veces por segundo.

**Procesamiento local (del lado del cliente).** Todo el análisis sucede en el
navegador del propio equipo; el video y el audio no se suben a ningún servidor.

**face-api.js / TensorFlow.js.** Las bibliotecas de código abierto que permiten
correr los modelos de rostro directamente en el navegador.

**CDN.** Red de servidores que entrega archivos comunes (aquí, los modelos de
IA) de forma rápida. Por eso la primera vez se necesita internet.

**HTTPS / contexto seguro.** Conexión cifrada. El navegador solo permite usar la
cámara y el micrófono en páginas seguras (https o localhost).

---

## 2. Análisis del rostro

**Detección de rostro.** Localizar dónde está la cara dentro de la imagen.

**Recuadro (bounding box).** El rectángulo que marca el rostro detectado.

**Puntos faciales (landmarks).** 68 puntos de referencia (ojos, cejas, nariz,
boca, contorno) que permiten medir la geometría de la cara.

**Emociones de Ekman.** Las seis emociones básicas universales según Paul Ekman:
alegría, tristeza, enojo, miedo, asco y sorpresa (la app añade "neutral").

**Edad y género (estimados).** Predicción aproximada del modelo; es una
estimación, no un dato real de la persona.

**Orientación de la cabeza (head pose).** Hacia dónde mira la cabeza, en tres
giros:
- **Roll (inclinación):** ladear la cabeza hacia un hombro.
- **Yaw (giro):** voltear a izquierda o derecha.
- **Pitch (cabeceo):** subir o bajar la barbilla.

**EAR (Eye Aspect Ratio).** Medida de apertura del ojo a partir de los puntos
faciales; sirve para saber si los ojos están abiertos o cerrados.

**Atención.** Estimación de 0 a 100 % de cuánto está la persona mirando de
frente y con los ojos abiertos (combina EAR y la orientación de la cabeza).

---

## 3. Afecto (cómo se siente la persona)

**Valencia (valence).** Qué tan agradable o desagradable es la emoción, de
negativo (−1) a positivo (+1).

**Activación (arousal).** Qué tan intensa o calmada es la emoción, de baja a
alta energía.

**Modelo circumplejo.** Representación de las emociones en un círculo con dos
ejes: valencia (horizontal) y activación (vertical). Es el gráfico circular de
la app.

**Cuadrantes.** Las cuatro zonas del círculo (p. ej. valencia positiva con
activación alta = entusiasmo; valencia negativa con activación alta = tensión).

**98 afectos (Russell, Scherer y Klaus) / 38 afectos (Paltoglou y Thelwall).**
Dos catálogos de palabras emocionales ubicadas en el círculo. La app dice cuál
es la más cercana a tu punto de valencia y activación.

**Positividad.** Resumen de 0 a 100 % de qué tan positiva es la emoción,
derivado de la valencia.

---

## 4. Análisis de voz

**Prosodia.** El "cómo" se dice algo: tono, volumen, ritmo y entonación (no las
palabras en sí).

**Pitch (f0, frecuencia fundamental).** Qué tan agudo o grave es el tono de voz,
medido en hercios (Hz).

**Energía (volumen).** Qué tan fuerte se habla.

**Activación vocal.** Estimación de qué tan intensa suena la voz, a partir de
energía, nivel de pitch y su variación.

**Autocorrelación.** Técnica matemática que la app usa para calcular el pitch a
partir de la onda de sonido.

**Web Audio API.** Herramienta del navegador que permite analizar el micrófono
en tiempo real.

---

## 5. Fusión e interacción cliente–servicio

**Fusión multimodal.** Combinar dos fuentes de información (rostro y voz) en una
sola lectura más completa.

**Satisfacción del cliente (estimada).** Señal que combina la valencia facial
del cliente con su activación vocal cuando habla, en una escala de −1 a +1.

**Momentos clave.** Instantes que la app marca automáticamente como posible
satisfacción o insatisfacción (por ejemplo, una caída de valencia o una voz
tensa), con su hora.

**Botón de rol (Cliente / Asesor).** Marca manual de quién habla en cada momento.

**Diarización de hablantes.** Separar automáticamente quién habla en un audio.
Es difícil y poco fiable con un solo micrófono; por eso usamos el botón de rol
en su lugar.

**Suavizado (media móvil / EMA).** Promediar los valores recientes para que la
señal no salte bruscamente y se vea estable.

**Umbral (threshold).** Valor de corte que decide la etiqueta (por ejemplo, por
encima de cierto número se considera "satisfecho").

**Sesgo (bias).** Tendencia sistemática de un modelo a equivocarse hacia un lado.
Aquí el modelo de rostro tiende a marcar "alegría" de más, y lo corregimos.

---

## 6. Datos para análisis

**Muestreo.** Tomar una lectura cada cierto intervalo (por ejemplo, cada 1 s).

**Registro (log).** La lista de lecturas que se va guardando durante la sesión.

**CSV.** Archivo de tabla (se abre en Excel) con una fila por lectura; sirve para
analizar la sesión después.

---

## 7. Publicación

**GitHub Pages.** Servicio gratuito que publica la app como página web con una
dirección (URL) para compartir.

**Caché.** Memoria temporal del navegador que guarda los modelos tras la primera
carga, para que después abra más rápido y sin depender tanto de internet.

---

## 8. Ética y límites (importante mencionarlo)

**Consentimiento informado.** Avisar y obtener permiso de la persona antes de
analizar su rostro y su voz.

**LFPDPPP.** Ley Federal de Protección de Datos Personales en Posesión de los
Particulares (México); rige el manejo de datos personales.

**Privacidad.** En esta app todo es local: no se sube ni se guarda video ni
audio.

**Señal exploratoria, no medición clínica.** Los resultados (afecto, voz,
satisfacción) son aproximaciones útiles para explorar, no diagnósticos ni
verdades sobre lo que la persona siente en realidad.

---

## 9. Preguntas que les pueden hacer (y cómo responder)

**¿Es lo mismo que MorphCast?** No. Nos inspiramos en sus categorías (valencia,
arousal, cuadrantes, 98 y 38 afectos, atención), pero aquí están implementadas
con modelos abiertos y aproximaciones propias; MorphCast es un producto
comercial con modelos propietarios.

**¿De verdad lee las emociones de la persona?** No literalmente. Estima
expresiones y señales de voz; es una interpretación probabilística, no una
lectura de la mente.

**¿Se guarda o se sube el video?** No. Todo el análisis ocurre en el navegador
del equipo; nada sale de él.

**¿Por qué hay un botón de Cliente / Asesor?** Porque separar voces
automáticamente con un micrófono no es confiable; así marcamos manualmente quién
habla y la voz se atribuye correctamente.

**¿Por qué a veces marcaba "satisfecho" casi siempre?** El modelo de rostro
tiende a detectar "alegría" de más; lo corregimos restando ese sesgo y ajustando
los umbrales.

**¿Necesita internet?** Solo la primera vez, para descargar los modelos; después
quedan en caché.
