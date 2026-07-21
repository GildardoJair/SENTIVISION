#!/bin/bash
# Reorganización de sistema_emociones_IRIS-AURA
# Mueve SOLO experimentos, evaluación y docs. La producción
# (index.html, app.js, voice.js, affect.js, metrics.js,
# emotion-fusion.js, facemesh.js, style.css, README.md) NO se toca.
#
# Uso: colócalo en la raíz del proyecto y corre:
#   bash reorganizar_proyecto.sh
#
set -e

# NOTA: usamos mv normal, no git mv. git mv exige que el archivo ya esté
# rastreado (git add) y en este proyecto no lo estaban. Con mv normal +
# "git add -A" al final, git detecta los renombres igual (por similitud de
# contenido) al hacer commit o `git status`, así que no se pierde nada.
MV="mv"
IS_GIT=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IS_GIT=1
  echo "Repo git detectado: se usará 'mv' + 'git add -A' al final."
else
  echo "No es un repo git: se usará 'mv' normal."
fi

mkdir -p docs experimentos evaluacion

echo
echo "== Moviendo documentación a docs/ =="
for f in GLOSARIO.md INSTRUCCIONES.md QUE_HACE_Y_FINALIDAD.md NOVEDADES_FASE2.md; do
  [ -f "$f" ] && $MV "$f" "docs/$f" && echo "  $f -> docs/"
done

echo
echo "== Moviendo experimentos (alternar-*, explorar-*) a experimentos/ =="
for f in alternar-i1.html alternar-i1.js \
         alternar-i5.html alternar-i5.js \
         alternar-emociones.html alternar-emociones.js \
         alternar-edad-human.html \
         explorar-human-emotion.html explorar-human-emotion.js; do
  [ -f "$f" ] && $MV "$f" "experimentos/$f" && echo "  $f -> experimentos/"
done

echo
echo "== Moviendo evaluación (evaluar-dataset, facemesh-image, analizar-sesion) a evaluacion/ =="
for f in evaluar-dataset.html evaluar-dataset.js facemesh-image.js analizar-sesion.html; do
  [ -f "$f" ] && $MV "$f" "evaluacion/$f" && echo "  $f -> evaluacion/"
done

echo
echo "== _pruebas-diagnostico-interno/ se deja igual (ya estaba aislado) =="

echo
echo "======================================================================"
echo "AUDITORÍA: buscando referencias relativas a archivos locales que"
echo "puedan haberse roto al mover de raíz a una subcarpeta."
echo "(Ignora las URLs http/https, esas no cambian.)"
echo "======================================================================"

for dir in docs experimentos evaluacion; do
  for f in "$dir"/*.html "$dir"/*.js; do
    [ -f "$f" ] || continue
    matches=$(grep -nE '(src|href)\s*=\s*"(\./|[a-zA-Z0-9_-]+\.(js|html|json|css))"' "$f" 2>/dev/null | grep -v 'http')
    if [ -n "$matches" ]; then
      echo
      echo "⚠  $f — posibles rutas relativas a revisar/ajustar a ../:"
      echo "$matches" | sed 's/^/    /'
    fi
  done
done

echo
echo "Listo. Revisa las advertencias de arriba: cualquier ruta relativa a"
echo "un archivo que sigue en la raíz (p.ej. affect.js, voice.js) ahora"
echo "necesita el prefijo ../ dentro de su nueva subcarpeta."
echo
echo "Estructura final esperada:"
tree -L 2 2>/dev/null || find . -maxdepth 2 -not -path '*/.git*' | sort

if [ "$IS_GIT" -eq 1 ]; then
  echo
  echo "======================================================================"
  echo "git add -A para que git registre los archivos (y detecte renombres)"
  echo "======================================================================"
  git add -A
  echo
  echo "git status (busca líneas 'renamed:' — confirma que detectó el movimiento):"
  git status
  echo
  echo "Nada se ha comiteado todavía. Revisa 'git status' arriba y cuando estés"
  echo "conforme: git commit -m \"Reorganiza proyecto: docs/, experimentos/, evaluacion/\""
fi
