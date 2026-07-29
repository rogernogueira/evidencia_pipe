#!/usr/bin/env bash
# Instala as units systemd do evidencia_pipe: copia para /etc/systemd/system/,
# recarrega o systemd, habilita (auto-start no boot) e sobe o stack.
#
# Uso:  sudo ./install.sh            # instala, habilita e inicia
#       sudo ./install.sh --no-start # instala e habilita, sem iniciar agora
set -euo pipefail

UNIT_DIR="/etc/systemd/system"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNITS=(
  evidencia-compose.service
  evidencia-api.service
  evidencia-worker-light.service
  evidencia-worker-gpu.service
  evidencia.target
)

if [[ $EUID -ne 0 ]]; then
  echo "Precisa de root (use sudo)." >&2
  exit 1
fi

# Sanidade: os units assumem WorkingDirectory=/app/evidencia_pipe e uv em
# /root/.local/bin/uv. Avisa (sem abortar) se o host divergir.
[[ -d /app/evidencia_pipe ]] || echo "AVISO: /app/evidencia_pipe não existe — ajuste os units antes." >&2
[[ -x /root/.local/bin/uv ]] || echo "AVISO: /root/.local/bin/uv não encontrado — ajuste o PATH/ExecStart dos units." >&2

echo "Copiando ${#UNITS[@]} units para ${UNIT_DIR}..."
for u in "${UNITS[@]}"; do
  install -m 0644 "${SRC_DIR}/${u}" "${UNIT_DIR}/${u}"
  echo "  + ${u}"
done

echo "Recarregando o systemd..."
systemctl daemon-reload

echo "Habilitando (auto-start no boot)..."
systemctl enable "${UNITS[@]}"

if [[ "${1:-}" == "--no-start" ]]; then
  echo "Instalado e habilitado (sem iniciar; use: systemctl start evidencia.target)."
else
  echo "Iniciando o stack..."
  systemctl start evidencia.target
  sleep 3
  systemctl --no-pager --plain list-units 'evidencia*'
fi

echo "Pronto."
