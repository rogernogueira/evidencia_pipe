#!/usr/bin/env bash
# Remove as units systemd do evidencia_pipe: para o stack, desabilita e apaga os
# arquivos de /etc/systemd/system/. NÃO mexe nos contêineres Docker nem nos dados.
#
# Uso:  sudo ./uninstall.sh
set -euo pipefail

UNIT_DIR="/etc/systemd/system"
UNITS=(
  evidencia.target
  evidencia-api.service
  evidencia-worker-light.service
  evidencia-worker-gpu.service
  evidencia-compose.service
)

if [[ $EUID -ne 0 ]]; then
  echo "Precisa de root (use sudo)." >&2
  exit 1
fi

echo "Parando e desabilitando..."
systemctl stop evidencia.target 2>/dev/null || true
systemctl disable "${UNITS[@]}" 2>/dev/null || true

echo "Removendo arquivos..."
for u in "${UNITS[@]}"; do
  rm -f "${UNIT_DIR}/${u}" && echo "  - ${u}"
done

systemctl daemon-reload
echo "Removido. (Contêineres Docker e dados não foram tocados.)"
