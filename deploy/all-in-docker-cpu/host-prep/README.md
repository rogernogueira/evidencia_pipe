# Preparação do host (CPU-only) — rdapp

Passos de SO a rodar **uma vez**, como root, antes de subir o stack. Derivados do
diagnóstico de `rdapp` (Ubuntu 24.04, 16 vCPU, 15 GiB RAM, sem GPU, UFW inativo).
Todos são idempotentes.

## 1. Atualizar e reiniciar (reboot pendente)

O host tem `libc6` + kernels novos pendentes (`É necessário reiniciar`). Faça numa
janela ANTES do deploy:

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

## 2. Parâmetros de kernel (sysctl)

Para o Redis (overcommit) e o Qdrant (mmap). Ver `99-evidencia-pipe.conf`.

```bash
sudo cp 99-evidencia-pipe.conf /etc/sysctl.d/99-evidencia-pipe.conf
sudo sysctl --system
# conferir:
sysctl vm.overcommit_memory vm.max_map_count vm.swappiness
```

## 3. Desativar Transparent Huge Pages (THP)

O Redis recomenda THP desligado (latência/uso de memória no fork do BGSAVE).
Persistir via systemd (sobrevive a reboot):

```bash
sudo tee /etc/systemd/system/disable-thp.service >/dev/null <<'EOF'
[Unit]
Description=Disable Transparent Huge Pages (THP)
DefaultDependencies=no
After=sysinit.target local-fs.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled'
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/defrag'

[Install]
WantedBy=basic.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now disable-thp.service
# conferir (deve mostrar [never]):
cat /sys/kernel/mm/transparent_hugepage/enabled
```

## 4. (Recomendado) Aumentar o swap para 8 GiB

O gargalo do host é RAM: 15 GiB com o swap atual de só 2 GiB. bge-m3 (~4,6 GB) e os
modelos do MinerU competem pela RAM. Subir o swap dá folga para picos sem OOM-kill.
O swap atual é um arquivo (`/swap.img`, 2 GiB, em `/etc/fstab`):

```bash
sudo swapoff /swap.img
sudo fallocate -l 8G /swap.img      # ou: dd if=/dev/zero of=/swap.img bs=1M count=8192
sudo chmod 600 /swap.img
sudo mkswap /swap.img
sudo swapon /swap.img
free -h
# /etc/fstab já tem a entrada do /swap.img — não precisa mexer.
```

> Há 86 GiB livres em `/`, então 8 GiB de swap cabem. Mas o LVM está 100% alocado
> (`VFree 0`): não dá para estender o filesystem sem adicionar disco — monitore o
> crescimento dos volumes `minio_data`/`qdrant_data`.

## 5. Firewall (nota)

O UFW está **inativo** e a política `INPUT` é `ACCEPT`. O `docker-compose.cpu.yml`
agora publica todas as portas em `127.0.0.1` justamente por isso — nada do stack
fica exposto na LAN sem você decidir. Se for abrir a API para outras máquinas,
prefira um reverse proxy com TLS/auth em vez de publicar a porta direto.
