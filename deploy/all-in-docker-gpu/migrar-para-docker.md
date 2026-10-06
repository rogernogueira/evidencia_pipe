# Migrar a produção desta máquina para "tudo via Docker"

Troca o modelo híbrido atual (API/workers no host via systemd + Qdrant legado avulso)
pelo stack `all-in-docker-gpu`, **reusando os dados de produção in-place** (MinIO e
Qdrant existentes — sem cópia). Usa `docker-compose.prod.yml` como override.

> Os comandos `docker`/`systemctl` abaixo são executados por **você** (o assistente
> não tem permissão). As validações marcadas "(assistente)" eu faço por leitura.

## 0. Rede de segurança (antes de tudo)

O stack novo abre o MESMO `/app/minerU/qdrant_storage`; o Qdrant 1.17.1 pode migrar o
formato on-disk. O **backup já existe** (`/app/backup-rdapp-20261006-151212`,
snapshot + MinIO). Confirme que está lá antes de prosseguir — é o rollback real do dado.

## 1. Aposentar os processos do host (systemd)

```bash
sudo systemctl disable --now evidencia-api evidencia-worker-light evidencia-worker-gpu
# impede o compose legado de ressubir os contêineres antigos no boot:
sudo systemctl disable --now evidencia-compose.service
```

## 2. Parar os contêineres legados que seguram dados/portas/GPU

```bash
# Qdrant legado — LIBERA /app/minerU/qdrant_storage (senão o novo não abre: lock)
docker stop mineru_qdrant
# Os demais já devem estar parados do teste anterior; garanta que seguem parados
# (NÃO os suba): evidencia_minio, evidencia_flower, evidencia_mineru_pipeline,
# vllm-bge-m3, vllm-bge-m3-sparse
docker ps --format '{{.Names}}' | grep -E 'mineru_qdrant|evidencia_(minio|flower|mineru_pipeline)|vllm-bge-m3' || echo 'legados parados OK'
```

## 3. Subir o stack de produção (base + override)

```bash
cd deploy/all-in-docker-gpu
docker compose -f docker-compose.gpu.yml -f docker-compose.prod.yml up -d
docker compose -f docker-compose.gpu.yml -f docker-compose.prod.yml ps
```

## 4. Validar (assistente, por leitura)

```bash
# Qdrant com os dados reais:
curl -s http://127.0.0.1:6333/collections/evidencia_chunks \
  | python3 -c "import sys,json;r=json.load(sys.stdin)['result'];print('points:',r['points_count'],'status:',r['status'])"
# esperado: points: 42853  status: green

# MinIO com o bucket real (objetos presentes):  console em 127.0.0.1:9001
# API no ar nas duas portas:
curl -s -o /dev/null -w "api 8020 -> %{http_code}\n" http://127.0.0.1:8020/docs
curl -s -o /dev/null -w "api 8181 -> %{http_code}\n" http://127.0.0.1:8181/docs
```

Uma busca de fumaça (com token de admin do DSpace) fecha a validação fim a fim.

## Rollback (volta ao híbrido)

```bash
cd deploy/all-in-docker-gpu
docker compose -f docker-compose.gpu.yml -f docker-compose.prod.yml down
docker start mineru_qdrant
sudo systemctl enable --now evidencia-compose.service
sudo systemctl enable --now evidencia-api evidencia-worker-light evidencia-worker-gpu
```

Se o Qdrant on-disk tiver migrado e algo quebrar no legado, restaure o snapshot do
backup (ver `../all-in-docker-cpu/host-prep/backup-restore.md`).

## Notas

- **Redis/job_store começa limpo**: o acervo (MinIO+Qdrant) é reusado, mas o histórico
  de status dos jobs não. É de propósito — evita re-disparar tarefas pendentes do broker.
- **API em `:8181` (0.0.0.0)**: espelha o que a produção no host expunha. Se só há
  consumidor local, troque por `127.0.0.1:8181:8020` no override.
- **Nunca rode os dois modelos ao mesmo tempo**: disputam a GPU, as portas
  `8000/8001/8012/9000/9001/5555` e o lock do storage do Qdrant.
- **Credenciais MinIO**: o stack lê o `.env` da raiz, o mesmo que a produção usava →
  casam com o volume reusado.
