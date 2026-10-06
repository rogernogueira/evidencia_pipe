# Backup e restore dos dados (Qdrant + MinIO)

Como mover o **acervo já processado** entre máquinas sem reprocessar MinerU/BGE.
Os dados vivem em dois stores; o resto (Redis/job_store, filas) é descartável.

| Store | Conteúdo | Formato de backup | Restore |
|---|---|---|---|
| **Qdrant** | vetores + payloads (chunks indexados) | **snapshot** da coleção (API) | upload via API |
| **MinIO** | PDFs + JSON/markdown do MinerU + manifestos | **tar do `/data` bruto** | untar no volume |

> **Por que métodos diferentes:** o snapshot é o formato portátil oficial do Qdrant
> e é consistente **sem parar** o serviço; o `/data` do MinIO são objetos imutáveis,
> então o tar ao vivo é seguro. Só o restore do MinIO para o serviço por segundos
> (troca o `/data` por baixo).

## ⚠️ Duas travas de correção

1. **Versão do Qdrant tem que casar** entre origem e destino. A origem de referência
   roda **1.17.1** → o `docker-compose.remote.yml`/`.cpu.yml` fixam `qdrant/qdrant:v1.17.1`.
   Snapshot tolera mais que dir bruto, mas mantenha a paridade. Confira a origem com
   `curl -s http://<qdrant>:6333/ | python3 -c "import sys,json;print(json.load(sys.stdin)['version'])"`.
2. **Nomes batem com o `.env`**: `QDRANT_COLLECTION` (`evidencia_chunks`) e
   `MINIO_BUCKET` (`evidencia-pipe`) iguais nos dois lados.

---

## 1. Gerar o backup (na máquina de origem)

Ajuste `QDRANT`, o nome do volume do MinIO e a coleção conforme o ambiente.

```bash
set -e
QDRANT=http://192.168.105.8:6333          # endpoint do Qdrant de origem
COLL=evidencia_chunks                      # = QDRANT_COLLECTION
MINIO_VOL=evidencia_pipe_minio_data        # docker volume do MinIO de origem
DIR=/app/backup-rdapp-$(date +%Y%m%d-%H%M%S); mkdir -p "$DIR"

# Qdrant: cria snapshot, baixa e remove o do servidor (não deixa lixo em produção)
SNAP=$(curl -s -X POST "$QDRANT/collections/$COLL/snapshots" | python3 -c "import sys,json;print(json.load(sys.stdin)['result']['name'])")
curl -s "$QDRANT/collections/$COLL/snapshots/$SNAP" -o "$DIR/qdrant_${COLL}.snapshot"
curl -s -X DELETE "$QDRANT/collections/$COLL/snapshots/$SNAP" >/dev/null

# MinIO: tar do /data bruto (volume montado read-only)
docker run --rm -v "$MINIO_VOL":/data:ro -v "$DIR":/out alpine \
  tar czf /out/minio_data.tar.gz -C /data .

# Integridade
( cd "$DIR" && sha256sum qdrant_${COLL}.snapshot minio_data.tar.gz > SHA256SUMS )
echo "Backup em: $DIR"; ls -lh "$DIR"
```

Tamanhos de referência (acervo de 42.853 pontos): snapshot ~455 MB, MinIO ~2,2 GB.

## 2. Transferir

O SSH do rdapp está na **porta 25000**.

```bash
scp -P 25000 -r /app/backup-rdapp-<timestamp> usuario@172.16.24.66:/restore/
# no destino:
cd /restore/backup-rdapp-<timestamp> && sha256sum -c SHA256SUMS   # "OK" nos dois
```

## 3. Restaurar (no rdapp)

A partir de `deploy/all-in-docker-cpu/`, com `BK` = caminho do backup transferido.

```bash
BK=/restore/backup-rdapp-<timestamp>
docker compose -f docker-compose.remote.yml up -d qdrant minio

# --- MinIO: troca o /data com o serviço parado (segundos) ---
docker compose -f docker-compose.remote.yml stop minio
docker run --rm -v evidencia-pipe-remote_minio_data:/data -v "$BK":/in:ro alpine \
  sh -c 'rm -rf /data/* && tar xzf /in/minio_data.tar.gz -C /data'
docker compose -f docker-compose.remote.yml up -d minio

# --- Qdrant: sobe a coleção do snapshot, sem parar ---
curl -s -X POST 'http://127.0.0.1:6333/collections/evidencia_chunks/snapshots/upload?priority=snapshot' \
  -F "snapshot=@$BK/qdrant_evidencia_chunks.snapshot"
```

> O nome do volume do MinIO depende do `name:` do compose: `remote` →
> `evidencia-pipe-remote_minio_data`; `cpu` → `evidencia-pipe-cpu_minio_data`.
> Confira com `docker volume ls | grep minio`.

## 4. Validar (tem que bater com a origem)

```bash
curl -s http://127.0.0.1:6333/collections/evidencia_chunks \
  | python3 -c "import sys,json;r=json.load(sys.stdin)['result'];print('points:',r['points_count'],'status:',r['status'])"
# esperado: points: 42853  status: green

docker compose -f docker-compose.remote.yml up -d          # api + workers
curl -s http://127.0.0.1:8020/api/status | python3 -m json.tool | head -40
```

`points` igual ao da origem + `/api/status` verde = restore completo. MinerU e BGE
só são necessários para processar documentos **novos** depois disso, não para o restore.
