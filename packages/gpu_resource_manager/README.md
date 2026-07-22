# gpu-resource-manager

Coordenação **distribuída** de recursos de GPU sobre Redis: exclusão mútua, fila
global com **prioridade + FIFO + aging**, **lease com TTL/heartbeat**, recuperação
após falhas e observabilidade. Biblioteca **independente** — o núcleo conhece apenas
Redis e o protocolo de coordenação; **não** importa Celery, FastAPI, PyTorch, MinerU,
BGE-M3 ou Qdrant. Qualquer processo (worker, subprocesso, script, contêiner) que use
o mesmo Redis e o mesmo nome de recurso participa da mesma coordenação global.

## Instalação

```bash
# como membro do workspace uv (dentro do monorepo)
uv sync

# como dependência local editável (outro projeto no mesmo ambiente)
uv add --editable ./packages/gpu_resource_manager

# como wheel (outro repositório)
uv build packages/gpu_resource_manager        # gera dist/*.whl
uv add ./dist/gpu_resource_manager-0.1.0-py3-none-any.whl
```

Única dependência de runtime: `redis>=5.0`.

## Uso

```python
from gpu_resource_manager import (
    GPUResourceManager, GPUManagerConfig, GPUAcquisitionTimeout,
)

config = GPUManagerConfig.from_env()          # lê GPU_MANAGER_* / GPU_RESOURCE_NAME
manager = GPUResourceManager(config)

with manager.acquire(
    resource="gpu0",
    service="script-externo",
    priority=10,                              # menor número = maior prioridade
    task_id=None,
    document_id=None,
    metadata={"operation": "treinamento-especial"},
) as lease:
    executar_codigo_cuda()
    lease.ensure_valid()                      # confirma a posse antes de um passo crítico
```

O objeto `lease` expõe: `request_id`, `token`, `resource`, `service`, `priority`,
`acquired_at`, `wait_time_seconds`, `is_valid`, `ensure_valid()`, `held_time_seconds`
e `metrics` (dict).

Consulta e administração:

```python
manager.get_status(resource="gpu0")           # ResourceStatus (dono, TTL, próximo)
manager.get_queue(resource="gpu0", limit=100) # lista de QueueEntry (prioridade efetiva)
manager.cancel_request(request_id)            # remove da fila (idempotente); NÃO libera lock
manager.healthcheck()                          # verifica o backend (fail-closed)
```

## CLI

```bash
gpu-manager status  --resource gpu0
gpu-manager queue   --resource gpu0
gpu-manager health
gpu-manager cancel  --request-id <uuid>
```

> **Sem `force-unlock`.** Um desbloqueio forçado pode causar uso simultâneo da GPU se
> o dono ainda estiver ativo. Se um dia for indispensável, deve exigir opção
> explícita, confirmação, verificação de TTL e registro de auditoria.

## Configuração (variáveis de ambiente)

| Variável | Padrão | Descrição |
|---|---|---|
| `GPU_MANAGER_ENABLED` | `true` | Liga/desliga a coordenação no consumidor |
| `GPU_MANAGER_REDIS_URL` | `redis://localhost:6379/2` | Redis **exclusivo** (não use DB 0/1) |
| `GPU_RESOURCE_NAME` | `gpu0` | Recurso padrão |
| `GPU_MANAGER_KEY_PREFIX` | `gpu` | Prefixo das chaves |
| `GPU_MANAGER_LOCK_TTL_SECONDS` | `300` | TTL do lock (deve ser > heartbeat) |
| `GPU_MANAGER_HEARTBEAT_SECONDS` | `30` | Intervalo de renovação do lock |
| `GPU_MANAGER_WAIT_TIMEOUT_SECONDS` | `1800` | Timeout máximo de espera |
| `GPU_MANAGER_POLL_INTERVAL_SECONDS` | `2` | Intervalo de polling |
| `GPU_MANAGER_POLL_JITTER_SECONDS` | `1` | Jitter aleatório (anti-thundering-herd) |
| `GPU_MANAGER_WAIT_LOG_INTERVAL_SECONDS` | `30` | Intervalo dos logs de espera |
| `GPU_MANAGER_REQUEST_TTL_SECONDS` | `120` | TTL da solicitação em espera (órfãs) |
| `GPU_MANAGER_REQUEST_HEARTBEAT_SECONDS` | `30` | Renovação da solicitação em espera |
| `GPU_MANAGER_DEFAULT_PRIORITY` | `50` | Prioridade padrão |
| `GPU_MANAGER_MIN_PRIORITY` / `MAX_PRIORITY` | `0` / `1000` | Intervalo válido |
| `GPU_MANAGER_AGING_ENABLED` | `true` | Liga o aging |
| `GPU_MANAGER_AGING_INTERVAL_SECONDS` | `300` | Intervalo de decremento |
| `GPU_MANAGER_AGING_STEP` | `1` | Passo de decremento por intervalo |
| `GPU_MANAGER_MIN_EFFECTIVE_PRIORITY` | `0` | Piso da prioridade efetiva |
| `GPU_MANAGER_REDIS_SOCKET_TIMEOUT_SECONDS` | `5` | Timeout de socket |
| `GPU_MANAGER_REDIS_CONNECT_TIMEOUT_SECONDS` | `5` | Timeout de conexão |
| `GPU_MANAGER_REDIS_HEALTH_CHECK_INTERVAL_SECONDS` | `30` | Health check do redis-py |

## Protocolo

### Chaves Redis (derivadas do recurso — sem `global` hardcoded)

```
gpu:{resource}:lock        string   SET token NX PX ttl  (única fonte de exclusão)
gpu:{resource}:owner       string   JSON de diagnóstico do dono (TTL = lease)
gpu:{resource}:queue       zset     member=request_id, score=priority*1e10+sequence
gpu:{resource}:sequence    int      INCR — FIFO monotônico por recurso
gpu:{resource}:request:{id} string  metadados + TTL de espera (waiter renova)
```

### Aquisição (tudo atômico via Lua)

`enqueue` → loop de espera → `try_acquire`:

1. varre a fila (O(n)), remove **órfãos** (membros sem `request:{id}`);
2. calcula a **prioridade efetiva** de cada candidato (aging = `max(min_eff,
   base − floor(wait/interval)·step)`), usando `now` do cliente;
3. ordena por `(prioridade_efetiva, sequence)` → menor vence; empate = **FIFO**;
4. se o chamador **for** o vencedor **e** o lock estiver livre, cria o lock
   (`SET NX PX`), grava o `owner` e remove o request da fila — **num único script**.

> Seleção **O(n)** por Lua: aceitável para filas pequenas (dezenas de solicitações),
> que é o caso de uma GPU compartilhada. Para filas grandes, migrar a seleção para
> uma estrutura auxiliar (ex.: manter mínimo por bucket de prioridade) sem mudar a
> API pública.

### Liberação / heartbeat (validam o token)

`release` e `renew` só agem se `GET lock == token` — **um processo nunca libera nem
renova o lock de outro**. O heartbeat renova o TTL; se a renovação falhar (token não
confere), o lease é marcado como perdido e `ensure_valid()` levanta
`GPULockLostError`. Em `kill -9`, o TTL expira e o recurso é recuperado
automaticamente.

## Exceções

`GPUManagerError` (base) · `GPUAcquisitionTimeout` · `GPUBackendUnavailable` ·
`GPULockLostError` · `GPUInvalidConfiguration` · `GPURequestCancelled` ·
`GPUReleaseError`. Todas expõem `.details` (dict) para logs/APIs.

## Observabilidade

- Logger próprio `gpu_resource_manager` (com `NullHandler`; o consumidor conecta seus
  handlers). **Nunca** loga senha/credenciais/payloads sensíveis.
- Hook de eventos via `GPUManagerConfig(on_event=callback)`; eventos:
  `request_created`, `request_waiting`, `lock_acquired`, `lock_released`,
  `lock_timeout`, `lock_lost`, `heartbeat_failed`, `backend_unavailable`,
  `request_cancelled`.
- Integração Prometheus **opcional** em `gpu_resource_manager.metrics_prometheus`
  (extra `[prometheus]`), nunca obrigatória.

## Semântica

- **Não preemptiva**: quem já adquiriu conclui; uma solicitação prioritária só é a
  **próxima** elegível — nunca interrompe um dono em execução (não matamos CUDA de
  terceiros). Preempção é evolução futura.
- **Fail-closed**: sem Redis, `GPUBackendUnavailable` — **nunca** há fallback em
  memória (quebraria a exclusão entre processos).
- **Leases não são reentrantes**: não adquira o mesmo recurso duas vezes aninhado no
  mesmo fluxo (causaria auto-deadlock até o timeout).

## Limitações conhecidas

- Seleção de fila O(n) em Lua (ver acima).
- Assume **uma instância Redis** (DB dedicado). Os scripts acessam chaves de
  solicitação por prefixo em ARGV — **não** é Redis-Cluster-safe (chaves em slots
  distintos). Para cluster, alocar todas as chaves de um recurso no mesmo hash slot
  (hash tags) ou usar um nó dedicado por recurso.
- O lock **não** libera VRAM: o consumidor deve gerenciar o ciclo de vida do modelo
  (ver `BGEModelManager` no evidencia_pipe).

## Múltiplas GPUs (futuro)

Os recursos já são **nomeados** (`gpu0`, `gpu1`, `gpu:rtx4090`, …) e cada um tem seu
próprio conjunto de chaves e fila independentes. Para expandir: cada GPU física vira
um recurso; um agendador de nível acima escolhe o recurso menos ocupado
(`get_status` por recurso) antes de `acquire`. A API pública não muda.

## Testes

```bash
uv run --group test pytest packages/gpu_resource_manager/tests/unit -q      # sem GPU/Redis (fakeredis)
uv run --group test pytest packages/gpu_resource_manager/tests/integration   # requer Redis real
```
