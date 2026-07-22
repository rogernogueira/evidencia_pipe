"""Scripts Lua atômicos.

Todos rodam server-side no Redis, garantindo atomicidade entre GET/SET/DEL e entre
a seleção do vencedor da fila e a criação do lock (elimina condições de corrida).

Nota sobre chaves: ``TRY_ACQUIRE``/``ENQUEUE``/``CANCEL`` acessam as chaves de
solicitação (``gpu:{r}:request:{id}``) por prefixo passado em ARGV, e não via KEYS.
Isso é seguro em uma instância Redis única (nosso caso: DB2 dedicado), mas NÃO é
Redis-Cluster-safe (chaves em slots distintos). Ver "limitações conhecidas" no README.

A prioridade EFETIVA (aging) é calculada dentro do Lua a partir de ``now`` passado
pelo cliente (evita depender de ``TIME`` não-determinístico e mantém testabilidade).
Seleção O(n) sobre a fila — aceitável para filas pequenas (poucas dezenas). Ver README.
"""

from __future__ import annotations

# Fator de composição do score do ZSET: priority * SCORE_PRIORITY_FACTOR + sequence.
# Mantém a fila aproximadamente ordenada por (prioridade, chegada) para exibição;
# a seleção real (com aging) é recalculada no Lua.
SCORE_PRIORITY_FACTOR = 10_000_000_000  # 1e10


# --------------------------------------------------------------------------- #
# ENQUEUE: INCR sequence + grava request key (TTL) + ZADD — atômico.
#   KEYS[1] = sequence key
#   KEYS[2] = queue key (zset)
#   ARGV[1] = request_id
#   ARGV[2] = request_json (sem 'sequence'; o script injeta)
#   ARGV[3] = priority (int)
#   ARGV[4] = request_ttl_ms
#   ARGV[5] = request_key_prefix
#   ARGV[6] = score_priority_factor
#   -> retorna sequence
# --------------------------------------------------------------------------- #
ENQUEUE = """
local seq = redis.call('INCR', KEYS[1])
local req = cjson.decode(ARGV[2])
req.sequence = seq
local factor = tonumber(ARGV[6])
local score = tonumber(ARGV[3]) * factor + seq
redis.call('SET', ARGV[5] .. ARGV[1], cjson.encode(req), 'PX', tonumber(ARGV[4]))
redis.call('ZADD', KEYS[2], score, ARGV[1])
return seq
"""


# --------------------------------------------------------------------------- #
# TRY_ACQUIRE: limpa órfãos, elege vencedor (prioridade efetiva + FIFO), e SÓ se
# o chamador for o vencedor e o lock estiver livre, adquire — tudo atômico.
#   KEYS[1] = lock key
#   KEYS[2] = owner key
#   KEYS[3] = queue key
#   ARGV[1]  = request_id (chamador)
#   ARGV[2]  = token
#   ARGV[3]  = now (epoch seconds, float)
#   ARGV[4]  = lock_ttl_ms
#   ARGV[5]  = owner_json
#   ARGV[6]  = request_key_prefix
#   ARGV[7]  = aging_enabled ('1'/'0')
#   ARGV[8]  = aging_interval_seconds
#   ARGV[9]  = aging_step
#   ARGV[10] = min_effective_priority
#   -> retorna { code, detail, position }
#      code: 1=adquirido, 0=aguardando, -1=não está na fila (órfão/cancelado)
#      detail: 'acquired' | 'locked' | 'not_next' | 'not_queued'
#      position: 1-based do chamador na ordem efetiva (0 se ausente)
# --------------------------------------------------------------------------- #
TRY_ACQUIRE = """
local members = redis.call('ZRANGE', KEYS[3], 0, -1)
local now = tonumber(ARGV[3])
local aging = ARGV[7] == '1'
local interval = tonumber(ARGV[8])
local step = tonumber(ARGV[9])
local mineff = tonumber(ARGV[10])

-- coleta candidatos vivos e calcula prioridade efetiva
local cand = {}
for _, id in ipairs(members) do
    local raw = redis.call('GET', ARGV[6] .. id)
    if not raw then
        redis.call('ZREM', KEYS[3], id)  -- órfão: sem request key
    else
        local ok, req = pcall(cjson.decode, raw)
        if ok and req then
            local eff = req.priority
            if aging and interval > 0 then
                local waited = now - (req.enqueued_at or now)
                if waited < 0 then waited = 0 end
                local steps = math.floor(waited / interval)
                eff = req.priority - steps * step
                if eff < mineff then eff = mineff end
            end
            cand[#cand + 1] = { id = id, eff = eff, seq = req.sequence or 0 }
        else
            redis.call('ZREM', KEYS[3], id)  -- payload corrompido
        end
    end
end

-- ordena por (eff asc, seq asc): menor prioridade efetiva vence; empate = FIFO
table.sort(cand, function(a, b)
    if a.eff ~= b.eff then return a.eff < b.eff end
    return a.seq < b.seq
end)

-- posição do chamador
local position = 0
for i, c in ipairs(cand) do
    if c.id == ARGV[1] then position = i break end
end
if position == 0 then
    return { -1, 'not_queued', 0 }
end
if position ~= 1 then
    return { 0, 'not_next', position }
end

-- chamador é o próximo: o lock está livre?
if redis.call('EXISTS', KEYS[1]) == 1 then
    return { 0, 'locked', position }
end

-- adquire atomicamente
redis.call('SET', KEYS[1], ARGV[2], 'PX', tonumber(ARGV[4]))
redis.call('SET', KEYS[2], ARGV[5], 'PX', tonumber(ARGV[4]))
redis.call('ZREM', KEYS[3], ARGV[1])
redis.call('DEL', ARGV[6] .. ARGV[1])
return { 1, 'acquired', 1 }
"""


# --------------------------------------------------------------------------- #
# RELEASE: só libera se o token confere (nunca libera lock de outro processo).
#   KEYS[1] = lock key, KEYS[2] = owner key ; ARGV[1] = token
#   -> 1 liberado, 0 token não confere
# --------------------------------------------------------------------------- #
RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    redis.call('DEL', KEYS[1])
    local o = redis.call('GET', KEYS[2])
    if o then
        local ok, owner = pcall(cjson.decode, o)
        if ok and owner and owner.token == ARGV[1] then
            redis.call('DEL', KEYS[2])
        end
    end
    return 1
else
    return 0
end
"""


# --------------------------------------------------------------------------- #
# RENEW: renova TTL do lock+owner validando o token (heartbeat do lease).
#   KEYS[1] = lock key, KEYS[2] = owner key
#   ARGV[1] = token, ARGV[2] = ttl_ms, ARGV[3] = owner_json (com last_heartbeat novo)
#   -> 1 renovado, 0 propriedade perdida
# --------------------------------------------------------------------------- #
RENEW = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]))
    redis.call('SET', KEYS[2], ARGV[3], 'PX', tonumber(ARGV[2]))
    return 1
else
    return 0
end
"""


# --------------------------------------------------------------------------- #
# WAIT_HEARTBEAT: renova o TTL da request key (mantém a solicitação viva na fila).
#   ARGV[1] = request key, ARGV[2] = ttl_ms
#   -> 1 renovado, 0 solicitação já expirou/foi removida (órfã/cancelada)
# --------------------------------------------------------------------------- #
WAIT_HEARTBEAT = """
if redis.call('EXISTS', ARGV[1]) == 1 then
    redis.call('PEXPIRE', ARGV[1], tonumber(ARGV[2]))
    return 1
else
    return 0
end
"""


# --------------------------------------------------------------------------- #
# CANCEL: remove da fila + apaga request key. NUNCA toca o lock. Idempotente.
#   KEYS[1] = queue key ; ARGV[1] = request_id, ARGV[2] = request key
#   -> 1 se removeu da fila (ou request key existia), 0 se nada existia
# --------------------------------------------------------------------------- #
CANCEL = """
local removed = redis.call('ZREM', KEYS[1], ARGV[1])
local had = redis.call('DEL', ARGV[2])
if removed > 0 or had > 0 then return 1 else return 0 end
"""
