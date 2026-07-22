"""Cenário 24 (§39): configuração inválida é rejeitada; from_env/from_dict."""

import pytest

from gpu_resource_manager import GPUInvalidConfiguration, GPUManagerConfig


@pytest.mark.parametrize("kw", [
    {"lock_ttl_seconds": 10, "heartbeat_seconds": 20},           # ttl <= hb
    {"request_ttl_seconds": 10, "request_heartbeat_seconds": 20},  # req ttl <= req hb
    {"poll_jitter_seconds": -1},                                  # jitter negativo
    {"min_priority": 10, "max_priority": 5},                      # intervalo invertido
    {"default_priority": 9999},                                   # fora do intervalo
    {"resource_name": "bad name!"},                               # nome inválido
    {"redis_url": "not-a-url"},                                   # url inválida
    {"aging_step": -1},                                           # aging inconsistente
])
def test_invalid_config_rejected(kw):
    with pytest.raises(GPUInvalidConfiguration):
        GPUManagerConfig(**kw)


def test_from_env_reads_gpu_manager_vars():
    env = {
        "GPU_MANAGER_REDIS_URL": "redis://h:6379/2",
        "GPU_RESOURCE_NAME": "gpu:rtx4090",
        "GPU_MANAGER_LOCK_TTL_SECONDS": "600",
        "GPU_MANAGER_HEARTBEAT_SECONDS": "60",
        "GPU_MANAGER_DEFAULT_PRIORITY": "40",
        "GPU_MANAGER_AGING_ENABLED": "false",
    }
    cfg = GPUManagerConfig.from_env(env=env)
    assert cfg.redis_url == "redis://h:6379/2"
    assert cfg.resource_name == "gpu:rtx4090"
    assert cfg.lock_ttl_seconds == 600 and cfg.heartbeat_seconds == 60
    assert cfg.default_priority == 40
    assert cfg.aging_enabled is False


def test_key_derivation_uses_resource():
    cfg = GPUManagerConfig(resource_name="gpu0")
    assert cfg.key_lock("gpu0") == "gpu:gpu0:lock"
    assert cfg.key_queue("gpu1") == "gpu:gpu1:queue"
    assert cfg.key_request("gpu0", "abc") == "gpu:gpu0:request:abc"
    # sem 'global' hardcoded: recurso arbitrário
    assert cfg.key_owner("mineru-node") == "gpu:mineru-node:owner"


def test_clamp_priority():
    cfg = GPUManagerConfig(min_priority=0, max_priority=100, default_priority=50)
    assert cfg.clamp_priority(None) == 50
    assert cfg.clamp_priority(-5) == 0
    assert cfg.clamp_priority(999) == 100
    assert cfg.clamp_priority(30) == 30
