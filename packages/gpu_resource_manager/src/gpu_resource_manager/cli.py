"""CLI administrativa: ``gpu-manager status|queue|health|cancel``.

Usa a mesma biblioteca e o mesmo Redis. Não expõe ``force-unlock`` (um desbloqueio
forçado pode causar uso simultâneo da GPU se o dono ainda estiver ativo — ver README).
Comandos são somente-leitura, exceto ``cancel`` (que só remove uma solicitação da
fila; nunca libera lock adquirido).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

from .config import GPUManagerConfig
from .exceptions import GPUBackendUnavailable, GPUManagerError
from .manager import GPUResourceManager


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _build_manager(args) -> GPUResourceManager:
    overrides = {}
    if getattr(args, "redis_url", None):
        overrides["redis_url"] = args.redis_url
    config = GPUManagerConfig.from_env()
    if overrides:
        config = GPUManagerConfig.from_dict(config.__dict__, **overrides)
    return GPUResourceManager(config)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(prog="gpu-manager", description="Administração do GPUResourceManager.")
    parser.add_argument("--redis-url", help="Sobrescreve GPU_MANAGER_REDIS_URL.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Status de um recurso (dono, TTL, fila).")
    p_status.add_argument("--resource", default=None)

    p_queue = sub.add_parser("queue", help="Lista a fila de um recurso.")
    p_queue.add_argument("--resource", default=None)
    p_queue.add_argument("--limit", type=int, default=100)

    sub.add_parser("health", help="Verifica o backend Redis.")

    p_cancel = sub.add_parser("cancel", help="Cancela uma solicitação da fila (não libera lock).")
    p_cancel.add_argument("--request-id", required=True)
    p_cancel.add_argument("--resource", default=None)

    args = parser.parse_args(argv)

    try:
        manager = _build_manager(args)
    except GPUManagerError as exc:
        print(f"erro de configuração: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "status":
            _print_json(manager.get_status(resource=args.resource).to_dict())
        elif args.command == "queue":
            _print_json(manager.get_queue(resource=args.resource, limit=args.limit))
        elif args.command == "health":
            _print_json(manager.healthcheck())
        elif args.command == "cancel":
            ok = manager.cancel_request(args.request_id, resource=args.resource)
            _print_json({"request_id": args.request_id, "cancelled": ok})
    except GPUBackendUnavailable as exc:
        print(f"backend indisponível: {exc}", file=sys.stderr)
        return 3
    except GPUManagerError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    finally:
        manager.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
