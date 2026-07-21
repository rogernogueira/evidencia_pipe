import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("mineru.viewer")
log_idx = logging.getLogger("mineru.index")
log_api = logging.getLogger("mineru.api")
