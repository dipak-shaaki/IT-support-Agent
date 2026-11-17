import logging


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are duplicated by the app's chat lifecycle and LLM timing logs.
    for name in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)
