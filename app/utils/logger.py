import logging

logger = logging.getLogger("text2sql")


def log_step(step: str, message: str, **kwargs: object) -> None:
    extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    line = f"[STEP] {step} | {message}"
    if extra:
        line += f" | {extra}"
    logger.info(line)
