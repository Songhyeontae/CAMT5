import hydra


def to_absolute_path(path: str) -> str:
    return hydra.utils.to_absolute_path(path)
