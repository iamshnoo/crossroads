import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _env_path(name, default):
    value = os.getenv(name)
    if value:
        return Path(value).expanduser()
    return Path(default).expanduser()


def repo_path(*parts):
    return REPO_ROOT.joinpath(*parts)


def data_root():
    return _env_path("CROSSROADS_DATA_ROOT", REPO_ROOT)


def secrets_path():
    return _env_path("CROSSROADS_SECRETS_FILE", repo_path("secrets.json"))


def load_secrets():
    with open(secrets_path(), "r") as f:
        return json.load(f)


def dollarstreet_cache_dir():
    return _env_path(
        "CROSSROADS_DOLLARSTREET_CACHE",
        repo_path(".cache", "dollarstreet"),
    )


def model_cache_dir(model_name=None):
    base = _env_path("CROSSROADS_MODEL_CACHE", repo_path(".cache", "models"))
    return base / model_name if model_name else base


def instructblip_cache_dir():
    return _env_path(
        "CROSSROADS_INSTRUCTBLIP_CACHE",
        repo_path(".cache", "instructblip"),
    )


def edits_cache_dir():
    return _env_path("CROSSROADS_EDITS_CACHE", repo_path(".cache", "edits"))


def resolve_data_path(path_value):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return data_root() / path
