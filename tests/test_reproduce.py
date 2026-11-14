"""Неповний пакет і LOG3_CONFIG не повинні позначатись як прогін викладача."""
import pytest

from log3 import config
from scripts import reproduce


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.delenv("LOG3_CONFIG", raising=False)
    paths = {"MATERIALS_CONFIG": tmp_path / "team_config.json", "DEV_CONFIG": tmp_path / "dev.json",
             "SESSION_CSV": tmp_path / "session.csv", "REPLICATION_PY": tmp_path / "replication.py"}
    for name in ("MATERIALS_CONFIG", "DEV_CONFIG"):
        monkeypatch.setattr(config, name, paths[name])
        monkeypatch.setattr(reproduce, name, paths[name], raising=False)
    for name in ("SESSION_CSV", "REPLICATION_PY"):
        monkeypatch.setattr(reproduce, name, paths[name], raising=False)
    paths["DEV_CONFIG"].touch()
    return paths


def test_partial_materials_are_labelled_as_dev(sources):
    sources["MATERIALS_CONFIG"].touch()
    assert "dev" in reproduce.env_note()


@pytest.mark.parametrize("present, expected", [
    ((), {"config": "dev", "session": "dev", "replication": "dev"}),
    (("MATERIALS_CONFIG",), {"config": "materials", "session": "dev", "replication": "dev"}),
    (("SESSION_CSV", "REPLICATION_PY"), {"config": "dev", "session": "materials", "replication": "materials"}),
    (("MATERIALS_CONFIG", "SESSION_CSV", "REPLICATION_PY"),
     {"config": "materials", "session": "materials", "replication": "materials"}),
])
def test_environment_tracks_each_actual_source(sources, present, expected):
    for name in present:
        sources[name].touch()
    assert reproduce.environment_sources() == expected


def test_config_override_is_reported_with_complete_materials(sources, monkeypatch, tmp_path):
    for path in sources.values():
        path.touch()
    monkeypatch.setenv("LOG3_CONFIG", str(sources["DEV_CONFIG"]))
    assert reproduce.environment_sources() == {"config": "dev", "session": "materials", "replication": "materials"}

    custom = tmp_path / "custom.json"
    custom.touch()
    monkeypatch.setenv("LOG3_CONFIG", str(custom))
    assert reproduce.environment_sources() == {"config": "custom", "session": "materials", "replication": "materials"}
