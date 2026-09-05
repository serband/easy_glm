from __future__ import annotations

import polars as pl

import easy_glm
from easy_glm.core import data as data_module


def test_swedish_motorcycle_loader_uses_the_public_casdatasets_table(monkeypatch):
    expected = pl.DataFrame({"ClaimAmount": [0, 100]})
    seen: dict[str, object] = {}

    def fake_load(url: str, object_name: str, *, cache: bool) -> pl.DataFrame:
        seen.update(url=url, object_name=object_name, cache=cache)
        return expected

    monkeypatch.setattr(data_module, "_load_rda_dataframe", fake_load)

    assert easy_glm.load_swedish_motorcycle_data(cache=False).equals(expected)
    assert seen == {
        "url": data_module._SWEDISH_MOTORCYCLE_URL,
        "object_name": "swmotorcycle",
        "cache": False,
    }
