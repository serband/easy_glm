from __future__ import annotations

from easy_glm.app import readiness


def test_split_ready_prefers_split_ready_helper():
    class State:
        @staticmethod
        def split_ready() -> bool:
            return True

        @staticmethod
        def status() -> dict[str, bool]:
            return {"split": False}

    assert readiness.split_ready(State) is True


def test_split_ready_falls_back_to_status_when_helper_missing():
    class State:
        @staticmethod
        def status() -> dict[str, bool]:
            return {"split": True}

    assert readiness.split_ready(State) is True


def test_split_ready_returns_false_when_state_contract_missing():
    class State:
        pass

    assert readiness.split_ready(State) is False
