"""Bersama semua test (T-607): `stage_common.set_message_config` = status proses (cli.main / main() stage menyetel --config yang
dipakai supaya perintah siap tempel membawanya). Direset sesudah setiap test supaya --config satu test tidak bocor ke pesan test lain."""

from __future__ import annotations

import pytest

from rotoscope import stage_common


@pytest.fixture(autouse=True)
def _reset_message_config():
    yield
    stage_common.set_message_config(None)
