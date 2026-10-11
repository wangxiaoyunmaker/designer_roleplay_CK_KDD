import json

import pytest

from tools.prompt_baselines.finalize_main_method_r4_output import (
    finalize_rows,
    merge_retry_rows,
    recover_aligned_response,
)


def test_recovers_unescaped_quotes_without_changing_text():
    raw = (
        '{"units":[{"state":"K","text":"已有"内部"引号"},'
        '{"state":"KtoC","text":"因此提出方案"}]}'
    )
    parsed = recover_aligned_response(raw)
    assert parsed["state_sequence"] == ["K", "KtoC"]
    assert parsed["unit_texts"] == ["已有\"内部\"引号。", "因此提出方案。"]
    assert parsed["parse_repair"]["dropped_truncated_tail"] is False


def test_recovers_complete_prefix_and_marks_truncated_tail():
    raw = (
        '{"units":[{"state":"K","text":"完整单元"},'
        '{"state":"C","text":"没有结束'
    )
    parsed = recover_aligned_response(raw)
    assert parsed["state_sequence"] == ["K"]
    assert parsed["unit_texts"] == ["完整单元。"]
    assert parsed["parse_repair"]["dropped_truncated_tail"] is True


def test_finalizer_does_not_convert_network_failure():
    rows, report = finalize_rows([{
        "condition": "aligned_full",
        "target_id": "D011",
        "segment": "seg1",
        "ok": False,
        "error": "RuntimeError: API HTTP 429",
    }])
    assert rows[0]["ok"] is False
    assert report["unresolved_target_ids"] == ["D011/seg1"]


def test_finalizer_rejects_duplicate_keys():
    row = {"condition": "aligned_full", "target_id": "D011", "ok": True}
    with pytest.raises(ValueError, match="duplicate"):
        finalize_rows([row, row])


def test_merge_retry_replaces_only_matching_failed_row():
    base = [
        {"condition": "aligned_full", "target_id": "D011", "ok": False, "error": "timeout"},
        {"condition": "aligned_full", "target_id": "D012", "ok": True},
    ]
    retry = [{"condition": "aligned_full", "target_id": "D011", "ok": True}]
    merged = merge_retry_rows(base, retry)
    assert merged[0]["ok"] is True
    assert merged[0]["retry_replaced_error"] == "timeout"
    assert merged[1] == base[1]


def test_merge_retry_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown"):
        merge_retry_rows(
            [{"condition": "aligned_full", "target_id": "D011", "ok": False}],
            [{"condition": "aligned_full", "target_id": "D999", "ok": True}],
        )
