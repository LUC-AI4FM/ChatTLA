import json
from src.shared.schemas.dataset_schema import (
    DatasetRecord,
    HarmonyMessage,
)


def _sample_tla():
    return """---- MODULE Hello
VARIABLE x
Init == x = 0
Next == x' = x + 1
===="""


def test_make_id_is_stable_and_trims_whitespace():
    a = _sample_tla()
    b = "\n" + a + "\n"
    assert DatasetRecord.make_id(a) == DatasetRecord.make_id(b)


def test_validate_success_and_roundtrip():
    tla = _sample_tla()
    rec = DatasetRecord(
        tla_content=tla,
        source="formalllm:1",
        license="MIT",
    )
    rec.id = DatasetRecord.make_id(tla)

    errs = rec.validate()
    assert errs == []

    # to_dict / to_json / from_dict round-trip
    d = rec.to_dict()
    s = rec.to_json()
    assert isinstance(s, str) and "tla_content" in s

    rec2 = DatasetRecord.from_dict(d)
    assert rec2.id == rec.id
    assert rec2.tla_content == rec.tla_content
    assert rec2.source == rec.source


def test_validate_detects_missing_and_mismatch():
    rec = DatasetRecord()
    # empty fields should produce errors
    errs = rec.validate()
    assert "id is empty" in errs
    assert "tla_content is empty" in errs
    assert "source is empty" in errs

    # mismatched id should be reported
    rec.tla_content = _sample_tla()
    rec.id = "bad-id"
    rec.source = "formalllm:1"
    errs = rec.validate()
    assert any("id does not match" in e for e in errs)


def test_harmony_messages_roundtrip():
    tla = _sample_tla()
    rec = DatasetRecord(tla_content=tla, source="formalllm:1")
    rec.id = DatasetRecord.make_id(tla)
    rec.harmony_messages = {
        "spec_generation": [HarmonyMessage(role="user", content="Do X")]
    }

    d = rec.to_dict()
    rec2 = DatasetRecord.from_dict(d)
    msgs = rec2.harmony_messages.get("spec_generation")
    assert isinstance(msgs, list)
    assert msgs and msgs[0].role == "user" and msgs[0].content == "Do X"
