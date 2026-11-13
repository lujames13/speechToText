"""CI-safe tests: fake data only. No GPU, no audio, no HF downloads, no .env.

These never import whisperx / torch / faster_whisper; transcribe.py only
imports them lazily inside run_whisperx().
"""

import json
import sys

import pytest
from opencc import OpenCC

import transcribe
from transcribe import (
    SEGMENTS_SUFFIX,
    Segment,
    _raw_stem,
    _speaker_kwargs,
    build_segments,
    load_segments,
    parse_args,
    segments_to_lines,
    write_segments,
)

KEYS = {"id", "start", "end", "speaker", "text"}


class IdentityConverter:
    def convert(self, text: str) -> str:
        return text


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _sample() -> list[Segment]:
    return [
        Segment(0, 0.0, 2.5, "SPEAKER_00", "請自我介紹。"),
        Segment(1, 2.5, 10.25, "SPEAKER_01", "我是軟體工程師。"),
        Segment(2, 10.25, 12.0, None, "嗯。"),
    ]


# --- build_segments ---------------------------------------------------------


def test_build_segments_converts_to_traditional():
    segs = build_segments([{"start": 0, "end": 1, "text": "软件开发", "speaker": "SPEAKER_00"}], OpenCC("s2twp"))
    assert segs[0].text == "軟體開發"


def test_build_segments_drops_empty_and_numbers_sequentially():
    raw = [
        {"start": 0.0, "end": 1.0, "text": "一"},
        {"start": 1.0, "end": 2.0, "text": "   "},
        {"start": 2.0, "end": 3.0, "text": ""},
        {"start": 3.0, "end": 4.0, "text": " 二 "},
    ]
    segs = build_segments(raw, IdentityConverter())
    assert [s.id for s in segs] == [0, 1]
    assert [s.text for s in segs] == ["一", "二"]


def test_build_segments_missing_speaker_is_none_and_rounds():
    segs = build_segments([{"start": 1.23456, "end": 7.891, "text": "x"}], IdentityConverter())
    assert segs[0].speaker is None
    assert (segs[0].start, segs[0].end) == (1.23, 7.89)


def test_build_segments_missing_timestamps_fall_back_without_dropping_text():
    raw = [
        {"start": 1.0, "end": 2.0, "text": "a"},
        {"text": "b"},
        {"start": 5.0, "text": "c"},
    ]
    segs = build_segments(raw, IdentityConverter())
    assert [s.text for s in segs] == ["a", "b", "c"]
    assert (segs[1].start, segs[1].end) == (2.0, 2.0)
    assert (segs[2].start, segs[2].end) == (5.0, 5.0)


# --- write_segments ---------------------------------------------------------


def test_write_segments_json_round_trip(in_tmp):
    json_path, txt_path = write_segments(_sample(), "interview")
    assert json_path == transcribe.RAW_DIR / f"interview{SEGMENTS_SUFFIX}"
    raw = json_path.read_text(encoding="utf-8")
    assert "請自我介紹" in raw  # ensure_ascii=False keeps CJK readable
    data = json.loads(raw)
    assert isinstance(data, list) and len(data) == 3
    assert all(set(item) == KEYS for item in data)
    assert data[2]["speaker"] is None
    assert load_segments(json_path) == _sample()
    assert not list(json_path.parent.glob("*.tmp"))


def test_write_segments_view_has_speaker_prefix(in_tmp):
    _, txt_path = write_segments(_sample(), "interview")
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[00000.00 --> 00002.50] SPEAKER_00: 請自我介紹。"
    assert lines[2] == "[00010.25 --> 00012.00] 嗯。"


# --- load_segments: JSON ----------------------------------------------------


def _write_json(tmp_path, data, name="x.segments.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_load_json_empty_list(tmp_path):
    assert load_segments(_write_json(tmp_path, [])) == []


def test_load_json_accepts_int_timestamps(tmp_path):
    p = _write_json(tmp_path, [{"id": 0, "start": 1, "end": 2, "speaker": None, "text": "t"}])
    assert load_segments(p) == [Segment(0, 1.0, 2.0, None, "t")]


@pytest.mark.parametrize(
    "data, msg",
    [
        ({"id": 0}, "最外層必須是 list"),
        (["nope"], "必須是 object"),
        ([{"id": 0, "start": 0, "end": 1, "text": "t"}], "缺少欄位: speaker"),
        ([{"id": "0", "start": 0, "end": 1, "speaker": None, "text": "t"}], "型別錯誤"),
        ([{"id": True, "start": 0, "end": 1, "speaker": None, "text": "t"}], "型別錯誤"),
        ([{"id": 0, "start": "0", "end": 1, "speaker": None, "text": "t"}], "型別錯誤"),
        ([{"id": 0, "start": 0, "end": 1, "speaker": 1, "text": "t"}], "型別錯誤"),
        ([{"id": 0, "start": 0, "end": 1, "speaker": None, "text": None}], "型別錯誤"),
    ],
)
def test_load_json_rejects_malformed(tmp_path, data, msg):
    with pytest.raises(SystemExit) as exc:
        load_segments(_write_json(tmp_path, data))
    assert msg in str(exc.value.code)


def test_load_json_rejects_invalid_json(tmp_path):
    p = tmp_path / "bad.segments.json"
    p.write_text("[{", encoding="utf-8")
    with pytest.raises(SystemExit, match="不是合法的 JSON"):
        load_segments(p)


# --- load_segments: legacy .txt ---------------------------------------------


def test_load_legacy_txt(tmp_path):
    p = tmp_path / "old_20260101_120000.txt"
    p.write_text(
        "[00007.00 --> 00009.50] 你好\n"
        "garbage line\n"
        "[00065.20 --> 00070.00] 第二段\n",
        encoding="utf-8",
    )
    segs = load_segments(p)
    assert segs == [
        Segment(0, 7.0, 9.5, None, "你好"),
        Segment(1, 65.2, 70.0, None, "第二段"),
    ]
    # Same compact lines Step A got from the old _segment_lines().
    assert segments_to_lines(segs) == ["[00:07] 你好", "[01:05] 第二段"]


def test_load_rejects_speaker_view_txt(in_tmp):
    _, txt_path = write_segments(_sample(), "interview")
    with pytest.raises(SystemExit) as exc:
        load_segments(txt_path)
    assert f"interview{SEGMENTS_SUFFIX}" in str(exc.value.code)


def test_load_rejects_unknown_suffix(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="不支援"):
        load_segments(p)


# --- helpers ----------------------------------------------------------------


def test_segments_to_lines_has_no_speaker_labels():
    assert segments_to_lines(_sample()) == ["[00:00] 請自我介紹。", "[00:02] 我是軟體工程師。", "[00:10] 嗯。"]


def test_speaker_kwargs():
    assert _speaker_kwargs(None) == {}
    assert _speaker_kwargs(2) == {"min_speakers": 2, "max_speakers": 2}


def test_raw_stem():
    from pathlib import Path

    assert _raw_stem(Path("output/raw/a.b.segments.json")) == "a.b"
    assert _raw_stem(Path("output/foo_20260101_120000.txt")) == "foo_20260101_120000"


# --- check_diarize_access (auth_check monkeypatched; no network) ------------


def _patch_auth(monkeypatch, fail_on: dict):
    import huggingface_hub

    calls = []

    def fake_auth_check(repo, token=None):
        calls.append(repo)
        if repo in fail_on:
            raise fail_on[repo]

    monkeypatch.setattr(huggingface_hub, "auth_check", fake_auth_check)
    return calls


def test_diarize_access_ok_checks_sub_models(monkeypatch):
    calls = _patch_auth(monkeypatch, {})
    transcribe.check_diarize_access("pyannote/speaker-diarization-3.1", "tok")
    # community-1 is always checked: pyannote 4 loads its PLDA for every pipeline.
    assert calls == [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
        "pyannote/speaker-diarization-community-1",
    ]


def test_diarize_access_default_checked_once(monkeypatch):
    calls = _patch_auth(monkeypatch, {})
    transcribe.check_diarize_access(transcribe.DEFAULT_DIARIZE_MODEL, "tok")
    assert calls == [transcribe.DEFAULT_DIARIZE_MODEL]


def test_diarize_access_gated_exits_with_url(monkeypatch):
    from huggingface_hub.errors import GatedRepoError

    _patch_auth(monkeypatch, {"pyannote/segmentation-3.0": GatedRepoError("403")})
    with pytest.raises(SystemExit) as exc:
        transcribe.check_diarize_access("pyannote/speaker-diarization-3.1", "tok")
    assert "https://huggingface.co/pyannote/segmentation-3.0" in str(exc.value.code)


def test_diarize_access_network_error_only_warns(monkeypatch, capsys):
    _patch_auth(monkeypatch, {transcribe.DEFAULT_DIARIZE_MODEL: OSError("offline")})
    transcribe.check_diarize_access(transcribe.DEFAULT_DIARIZE_MODEL, "tok")
    assert "警告" in capsys.readouterr().out


# --- parse_args -------------------------------------------------------------


@pytest.fixture
def files(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"")  # empty placeholder, never decoded
    seg = _write_json(tmp_path, [])
    return audio, seg


def _error(argv, capsys) -> str:
    with pytest.raises(SystemExit):
        parse_args(argv)
    return capsys.readouterr().err


def test_parse_speakers(files):
    audio, _ = files
    assert parse_args([str(audio), "--speakers", "2"]).speakers == 2
    assert parse_args([str(audio)]).speakers is None


@pytest.mark.parametrize("n", ["0", "-1", "x"])
def test_parse_speakers_rejects_non_positive(files, capsys, n):
    audio, _ = files
    assert "--speakers" in _error([str(audio), "--speakers", n], capsys)


def test_parse_from_raw_segments_json(files):
    _, seg = files
    args = parse_args(["--from-raw", str(seg)])
    assert args.from_raw == seg and args.audio is None


@pytest.mark.parametrize("extra", [["--speakers", "2"], ["--no-diarize"], ["--diarize-model", "m"]])
def test_parse_from_raw_rejects_stt_flags(files, capsys, extra):
    _, seg = files
    assert "--from-raw" in _error(["--from-raw", str(seg), *extra], capsys)


@pytest.mark.parametrize("extra", [["--speakers", "2"], ["--diarize-model", "m"]])
def test_parse_no_diarize_conflicts(files, capsys, extra):
    audio, _ = files
    assert "--no-diarize" in _error([str(audio), "--no-diarize", *extra], capsys)


def test_parse_existing_rules_still_hold(files, capsys):
    audio, seg = files
    assert "--no-llm" in _error(["--from-raw", str(seg), "--no-llm"], capsys)
    assert "audio" in _error([str(audio), "--from-raw", str(seg)], capsys)
    assert "請提供" in _error([], capsys)


def test_transcribe_module_does_not_import_heavy_deps():
    import subprocess
    from pathlib import Path

    code = (
        "import sys, transcribe; "
        "heavy = [m for m in ('whisperx', 'torch', 'faster_whisper') if m in sys.modules]; "
        "assert not heavy, heavy"
    )
    root = Path(transcribe.__file__).parent
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
