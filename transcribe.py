"""Transcribe an interview recording and turn it into a speaker-labelled Q&A script.

Pipeline: WhisperX (GPU: transcribe -> align -> pyannote diarization)
-> output/raw/<stem>.segments.json (+ .txt view) -> Gemini -> Q&A .md.

    uv run transcribe.py records/foo.mp3                    # STT + Q&A (asks for JD)
    uv run transcribe.py records/foo.mp3 --no-llm --speakers 2   # STT only
    uv run transcribe.py --from-raw output/raw/foo.segments.json # LLM step only

Forces GPU execution on the RTX 4070 (device="cuda", compute_type="float16",
model="large-v3").
"""

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

OUTPUT_DIR = Path("output")
RAW_DIR = OUTPUT_DIR / "raw"
CLEANED_DIR = OUTPUT_DIR / "cleaned"
PROMPTS_DIR = Path(__file__).parent / "prompts"
SEGMENTS_SUFFIX = ".segments.json"
MODEL_SIZE = "large-v3"
LANGUAGE = "zh"
INITIAL_PROMPT = "以下是繁體中文的對話逐字稿，包含常見的標點符號。"
# Lower to 8 via env if the 12 GB card runs out of memory during transcription.
BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", 16))
# whisperx 3.8.6's DiarizationPipeline default; passed explicitly so the
# preflight access check and the actual load use the same repo.
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"
# Gated sub-models a pipeline loads, so the preflight catches them too.
# pyannote-audio 4's SpeakerDiarization defaults its PLDA to community-1 and
# the 3.1 config doesn't override it, so community-1 is needed in every case.
DIARIZE_EXTRA_REPOS = {"pyannote/speaker-diarization-3.1": ("pyannote/segmentation-3.0",)}

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
# Step A rewrites the whole transcript, so the output limit (not the input) is
# the bottleneck. Conservative until measured on a real recording.
MAX_SINGLE_PASS_TOKENS = int(os.environ.get("MAX_SINGLE_PASS_TOKENS", 40_000))
CHUNK_SEGMENTS = int(os.environ.get("CHUNK_SEGMENTS", 400))
CHUNK_OVERLAP = 15  # segments of read-only context shown before each chunk
PREV_TAIL_CHARS = 3000  # tail of the previous chunk's output, for speaker state

JD_MARKER = "----- 以下貼上 JD -----"

# Legacy raw line: '[00007.00 --> 00009.00] text'.
SEGMENT_RE = re.compile(r"^\[(\d+(?:\.\d+)?) --> (\d+(?:\.\d+)?)\] (.*)$")
# Human-readable view line with a speaker: '[00007.00 --> 00009.00] SPEAKER_00: text'.
VIEW_SPEAKER_RE = re.compile(r"^\[\d+(?:\.\d+)? --> \d+(?:\.\d+)?\] SPEAKER_\w+: ")


@dataclass
class InterviewMeta:
    company: str
    round: str
    date: date
    jd: str


@dataclass
class Segment:
    id: int
    start: float
    end: float
    speaker: str | None
    text: str


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("必須是 >= 1 的整數")
    return n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio", nargs="?", type=Path, help="錄音檔路徑")
    parser.add_argument("--no-llm", action="store_true", help="只做 STT，不呼叫 LLM、不問任何輸入")
    parser.add_argument(
        "--from-raw", type=Path, metavar="PATH",
        help=f"跳過 STT，直接用既有逐字稿（{SEGMENTS_SUFFIX}，或舊版 [start --> end] .txt）",
    )
    parser.add_argument("--date", type=date.fromisoformat, metavar="YYYY-MM-DD", help="面試日期")
    parser.add_argument(
        "--speakers", type=_positive_int, metavar="N",
        help="說話者人數（預設由 pyannote 自行判斷）",
    )
    parser.add_argument("--no-diarize", action="store_true", help="不做說話者分離（不需要 HF_TOKEN）")
    parser.add_argument(
        "--diarize-model", metavar="HF_REPO",
        help="pyannote 模型（預設用 whisperx 的預設值；例：pyannote/speaker-diarization-3.1）",
    )
    args = parser.parse_args(argv)

    if args.from_raw and args.no_llm:
        parser.error("--no-llm 與 --from-raw 不能同時使用（兩者都沒有事可做）")
    if args.from_raw and args.audio:
        parser.error("--from-raw 不需要 audio 參數")
    if args.from_raw and (args.speakers or args.no_diarize or args.diarize_model):
        parser.error("--from-raw 會跳過 STT，不能搭配 --speakers / --no-diarize / --diarize-model")
    if args.no_diarize and (args.speakers or args.diarize_model):
        parser.error("--no-diarize 不能搭配 --speakers / --diarize-model")
    if not args.from_raw and not args.audio:
        parser.error("請提供錄音檔路徑，或用 --from-raw 指定既有逐字稿")
    if args.audio and not args.audio.is_file():
        parser.error(f"找不到錄音檔: {args.audio}")
    if args.from_raw and not args.from_raw.is_file():
        parser.error(f"找不到逐字稿: {args.from_raw}")
    return args


# --- GPU / CUDA -------------------------------------------------------------


def _nvidia_lib_dir(package: str) -> str | None:
    spec = importlib.util.find_spec(f"{package}.lib")
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def configure_cuda_library_path() -> None:
    """Point the dynamic loader at the pip-installed cuBLAS/cuDNN .so files.

    No system CUDA toolkit is installed in this WSL2 environment, so
    ctranslate2 (faster-whisper's backend, used by WhisperX) must load the
    CUDA runtime libraries shipped inside the nvidia-cublas-cu12 /
    nvidia-cudnn-cu12 wheels instead. Those wheels are now pulled in by torch
    (exact pins), so torch and ctranslate2 share one copy. glibc's dynamic
    loader only reads LD_LIBRARY_PATH when
    a process starts, so setting os.environ from inside the already-running
    interpreter has no effect on later dlopen() calls. We re-exec this same
    script once with the corrected environment to work around that.
    """
    if os.environ.get("_CUDA_LIBS_CONFIGURED") == "1":
        return
    lib_dirs = [
        d
        for d in (_nvidia_lib_dir("nvidia.cublas"), _nvidia_lib_dir("nvidia.cudnn"))
        if d
    ]
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    os.environ["_CUDA_LIBS_CONFIGURED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def check_gpu_environment() -> None:
    print("=== GPU / CUDA 環境檢查 ===")

    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if nvidia_smi.returncode != 0:
        sys.exit(
            "找不到可用的 GPU（nvidia-smi 執行失敗）。\n"
            "請確認：\n"
            "  1. Windows 端已安裝支援 WSL2 的 NVIDIA 顯示卡驅動程式\n"
            "  2. 這是 WSL2（非 WSL1），執行 `wsl.exe -l -v` 確認版本\n"
            "之後重新開啟 WSL2 再試一次。"
        )
    print(f"GPU: {nvidia_smi.stdout.strip()}")

    missing = [
        pkg for pkg in ("nvidia.cublas.lib", "nvidia.cudnn.lib")
        if importlib.util.find_spec(pkg) is None
    ]
    if missing:
        sys.exit(
            "缺少 CUDA 執行所需的函式庫: "
            + ", ".join(missing)
            + "\n這些由 torch 帶入，請執行: uv sync"
        )
    print("cuBLAS / cuDNN (pip 套件) 可用。")
    print()


# --- STT --------------------------------------------------------------------


def check_diarize_access(model_name: str, token: str) -> None:
    """Fail fast on gated-model 401/403 before spending an hour on STT.

    Network errors only warn: the models may already be in the local cache.
    """
    from huggingface_hub import auth_check
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    repos = dict.fromkeys(
        (model_name, *DIARIZE_EXTRA_REPOS.get(model_name, ()), DEFAULT_DIARIZE_MODEL)
    )
    for repo in repos:
        try:
            auth_check(repo, token=token)
        except GatedRepoError:
            sys.exit(
                f"HF_TOKEN 沒有 {repo} 的存取權（gated model）。\n"
                f"請登入 Hugging Face 後到 https://huggingface.co/{repo} 同意使用條款，再重跑。"
            )
        except RepositoryNotFoundError:
            sys.exit(f"找不到 {repo}，或 HF_TOKEN 無效。請檢查 --diarize-model 與 HF_TOKEN。")
        except (HfHubHTTPError, OSError) as e:
            print(f"警告：無法確認 {repo} 的存取權（{type(e).__name__}），若模型已快取則可繼續。")


@contextmanager
def stage(name: str):
    """Print wall time for one pipeline stage."""
    print(f"[{name}] 開始...")
    t0 = time.perf_counter()
    yield
    print(f"[{name}] 完成，耗時 {time.perf_counter() - t0:.1f} 秒")


def _free_gpu() -> None:
    """Call after `del`-ing the last reference to a model, before the next stage."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _speaker_kwargs(speakers: int | None) -> dict[str, int]:
    if speakers is None:
        return {}
    return {"min_speakers": speakers, "max_speakers": speakers}


def run_whisperx(
    audio_path: Path,
    speakers: int | None = None,
    diarize: bool = True,
    diarize_model: str | None = None,
) -> list[Segment]:
    """Transcribe, align and (optionally) diarize one file on the GPU."""
    import whisperx
    from faster_whisper.audio import decode_audio
    from opencc import OpenCC

    device = "cuda"

    with stage("load"):
        # whisperx.load_audio shells out to the ffmpeg CLI; PyAV (bundled
        # ffmpeg, via faster-whisper) gives the same 16 kHz float32 mono array
        # without a system package.
        audio = decode_audio(str(audio_path), sampling_rate=16000)
        print(f"音訊長度 {len(audio) / 16000 / 60:.1f} 分鐘")
        print(f"載入模型 {MODEL_SIZE}（device={device}, compute_type=float16, batch_size={BATCH_SIZE}）")
        model = whisperx.load_model(
            MODEL_SIZE,
            device,
            compute_type="float16",
            language=LANGUAGE,
            asr_options={"initial_prompt": INITIAL_PROMPT},
        )

    with stage("transcribe"):
        result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
        del model
        _free_gpu()

    with stage("align"):
        align_model, align_meta = whisperx.load_align_model(language_code=LANGUAGE, device=device)
        result = whisperx.align(
            result["segments"], align_model, align_meta, audio, device,
            return_char_alignments=False,
        )
        del align_model
        _free_gpu()

    if diarize:
        from whisperx.diarize import DiarizationPipeline

        with stage("diarize"):
            pipeline = DiarizationPipeline(
                model_name=diarize_model or DEFAULT_DIARIZE_MODEL,
                token=os.environ["HF_TOKEN"],
                device=device,
            )
            diarize_df = pipeline(audio, **_speaker_kwargs(speakers))
            result = whisperx.assign_word_speakers(diarize_df, result, fill_nearest=True)
            del pipeline
            _free_gpu()

    return build_segments(result["segments"], OpenCC("s2twp"))


def build_segments(raw_segments: list[dict], converter) -> list[Segment]:
    """WhisperX segment dicts -> numbered Segments in Traditional Chinese.

    Empty text is dropped. A missing start falls back to the previous end
    (0.0 at the beginning); a missing end falls back to start. No text is
    ever dropped for lack of a timestamp.
    """
    segments: list[Segment] = []
    prev_end = 0.0
    for raw in raw_segments:
        text = converter.convert((raw.get("text") or "").strip())
        if not text:
            continue
        start = raw.get("start")
        start = prev_end if start is None else float(start)
        end = raw.get("end")
        end = start if end is None else float(end)
        segments.append(
            Segment(
                id=len(segments),
                start=round(start, 2),
                end=round(end, 2),
                speaker=raw.get("speaker"),
                text=text,
            )
        )
        prev_end = end
    return segments


def _raw_stem(path: Path) -> str:
    """'foo.segments.json' -> 'foo'; anything else -> path.stem."""
    if path.name.endswith(SEGMENTS_SUFFIX):
        return path.name[: -len(SEGMENTS_SUFFIX)]
    return path.stem


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def segments_view(segments: list[Segment]) -> str:
    """Human-readable view. Never parsed back; the JSON is canonical."""
    lines = []
    for s in segments:
        who = f"{s.speaker}: " if s.speaker else ""
        lines.append(f"[{s.start:08.2f} --> {s.end:08.2f}] {who}{s.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_segments(segments: list[Segment], stem: str) -> tuple[Path, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RAW_DIR / f"{stem}{SEGMENTS_SUFFIX}"
    txt_path = RAW_DIR / f"{stem}.txt"
    payload = json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2)
    _atomic_write(json_path, payload + "\n")
    _atomic_write(txt_path, segments_view(segments))
    return json_path, txt_path


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_segments_json(path: Path) -> list[Segment]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"{path} 不是合法的 JSON: {e}")
    if not isinstance(data, list):
        sys.exit(f"{path} 格式錯誤：最外層必須是 list。")

    segments = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            sys.exit(f"{path} 第 {i} 筆格式錯誤：必須是 object。")
        missing = [k for k in ("id", "start", "end", "speaker", "text") if k not in item]
        if missing:
            sys.exit(f"{path} 第 {i} 筆缺少欄位: {', '.join(missing)}")
        if not (
            isinstance(item["id"], int) and not isinstance(item["id"], bool)
            and _is_number(item["start"])
            and _is_number(item["end"])
            and (item["speaker"] is None or isinstance(item["speaker"], str))
            and isinstance(item["text"], str)
        ):
            sys.exit(f"{path} 第 {i} 筆欄位型別錯誤（id:int, start/end:number, speaker:str|null, text:str）")
        segments.append(
            Segment(item["id"], float(item["start"]), float(item["end"]), item["speaker"], item["text"])
        )
    return segments


def _load_segments_legacy_txt(path: Path) -> list[Segment]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(VIEW_SPEAKER_RE.match(line) for line in lines):
        json_path = path.with_name(f"{path.stem}{SEGMENTS_SUFFIX}")
        sys.exit(
            f"{path} 是帶說話者的檢視用 .txt，不能當輸入。請改用 --from-raw {json_path}"
        )
    segments = []
    for line in lines:
        m = SEGMENT_RE.match(line)
        if m:
            segments.append(
                Segment(len(segments), float(m.group(1)), float(m.group(2)), None, m.group(3))
            )
    return segments


def load_segments(path: Path) -> list[Segment]:
    if path.name.endswith(".json"):
        return _load_segments_json(path)
    if path.suffix == ".txt":
        return _load_segments_legacy_txt(path)
    sys.exit(f"不支援的逐字稿格式: {path}（需為 {SEGMENTS_SUFFIX} 或舊版 .txt）")


# --- Interactive input ------------------------------------------------------


def prompt_meta(default_date: date) -> InterviewMeta:
    """Press Enter, then fill company/round/date and paste the JD in $EDITOR."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or next(
        (e for e in ("nano", "vim", "vi") if shutil.which(e)), None
    )
    if editor is None:
        sys.exit("找不到文字編輯器，請設定 EDITOR 環境變數（例如 export EDITOR=nano）。")

    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", prefix="jd_", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "# 填寫公司、輪次、日期，並把 JD 貼在分隔線之後；存檔並關閉編輯器即可繼續。\n"
            "公司：\n"
            "輪次：一面\n"
            f"日期：{default_date.isoformat()}\n"
            f"{JD_MARKER}\n"
        )
        tmp_path = Path(f.name)
    try:
        input(f"按 Enter 以 {editor} 開啟編輯畫面（填公司、輪次、日期並貼上 JD）...")
        result = subprocess.run([*shlex.split(editor), str(tmp_path)])
        if result.returncode != 0:
            sys.exit(f"編輯器異常結束（exit {result.returncode}），已中止。")
        content = tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)

    return _parse_meta_file(content, default_date)


def _parse_meta_file(content: str, default_date: date) -> InterviewMeta:
    head, _, jd = content.partition(JD_MARKER)
    fields = {}
    for line in head.splitlines():
        key, sep, value = line.replace("：", ":", 1).partition(":")
        if sep and not line.lstrip().startswith("#"):
            fields[key.strip()] = value.strip()

    company = fields.get("公司", "")
    if not company:
        sys.exit("公司不能為空，已中止。")
    try:
        interview_date = date.fromisoformat(fields["日期"]) if fields.get("日期") else default_date
    except ValueError:
        sys.exit(f"日期格式需為 YYYY-MM-DD: {fields['日期']}")
    jd = jd.strip()
    if not jd:
        sys.exit("JD 是空的，已中止。")

    print(f"已收到：{company} {fields.get('輪次') or '一面'} {interview_date}；"
          f"JD {len(jd.splitlines())} 行，最後一行：{jd.splitlines()[-1][:40]}")
    return InterviewMeta(company, fields.get("輪次") or "一面", interview_date, jd)


# --- LLM --------------------------------------------------------------------


@dataclass
class LlmResult:
    text: str
    truncated: bool


_client = None
_model_reported = False


def _gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def call_llm(system: str, user: str, thinking_level: str, schema=None) -> LlmResult:
    """The only place that touches the Gemini SDK."""
    global _model_reported
    from google.genai import types

    # No temperature/top_p/top_k/candidate_count: gemini-3.8-flash rejects them.
    config = types.GenerateContentConfig(
        system_instruction=system,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level.upper()),
    )
    if schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = schema

    response = _get_client().models.generate_content(
        model=_gemini_model(), contents=user, config=config
    )
    if not _model_reported:
        print(f"實際回應模型: {response.model_version or _gemini_model()}")
        _model_reported = True

    candidate = response.candidates[0] if response.candidates else None
    truncated = candidate is not None and candidate.finish_reason == types.FinishReason.MAX_TOKENS
    return LlmResult(text=response.text or "", truncated=truncated)


def count_llm_tokens(text: str) -> int:
    return _get_client().models.count_tokens(model=_gemini_model(), contents=text).total_tokens


# --- Step A: Q&A transcript -------------------------------------------------


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def segments_to_lines(segments: list[Segment]) -> list[str]:
    """Segments -> compact '[00:07] text' lines for Step A (no speaker labels; see #5)."""
    return [f"[{_fmt_ts(s.start)}] {s.text}" for s in segments]


def _user_prompt(
    meta: InterviewMeta,
    core: list[str],
    context: list[str] | None = None,
    prev_tail: str | None = None,
) -> str:
    parts = [
        f"公司：{meta.company}\n輪次：{meta.round}",
        f"<jd>\n{meta.jd}\n</jd>",
    ]
    if prev_tail:
        parts.append(
            "這份逐字稿分段處理。以下是上一段輸出的結尾，請沿用其中的說話者標註，"
            "若最後一題尚未結束就直接接續，不要重複標題；新題目的編號接續上一段。\n"
            f"<previous_output_tail>\n{prev_tail}\n</previous_output_tail>"
        )
    if context:
        parts.append(
            "以下是已處理過的前文，僅供判斷說話者與語境，不要輸出：\n"
            "<context_only>\n" + "\n".join(context) + "\n</context_only>"
        )
    parts.append("請整理以下逐字稿：\n<transcript>\n" + "\n".join(core) + "\n</transcript>")
    return "\n\n".join(parts)


def chunk_segments(lines: list[str], size: int, overlap: int) -> list[tuple[list[str], list[str]]]:
    """Split into (context, core) pairs. Context is read-only overlap before core."""
    chunks = []
    for start in range(0, len(lines), size):
        chunks.append((lines[max(0, start - overlap):start], lines[start:start + size]))
    return chunks


def _build_qa_chunked(system: str, meta: InterviewMeta, lines: list[str], size: int) -> str:
    chunks = chunk_segments(lines, size, CHUNK_OVERLAP)
    print(f"分批處理：{len(chunks)} 批（每批 {size} 段）")
    outputs: list[str] = []
    for i, (context, core) in enumerate(chunks, 1):
        print(f"  Step A 第 {i}/{len(chunks)} 批...")
        prev_tail = outputs[-1][-PREV_TAIL_CHARS:] if outputs else None
        result = _call_step_a(system, _user_prompt(meta, core, context, prev_tail))
        if result.truncated:
            if size <= 20:
                sys.exit("單批已很小仍被截斷，已中止，請檢查 prompt 或模型設定。")
            print(f"  第 {i} 批輸出被截斷，改用較小的批次重新處理整份。")
            return _build_qa_chunked(system, meta, lines, size // 2)
        outputs.append(result.text.strip())
    return "\n\n".join(outputs)


def _call_step_a(system: str, user: str) -> LlmResult:
    return call_llm(system, user, thinking_level="medium")


def build_qa_transcript(lines: list[str], meta: InterviewMeta) -> str:
    system = (PROMPTS_DIR / "qa_transcript.md").read_text(encoding="utf-8")
    if not lines:
        sys.exit("逐字稿中沒有可用的段落。")

    user = _user_prompt(meta, lines)
    tokens = count_llm_tokens(system + user)
    print(f"Step A 輸入約 {tokens} tokens（單次上限 {MAX_SINGLE_PASS_TOKENS}）")

    if tokens <= MAX_SINGLE_PASS_TOKENS:
        result = _call_step_a(system, user)
        if not result.truncated:
            return result.text.strip()
        print("單次輸出被截斷，改用分批處理。")
    return _build_qa_chunked(system, meta, lines, CHUNK_SEGMENTS)


def write_qa_file(raw_path: Path, meta: InterviewMeta, body: str) -> Path:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEANED_DIR / f"{_raw_stem(raw_path)}_qa.md"
    header = f"# {meta.company} {meta.round}（{meta.date.isoformat()}）面試問答整理\n\n"
    out_path.write_text(header + body + "\n", encoding="utf-8")
    return out_path


# --- main -------------------------------------------------------------------


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()
    use_llm = not args.no_llm

    if use_llm and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("缺少 GEMINI_API_KEY（可寫在 .env，範例見 .env.example）。")
    if args.audio and not args.no_diarize and not os.environ.get("HF_TOKEN"):
        sys.exit(
            "缺少 HF_TOKEN（說話者分離需要；可寫在 .env，範例見 .env.example，"
            "申請方式見 README）。或加 --no-diarize 略過說話者分離。"
        )
    if args.audio and not args.no_diarize:
        check_diarize_access(args.diarize_model or DEFAULT_DIARIZE_MODEL, os.environ["HF_TOKEN"])

    if args.audio:
        check_gpu_environment()
        configure_cuda_library_path()

    if use_llm:
        print("注意：逐字稿與 JD 將送至 Google Gemini API。")
        default_date = (
            datetime.fromtimestamp(args.audio.stat().st_mtime).date() if args.audio else date.today()
        )
        meta = prompt_meta(args.date or default_date)

    if args.from_raw:
        raw_path = args.from_raw
        segments = load_segments(raw_path)
    else:
        total_t0 = time.perf_counter()
        segments = run_whisperx(
            args.audio,
            speakers=args.speakers,
            diarize=not args.no_diarize,
            diarize_model=args.diarize_model,
        )
        with stage("write"):
            raw_path, view_path = write_segments(segments, args.audio.stem)
        speakers = sorted({s.speaker or "（無）" for s in segments})
        print(f"已輸出: {raw_path}（{len(segments)} 段；說話者: {', '.join(speakers)}）")
        print(f"檢視用: {view_path}")
        print(f"STT 總耗時 {time.perf_counter() - total_t0:.1f} 秒")
    if not use_llm:
        return

    body = build_qa_transcript(segments_to_lines(segments), meta)
    print(f"已輸出: {write_qa_file(raw_path, meta, body)}")


if __name__ == "__main__":
    main()
