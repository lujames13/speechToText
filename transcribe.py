"""Transcribe an interview recording and turn it into a speaker-labelled Q&A script.

Pipeline: faster-whisper (GPU) -> timestamped raw .txt -> Gemini -> Q&A .md.

    uv run transcribe.py records/foo.mp3              # STT + Q&A (asks for JD)
    uv run transcribe.py records/foo.mp3 --no-llm     # STT only (old behaviour)
    uv run transcribe.py --from-raw output/foo_*.txt  # skip STT, LLM step only

Forces GPU execution on the RTX 4070 (device="cuda", compute_type="float16",
model="large-v3").
"""

import argparse
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

OUTPUT_DIR = Path("output")
CLEANED_DIR = OUTPUT_DIR / "cleaned"
PROMPTS_DIR = Path(__file__).parent / "prompts"
MODEL_SIZE = "large-v3"
INITIAL_PROMPT = "以下是繁體中文的對話逐字稿，包含常見的標點符號。"

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
# Step A rewrites the whole transcript, so the output limit (not the input) is
# the bottleneck. Conservative until measured on a real recording.
MAX_SINGLE_PASS_TOKENS = int(os.environ.get("MAX_SINGLE_PASS_TOKENS", 40_000))
CHUNK_SEGMENTS = int(os.environ.get("CHUNK_SEGMENTS", 400))
CHUNK_OVERLAP = 15  # segments of read-only context shown before each chunk
PREV_TAIL_CHARS = 3000  # tail of the previous chunk's output, for speaker state

JD_MARKER = "----- 以下貼上 JD -----"

SEGMENT_RE =re.compile(r"^\[(\d+(?:\.\d+)?) --> (\d+(?:\.\d+)?)\] (.*)$")


@dataclass
class InterviewMeta:
    company: str
    round: str
    date: date
    jd: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio", nargs="?", type=Path, help="錄音檔路徑")
    parser.add_argument("--no-llm", action="store_true", help="只做 STT，不呼叫 LLM、不問任何輸入")
    parser.add_argument("--from-raw", type=Path, metavar="PATH", help="跳過 STT，直接用既有逐字稿")
    parser.add_argument("--date", type=date.fromisoformat, metavar="YYYY-MM-DD", help="面試日期")
    args = parser.parse_args(argv)

    if args.from_raw and args.no_llm:
        parser.error("--no-llm 與 --from-raw 不能同時使用（兩者都沒有事可做）")
    if args.from_raw and args.audio:
        parser.error("--from-raw 不需要 audio 參數")
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
    ctranslate2 (faster-whisper's backend) must load the CUDA runtime
    libraries shipped inside the nvidia-cublas-cu12 / nvidia-cudnn-cu12
    wheels instead. glibc's dynamic loader only reads LD_LIBRARY_PATH when
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
            + "\n請執行: uv add nvidia-cublas-cu12 nvidia-cudnn-cu12"
        )
    print("cuBLAS / cuDNN (pip 套件) 可用。")
    print()


# --- STT --------------------------------------------------------------------


def transcribe_file(audio_path: Path) -> Path:
    """Run Whisper on one file and write the timestamped raw transcript."""
    from faster_whisper import WhisperModel
    from opencc import OpenCC

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = OpenCC("s2twp")  # Simplified -> Traditional (Taiwan, with phrases)

    print(f"載入模型 {MODEL_SIZE}（device=cuda, compute_type=float16）...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    print(f"轉錄中: {audio_path.name}")
    segments, info = model.transcribe(
        str(audio_path),
        language="zh",
        initial_prompt=INITIAL_PROMPT,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"{audio_path.stem}_{timestamp}.txt"

    with out_path.open("w", encoding="utf-8") as f:
        for segment in segments:
            text = converter.convert(segment.text.strip())
            start = f"{segment.start:08.2f}"
            end = f"{segment.end:08.2f}"
            f.write(f"[{start} --> {end}] {text}\n")

    print(f"已輸出: {out_path}")
    return out_path


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


def _segment_lines(raw_text: str) -> list[str]:
    """Raw '[00007.00 --> 00009.00] text' lines -> compact '[00:07] text' lines."""
    out = []
    for line in raw_text.splitlines():
        m = SEGMENT_RE.match(line)
        if m:
            out.append(f"[{_fmt_ts(float(m.group(1)))}] {m.group(3)}")
    return out


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


def build_qa_transcript(raw_text: str, meta: InterviewMeta) -> str:
    system = (PROMPTS_DIR / "qa_transcript.md").read_text(encoding="utf-8")
    lines = _segment_lines(raw_text)
    if not lines:
        sys.exit("逐字稿中沒有可解析的 '[start --> end] text' 行。")

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
    out_path = CLEANED_DIR / f"{raw_path.stem}_qa.md"
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

    if args.audio:
        check_gpu_environment()
        configure_cuda_library_path()

    if use_llm:
        print("注意：逐字稿與 JD 將送至 Google Gemini API。")
        default_date = (
            datetime.fromtimestamp(args.audio.stat().st_mtime).date() if args.audio else date.today()
        )
        meta = prompt_meta(args.date or default_date)

    raw_path = args.from_raw or transcribe_file(args.audio)
    if not use_llm:
        return

    body = build_qa_transcript(raw_path.read_text(encoding="utf-8"), meta)
    print(f"已輸出: {write_qa_file(raw_path, meta, body)}")


if __name__ == "__main__":
    main()
