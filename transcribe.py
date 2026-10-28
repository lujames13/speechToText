"""Transcribe every ./records/*.mp3 into a traditional-Chinese .txt in ./output/.

Forces GPU execution on the RTX 4070 (device="cuda", compute_type="float16",
model="large-v3"). Run with: uv run transcribe.py
"""

import importlib.util
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RECORDS_DIR = Path("records")
OUTPUT_DIR = Path("output")
MODEL_SIZE = "large-v3"
INITIAL_PROMPT = "以下是繁體中文的對話逐字稿，包含常見的標點符號。"


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


def transcribe_all() -> None:
    from faster_whisper import WhisperModel
    from opencc import OpenCC

    mp3_files = sorted(RECORDS_DIR.glob("*.mp3"))
    if not mp3_files:
        print(f"在 {RECORDS_DIR}/ 中找不到任何 .mp3 檔案。")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = OpenCC("s2twp")  # Simplified -> Traditional (Taiwan, with phrases)

    print(f"載入模型 {MODEL_SIZE}（device=cuda, compute_type=float16）...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    for mp3_path in mp3_files:
        print(f"轉錄中: {mp3_path.name}")
        segments, info = model.transcribe(
            str(mp3_path),
            language="zh",
            initial_prompt=INITIAL_PROMPT,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"{mp3_path.stem}_{timestamp}.txt"

        with out_path.open("w", encoding="utf-8") as f:
            for segment in segments:
                text = converter.convert(segment.text.strip())
                start = f"{segment.start:08.2f}"
                end = f"{segment.end:08.2f}"
                f.write(f"[{start} --> {end}] {text}\n")

        print(f"已輸出: {out_path}")


def main() -> None:
    check_gpu_environment()
    configure_cuda_library_path()
    transcribe_all()


if __name__ == "__main__":
    main()
