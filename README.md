# speechToText

把面試錄音轉成繁體中文逐字稿，並用聲紋分離標出說話者。

流程：WhisperX（`large-v3` 轉錄 → wav2vec2 字級對齊 → pyannote 說話者分離，全部在本機 GPU 上跑）
→ `output/raw/<檔名>.segments.json`。之後可選擇把逐字稿送到 Gemini 整理成問答稿（`output/cleaned/<檔名>_qa.md`）。

> 錄音（`records/`）與逐字稿（`output/`）是私人面試資料，已被 `.gitignore` 排除，請不要上傳或貼到 issue。

## 需求

- Linux 或 WSL2，NVIDIA GPU（在 RTX 4070 12 GB 上開發）
- [uv](https://docs.astral.sh/uv/)（Python 3.12 由 uv 自動處理）
- 不需要安裝系統 CUDA toolkit，也不需要系統 `ffmpeg`：
  - CUDA 執行函式庫（cuBLAS、cuDNN）來自 pip 的 `nvidia-*-cu12` 套件，由 `torch` 帶入
  - 音訊解碼用 PyAV（內含 ffmpeg）
- WSL2：GPU 驅動裝在 **Windows** 端（支援 WSL 的 NVIDIA 驅動），WSL 內不要另裝驅動。用 `nvidia-smi` 確認看得到顯示卡。

## 安裝

```bash
git clone https://github.com/lujames13/speechToText.git
cd speechToText
uv sync
```

第一次 `uv sync` 會下載 torch 等套件（數 GB）。檢查 GPU 是否可用：

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

應印出 `True 12.8`。

## Hugging Face token（說話者分離需要）

pyannote 的說話者分離模型是 gated model，要先同意使用條款並建立 token。

1. 註冊／登入 [Hugging Face](https://huggingface.co/)。
2. 打開下列頁面，各自按下同意（填表後立即生效）：
   - [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)（**必要**：預設模型。pyannote-audio 4 不論用哪個模型都會讀取它的 PLDA 檔案）
   - 只有在使用 `--diarize-model pyannote/speaker-diarization-3.1` 時，才另外需要 [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) 與 [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
3. 到 [Settings → Access Tokens](https://huggingface.co/settings/tokens) 建立一個 **Read** 權限的 token。
4. 複製範例檔並填入 token：

   ```bash
   cp .env.example .env
   ```

   編輯 `.env`，填入 `HF_TOKEN=hf_...`（要用 Gemini 整理問答時，也填 `GEMINI_API_KEY`）。`.env` 已被 git 忽略。

程式在開始轉錄**之前**會先檢查 token 對模型的存取權。沒同意條款時會立刻停下，並顯示要去哪個頁面同意。

## 使用

```bash
# 只做 STT + 說話者分離，已知有 2 人
uv run transcribe.py records/interview.mp3 --no-llm --speakers 2

# 讓 pyannote 自行判斷人數
uv run transcribe.py records/interview.mp3 --no-llm

# 不做說話者分離（不需要 HF_TOKEN）
uv run transcribe.py records/interview.mp3 --no-llm --no-diarize

# STT 後接 Gemini 問答整理（會開編輯器讓你填公司、輪次、日期、JD）
uv run transcribe.py records/interview.mp3 --speakers 2

# 跳過 STT，用既有逐字稿做 Gemini 問答整理
uv run transcribe.py --from-raw output/raw/interview.segments.json
```

| 參數 | 說明 |
| --- | --- |
| `--speakers N` | 指定說話者人數（傳給 pyannote 的 `min_speakers = max_speakers = N`）。不給則自動判斷 |
| `--no-diarize` | 跳過說話者分離，`speaker` 全為 `null` |
| `--diarize-model REPO` | 改用其他 pyannote 模型，例如 `pyannote/speaker-diarization-3.1` |
| `--no-llm` | 只做 STT，不呼叫 Gemini |
| `--from-raw PATH` | 跳過 STT。接受 `.segments.json`，也接受舊版 `[start --> end] text` 格式的 `.txt`（說話者為空） |
| `--date YYYY-MM-DD` | 面試日期（Gemini 表單的預設值） |

環境變數：

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `HF_TOKEN` | — | 說話者分離必要 |
| `WHISPERX_BATCH_SIZE` | `16` | 轉錄批次大小。顯示卡記憶體不足（CUDA OOM）時改成 `8` |
| `GEMINI_API_KEY` | — | 只有 Gemini 步驟需要 |

第一次執行會從 Hugging Face 下載模型並快取在 `~/.cache/huggingface`：`large-v3`（約 3 GB）、中文對齊模型 `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`（約 1.2 GB）、pyannote 分離模型。下載的只有模型，音訊不會上傳；之後的執行可以離線。

## 輸出

```
output/raw/<檔名>.segments.json   # 正式輸出，供後續步驟使用
output/raw/<檔名>.txt             # 方便閱讀的檢視版，不會被讀回
output/cleaned/<檔名>_qa.md       # Gemini 問答稿（沒加 --no-llm 時）
```

同一個錄音重跑會覆寫同名檔案。

`segments.json` 是一個 list，每段：

```json
{"id": 0, "start": 12.34, "end": 18.9, "speaker": "SPEAKER_00", "text": "繁體中文…"}
```

- `start` 與 `end` 以秒為單位，取到小數 2 位
- `text` 經 OpenCC `s2twp` 轉成台灣繁體
- `speaker` 是 pyannote 的標籤（`SPEAKER_00`、`SPEAKER_01`…）；用 `--no-diarize` 時為 `null`。標籤不代表誰是面試官，角色對應不在這一步處理。

執行時會印出各階段（load、transcribe、align、diarize、write）的耗時。

## 疑難排解

- **`HF_TOKEN 沒有 ... 的存取權`**：到訊息中的網址同意使用條款，再重跑。
- **CUDA out of memory**：`WHISPERX_BATCH_SIZE=8 uv run transcribe.py ...`
- **啟動時出現 `torchcodec is not installed correctly` 警告**：這是因為系統沒有裝 ffmpeg 函式庫，可以忽略；本程式自行解碼音訊，不經過 torchcodec。
- **`libcudnn` / `libcublas` 找不到**：執行 `uv sync`。程式啟動時會把 pip 套件內的 CUDA 函式庫加到 `LD_LIBRARY_PATH` 並自動重新啟動一次。

## 測試

```bash
uv run pytest
```

測試只用假資料，不需要 GPU、音訊或網路。
