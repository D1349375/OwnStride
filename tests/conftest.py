"""
測試環境設定：API 啟動時不要在背景預載本機 LLM（避免測試依賴 Ollama 是否正在執行）。
"""
import os

os.environ.setdefault("OWNSTRIDE_SKIP_LLM_WARMUP", "1")
