"""下载人脸、表情和手部模型。地址是固定的，不接受外部传入的链接。"""

import shutil
import urllib.request
from pathlib import Path

from .config import MODEL_FILES, MODELS_DIR

MIN_BYTES = 100_000


def model_path(key: str) -> Path:
    filename, _ = MODEL_FILES[key]
    return MODELS_DIR / filename


def ensure_models() -> dict[str, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (filename, url) in MODEL_FILES.items():
        dest = MODELS_DIR / filename
        if not dest.exists() or dest.stat().st_size < MIN_BYTES:
            _download(url, dest)
        paths[key] = dest
    return paths


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"正在下载 {dest.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "face-rec-demo"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"模型下载失败：{dest.name}。请检查网络后重试。") from exc
    if tmp.stat().st_size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"模型文件不完整：{dest.name}")
    tmp.replace(dest)
