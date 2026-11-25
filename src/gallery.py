"""人脸库。向量存在 npy 里，名字存在 json 里，不用 pickle。"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_DIR


@dataclass
class Match:
    name: str | None
    score: float


class Gallery:
    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        names: list[str] | None = None,
        directory: Path | None = None,
    ):
        self.directory = DATA_DIR if directory is None else directory
        self.embeddings = (
            np.zeros((0, 128), dtype=np.float32) if embeddings is None else embeddings.astype(np.float32)
        )
        self.names = [] if names is None else list(names)
        self._check()

    @classmethod
    def load(cls, directory: Path | None = None) -> "Gallery":
        directory = DATA_DIR if directory is None else directory
        embeddings_path = directory / "embeddings.npy"
        names_path = directory / "names.json"
        if not embeddings_path.exists() or not names_path.exists():
            return cls(directory=directory)
        embeddings = np.load(embeddings_path, allow_pickle=False)
        names = json.loads(names_path.read_text(encoding="utf-8"))
        if embeddings.ndim != 2 or len(names) != len(embeddings):
            raise RuntimeError("人脸库文件损坏。可以删掉 data/embeddings.npy 和 data/names.json 后重新注册。")
        return cls(embeddings, names, directory)

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "embeddings.npy", self.embeddings)
        (self.directory / "names.json").write_text(
            json.dumps(self.names, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def empty(self) -> bool:
        return len(self.names) == 0

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for name in self.names:
            result[name] = result.get(name, 0) + 1
        return result

    def replace(self, name: str, embeddings: list[np.ndarray]) -> None:
        clean = safe_person_name(name)
        vectors = [_normalize(item) for item in embeddings]
        if not vectors:
            raise ValueError("没有可用的人脸特征")
        self.remove(clean)
        stacked = np.stack(vectors).astype(np.float32)
        if self.embeddings.size == 0:
            self.embeddings = stacked
        else:
            if stacked.shape[1] != self.embeddings.shape[1]:
                raise ValueError("特征维度和现有人脸库不一致")
            self.embeddings = np.concatenate([self.embeddings, stacked], axis=0)
        self.names.extend([clean] * len(vectors))

    def remove(self, name: str) -> int:
        keep = [index for index, person in enumerate(self.names) if person != name]
        removed = len(self.names) - len(keep)
        if removed and len(self.embeddings):
            self.embeddings = self.embeddings[keep]
        self.names = [self.names[index] for index in keep]
        return removed

    def match(self, embedding: np.ndarray, threshold: float) -> Match:
        if self.empty():
            return Match(None, 0.0)
        query = _normalize(embedding)
        scores = self.embeddings @ query
        best_name = None
        best_score = -1.0
        for person in dict.fromkeys(self.names):
            indexes = [index for index, item in enumerate(self.names) if item == person]
            score = float(scores[indexes].max())
            if score > best_score:
                best_name = person
                best_score = score
        if best_score < threshold:
            return Match(None, best_score)
        return Match(best_name, best_score)

    def _check(self) -> None:
        if len(self.names) != len(self.embeddings):
            raise ValueError("名字数量和特征数量不一致")


def safe_person_name(name: str) -> str:
    clean = name.strip()
    if not clean or clean in {".", ".."} or any(token in clean for token in ("/", "\\", "..")):
        raise ValueError("名字不能为空，也不能包含路径符号")
    return clean


def _normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm < 1e-6:
        raise ValueError("人脸特征是空的")
    return array / norm
