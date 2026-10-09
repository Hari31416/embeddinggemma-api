from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreparedMedia:
    upload_id: str
    filepath: str
    modality: str  # "image", "audio", "video"


@dataclass
class PreparedItem:
    id: str
    text: str | None = None
    media: list[PreparedMedia] = field(default_factory=list)

    def to_input(self) -> Any:
        # If text only
        if not self.media:
            return self.text or ""

        # If media only or mixed
        images = [m.filepath for m in self.media if m.modality == "image"]
        audios = [m.filepath for m in self.media if m.modality == "audio"]
        videos = [m.filepath for m in self.media if m.modality == "video"]

        payload: dict[str, Any] = {}
        if self.text is not None:
            payload["text"] = self.text

        if images:
            payload["image"] = images[0] if len(images) == 1 else images
        if audios:
            payload["audio"] = audios[0] if len(audios) == 1 else audios
        if videos:
            payload["video"] = videos[0] if len(videos) == 1 else videos

        return payload


class ModelAdapter(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @property
    @abstractmethod
    def revision(self) -> str:
        pass

    @property
    @abstractmethod
    def actual_device(self) -> str:
        pass

    @property
    @abstractmethod
    def requested_device(self) -> str:
        pass

    @property
    @abstractmethod
    def device_fallback(self) -> bool:
        pass

    @property
    @abstractmethod
    def supported_inputs(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def output_dimensions(self) -> list[int]:
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def encode(
        self,
        items: list[PreparedItem],
        task: str = "document",
        truncate_dim: int | None = None,
    ) -> list[list[float]]:
        pass
