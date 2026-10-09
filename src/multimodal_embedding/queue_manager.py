import asyncio
import logging
from typing import Callable, TypeVar
import anyio

from multimodal_embedding.exceptions import QueueSaturationError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class InferenceQueueManager:
    def __init__(self, max_concurrency: int = 1, max_queue_size: int = 10) -> None:
        self.max_concurrency = max_concurrency
        self.max_queue_size = max_queue_size
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._waiting_count = 0
        self._active_count = 0
        self._lock = asyncio.Lock()

    @property
    def queue_depth(self) -> int:
        return self._waiting_count

    @property
    def active_workers(self) -> int:
        return self._active_count

    async def run(self, func: Callable[..., T], *args, **kwargs) -> T:
        # Check if queue has space
        async with self._lock:
            if self._waiting_count >= self.max_queue_size:
                logger.warning(
                    "Queue saturation: waiting=%d >= max_queue_size=%d",
                    self._waiting_count,
                    self.max_queue_size,
                )
                raise QueueSaturationError(
                    f"Inference queue saturated. Currently {self._waiting_count} requests waiting. Max capacity is {self.max_queue_size}."
                )
            self._waiting_count += 1

        try:
            async with self._semaphore:
                async with self._lock:
                    self._waiting_count -= 1
                    self._active_count += 1

                try:
                    # Run blocking inference in worker thread outside async event loop
                    result = await anyio.to_thread.run_sync(
                        lambda: func(*args, **kwargs)
                    )
                    return result
                finally:
                    async with self._lock:
                        self._active_count -= 1
        except Exception:
            # In case acquire failed or raised before entering block
            async with self._lock:
                if self._waiting_count > 0 and self._active_count == 0:
                    pass
            raise
