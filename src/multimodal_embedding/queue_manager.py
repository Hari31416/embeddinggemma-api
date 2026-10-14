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

    @property
    def queue_depth(self) -> int:
        return self._waiting_count

    @property
    def active_workers(self) -> int:
        return self._active_count

    async def run(self, func: Callable[..., T], *args, **kwargs) -> T:
        # These counters are only accessed on the event loop, without awaits.
        if (
            self._active_count + self._waiting_count
            >= self.max_concurrency + self.max_queue_size
        ):
            raise QueueSaturationError("Inference queue saturated.")
        self._waiting_count += 1
        try:
            await self._semaphore.acquire()
        finally:
            # Cancelled waiters must release their queue reservation too.
            self._waiting_count -= 1
        self._active_count += 1
        try:
            # Shield native asyncio cancellation as well as AnyIO cancellation.
            # A running inference thread cannot be stopped; retain its slot until done.
            work = asyncio.create_task(
                anyio.to_thread.run_sync(lambda: func(*args, **kwargs))
            )
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:

                def release_when_done(task):
                    if not task.cancelled():
                        task.exception()
                    self._active_count -= 1
                    self._semaphore.release()

                work.add_done_callback(release_when_done)
                work = None
                raise
        finally:
            if work is not None:
                self._active_count -= 1
                self._semaphore.release()
