import asyncio
import random
from typing import Awaitable, Callable, Optional

from ten_ai_base.message import (
    ModuleError,
    ModuleErrorCode,
    ModuleErrorVendorInfo,
)

from .const import MODULE_NAME_ASR


class ReconnectManager:
    """Bounded reconnect with exponential backoff (0.5, 1, 2, 4, 4 ... s).

    Same contract as xai_asr_python's manager: the counter resets in
    ``mark_connection_successful`` (called from on_open), the last allowed
    attempt reports a FATAL error, earlier ones NON_FATAL.
    """

    def __init__(
        self,
        base_delay: float = 0.5,
        max_delay: float = 4.0,
        max_attempts: int = 4,
        logger=None,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.logger = logger
        self.attempts = 0
        self._connection_successful = False

    def mark_connection_successful(self) -> None:
        self._connection_successful = True
        self.attempts = 0

    def get_attempts_info(self) -> dict:
        return {
            "current_attempts": self.attempts,
            "max_attempts": self.max_attempts,
        }

    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts

    def next_delay(self) -> float:
        return min(self.base_delay * (2 ** max(self.attempts - 1, 0)), self.max_delay)

    async def handle_reconnect(
        self,
        connection_func: Callable[[], Awaitable[None]],
        error_handler: Optional[Callable[..., Awaitable[None]]] = None,
        vendor_name: str | None = None,
        vendor_code: str = "connect_failed",
    ) -> bool:
        """One attempt after the backoff delay. True if connection_func ran."""
        self._connection_successful = False
        self.attempts += 1
        delay = self.next_delay()
        if self.logger:
            self.logger.log_warn(
                f"Attempting reconnection #{self.attempts} after {delay:.2f}s..."
            )
        try:
            await asyncio.sleep(delay * random.uniform(0.5, 1.0))  # jitter vs thundering herd
            await connection_func()
            return True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            is_fatal = self.attempts >= self.max_attempts
            if self.logger:
                self.logger.log_error(
                    f"Reconnection attempt #{self.attempts} failed: {e}. "
                    f"{'Giving up.' if is_fatal else 'Will retry...'}"
                )
            if error_handler:
                error = ModuleError(
                    module=MODULE_NAME_ASR,
                    code=(
                        ModuleErrorCode.FATAL_ERROR.value
                        if is_fatal
                        else ModuleErrorCode.NON_FATAL_ERROR.value
                    ),
                    message=f"Reconnection attempt #{self.attempts} failed: {e}",
                )
                if vendor_name:
                    await error_handler(
                        error,
                        ModuleErrorVendorInfo(
                            vendor=vendor_name, code=vendor_code, message=str(e)
                        ),
                    )
                else:
                    await error_handler(error)
            return False
