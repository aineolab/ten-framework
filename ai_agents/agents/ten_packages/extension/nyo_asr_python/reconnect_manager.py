import asyncio
import random
import time
from typing import Awaitable, Callable, Optional

from ten_ai_base.message import (
    ModuleError,
    ModuleErrorCode,
    ModuleErrorVendorInfo,
)

from .const import (
    CONNECT_ERROR_REPORT_INTERVAL_S,
    DEFAULT_RECONNECT_WINDOW_MS,
    MODULE_NAME_ASR,
    RECONNECT_BASE_DELAY_S,
    RECONNECT_MAX_DELAY_S,
    RETRY_AFTER_CAP_S,
    STABLE_CONNECTION_S,
)


class ReconnectManager:
    """Time-bounded reconnect with jittered exponential backoff (0.5, 1, 2, 4, 5, 5 ... s).

    An *outage* starts at the first failure / unexpected close and ends in
    ``mark_connection_successful`` once a connection has proved stable: it delivered a result or
    finalize_end (``mark_stable``), or stayed up at least ``stable_s`` (checked in ``note_closed``).
    A socket that opens and dies sooner (``note_open`` only) does not end the outage or reset the
    backoff, so a gateway that accepts and kills every socket ends in one FATAL. A failed attempt is FATAL only
    once the outage has lasted longer than ``window_ms`` (or, if given, after
    ``max_attempts`` attempts); earlier failures are NON_FATAL and rate-limited: the first
    failure of an outage is always reported, then at most one per ``report_interval_s``.
    A ``Retry-After`` hint (429/503 handshake) stretches the next delay, capped at 15 s.
    """

    def __init__(
        self,
        base_delay: float = RECONNECT_BASE_DELAY_S,
        max_delay: float = RECONNECT_MAX_DELAY_S,
        max_attempts: int | None = None,
        logger=None,
        window_ms: int = DEFAULT_RECONNECT_WINDOW_MS,
        report_interval_s: float = CONNECT_ERROR_REPORT_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        stable_s: float = STABLE_CONNECTION_S,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.logger = logger
        self.window_ms = window_ms
        self.report_interval_s = report_interval_s
        self.clock = clock
        self.attempts = 0
        self.outage_start: float | None = None
        self.retry_after: float | None = None
        self._last_report: float | None = None
        self._connection_successful = False
        self.stable_s = stable_s
        self._open_at: float | None = None

    # ------------------------------------------------------------- outage
    def start_outage(self) -> None:
        if self.outage_start is None:
            self.outage_start = self.clock()

    def mark_connection_successful(self) -> None:
        self._connection_successful = True
        self.attempts = 0
        self.outage_start = None
        self.retry_after = None
        self._last_report = None

    def note_open(self) -> None:
        """A socket opened (on_open). It ends the outage only once it proves stable."""
        self._open_at = self.clock()
        if self.outage_start is None and self.attempts == 0:
            self.mark_connection_successful()  # first connect, no outage: nothing to prove

    def mark_stable(self) -> None:
        """The current socket delivered a result / finalize_end: the outage (if any) is over."""
        if self.outage_start is not None or self.attempts or self._last_report is not None:
            self.mark_connection_successful()

    def note_closed(self) -> None:
        """The current socket closed. If it stayed up stable_s, the outage it ended is over."""
        opened, self._open_at = self._open_at, None
        if opened is not None and self.clock() - opened >= self.stable_s:
            self.mark_connection_successful()

    def outage_ms(self) -> float:
        return 0.0 if self.outage_start is None else (self.clock() - self.outage_start) * 1000

    def window_exceeded(self) -> bool:
        if self.max_attempts is not None and self.attempts >= self.max_attempts:
            return True
        return self.outage_start is not None and self.outage_ms() > self.window_ms

    def can_retry(self) -> bool:
        return not self.window_exceeded()

    def set_retry_after(self, seconds: float | None) -> None:
        if seconds is not None and seconds > 0:
            self.retry_after = min(float(seconds), RETRY_AFTER_CAP_S)

    def should_report(self) -> bool:
        """Rate limit for NON_FATAL connect errors; True also records the report."""
        now = self.clock()
        if self._last_report is not None and now - self._last_report < self.report_interval_s:
            return False
        self._last_report = now
        return True

    def get_attempts_info(self) -> dict:
        return {
            "current_attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "outage_ms": round(self.outage_ms()),
            "window_ms": self.window_ms,
        }

    def next_delay(self) -> float:
        return min(self.base_delay * (2 ** max(self.attempts - 1, 0)), self.max_delay)

    async def handle_reconnect(
        self,
        connection_func: Callable[[], Awaitable[None]],
        error_handler: Optional[Callable[..., Awaitable[None]]] = None,
        vendor_name: str | None = None,
        vendor_code: str = "connect_failed",
    ) -> bool:
        """One attempt after the backoff delay. True if connection_func returned."""
        self.start_outage()
        self._connection_successful = False
        self.attempts += 1
        delay = self.next_delay() * random.uniform(0.5, 1.0)  # jitter vs thundering herd
        if self.retry_after is not None:
            delay, self.retry_after = max(delay, self.retry_after), None
        if self.logger:
            self.logger.log_warn(
                f"Attempting reconnection #{self.attempts} after {delay:.2f}s "
                f"(outage {self.outage_ms() / 1000:.1f}s of {self.window_ms / 1000:.0f}s)..."
            )
        try:
            await asyncio.sleep(delay)
            await connection_func()
            return True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            is_fatal = self.window_exceeded()
            if self.logger:
                self.logger.log_error(
                    f"Reconnection attempt #{self.attempts} failed: {e}. "
                    f"{'Giving up.' if is_fatal else 'Will retry...'}"
                )
            if error_handler and (is_fatal or self.should_report()):
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
                    # the attempt may name its own vendor code (e.g. the handshake's HTTP status)
                    code = getattr(e, "vendor_code", None) or vendor_code
                    await error_handler(
                        error,
                        ModuleErrorVendorInfo(
                            vendor=vendor_name, code=str(code), message=str(e)
                        ),
                    )
                else:
                    await error_handler(error)
            return False
