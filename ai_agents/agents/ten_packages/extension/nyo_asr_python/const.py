DUMP_FILE_NAME = "nyo_asr_in.pcm"
MODULE_NAME_ASR = "asr"
VENDOR_NAME = "nyo"

# NRSP v1 websocket subprotocol offered at the handshake.
NRSP_SUBPROTOCOL = "nrsp.v1"

# No default host: a deployment must set params.base_url (no audio or key goes to an
# address the operator did not choose).
DEFAULT_BASE_URL = ""
DEFAULT_ENDPOINTING_MS = 1000
DEFAULT_SAMPLE_RATE = 16000

# "Keep" buffer window while the socket is down (plan 3.2: bounded 2 s).
DEFAULT_BUFFER_MS = 2000

# Reconnect budget is time-based: jittered backoff 0.5, 1, 2, 4, 5, 5... s, FATAL only once an outage
# has lasted longer than params.reconnect_window_ms.
DEFAULT_RECONNECT_WINDOW_MS = 120_000
RECONNECT_BASE_DELAY_S = 0.5
RECONNECT_MAX_DELAY_S = 5.0
# A reconnect ends an outage (resets the window, backoff and error rate limit) only once the socket stayed
# up this long or delivered a result / finalize_end.
STABLE_CONNECTION_S = 10.0
# Retry-After on a 429/503 handshake is honoured up to this many seconds.
RETRY_AFTER_CAP_S = 15.0
# NON_FATAL connect errors during one outage: the first is always reported, then at most one per interval.
CONNECT_ERROR_REPORT_INTERVAL_S = 10.0

# The gateway queues opens up to 15 s during an engine swap, so the handshake budget is above that.
CONNECT_TIMEOUT_S = 20

# Server fatal error codes that end the socket but not the service: reported NON_FATAL and reconnected.
RECONNECTABLE_CODES = frozenset({"engine_restarting", "engine_unavailable", "slow_consumer"})
# Server error for a model it does not serve right now (an engine that is down, or a typo): NON_FATAL
# model_not_found, retried inside reconnect_window_ms (the real gateway denies the handshake with 404 instead).
UNKNOWN_MODEL_CODE = "unknown_model"
# Server closed an idle socket: not an error for the graph; reconnect on the next audio frame.
IDLE_TIMEOUT_CODE = "idle_timeout"

# Default finalize wait for streaming models: above the gateway's own 3000 ms finalize timer, so its
# finalize_end (and any final it still forwards) arrives first.
DEFAULT_FINALIZE_TIMEOUT_MS = 3500

# Models the gateway (serve/) treats specially.
NEMOTRON_MODEL = "nvidia/nemotron-3.5-transcribe"
# The gateway's Nemotron finalize timer is 30 s (a final is a batch decode); wait a little longer by default.
NEMOTRON_FINALIZE_TIMEOUT_MS = 32_000

# Budget for the GET /v1/models preflight (transport.fetch_models).
MODELS_TIMEOUT_S = 5
