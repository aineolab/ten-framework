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

# Bounded reconnect ceiling; backoff is 0.5, 1, 2, 4, 4... seconds.
RECONNECT_MAX_ATTEMPTS = 10

CONNECT_TIMEOUT_S = 10

# Budget for the GET /v1/models preflight (transport.fetch_models).
MODELS_TIMEOUT_S = 5
