MODULE_NAME_TTS = "tts"
VENDOR_NAME = "nyo"
NRSP_SUBPROTOCOL = "nrsp.v1"

# No default host: a deployment must set params.base_url (no audio or key goes to an
# address the operator did not choose).
DEFAULT_BASE_URL = ""
# Used only until the server's audio_start announces the real rate.
DEFAULT_SAMPLE_RATE = 24000  # Orpheus + SNAC-24; SpeechT5 fallback announces 16000
VALID_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}

CONNECT_TIMEOUT_S = 10
# Timeout for the GET /v1/models preflight that runs before the first connect.
MODELS_TIMEOUT_S = 5
# How long a request may stay silent after its text_input_end before we give up.
DEFAULT_REQUEST_TIMEOUT_MS = 30000
RECONNECT_MAX_ATTEMPTS = 5
# Bounded memory for stale-seq / cancelled-request bookkeeping.
STALE_SET_MAX = 1024
