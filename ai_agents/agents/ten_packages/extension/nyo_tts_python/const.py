MODULE_NAME_TTS = "tts"
VENDOR_NAME = "nyo"
NRSP_SUBPROTOCOL = "nrsp.v1"

# No default host: a deployment must set params.base_url (no audio or key goes to an
# address the operator did not choose).
DEFAULT_BASE_URL = ""
# Used only until the server's audio_start announces the real rate.
DEFAULT_SAMPLE_RATE = 24000  # Orpheus + SNAC-24; SpeechT5 fallback announces 16000
VALID_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}

# The gateway holds new opens up to 15 s during an engine swap (nrsp-v1 s.4), so wait longer than that.
CONNECT_TIMEOUT_S = 20
# Timeout for the GET /v1/models preflight that runs before the first connect.
MODELS_TIMEOUT_S = 5
# How long a request may stay silent after its text_input_end before we give up.
DEFAULT_REQUEST_TIMEOUT_MS = 30000
# A TEN request without text_input_end (voice-assistant-with-turn-detection never sends one for an LLM reply) is cut
# into NRSP requests: after this long without a new chunk the open NRSP request gets end:true, so the engine speaks
# its buffered last sentence; the next chunk opens a new NRSP request (<request_id>~<n>). 0 turns it off.
DEFAULT_SEGMENT_IDLE_MS = 1000
# A chunk that ends a sentence (final punctuation, not after a digit) and is not followed by another chunk within this
# long gets a single " " text increment, so the engine (which splits at punctuation FOLLOWED BY whitespace) speaks
# that sentence now. 0 turns it off.
DEFAULT_BOUNDARY_HINT_MS = 250
# A segment open this long is ended at the next chunk that ends a sentence (as above), so SpeechT5, which waits for
# the end of its request, never stays silent past the gateway's stall timer (tts_stall_s, 10 s). 0 turns it off.
DEFAULT_SEGMENT_MAX_MS = 6000
# Reconnect budget is time-based: retry (jittered backoff, capped) until the outage is older than this.
DEFAULT_RECONNECT_WINDOW_MS = 120000
RECONNECT_BACKOFF_CAP_S = 5.0
# Retry-After on a 429/503 upgrade rejection is honoured up to this many seconds.
RETRY_AFTER_CAP_S = 15.0
# Fatal on the wire but cured by a new socket: end the in-flight request (NON_FATAL) and reconnect.
RECONNECTABLE_CODES = frozenset({"engine_restarting", "engine_unavailable", "slow_consumer"})
# The gateway closed an idle socket: not an error; the next request reconnects.
IDLE_TIMEOUT_CODE = "idle_timeout"
# Outage clock (reconnect_window_ms) resets only after a connection stayed up this long or completed a request.
STABLE_CONNECTION_S = 10.0
# NON_FATAL connection errors during one outage: the first always, then at most one per this many seconds.
RECONNECT_ERROR_INTERVAL_S = 10.0
# Voice names the gateway maps to the engine's default speaker on a route WITH a voice catalogue (voice_details;
# serve/speech.py DEFAULT_VOICE_NAMES, case-insensitive). A route without one accepts only "", "default" (exact)
# and the names its /v1/models `voices` allow-list holds (SAPIN_TTS_VOICES).
DEFAULT_VOICE_ALIASES = frozenset({"", "default", "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx",
                                   "sage", "shimmer", "verse"})
# Largest NRSP text message sent (the gateway closes the socket with fatal frame_too_large above 64 KiB);
# a bigger text increment goes out as several text messages of the same request.
TEXT_FRAME_MAX_BYTES = 32 * 1024
# Plain voice names a route without a voice catalogue always accepts (serve/speech.py pick_voice).
PLAIN_DEFAULT_VOICES = frozenset({"", "default"})
# Bounded memory for stale-seq / cancelled-request bookkeeping.
STALE_SET_MAX = 1024
