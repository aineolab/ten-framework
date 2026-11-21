# nyo_tts_python

TEN TTS extension for the NYO speech plane (NRSP v1), built on `AsyncTTS2BaseExtension`
(ten_ai_base 0.7) in WebSocket mode, modelled on `deepgram_tts`. One socket carries many
sequential requests; text increments stream up, `[u32 request_seq][u32 audio_offset_ms][PCM16]`
frames stream down. Full guide: `docs/ten.md` in nyo-serve.

## What is different from the other WS TTS addons

* **Cancel without reconnect.** `tts_flush` -> `cancel_tts()` sends NRSP `flush`, marks the
  request's `request_seq` stale and emits `tts_audio_end(INTERRUPTED)` immediately. Frames that
  were already in flight carry the old seq and are dropped by header check. The socket is never
  recycled (deepgram/xai reconnect, 100-300 ms).
* **Sample rate comes from the server** (`audio_start.sample_rate`): 24000 for Orpheus + SNAC,
  16000 for the SpeechT5 fallback. `params.sample_rate` is only the pre-first-request default.
* **Frame timestamps are the server's `audio_offset_ms`**, so TEN's subtitle/timeline checks see the
  true position; gaps >20 ms are logged and counted.
* **Speech without `text_input_end`.** voice-assistant-with-turn-detection never ends an LLM reply, and the NYO engines
  split sentences only at punctuation followed by whitespace (SpeechT5 waits for `end`). main_python `parse_sentences`
  cuts after every `. , ? !` and leaves the whitespace at the start of the NEXT chunk, so only a reply's last sentence
  lacks it; it also cuts inside `₱3.50` or `nyolab.ai`, whose next chunk has no whitespace. Text therefore always
  goes out verbatim, and:
  (a) **boundary hint**: a chunk that ends with `. ! ? …` NOT after a digit, and that no next chunk follows within
  `boundary_hint_ms` (default 250), gets one `" "` text increment, so the engine speaks that sentence now (a next
  chunk in time brings its own whitespace and cancels it; `3.` is never hinted: the next chunk or (b) covers it). The
  engines' splitter (`modules/tts/frontend.py` `split_sentences`, `(?<=[.!?…])\s+`) cuts only when the whitespace
  follows the punctuation directly, so a sentence that ends in a closing quote or bracket (`po."`, `po.)`; `(po).` ends in `.` and is hinted)
  or a CJK stop gets no hint: a space there would not split, and the text is never altered. Such a last sentence is
  spoken at the idle end (b); mid-reply it is spoken together with the next sentence;
  (b) **idle end**: after `segment_idle_ms` (default 1000) without a new chunk the open NRSP request gets `end:true`;
  (c) **max segment**: a segment open for `segment_max_ms` (default 6000) is ended right after the next chunk that ends
  a sentence (`. ! ? … 。 ！ ？`, optionally followed by closing quotes/brackets, not after a digit), so an engine that speaks only at the end of its request (SpeechT5, `hil`) never stays silent
  past the gateway's stall timer (`tts_stall_s`, 10 s) on a long fast reply. **Hard cut**: a segment that reaches
  `segment_max_ms` + 2000 ms without such a sentence end (a numbered list `1.` ` Bigas,\n2.`, prices at every sentence
  end, a long comma chain) is ended right after the next chunk that ends with `, ; :` (or `，；：、`) not right after a digit,
  a list number at a line start (`\n2.`), or a newline, optionally followed by closing quotes/brackets. Chunks end only
  after punctuation, so this never cuts mid-word, and a mark right after a digit is never a cut: `₱3.50`, `₱1,250`,
  `₱1,234.50` and `10:30` stay whole (a chunk that only begins with digits, like `234.`, is not a list number).
  After (b) or (c) the next chunk of the same TEN request opens a new NRSP request `<request_id>~2`, `~3`, ... All
  segments are one TEN request: one `tts_audio_start`, one timeline (frame timestamps continue across segments),
  summed duration, TTFB from the first segment; `tts_audio_end` only on `text_input_end` (after the last segment's
  `audio_end`), flush (cancels the playing segment; later chunks are dropped), `stale_request`, a FATAL, or stop.
  Each of `boundary_hint_ms`, `segment_idle_ms`, `segment_max_ms` is off at `0`.
* **A failed segment is not a failed reply.** A failure of one segment (any non-fatal error that names one of the
  request's segments: `admission_rejected`, `engine_error`, the engines' in-band `tts_engine_error`,
  `synthesis_truncated`, `text_too_long`, future engine codes; the gateway's own cancel `server_cancelled`; a lost
  socket / `engine_restarting` / `engine_unavailable` / `slow_consumer`, also a socket that dies during an idle-close
  resend; vendor code `connection_lost`) costs that segment only. Not so
  `stale_request` and the codes that say the route is wrong (`unknown_voice`, `unsupported_voice`, `unknown_model`,
  `unsupported_language`, `unsupported_sample_rate`, `unauthorized`, `unsupported_version`): they end the TEN request.
  For a failed segment there is one NON_FATAL error (connection errors
  rate-limited as below) and the TEN request goes on, its next chunk opening a new segment (on a fresh socket if
  needed). When the TEN request ends it is `REQUEST_END` if any audio was produced, else `ERROR`. A request that is
  still waiting for its first socket, the watchdog, `stale_request` and FATAL errors end the request with `ERROR` as
  before.
* A watchdog fails a request the gateway owes something (a segment's `audio_end`, or the end after `text_input_end`)
  that stays silent for `request_timeout_ms`. A request whose segments are all spoken and that waits for the LLM's next
  chunk owes nothing: a slow LLM never times it out, and a connection loss or idle close in that pause does not fail
  it (its next chunk reconnects).
* **The gateway's own cancel is an error.** An `audio_end` with reason `interrupted` / `flushed` for a request this
  extension did not flush (engine swap, shutdown drain) ends it with `tts_audio_end(ERROR)` + a NON_FATAL error
  `server_cancelled`; only a `tts_flush` gives `INTERRUPTED`.
* **No oversized frames.** A text increment whose NRSP message would exceed 32 KiB (the gateway closes the socket on
  frames over 64 KiB) is sent as several `text` messages of the same request, `end` only on the last.
* **Reconnect, not die.** Fatal-on-the-wire `engine_restarting` / `engine_unavailable` / `slow_consumer` cost the
  in-flight segment (a NON_FATAL error, 1000; see "A failed segment" above) and reconnect in the background; while idle
  they emit nothing. `idle_timeout` (the gateway closes sockets idle for `SAPIN_IDLE_TIMEOUT_S`, default 300 s) is
  logged at info and reported only as a connection-status change; the next request reconnects. The extension sends no
  app-level ping. Once the gateway announces such a close, the socket takes no new request, so a request that starts
  before the close frame lands opens a fresh socket instead of failing. A close or error only fails the request whose
  text went to that socket; a request still waiting for a connection connects on its own. If the idle close crosses
  the first text of a segment (the gateway never saw it: nothing came back for that segment), the segment is resent
  once on a fresh socket instead of failing (segments already spoken are never resent); later chunks of that request
  wait for the resend, so the fresh socket gets every chunk once and in order. A chunk whose send fails on that closing socket is handed to the resend too.
* **Late chunks.** A chunk of a request this extension already ended (error mid-stream, flushed, or completed) is not
  synthesised and releases the base input loop at once, also after a `tts_flush`; a duplicate chunk of a request
  still playing is ignored without ending it. The id of the last request stays ended after a flush; an older id
  reused after a flush is a new request.
* **`stale_request` for the open request** (an id the gateway already ended on that socket, e.g. an older id reused
  after a flush) ends it at once with `tts_audio_end(ERROR)` + NON_FATAL `stale_request`; for an id that already finished it
  is ignored.
* **Time-based reconnect budget.** Jittered backoff (0.5 s doubling, capped at 5 s; a 429/503 `Retry-After` is honoured
  up to 15 s) until the outage is older than `reconnect_window_ms`. Then, if a request is waiting or in flight, one
  FATAL `connect_failed`; if the extension is idle, nothing is reported and the background retries stop: the next
  request reconnects, and only if that fails while the outage is still older than the window is it the FATAL. Connect timeout is
  20 s (the gateway queues opens up to 15 s during an engine swap). A request that arrives during an outage waits for
  the socket up to `request_timeout_ms`, then ends with `tts_audio_end(ERROR)` + NON_FATAL; a flush releases it (and the
  base input loop) at once, even during the request's own first handshake, and nothing of it reaches the server; the
  handshake finishes in the background for the next request.
* **Outage clock.** The window counts from the first failure and resets only after a socket stayed up 10 s or completed
  a request: a gateway that accepts and drops connections at once still ends in FATAL after the window.
* **Rate-limited NON_FATAL errors.** Within one outage the first connection error (`connection_lost`, a reconnectable
  server code, "not connected within request_timeout_ms") is reported, then at most one per 10 s; every affected
  request still ends with `tts_audio_end(ERROR)`.
* **Handshake 400/404** (also answered by a restarting gateway) re-check `GET /v1/models`: FATAL only when the listing
  conclusively lacks the route (below); otherwise retried inside the window. The rejection body's NRSP `code` is read:
  `unknown_voice` / `unsupported_voice` -> FATAL `unknown_voice` at once (unless the listing says the route itself is
  not served now: then retried), `unsupported_sample_rate` FATAL, `unknown_model` / `unsupported_language` ->
  `model_not_found` / `language_not_supported` (retried, rule A'). 401/403 are FATAL at once.
* **A missing model or language is never FATAL at once.** The gateway lists only engines that are ready now, so a model
  missing from `/v1/models` (even next to other `halohalo/` TTS models), or a language no listed TTS model declares
  (`languages` or `language`), may be an engine that is down. Before the first connect and after a handshake 400/404
  the extension then reports one NON_FATAL `model_not_found` / `language_not_supported` (rate-limited as above), keeps
  retrying inside `reconnect_window_ms`, and if it is still missing when the window ends the one FATAL carries that
  code. Requests meanwhile end with `tts_audio_end(ERROR)` after `request_timeout_ms`.
* **Conclusive "not served"** (FATAL at once): the id is listed as another kind (an ASR model); a voice the effective
  language's route refuses, checked as the gateway does: against its voice catalogue (`voice_details`,
  case-insensitive; the default aliases `default`, `alloy`, `nova`, ... pass), or on a route without one (SpeechT5,
  `hil` today) against its `voices` allow-list (exact names; only `""` and `default` always pass; an empty list lets
  any name through). No answer, a non-200 or no TTS model in our namespace is inconclusive: the
  extension proceeds and keeps retrying.
* **FATAL after the window only when a request is affected.** A TEN FATAL means "this extension cannot work without
  operator action". An outage that ends while the extension is idle (a planned restart or model reload longer than
  the window) is not FATAL: the next request simply reconnects. When a request is waiting or in flight and the outage
  is older than `reconnect_window_ms`, the one FATAL fires; afterwards the extension stops retrying and every later
  request ends with `tts_audio_end(ERROR)` at once instead of hanging, so the agent can fail over or alert. It does not
  recover by itself after that FATAL; restart the graph.

* **Stop.** `on_stop` cancels the extension's background tasks (segment timers, idle-close resends, background
  connects) and waits for them at most 1 s; a handshake in flight is abandoned, no socket stays open.
* **Known limit (lost-socket race).** If the gateway connection drops without an NRSP error (TCP reset, gateway crash)
  and the next chunk reconnects in the short window before the old socket's reader has reported the close, the
  segments still owed on the old socket are not marked lost: that TEN request ends only via the watchdog
  (`request_timeout_ms`, `tts_audio_end(ERROR)`). A drop announced by the gateway (`engine_restarting`,
  `idle_timeout`, ...) or reported first by the reader is not affected.

## Properties

| property | default | notes |
|---|---|---|
| `params.api_key` | `${env:NYO_API_KEY\|}` | Empty or HTTP 401/403 -> FATAL (-1000). |
| `params.base_url` | required, no default (`${env:NYO_BASE_URL\|}`) | addon appends `/v1/tts`. `ws://` only to loopback unless `allow_insecure`. |
| `params.allow_insecure` | `false` | `true` permits `ws://` to a non-local host; otherwise FATAL `insecure_url` at init. |
| `params.tls_verify` | `true` | verify the `wss://` certificate; a bad one is FATAL `tls_error`. |
| `params.model` | `halohalo/tts-fil` | any TTS id the gateway's `GET /v1/models` lists; checked before the first connect (missing: NON_FATAL `model_not_found` and retries; FATAL only when the window ends). |
| `params.voice` | `""` | forwarded as `?voice=`. Empty = the model's default speaker; see [Speakers](#speakers). Must be one the route serves (see "Conclusive" above): a catalogue id or default alias, or on a route without a catalogue `default` or a name from its `voices` list; else FATAL `unknown_voice`. |
| `params.language` | `""` (from model) | forwarded as `?language=`; overrides the model suffix at the gateway. Empty -> derived from `halohalo/tts-<lang>`. property.json ships it empty on purpose: TEN deep-merges a graph's `params` into property.json, so a non-empty default would override the language of every model a graph picks. Set it only together with a model that serves it. |
| `params.sample_rate` | `24000` | fallback only: the stream's rate is the model's (`audio_start.sample_rate`); NRSP has no rate request and the addon does not resample. So `tts_guarder` `basic_audio_setting` passes only when its two configs name models with different rates. |
| `params.request_timeout_ms` | `30000` | inactivity limit once `text_input_end` was sent; also how long a request waits for a socket during an outage. |
| `params.reconnect_window_ms` | `120000` | an outage longer than this ends in one FATAL `connect_failed`; shorter ones recover. |
| `params.segment_idle_ms` | `1000` | no new chunk for this long and no `text_input_end`: `end:true` for the open NRSP request so its last sentence is spoken; the next chunk opens `<request_id>~<n>`. `0` = off (only for graphs that always send `text_input_end`). |
| `params.boundary_hint_ms` | `250` | a chunk ending in `. ! ? …` (not after a digit, not before a closing quote) with no next chunk within this long gets one `" "` increment, so its sentence is spoken without waiting for the idle end. `0` = off. |
| `params.segment_max_ms` | `6000` | a segment open this long ends after the next sentence-final chunk (not after a digit); 2000 ms later also after a `, ; :` / list-number / newline chunk (hard cut): keeps SpeechT5 under the gateway's 10 s stall timer on long fast replies. `0` = off. |
| `params.enable_words` | `false` | NRSP `alignment` -> `tts_text_result` (subtitles). |
| `params.gc_freeze` | `false` | opt-in. `true` runs `gc.collect(); gc.freeze()` at the end of `on_init`, removing gen-2 GC passes over the TEN + pydantic init-time heap from the audio path (measured in a TEN host: max per-frame cost 4067 us -> 125 us, bench/results/ten.json). It is process-wide: it also freezes the init-time heap of every other extension in the same host, so enable it only for a host you own. |
| `dump`, `dump_path` | `false`, `/tmp` | per-request PCM `nyo_dump_<request_id>.pcm`. |

## Models on `api.nyolab.ai`

`GET /v1/models` is the authority; the extension reads it on start. Checked 2026-11-20 (gateway release
`gateway-aws-4`, from its `/healthz` and published model table):

| Model | What |
|---|---|
| `fw/flipvox-whisper-base-fil` (alias `halohalo/asr-fil`) | Filipino speech to text, the default |
| `fw/whisper-large-v3-pld-fil-norm` | Filipino speech to text, large, slower |
| `fw/whisper-large-v3-pld-hil-norm` | Hiligaynon speech to text |
| `halohalo/tts-fil` | Filipino text to speech (Orpheus, 24 kHz, ten speakers) |
| `halohalo/tts-hil` | Hiligaynon text to speech (SpeechT5, 16 kHz, one speaker) |

## Speakers

A model may offer several speakers; the gateway lists them in `GET /v1/models` (`voices`, `default_voice`) and in
`GET /v1/audio/voices`. Set `params.voice` to one of those ids, or leave it empty for the default. A voice the route
does not serve is FATAL `unknown_voice` (from the listing, or from the handshake rejection body), and
`update_configs` refuses it and keeps the old config.

The voice belongs to the session: the gateway fixes it when the socket opens and keeps it for every request on
that socket. `update_configs` with a new `voice` therefore opens a fresh socket: right away when idle, otherwise
before the next request.

Orpheus Filipino (`halohalo/tts-fil`) offers ten speakers, ids `FIL_0051`, `FIL_0059`, `FIL_0078` (default),
`FIL_0079`, `FIL_0081`, `FIL_0084`, `FIL_0107`, `FIL_0113`, `FIL_0115` and `FIL_0135`. A gateway in front of
another one must also allow the ids (`SAPIN_TTS_VOICES` on older releases).

SpeechT5 Hiligaynon (`halohalo/tts-hil`) has one speaker: leave `params.voice` empty. Its audio is 16 kHz, so
`halohalo/tts-fil` and `halohalo/tts-hil` are a pair of models with different rates for the `tts_guarder`
`basic_audio_setting` test (`tests/configs/property_fil.json`, `property_hil.json`).

## Graph snippet (agents/examples/voice-assistant)

```json
{
  "type": "extension",
  "name": "tts",
  "addon": "nyo_tts_python",
  "extension_group": "tts",
  "property": {
    "dump": false,
    "params": {
      "api_key": "${env:NYO_API_KEY}",
      "base_url": "wss://api.nyolab.ai",
      "model": "halohalo/tts-hil"
    }
  }
}
```

`language` is left out: it follows the model (`hil` here, SpeechT5 16 kHz; `halohalo/tts-fil` is Orpheus 24 kHz).

Add `{"path": "../../../ten_packages/extension/nyo_tts_python"}` to `tenapp/manifest.json`.

## Errors

`vendor_info.code` names the cause: the NRSP code (`engine_error`, `connection_lost`, ...), or for a request that could
not get a socket the connect error's code (`model_not_found`, `language_not_supported`, `engine_unavailable`, ...;
the HTTP status, if any, is in the message), else the HTTP status, else `connect_failed` / `request_failed`.

FATAL (-1000): bad key (401/403), `unknown_voice` (listing or handshake rejection body), `model_not_found` for an id listed as another kind (conclusive
`/v1/models` answers, before the first connect or after a handshake 400/404), `model_not_found` /
`language_not_supported` for a model / language still missing when `reconnect_window_ms` ends, `tls_error`, an outage longer than `reconnect_window_ms` that affects a request, and any fatal server error other than the reconnectable ones above.
Exactly one FATAL `tts_error` per extension instance, always sent with the request's `tts_audio_end` (the socket is
closed only after reporting, never by cancelling the task that reports); afterwards nothing reconnects, and every
request ends with `tts_audio_end(ERROR)` at its first chunk (final or not) without another error. `vendor_info.vendor` is always `nyo`; an engine's
name goes into the message.

`busy` (the engine is at its cap) is not fatal. On an open socket it names a segment, so it costs that segment
only: one NON_FATAL `tts_error` with `vendor_info.code = "busy"`, and the reply continues with its next chunk. On the
upgrade it is HTTP 503 with `Retry-After`, which the reconnect loop honours.

## Metrics

* TTFB: first text frame written to the socket -> first non-empty audio frame (connect/preflight time excluded).
* `request_event_interval_ms`: `tts_audio_start` -> `tts_audio_end` (excludes TTFB; 0 when no audio was produced).
* `request_total_audio_duration_ms`: emitted PCM bytes at the stream's sample rate.

## update_configs

`{"params": {"voice"?, "language"?, "model"?}}`. Validated first (a rejected update changes nothing and the cmd
returns ERROR): types and format, then the new route against `GET /v1/models` with the rules above. A model, language or
voice the listing conclusively does not serve, or a model / language missing from a listing of same-namespace TTS
models, is rejected and the old config stays (the old route still works); an inconclusive answer accepts it. Applied at once when idle (fresh socket, model re-checked against `/v1/models`); during a request it
is held and applied before the next request, so the running request keeps its route and sample rate. A socket that
was still opening when a change was applied (prewarm, reconnect, background connect) is closed and reopened with the
new values: no request goes out on stale settings. A new `model`
without `language` gets the language the gateway derives from `halohalo/tts-<lang>`.

## Tests

The TEN guarders need `pytest<9` (their `conftest.py` imports `_pytest.config.Notset`, removed in pytest 9).

```bash
cd ten/nyo_tts_python && PYTHONPATH=<runtime lib:interface:ten_ai_base/interface:ten> pytest tests -q
python3 ten/fake_nrsp_server.py --port 9100 &
NYO_API_KEY=fake-key task tts-guarder-test EXTENSION=nyo_tts_python
```
