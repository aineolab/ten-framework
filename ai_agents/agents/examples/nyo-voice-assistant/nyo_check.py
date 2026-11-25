"""Check that your NYO key works, end to end, before you build an agent on it.

Run (needs Python 3.10+ and the `websockets` package):
    export NYO_API_KEY='paste-your-key-here'
    pip install websockets          # or: uv run --with websockets python nyo_check.py
    python nyo_check.py

What it does, in order, against https://api.nyolab.ai (or NYO_BASE_URL if set):
  1. GET  /healthz                       service up, which server answered
  2. GET  /v1/models                     your key is accepted, which models are served
  3. POST /v1/audio/speech               Filipino text to speech over HTTP, saved as sample-fil-http.wav
  4. WS   /v1/tts   (Filipino)           streaming text to speech: time to first audio, saved as sample-fil.wav
  5. WS   /v1/asr   (Filipino)           streams the audio from step 4 back in real time and prints the transcript
  6. Hiligaynon text to speech and speech to text, the same way, if the answering server lists them
     (Hiligaynon speech to text by its full id fw/whisper-large-v3-pld-hil-norm)
Exit code 0 when every step passes. The key is never printed or written to disk.
"""

import array
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

try:
    import websockets
except ImportError:
    sys.exit("This check needs the websockets package: pip install websockets")

BASE = os.environ.get("NYO_BASE_URL", "https://api.nyolab.ai").rstrip("/")
if BASE.endswith("/v1"):
    BASE = BASE[:-3]
HTTP = BASE.replace("wss://", "https://").replace("ws://", "http://")
WS = HTTP.replace("https://", "wss://").replace("http://", "ws://")
KEY = os.environ.get("NYO_API_KEY", "").strip()
SENTENCE = {
    "fil": "Magandang umaga po. Ito ay isang pagsubok ng boses na Filipino.",
    "hil": "Maayong aga. Ini isa ka pagtilaw sang tingog sa Hiligaynon.",
}
RESULTS: list[tuple[str, bool, str]] = []


def record(step: str, ok: bool, detail: str) -> None:
    RESULTS.append((step, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {step}: {detail}", flush=True)


def get_json(path: str, auth: bool) -> tuple[int, dict]:
    req = urllib.request.Request(
        HTTP + path, headers={"Authorization": "Bearer " + KEY} if auth else {}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:  # noqa: BLE001
            return e.code, {}


def resample_to_16k(pcm: bytes, rate: int) -> bytes:
    """Linear resampling of PCM16 mono to 16 kHz, enough for a transcription check."""
    if rate == 16000:
        return pcm
    src = array.array("h", pcm)
    n = int(len(src) * 16000 / rate)
    out = array.array("h", bytes(2 * n))
    step = rate / 16000
    for i in range(n):
        x = i * step
        j = int(x)
        f = x - j
        a = src[j]
        b = src[j + 1] if j + 1 < len(src) else a
        out[i] = int(a + (b - a) * f)
    return out.tobytes()


def save_wav(path: str, pcm: bytes, rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def http_speech(model: str, lang: str) -> None:
    body = json.dumps(
        {
            "model": model,
            "input": SENTENCE[lang],
            "language": lang,
            "response_format": "wav",
        }
    ).encode()
    req = urllib.request.Request(
        HTTP + "/v1/audio/speech",
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer " + KEY,
            "Content-Type": "application/json",
        },
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            ctype = r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        record(
            f"HTTP speech {lang}", False, f"HTTP {e.code} {e.read()[:200]!r}"
        )
        return
    ms = (time.perf_counter() - t0) * 1000
    ok = data[:4] == b"RIFF" and len(data) > 44 + 16000
    if ok:
        with open(f"sample-{lang}-http.wav", "wb") as f:
            f.write(data)
    rate = int.from_bytes(data[24:28], "little") if ok else 0
    secs = (len(data) - 44) / 2 / rate if rate else 0
    record(
        f"HTTP speech {lang}",
        ok,
        f"{ctype}, {secs:.2f} s of audio at {rate} Hz in {ms:.0f} ms"
        + (f", saved sample-{lang}-http.wav" if ok else ""),
    )


async def tts_stream(model: str, lang: str) -> tuple[bytes, int]:
    url = f"{WS}/v1/tts?model={urllib.parse.quote(model, safe='')}&language={lang}"
    t_conn = time.perf_counter()
    try:
        async with websockets.connect(
            url,
            subprotocols=["nrsp.v1", "bearer." + KEY],
            open_timeout=15,
            max_size=1 << 22,
        ) as ws:
            connect_ms = (time.perf_counter() - t_conn) * 1000
            t0 = time.perf_counter()
            await ws.send(
                json.dumps(
                    {
                        "type": "text",
                        "request_id": "check-1",
                        "text": SENTENCE[lang],
                        "end": True,
                    }
                )
            )
            rate, pcm, first, end, err = 0, bytearray(), None, {}, None
            while True:
                m = await asyncio.wait_for(ws.recv(), 60)
                if isinstance(m, bytes):
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                    pcm += m[8:]
                    continue
                msg = json.loads(m)
                t = msg.get("type")
                if t == "audio_start":
                    rate = int(msg.get("sample_rate") or 0)
                elif t == "error":
                    err = f"{msg.get('code')}: {msg.get('message')}"
                    if msg.get("fatal"):
                        break
                elif t == "audio_end":
                    end = msg
                    break
            await ws.send(json.dumps({"type": "close"}))
    except Exception as e:  # noqa: BLE001
        record(f"Streaming speech {lang}", False, f"{type(e).__name__}: {e}")
        return b"", 0
    secs = len(pcm) / 2 / rate if rate else 0
    ok = end.get("reason") == "end" and secs > 1.0 and not err
    if ok:
        save_wav(f"sample-{lang}.wav", bytes(pcm), rate)
    record(
        f"Streaming speech {lang}",
        ok,
        f"connect {connect_ms:.0f} ms, first audio {first or 0:.0f} ms (server ttfb_ms {end.get('ttfb_ms')}), "
        f"{secs:.2f} s at {rate} Hz, end reason {end.get('reason')}"
        + (f", error {err}" if err else "")
        + (f", saved sample-{lang}.wav" if ok else ""),
    )
    return (bytes(pcm), rate) if ok else (b"", 0)


async def asr_stream(model: str, lang: str, pcm16k: bytes) -> None:
    url = f"{WS}/v1/asr?model={urllib.parse.quote(model, safe='')}&language={lang}&sample_rate=16000"
    t_conn = time.perf_counter()
    try:
        async with websockets.connect(
            url, subprotocols=["nrsp.v1", "bearer." + KEY], open_timeout=15
        ) as ws:
            hello = json.loads(await asyncio.wait_for(ws.recv(), 15))
            connect_ms = (time.perf_counter() - t_conn) * 1000
            await ws.send(
                json.dumps(
                    {
                        "type": "config",
                        "language": lang,
                        "interim": True,
                        "endpointing_ms": 0,
                    }
                )
            )
            finals, ttfw, ttlw, err = [], None, None, None
            done = asyncio.Event()

            async def reader():
                nonlocal ttfw, ttlw, err
                async for m in ws:
                    msg = json.loads(m)
                    t = msg.get("type")
                    if t == "result":
                        if msg.get("ttfw_ms") is not None:
                            ttfw = msg["ttfw_ms"]
                        if msg.get("final") and msg.get("text", "").strip():
                            finals.append(msg["text"].strip())
                    elif t == "error":
                        err = f"{msg.get('code')}: {msg.get('message')}"
                    elif t == "finalize_end" and msg.get("id") == "check-1":
                        ttlw = msg.get("ttlw_ms")
                        done.set()
                        return

            rd = asyncio.create_task(reader())
            chunk = 2048  # 64 ms of 16 kHz PCM16
            t_start = time.perf_counter()
            for i in range(0, len(pcm16k), chunk):
                await ws.send(pcm16k[i : i + chunk])
                await asyncio.sleep(
                    max(0, t_start + (i + chunk) / 32000 - time.perf_counter())
                )
            await ws.send(json.dumps({"type": "finalize", "id": "check-1"}))
            await asyncio.wait_for(done.wait(), 20)
            await ws.send(json.dumps({"type": "close"}))
            rd.cancel()
    except Exception as e:  # noqa: BLE001
        record(
            f"Round trip {lang} (speech, then speech to text)",
            False,
            f"{type(e).__name__}: {e}",
        )
        return
    text = " ".join(finals)
    # Round trip: the transcript of our own synthesised sentence should contain most of its words. A low score means
    # either the speech came back incomplete (listen to the saved sample) or the transcription missed it.
    norm = lambda s: [
        w.strip(".,!?").lower().replace("pilipino", "filipino")
        for w in s.split()
    ]
    ref, got = norm(SENTENCE[lang]), set(norm(text))
    score = sum(w in got for w in ref) / len(ref)
    ok = bool(text) and not err and score >= 0.6
    record(
        f"Round trip {lang} (speech, then speech to text)",
        ok,
        f"model {hello.get('model')}, connect {connect_ms:.0f} ms, server ttfw_ms {ttfw}, ttlw_ms {ttlw}, "
        f"transcript {text!r}, {score:.0%} of the sentence's words heard"
        + (f", error {err}" if err else "")
        + (
            ""
            if ok
            else f"; listen to sample-{lang}.wav to tell incomplete speech from a transcription miss"
        ),
    )


async def main() -> int:
    print(f"NYO check against {HTTP}\n")
    if not KEY:
        print(
            "Set NYO_API_KEY first:  export NYO_API_KEY='paste-your-key-here'"
        )
        return 2
    code, h = get_json("/healthz", auth=False)
    record(
        "Service health",
        code == 200 and h.get("ok") is True,
        f"HTTP {code}, answered by release {h.get('release')}, engines "
        + ", ".join(
            f"{k} {v.get('state')}" for k, v in (h.get("engines") or {}).items()
        ),
    )
    code, m = get_json("/v1/models", auth=True)
    if code != 200:
        record(
            "Your key",
            False,
            f"HTTP {code} {m.get('code')}: {m.get('message')}. Check NYO_API_KEY.",
        )
        return 1
    ids = [d["id"] for d in m.get("data", [])]
    record("Your key", True, "accepted; models served now: " + ", ".join(ids))

    if "halohalo/tts-fil" in ids:
        http_speech("halohalo/tts-fil", "fil")
    # Speech-to-text model per language: Filipino by its alias, Hiligaynon by its full id (see the guide, "Models").
    asr_model = {
        "fil": "halohalo/asr-fil",
        "hil": "fw/whisper-large-v3-pld-hil-norm",
    }
    for lang in ("fil", "hil"):
        tts, asr = f"halohalo/tts-{lang}", asr_model[lang]
        if tts not in ids and asr not in ids:
            print(
                f"[SKIP] {lang}: not served by the server that answered (see the model list above)"
            )
            continue
        pcm, rate = await tts_stream(tts, lang) if tts in ids else (b"", 0)
        if asr in ids and pcm:
            await asr_stream(asr, lang, resample_to_16k(pcm, rate))
    failed = [s for s, ok, _ in RESULTS if not ok]
    print(
        f"\n{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed."
        + (f" Failed: {', '.join(failed)}" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
