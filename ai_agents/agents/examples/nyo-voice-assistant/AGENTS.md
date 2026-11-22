# NYO voice assistant: instructions for coding agents

Version: 2026-11-22. For this example (`nyo-voice-assistant`) and the two NYO extensions. Human version: `README.md` here,
and the evaluation guide NYO sends with a key. If they disagree, stop and ask the person you work for. Repo-wide rules
are in `/AGENTS.md` at the repository root. `nyo_check.py`, next to this file, checks the key and the service.

## Goal

Run a TEN Framework voice agent that hears and speaks Filipino (`fil`) or Hiligaynon (`hil`), using two TEN extensions:
`nyo_asr_python` (streaming speech to text) and `nyo_tts_python` (streaming text to speech). Both call
`wss://api.nyolab.ai`.

Done means: `nyo_check.py` exits 0, the example agent starts with no fatal error, and a person has talked to it and
heard an answer. You cannot do the last part yourself. Ask the person to do it, and report it as "not verified" until
they have.

## Rules

1. Never print, log, commit or send `NYO_API_KEY`, and never put it in a URL. Read it from the environment or from
   `ai_agents/.env`, which must stay out of Git. If it is missing, ask for it. Do not search for one.
2. Test data only. No real customer calls and no personal data.
3. **Capacity is 2 speech-to-text sessions for the whole service, shared with other users.** One agent call holds one
   session for its whole length. Run one thing at a time: no parallel runs, load tests or retry loops, and do not run
   TEN's guarder suites against the live service. Ask whether a test window has been agreed before you open a session.
4. On `busy`, 429 or 503, wait for `Retry-After` and retry at most 3 times. Then stop and report.
5. Do not change extension code to get past an error. Report the error. Configuration lives in `.env` and
   `tenapp/property.json`.
6. Do not set `allow_insecure: true` or `tls_verify: false`.
7. Report only what you ran. Quote the exact error code or log line, with any key removed.

## Pinned facts

| Item | Value |
|---|---|
| Repository | `https://github.com/aineolab/ten-framework.git` (public, no login) |
| Branch | `feat/nyo-extensions`. Must contain commit `2b2a19d169bee46e7d20cf9ae295a4b3586288bc`. If NYO gave you a tag or commit, use that |
| Do not use | tag `nyo-eval-2026-11-18` (older, lacks fixes); any service address other than the two below |
| Extensions | `ai_agents/agents/ten_packages/extension/nyo_asr_python`, `.../nyo_tts_python` |
| Example | `ai_agents/agents/examples/nyo-voice-assistant` |
| Graphs | `nyo_voice_assistant` (fil, starts by default), `nyo_voice_assistant_hil` (hil) |
| TEN versions in the manifests | `ten_runtime_go` 0.11, `ten_ai_base` 0.7 |
| Container | `ten_agent_dev`, image `ghcr.io/ten-framework/ten_agent_build:0.7.14`, `linux/amd64` (emulated on ARM) |
| Host ports | 3000, 8000-9001, and `GRAPH_DESIGNER_SERVER_PORT` (49483 in `.env.example`) |
| HTTP base, WebSocket base | `https://api.nyolab.ai`, `wss://api.nyolab.ai` |
| Audio | PCM16 mono. ASR input 16 kHz. TTS output: fil 24 kHz, hil 16 kHz |

| Task | `language` | `model` |
|---|---|---|
| ASR Filipino (default) | `fil` | `halohalo/asr-fil` |
| ASR Filipino, larger and slower | `fil` | `fw/whisper-large-v3-pld-fil-norm` |
| ASR Hiligaynon | `hil` | `fw/whisper-large-v3-pld-hil-norm` (not `halohalo/asr-hil`) |
| TTS Filipino, 10 speakers | `fil` | `halohalo/tts-fil` (leave `voice` unset for the default) |
| TTS Hiligaynon, 1 speaker | `hil` | `halohalo/tts-hil` |

`GET /v1/models` with the key is the authority on what is served. If it disagrees with this table, trust it and say so.

## Procedure

Run the steps in order. Stop at the first failed check.

### Step 1. Preconditions

```bash
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
docker info --format '{{.ServerVersion}}'
curl -sS -m 15 https://api.nyolab.ai/healthz
test -n "$NYO_API_KEY" && echo "key is set"
```

Pass: all four succeed, and `/healthz` has `"ok":true` with `asr`, `tts` and `tts_fil` all `"state":"ready"`.
Note `failover`: expected `"off"`. If it is a list, audio may be processed outside Singapore. Stop and ask.

You also need these from the person: `AGORA_APP_ID`, `AGORA_APP_CERTIFICATE` (only if enabled in their Agora Console),
`OPENAI_API_KEY`, and a value for `OPENAI_MODEL`.

### Step 2. Get the code

Skip the clone if you are already in a checkout.

```bash
git clone -b feat/nyo-extensions https://github.com/aineolab/ten-framework.git
cd ten-framework
git merge-base --is-ancestor 2b2a19d169bee46e7d20cf9ae295a4b3586288bc HEAD && echo "commit ok"
git rev-parse HEAD
```

Pass: prints `commit ok`. Put the `git rev-parse HEAD` value in the report. If NYO named a tag or commit, check it out
first. A shallow clone cannot answer this check: clone without `--depth`.

### Step 3. Check the key and the service

Uses one ASR session at a time, briefly. Run it once. From the repository root; the sample files go to `/tmp/nyo-check`:

```bash
REPO=$(git rev-parse --show-toplevel)
mkdir -p /tmp/nyo-check && cd /tmp/nyo-check
python3 -m venv venv
venv/bin/pip install 'websockets>=15.0.1'
venv/bin/python "$REPO/ai_agents/agents/examples/nyo-voice-assistant/nyo_check.py"
```

Pass: exit code 0 and `7 of 7 checks passed`. Exit 1: a check failed. Exit 2: `NYO_API_KEY` is not set.
If only the `hil` lines fail, this is most likely the known Hiligaynon fault. Run it once more at most, then continue
with Filipino and report it. Output files: `sample-fil-http.wav`, `sample-fil.wav`, `sample-hil.wav`.

### Step 4. Fill in `.env`

From `ai_agents/`, create `.env` if there is none (`cp .env.example .env`; do not overwrite an existing one). Set the
existing lines `AGORA_APP_ID`, `AGORA_APP_CERTIFICATE`, `OPENAI_API_KEY`, `OPENAI_MODEL` and `NYO_API_KEY`.
Leave `NYO_BASE_URL=wss://api.nyolab.ai` as it is. Check without printing values:

```bash
for k in AGORA_APP_ID OPENAI_API_KEY OPENAI_MODEL NYO_API_KEY NYO_BASE_URL; do
  grep -qE "^$k=.+" .env && echo "$k set" || echo "$k MISSING"
done
git check-ignore .env
```

Pass: all five are set, and `git check-ignore` prints `.env`.

### Step 5. Build and run

On the host, in `ai_agents/`:

```bash
docker compose up -d
docker exec ten_agent_dev bash -lc 'cd /app/agents/examples/nyo-voice-assistant && task install'
docker exec -d ten_agent_dev bash -lc 'cd /app/agents/examples/nyo-voice-assistant && task run > /tmp/nyo-run.log 2>&1'
```

The first install takes several minutes. Then check that the page is up:

```bash
curl -sS -m 15 -o /dev/null -w '%{http_code}\n' http://localhost:3000
docker exec ten_agent_dev tail -n 50 /tmp/nyo-run.log
```

Pass: HTTP 200. This does not open a speech session; joining a call does. After changing `.env`, recreate the
container with `docker compose up -d --force-recreate`.

### Step 6. Hand over to a person

Ask them to open `http://localhost:3000`, pick `nyo_voice_assistant` (or `nyo_voice_assistant_hil`), join, speak, and
interrupt the agent once while it talks. Ask for yes or no on each: heard the greeting; understood; answered in speech;
stopped when interrupted. Ask them to leave the call when done, which frees the session.

### Step 7. Report

```text
commit:
healthz (release, ok, failover):
nyo_check.py: exit code, N of N, failed lines
agent started: yes/no, first error line if any
human call test: greeting / understood / answered / interruption, or "not verified"
TTLW and TTFB from the logs: number of turns, p50, p95
errors seen: code, count
not run:
```

Stop the agent afterwards: `docker compose down`.

## Use the extensions in another graph

Replace only the `stt` and `tts` nodes and keep the connections. Add both packages to `tenapp/manifest.json` as path
dependencies: `../../../ten_packages/extension/nyo_asr_python` and `../../../ten_packages/extension/nyo_tts_python`.

```json
{"type": "extension", "name": "stt", "addon": "nyo_asr_python", "extension_group": "stt",
 "property": {"params": {"api_key": "${env:NYO_API_KEY|}", "base_url": "${env:NYO_BASE_URL|}",
  "model": "halohalo/asr-fil", "language": "fil", "sample_rate": 16000, "endpointing_ms": 1000}}}
```

```json
{"type": "extension", "name": "tts", "addon": "nyo_tts_python", "extension_group": "tts",
 "property": {"params": {"api_key": "${env:NYO_API_KEY|}", "base_url": "${env:NYO_BASE_URL|}",
  "model": "halohalo/tts-fil", "language": "fil"}}}
```

- Set `model` and `language` together on each node.
- Keep `endpointing_ms` above 0 if the graph never sends `asr_finalize`. At 0 such a graph gets no finals and the LLM
  is never called.
- Have the LLM write numbers, prices, dates and web addresses as words. The voices read them badly otherwise.
- Every setting is in each extension's `README.md` in the fork. It also mentions a private
  repository (`nyo-serve`), which you do not need.

## Errors

| You see | Do |
|---|---|
| HTTP 401 | The key is missing or wrong. Stop and ask for a valid one |
| `model_not_found` (retried for up to 120 s, then fatal) | Use an id from `GET /v1/models` |
| `busy`, 429, 503 `admission_rejected` | Usually both ASR sessions are in use. Follow rule 4 |
| `base_url is required` | Set `NYO_BASE_URL` in `.env` |
| `insecure_url` | Use `wss://`. Follow rule 6 |
| `unknown_voice` (fatal) | Remove `voice`, or use one from `GET /v1/audio/voices` |
| Close code 1011 on `halohalo/asr-hil` (seen 18 Nov) | Use `fw/whisper-large-v3-pld-hil-norm` |
| Interims but no final | Set `endpointing_ms` to 1000 |
| Hiligaynon speech stops early, then silence (seen 18 Nov) | Known fault. Report it; test Filipino first |
| 413 `input_too_long_use_stream` | Non-stream HTTP speech is limited to 450 characters. Use streaming |
| HTTP 502 or 503 on every path | Wait 30 s and check `/healthz` once. If still down, stop and report |

## Limits

ASR: 2 open sessions, all users. TTS speaking at once, all users: 2 (fil), 1 (hil). Per key: 6 open sessions. Total: 16.
Streaming speech: 4,096 characters per request. Idle sessions close after 300 s. No automatic backup, no uptime or
latency commitment. Normal processing is in AWS Singapore.

## Verified and not verified

| Claim | Status |
|---|---|
| Service healthy, wrong key refused with 401 | Checked 22 Nov 2026 |
| Code clones without a login; graphs, models and `.env.example` lines as above | Checked 22 Nov 2026 |
| TEN guarders, live service | 21 Nov 2026: ASR 11 of 11 selected, TTS 60 passed, 11 skipped, 0 failed |
| `nyo_check.py` with a valid key | 18 Nov 2026, 7 of 7, on earlier servers. Not re-run since |
| Steps 5 and 6 (Docker build, Agora call) | **Not run by NYO**, as far as on record |
| Hiligaynon TTS fault, `halohalo/asr-hil` fault | Seen 18 Nov 2026. Not re-tested on the current servers |
