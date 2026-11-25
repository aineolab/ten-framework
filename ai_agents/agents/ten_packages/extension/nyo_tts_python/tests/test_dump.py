import os

from .helpers import harness, run, text_input, wait_for


def test_dump_writes_pcm_per_request_id(tmp_path):
    async def go():
        async with harness() as (srv, ext, rec):
            ext.config.dump = True
            ext.config.dump_path = str(tmp_path)
            await ext.request_tts(text_input("req-1", "dump me ", end=True))
            await wait_for(lambda: "finish" in rec.names())
            path = tmp_path / "nyo_dump_req-1.pcm"
            assert path.exists() and os.path.getsize(path) == 7 * 60 * 24000 * 2 // 1000

    run(go())
