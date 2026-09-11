# Stop cancellation on Python 3.11

One Stop request could leave the audio consumer running when a queue read
completed at the same moment. The Python 3.11 `asyncio.wait_for(queue.get())`
implementation can return the completed child result instead of propagating
that cancellation. The buffer then remains unfinalized until another Stop.

The production wait now uses `asyncio.timeout(0.1)` around `queue.get()` in the
consumer task. The idle timeout and stop-buffer policy are unchanged. This is
a shutdown repair, not a measured caption-latency improvement.

The failure first appeared in [CI on source `980c8ea`](https://github.com/wrbell/stark-translate/actions/runs/34516257463):
Python 3.11 failed one Stop test, with 2,833 other tests and 20 subtests passing.
The Python 3.12 test step passed, but its job was cancelled by the matrix failure;
that job is not a passing CI certificate. The [original failure log](stop-cancellation/ci-980c8ea-failed.log)
is retained.

A new deterministic case schedules cancellation exactly as the 25th queue read
returns. It requires one final containing all 12,800 buffered samples and
successful required persistence, alongside the existing Stop, Pause and
disconnect checks. Its mocked VAD runs inline to control scheduling; no native
audio or models are used.

The [focused validation](stop-cancellation/validation.json) passed 88 tests and
20 subtests. A test-only plugin restored the old wait expression in the
in-memory production function, leaving source files untouched: the new case
then failed with zero final calls. The [old-behavior failure](stop-cancellation/legacy-production-regression.log),
[stdlib reproduction](stop-cancellation/reproduction.json) and exact source hashes
are preserved in the [evidence index](stop-cancellation/evidence-index.json).
Final integrated checks and installed rehearsals follow under their own source
identity. Further live microphone and output playback remain prohibited this session.
