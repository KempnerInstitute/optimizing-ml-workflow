# aggregator.py
"""
Aggregator: computes the cross-run average loss per `step`.

It listens on channel:   {NS}:REPORTS:{SWEEP_ID}
It publishes averages:   {NS}:AVGS:{SWEEP_ID}

State per step is kept in a Redis hash:
  key = {NS}:STATE:{SWEEP_ID}:{STEP}
  fields: sum, count, deadline_ts

Env vars:
  SWEEP_ID        (required)
  REDIS_HOST      (required)
  REDIS_PORT      (optional, default 6379)
  REDIS_PASSWORD  (optional)
  EXPECTED_RUNS   (optional, default 12)   -> typical array size
  STEP_TIMEOUT    (optional, default 10.0) -> seconds before emitting partial average
  QUORUM          (optional, default 0)    -> emit early when >= QUORUM reports arrive (0 disables)
  LIVE_AVG_NS     (optional, default "loss")  -> set to "val" to aggregate validation
"""

import json
import os
import time
import threading

import redis


def main():
    sweep_id = os.environ["SWEEP_ID"]
    host = os.environ["REDIS_HOST"]
    port = int(os.environ.get("REDIS_PORT", "6379"))
    password = os.environ.get("REDIS_PASSWORD")
    expected = int(os.environ.get("EXPECTED_RUNS", "12"))
    timeout_s = float(os.environ.get("STEP_TIMEOUT", "10.0"))
    quorum = int(os.environ.get("QUORUM", "0"))
    ns = os.environ.get("LIVE_AVG_NS", "loss")

    r = redis.Redis(host=host, port=port, password=password, decode_responses=True)
    reports = f"{ns}:REPORTS:{sweep_id}"
    avgs = f"{ns}:AVGS:{sweep_id}"

    def emit(step: int, avg: float, n: int, complete: bool):
        msg = {"step": step, "avg": avg, "n": n, "complete": complete}
        print(f"[{ns}] Emitting average for step {step}: avg={avg:.4f}, n={n}, complete={complete}", flush=True)
        r.publish(avgs, json.dumps(msg))

    def maybe_finish_step(step: int):
        key = f"{ns}:STATE:{sweep_id}:{step}"
        data = r.hgetall(key)
        if not data:
            return
        s = float(data.get("sum", "0"))
        c = int(data.get("count", "0"))
        dl = float(data.get("deadline_ts", "0"))
        now = time.time()

        quorum_ok = (quorum and c >= quorum)
        timeout_ok = (now >= dl)
        full_ok = (c >= expected)

        if c == 0 or not (quorum_ok or timeout_ok or full_ok):
            return

        avg = s / c
        complete = c >= expected
        emit(step, avg, c, complete)
        if complete:
            r.delete(key)

    def reports_listener():
        pubsub = r.pubsub()
        pubsub.subscribe(reports)
        print(f"[{ns}] Subscribed to channel: {reports}", flush=True)
        for m in pubsub.listen():
            if m["type"] != "message":
                continue
            try:
                d = json.loads(m["data"])
                step = int(d["step"])
                loss = float(d["loss"])
                print(f"[{ns}] Received report for step {step}: loss={loss:.4f}", flush=True)
            except Exception as e:
                print(f"[{ns}] Error parsing message: {e}", flush=True)
                continue

            key = f"{ns}:STATE:{sweep_id}:{step}"
            pipe = r.pipeline()
            # first reporter sets the deadline
            pipe.hincrbyfloat(key, "sum", loss)
            pipe.hincrby(key, "count", 1)
            pipe.hsetnx(key, "deadline_ts", time.time() + timeout_s)
            pipe.execute()

            maybe_finish_step(step)

    def sweeper():
        # periodically check for timeouts across all active steps
        pattern = f"{ns}:STATE:{sweep_id}:*"
        while True:
            for key in r.scan_iter(pattern):
                try:
                    step = int(key.rsplit(":", 1)[1])
                except Exception:
                    continue
                maybe_finish_step(step)
            time.sleep(1.0)

    threading.Thread(target=sweeper, daemon=True).start()
    reports_listener()


if __name__ == "__main__":
    main()
