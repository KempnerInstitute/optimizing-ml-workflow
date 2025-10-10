"""
live_avg.py

Lightweight client for live cross-run averaging of train and validation losses.

Each worker:
- publishes {"run_id", "step", "loss"} to Redis channels for train and/or validation
- subscribes to average channels and logs {"loss_ave"} and/or {"val_loss_ave"} to its W&B run

Env vars:
- SWEEP_ID         (required)  -> the W&B sweep id (or any shared group id)
- REDIS_HOST       (required)  -> hostname or IP of a Redis server
- REDIS_PORT       (optional)  -> default 6379
- REDIS_PASSWORD   (optional)
- LIVE_AVG_NS      (optional)  -> train namespace; default "loss"
- LIVE_AVG_NS_VAL  (optional)  -> validation namespace; default "val"
"""

import json
import os
import threading

import redis
import wandb


class LiveAverager:
    def __init__(self):
        # Required context
        self.sweep_id = os.environ["SWEEP_ID"]
        self.redis_host = os.environ["REDIS_HOST"]

        # Optional context
        self.redis_port = int(os.environ.get("REDIS_PORT", "6379"))
        self.redis_password = os.environ.get("REDIS_PASSWORD")
        self.ns_train = os.environ.get("LIVE_AVG_NS", "loss")
        self.ns_val = os.environ.get("LIVE_AVG_NS_VAL", "val")

        # Ensure wandb.init() has been called
        if wandb.run is None:
            raise RuntimeError("wandb.init() must be called before creating LiveAverager")
        self.run_id = wandb.run.id
        
        self._r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=True,
        )

        # Channels
        self._pub_chan_train = f"{self.ns_train}:REPORTS:{self.sweep_id}"
        self._sub_chan_train = f"{self.ns_train}:AVGS:{self.sweep_id}"
        self._pub_chan_val = f"{self.ns_val}:REPORTS:{self.sweep_id}"
        self._sub_chan_val = f"{self.ns_val}:AVGS:{self.sweep_id}"

        # Latest known averages by step (train/val)
        self._latest_avg_train = {}
        self._latest_avg_val = {}

        # Background subscriber that logs arriving averages
        self._t = threading.Thread(target=self._subscriber_loop, daemon=True)
        self._t.start()

    def _subscriber_loop(self):
        p = self._r.pubsub()
        p.subscribe(self._sub_chan_train, self._sub_chan_val)
        for m in p.listen():
            if m.get("type") != "message":
                continue
            try:
                d = json.loads(m["data"])
                step = int(d["step"])
                avg = float(d["avg"])
            except Exception:
                continue

            chan = m.get("channel")
            if chan == self._sub_chan_train:
                self._latest_avg_train[step] = avg
                wandb.log({"Train Loss Average": avg}, step=step)
            elif chan == self._sub_chan_val:
                self._latest_avg_val[step] = avg
                wandb.log({"Validation Loss Average": avg}, step=step)

    def report(self, step: int, loss: float):
        """Publish a training loss for this step and log any known average."""
        msg = {"run_id": self.run_id, "step": int(step), "loss": float(loss)}
        self._r.publish(self._pub_chan_train, json.dumps(msg))
        if step in self._latest_avg_train:
            wandb.log({"Train Loss Average": self._latest_avg_train[step]}, step=step)

    def report_val(self, step: int, loss: float):
        """Publish a validation loss for this step and log any known average.
        Note: the training loop already logs base "Validation Loss"; this avoids duplicating it here.
        """
        msg = {"run_id": self.run_id, "step": int(step), "loss": float(loss)}
        self._r.publish(self._pub_chan_val, json.dumps(msg))
        if step in self._latest_avg_val:
            wandb.log({"Validation Loss Average": self._latest_avg_val[step]}, step=step)

    def close(self):
        # Subscriber thread is daemonized; nothing to close
        pass
