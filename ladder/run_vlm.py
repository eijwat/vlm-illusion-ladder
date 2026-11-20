"""VLM runs. Plain Messages API calls, no tools, one fresh conversation each.

    python3 -m ladder.run_vlm trial            # 3 illusions x all rungs, effort sweep (resumable)
    python3 -m ladder.run_vlm report [run]     # thinking length, invalid rate, p_ill by rung, cost
    python3 -m ladder.run_vlm main <model> ... [--dry]   # the main run as batches (half price)
    python3 -m ladder.run_vlm collect [run]    # fetch finished batches into ladder.db (run again until all are in)

Every call is stored in ladder.db (table calls) with the full prompt, raw reply, normalised answer, the
thinking summary and token counts. A call already stored (same run, model, setting, image, rung, paraphrase, rep)
is not sent again, so a run can be stopped and restarted. The API key is read from the environment
(ANTHROPIC_API_KEY) or, if that is not set, from ~/.config/vlm-illusion-ladder/anthropic_key.
"""
import base64
import json
import os
import re
import sqlite3
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .prompts import PARAS, RUNGS, SYSTEM, prompt, rows

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "ladder.db"
STIM = HERE.parent / "stimuli"
KEY = Path.home() / ".config" / "vlm-illusion-ladder" / "anthropic_key"
URL = "https://api.anthropic.com/v1/messages"

# USD per million tokens: input, output, cache write (5 min), cache read (pricing page, 2026-10-13)
PRICE = {"claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.10), "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
         "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20)}
# "off" is the lowest thinking setting each model accepts
OFF = {"claude-sonnet-5-5": {"type": "between_tools"}, "claude-haiku-5-5": {"type": "disabled"}}
ON = {"type": "adaptive", "display": "summarized"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    run          TEXT NOT NULL,       -- 'trial-1', 'main', ...
    model        TEXT NOT NULL,
    thinking     TEXT NOT NULL,       -- 'off' | 'on'
    effort       TEXT NOT NULL,
    image        TEXT NOT NULL,
    rung         INTEGER NOT NULL,
    para         INTEGER NOT NULL,
    rep          INTEGER NOT NULL,
    illusion     TEXT, condition TEXT, correct TEXT, illusion_answer TEXT, diff REAL,
    system       TEXT, prompt TEXT, request TEXT,          -- request = JSON of the settings sent (not the image)
    raw          TEXT,                                     -- reply text
    answer       TEXT,                                     -- one option, or 'invalid'
    thinking_text TEXT,                                    -- summary only; the raw thinking is never returned
    input_tokens INTEGER, output_tokens INTEGER, thinking_tokens INTEGER,
    cache_read   INTEGER, cache_write INTEGER,
    stop_reason  TEXT, request_id TEXT, error TEXT,
    seconds      REAL,
    at           TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (run, model, thinking, effort, image, rung, para, rep)
);
"""


def normalise(text: str, options: list[str]) -> str:
    words = re.findall(r"[A-Za-z]+", text or "")
    low = {o.lower(): o for o in options}
    hits = [low[w.lower()] for w in words if w.lower() in low]
    if len(words) >= 1 and words[0].lower() in low and len(set(hits)) == 1:
        return hits[0]
    if len(set(hits)) == 1 and len(words) <= 3:
        return hits[0]
    # reasoning written in the reply (it happens when thinking is allowed but not used): accept a last line that is
    # one option and nothing else. The raw reply is kept, and `verbose` in report() counts these.
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    if len(lines) > 1:
        last = re.findall(r"[A-Za-z]+", lines[-1])
        if len(last) == 1 and last[0].lower() in low:
            return low[last[0].lower()]
    return "invalid"


def renormalise(con, run):
    """Recompute answers from the stored raw replies (after a change to normalise())."""
    opts = {r["id"]: r["options"] for r in rows()}
    ups = [(normalise(raw, opts[img]), run, m, th, ef, img, g, p, rep) for m, th, ef, img, g, p, rep, raw in con.execute(
        "SELECT model, thinking, effort, image, rung, para, rep, raw FROM calls WHERE run = ? AND error IS NULL", (run,))]
    con.executemany("""UPDATE calls SET answer = ? WHERE run = ? AND model = ? AND thinking = ? AND effort = ? AND image = ?
                       AND rung = ? AND para = ? AND rep = ?""", ups)
    con.commit()


def build(job, image_b64, cache_ttl=None):
    """(prompt text, settings without the image, full request body) for one call."""
    model, thinking, effort, row, rung, para, rep, max_tokens = job
    text = prompt(row, rung, para)
    settings = {"model": model, "max_tokens": max_tokens, "temperature": 1.0,
                "thinking": ON if thinking == "on" else OFF[model], "output_config": {"effort": effort}}
    cache = {"type": "ephemeral"} if cache_ttl is None else {"type": "ephemeral", "ttl": cache_ttl}
    body = dict(settings, system=SYSTEM, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
         "cache_control": cache},
        {"type": "text", "text": text}]}])
    if cache_ttl:
        settings = dict(settings, cache_ttl=cache_ttl)
    return text, settings, body


def parse(j, options):
    u = j.get("usage", {})
    reply = " ".join(b.get("text", "") for b in j["content"] if b["type"] == "text").strip()
    think = "\n".join(b.get("thinking", "") for b in j["content"] if b["type"] == "thinking").strip()
    return {"error": None, "raw": reply, "answer": normalise(reply, options), "thinking_text": think,
            "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
            "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens"),
            "cache_read": u.get("cache_read_input_tokens"), "cache_write": u.get("cache_creation_input_tokens"),
            "stop_reason": j.get("stop_reason")}


def store(con, run, job, text, settings, rec):
    model, thinking, effort, row, rung, para, rep, max_tokens = job
    con.execute("""INSERT OR REPLACE INTO calls (run, model, thinking, effort, image, rung, para, rep, illusion,
        condition, correct, illusion_answer, diff, system, prompt, request, raw, answer, thinking_text, input_tokens,
        output_tokens, thinking_tokens, cache_read, cache_write, stop_reason, request_id, error, seconds)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run, model, thinking, effort, row["id"], rung, para, rep, row["illusion"], row["condition"],
         row["correct"], row["illusion_answer"], row["diff"], SYSTEM, text, json.dumps(settings),
         rec.get("raw"), rec.get("answer"), rec.get("thinking_text"), rec.get("input_tokens"),
         rec.get("output_tokens"), rec.get("thinking_tokens"), rec.get("cache_read"), rec.get("cache_write"),
         rec.get("stop_reason"), rec.get("request_id"), rec.get("error"), rec.get("seconds")))


class Runner:
    def __init__(self, run: str, workers: int = 12):
        self.run, self.workers = run, workers
        self.key = KEY.read_text().strip()
        self.lock = threading.Lock()
        self.con = sqlite3.connect(DB, timeout=300, check_same_thread=False)
        self.con.executescript(SCHEMA)
        self.img = {}
        self.done = {tuple(r) for r in self.con.execute(
            "SELECT model, thinking, effort, image, rung, para, rep FROM calls WHERE run = ? AND error IS NULL", (run,))}
        self.n = self.fail = 0
        self.t0 = time.time()

    def image(self, iid):
        if iid not in self.img:
            self.img[iid] = base64.b64encode((STIM / f"{iid}.png").read_bytes()).decode()
        return self.img[iid]

    def one(self, job):
        model, thinking, effort, row, rung, para, rep, max_tokens = job
        text, settings, body = build(job, self.image(row["id"]))
        rec = {"error": None}
        t = time.time()
        for attempt in range(8):
            try:
                r = requests.post(URL, json=body, timeout=900, headers={
                    "x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
            except requests.RequestException as e:
                rec["error"] = f"network: {e}"[:300]; time.sleep(min(60, 2 ** attempt)); continue
            if r.status_code in (429, 500, 502, 503, 504, 529):
                rec["error"] = f"HTTP {r.status_code}"
                time.sleep(float(r.headers.get("retry-after") or min(60, 2 ** attempt))); continue
            if not r.ok:
                rec["error"] = f"HTTP {r.status_code}: {r.text[:300]}"; break
            rec = parse(r.json(), row["options"])
            rec["request_id"] = r.headers.get("request-id")
            break
        rec["seconds"] = round(time.time() - t, 2)
        with self.lock:
            store(self.con, self.run, job, text, settings, rec)
            self.n += 1; self.fail += rec["error"] is not None
            if self.n % 200 == 0:
                self.con.commit()
                print(f"{self.n} calls, {self.fail} failed, {time.time() - self.t0:.0f} s", flush=True)

    def go(self, jobs):
        jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]["id"], j[4], j[5], j[6]) not in self.done]
        print(f"run {self.run}: {len(jobs)} calls to send", flush=True)
        # same image, model and setting next to each other so the cached image is reused
        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(self.one, jobs))
        self.con.commit()
        print(f"done: {self.n} calls, {self.fail} failed, {time.time() - self.t0:.0f} s", flush=True)


# ---------------------------------------------------------------- batches (the main run: half price, plus the cache)

BATCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id   TEXT PRIMARY KEY,
    run        TEXT NOT NULL,
    model      TEXT NOT NULL,
    n          INTEGER NOT NULL,
    status     TEXT NOT NULL,          -- in_progress | ended | collected
    created    TEXT NOT NULL DEFAULT (datetime('now')),
    collected  TEXT
);
CREATE TABLE IF NOT EXISTS batch_items (
    custom_id  TEXT NOT NULL,
    batch_id   TEXT NOT NULL,
    job        TEXT NOT NULL,          -- JSON: model, thinking, effort, image, rung, para, rep, max_tokens
    PRIMARY KEY (batch_id, custom_id)
);
"""
BATCH_URL = "https://api.anthropic.com/v1/messages/batches"
BATCH_BYTES = 200 * 2 ** 20            # under the 256 MB limit
BATCH_MAX = 90_000                     # under the 100,000 requests limit


def _headers():
    return {"x-api-key": os.environ.get("ANTHROPIC_API_KEY") or KEY.read_text().strip(), "anthropic-version": "2023-06-01", "content-type": "application/json"}


def submit(run, jobs, dry=False):
    """Send jobs as batches. Calls of one image (and one model) stay together and in order so the cached image
    (1-hour cache, as the batch docs advise) is reused. Jobs already answered or waiting in a batch are skipped."""
    con = sqlite3.connect(DB, timeout=300); con.executescript(SCHEMA); con.executescript(BATCH_SCHEMA)
    done = {tuple(r) for r in con.execute(
        "SELECT model, thinking, effort, image, rung, para, rep FROM calls WHERE run = ? AND error IS NULL", (run,))}
    waiting = set()
    for (j,) in con.execute("""SELECT i.job FROM batch_items i JOIN batches b ON b.batch_id = i.batch_id
                               WHERE b.run = ? AND b.status != 'collected'""", (run,)):
        j = json.loads(j); waiting.add((j[0], j[1], j[2], j[3], j[4], j[5], j[6]))
    skip = done | waiting
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]["id"], j[4], j[5], j[6]) not in skip]
    jobs.sort(key=lambda j: (j[0], j[3]["id"], j[1], j[4], j[5], j[6]))
    print(f"run {run}: {len(jobs)} calls to send as batches", flush=True)
    images, sent, chunk, size = {}, [], [], 0

    def flush():
        nonlocal chunk, size
        if not chunk:
            return
        model = chunk[0][1][0]
        reqs = [{"custom_id": cid, "params": body} for cid, job, text, settings, body in chunk]
        if dry:
            print(f"  (dry) batch of {len(reqs)} for {model}, {size / 2 ** 20:.0f} MB")
        else:
            for attempt in range(200):          # a full queue (429) can take hours to drain: keep waiting
                try:
                    r = requests.post(BATCH_URL, headers=_headers(), json={"requests": reqs}, timeout=1800)
                except requests.RequestException as e:
                    print(f"  batch create failed: {e}"[:300], flush=True); time.sleep(60); continue
                if r.ok:
                    break
                print(f"  batch create failed: HTTP {r.status_code} {r.text[:200]}", flush=True)
                if r.status_code not in (408, 429, 500, 502, 503, 504, 529):
                    break
                time.sleep(300 if r.status_code == 429 else 60)
            r.raise_for_status()
            bid = r.json()["id"]
            con.execute("INSERT INTO batches (batch_id, run, model, n, status) VALUES (?, ?, ?, ?, 'in_progress')",
                        (bid, run, model, len(reqs)))
            con.executemany("INSERT INTO batch_items VALUES (?, ?, ?)",
                            [(cid, bid, json.dumps([job[0], job[1], job[2], job[3]["id"], job[4], job[5], job[6], job[7]]))
                             for cid, job, text, settings, body in chunk])
            con.commit()
            print(f"  batch {bid}: {len(reqs)} calls for {model}, {size / 2 ** 20:.0f} MB", flush=True)
        sent.append(len(reqs)); chunk, size = [], 0

    for k, job in enumerate(jobs):
        iid = job[3]["id"]
        if iid not in images:
            images = {iid: base64.b64encode((STIM / f"{iid}.png").read_bytes()).decode()}   # keep one in memory
        text, settings, body = build(job, images[iid], cache_ttl="1h")
        b = len(images[iid]) + 2000
        if chunk and (chunk[0][1][0] != job[0] or size + b > BATCH_BYTES or len(chunk) >= BATCH_MAX):
            flush()
        chunk.append((f"c{k:07d}", job, text, settings, body)); size += b
    flush()
    print(f"submitted {sum(sent)} calls in {len(sent)} batches", flush=True)


def collect(run=None):
    """Fetch the results of ended batches into calls. Errored or expired requests are stored with the error,
    so a later submit() of the same jobs sends them again."""
    con = sqlite3.connect(DB, timeout=300); con.executescript(SCHEMA); con.executescript(BATCH_SCHEMA)
    rs = {r["id"]: r for r in rows()}
    q = "SELECT batch_id, run FROM batches WHERE status != 'collected'" + (" AND run = ?" if run else "")
    for bid, brun in con.execute(q, (run,) if run else ()).fetchall():
        b = requests.get(f"{BATCH_URL}/{bid}", headers=_headers(), timeout=60).json()
        st, counts = b.get("processing_status"), b.get("request_counts")
        print(f"batch {bid} ({brun}): {st} {counts}", flush=True)
        if st != "ended":
            continue
        jobs = {cid: json.loads(j) for cid, j in con.execute("SELECT custom_id, job FROM batch_items WHERE batch_id = ?", (bid,))}
        n = 0
        with requests.get(b["results_url"], headers=_headers(), stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                x = json.loads(line)
                m, th, ef, iid, rung, para, rep, mt = jobs[x["custom_id"]]
                job = (m, th, ef, rs[iid], rung, para, rep, mt)
                text, settings, _ = build(job, "", cache_ttl="1h")
                res = x["result"]
                if res["type"] == "succeeded":
                    rec = parse(res["message"], rs[iid]["options"]); rec["request_id"] = res["message"].get("id")
                else:
                    rec = {"error": f"batch {res['type']}: {json.dumps(res.get('error'))[:250]}"}
                store(con, brun, job, text, settings, rec); n += 1
        con.execute("UPDATE batches SET status = 'collected', collected = datetime('now') WHERE batch_id = ?", (bid,))
        con.commit()
        print(f"  stored {n} results", flush=True)


# decided after trial-1 (2026-10-13, user): thinking off at effort low (least reasoning written into the reply),
# thinking on at effort high (the only level at which Sonnet 5.5 thinks at all); max_tokens 8,000 (trial max 2,222)
MAIN_EFFORT = {"off": "low", "on": "high"}
MAIN_MAX_TOKENS = 8000


def main_jobs(models, reps=16, illusions=None):
    """The full design: every image x rung x paraphrase x thinking off/on x reps."""
    return [(m, th, MAIN_EFFORT[th], r, g, p, rep, MAIN_MAX_TOKENS) for m in models for r in rows()
            if illusions is None or r["illusion"] in illusions
            for th in ("off", "on") for g in RUNGS for p in range(PARAS) for rep in range(reps)]


TRIAL_ILLUSIONS = ("muller_lyer", "checker_shadow", "kanizsa")
TRIAL_SETTINGS = [("off", "low"), ("off", "high"), ("on", "low"), ("on", "medium"), ("on", "high")]


def trial():
    rs = [r for r in rows() if r["illusion"] in TRIAL_ILLUSIONS]
    jobs = [(m, th, ef, r, g, p, rep, 32000)
            for m in ("claude-sonnet-5-5", "claude-haiku-5-5") for th, ef in TRIAL_SETTINGS for r in rs
            for g in RUNGS for p in range(PARAS) for rep in range(2)]
    Runner("trial-1", workers=16).go(jobs)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else 0


def report(run="trial-1"):
    con = sqlite3.connect(DB, timeout=300); con.row_factory = sqlite3.Row
    renormalise(con, run)
    cs = con.execute("SELECT * FROM calls WHERE run = ?", (run,)).fetchall()
    print(f"run {run}: {len(cs)} calls, errors {sum(c['error'] is not None for c in cs)}")
    groups = {}
    for c in cs:
        if c["error"] is None:
            groups.setdefault((c["model"], c["thinking"], c["effort"]), []).append(c)
    print("\nmodel / setting: calls, thought (%), thinking tokens median / p90 / p99 / max, cut off, invalid, "
          "reasoning in the reply, cost per call (USD)")
    for (m, th, ef), g in sorted(groups.items()):
        tt = [c["thinking_tokens"] or 0 for c in g]
        pin, pout, pw, pr = PRICE[m]
        def one_cost(c):
            batch = "cache_ttl" in json.loads(c["request"] or "{}")      # batches use the 1-hour cache, at half price
            w = 2 * pin if batch else pw                                # 1-hour cache writes cost 2x input
            usd = ((c["input_tokens"] or 0) * pin + (c["output_tokens"] or 0) * pout + (c["cache_write"] or 0) * w
                   + (c["cache_read"] or 0) * pr) / 1e6
            return usd / 2 if batch else usd
        cost = sum(one_cost(c) for c in g)
        print(f"  {m} {th}/{ef}: {len(g)}, {100 * sum(t > 0 for t in tt) / len(g):.0f}%, "
              f"{statistics.median(tt):.0f} / {q(tt, .9)} / {q(tt, .99)} / {max(tt)}, "
              f"{sum(c['stop_reason'] == 'max_tokens' for c in g)}, {sum(c['answer'] == 'invalid' for c in g)}, "
              f"{sum(len((c['raw'] or '').split()) > 3 for c in g)}, "
              f"{cost / len(g):.5f}")
    print("\np_ill by rung (normal + reverse; share of answers in the illusion's direction)")
    print("  " + " " * 34 + "".join(f"{g:>6}" for g in RUNGS))
    for (m, th, ef), g in sorted(groups.items()):
        for ill in sorted({c["illusion"] for c in cs}, key=lambda x: [r["illusion"] for r in rows()].index(x)):
            line = []
            for rung in RUNGS:
                cc = [c for c in g if c["illusion"] == ill and c["rung"] == rung and c["illusion_answer"]]
                line.append(f"{sum(c['answer'] == c['illusion_answer'] for c in cc) / len(cc):6.2f}" if cc else "     -")
            print(f"  {m[7:]:>10} {th:>3}/{ef:<6} {ill:<22}" + "".join(line))
    print("\n`none`, equal images: share answering 'same' / 'no' (the correct answer), by rung")
    for (m, th, ef), g in sorted(groups.items()):
        line = []
        for rung in RUNGS:
            cc = [c for c in g if c["condition"] == "none" and not c["diff"] and c["rung"] == rung]
            line.append(f"{sum(c['answer'] == c['correct'] for c in cc) / len(cc):6.2f}" if cc else "     -")
        print(f"  {m[7:]:>10} {th:>3}/{ef:<6} {'':<14}" + "".join(line))
    print("\n+-10% images (none renders 2, 3): share with the physically correct answer, rungs 0 / 2")
    for (m, th, ef), g in sorted(groups.items()):
        out = []
        for rung in (0, 2):
            cc = [c for c in g if c["diff"] and c["rung"] == rung]
            out.append(f"{sum(c['answer'] == c['correct'] for c in cc) / len(cc):.2f}" if cc else "-")
        print(f"  {m[7:]:>10} {th:>3}/{ef:<6} " + " / ".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "trial":
        trial()
    elif cmd == "report":
        report(*sys.argv[2:3])
    elif cmd == "main":                   # main <model> [model ...] [--dry]
        submit("main", main_jobs([a for a in sys.argv[2:] if a != "--dry"]), dry="--dry" in sys.argv)
    elif cmd == "collect":
        collect(*sys.argv[2:3])
    else:
        print(__doc__)
