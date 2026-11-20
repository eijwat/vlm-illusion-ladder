"""Analysis of the VLM runs for the paper. Standard library only.

    python3 -m ladder.analyze [run]      # default run 'main'; reads data/calls_<run>.jsonl.gz, writes results/

Two kinds of items:
  two-sided  (13 illusions; options "A / B / same", e.g. top / bottom / same). The illusion pushes one side.
             The rung -1 and rung 6 wording ("..., even so, which looks longer?") pushes the model away from "same",
             so the share of illusion answers alone mixes the illusion with that pressure. The main measure is the
             direction among side answers, pooled over the normal and reversed images (the reversal swaps which side
             the illusion favours, so position and label preferences cancel):
                 D = n(illusion side) / (n(illusion side) + n(other side));  0.5 = no direction.
  yes/no     (12 illusions; e.g. "Do the long lines look parallel?"). The illusion pushes one word. The control image
             without the inducing elements ("none") has the same shape and is often recognised as the illusion, so the
             main measure is the stimulus effect
                 delta = p(illusion word | normal + reversed) - p(illusion word | none).

Uncertainty: percentile bootstrap (2,000 resamples) over the stimulus cells (image x paraphrase) for per-illusion
values, and over illusions for the means across illusions.
"""
import csv
import gzip
import json
import random
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE / "ladder.db"
OUT = HERE.parent / "results"
RUNGS = [-1, 0, 1, 2, 3, 4, 5, 6]
SETTINGS = [("claude-haiku-5-5", "off"), ("claude-haiku-5-5", "on"),
            ("claude-sonnet-5-5", "off"), ("claude-sonnet-5-5", "on")]
SHORT = {("claude-haiku-5-5", "off"): "Haiku/off", ("claude-haiku-5-5", "on"): "Haiku/on",
         ("claude-sonnet-5-5", "off"): "Sonnet/off", ("claude-sonnet-5-5", "on"): "Sonnet/on"}
B = 2000


def manifest():
    rows = list(csv.DictReader(open(HERE.parent / "stimuli" / "manifest.csv")))
    order, info = [], {}
    for r in rows:
        if r["illusion"] not in info:
            order.append(r["illusion"])
            info[r["illusion"]] = {"category": r["category"], "n": int(r["n"]),
                                   "kind": "two-sided" if len(r["options"].split(" / ")) == 3 else "yes/no"}
    for r in rows:
        if r["condition"] == "normal":
            info[r["illusion"]]["word"] = r["illusion_answer"]    # yes/no items: one word per illusion
    return order, info, {r["id"]: r for r in rows}


def load(run):
    """Rows of the calls table; from data/calls_<run>.jsonl.gz when present (the published data), else the SQLite DB."""
    path = HERE.parent / "data" / f"calls_{run}.jsonl.gz"
    if path.exists():
        keys = ["model", "thinking", "image", "rung", "para", "illusion", "condition", "correct", "illusion_answer",
                "diff", "answer", "thinking_tokens", "prompt"]
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [tuple(d[k] for k in keys) for d in map(json.loads, f)]
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    q = ("SELECT model, thinking, image, rung, para, illusion, condition, correct, illusion_answer, diff, answer, "
         "thinking_tokens, prompt FROM calls WHERE run = ? AND error IS NULL")
    return con.execute(q, (run,)).fetchall()


def boot(cells, stat, rng):
    """cells: list of per-cell tallies; stat: tallies -> value or None. Returns (value, lo, hi)."""
    v = stat(cells)
    if v is None:
        return None, None, None
    vals = []
    for _ in range(B):
        s = stat([cells[rng.randrange(len(cells))] for _ in cells])
        if s is not None:
            vals.append(s)
    vals.sort()
    return v, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def tally():
    return {"n": 0, "ill": 0, "opp": 0, "correct": 0, "invalid": 0, "word": 0}


def main(run="main"):
    order, info, man = manifest()
    rng = random.Random(1)
    # cells[(setting, illusion, rung, group)][(image, para)] -> tally;  group: 'ind' (normal + reverse), 'none0', 'diff'
    cells = defaultdict(lambda: defaultdict(tally))
    thought = defaultdict(lambda: defaultdict(tally))       # Sonnet/on split by whether it thought
    unparsed = 0
    for (model, th, image, rung, para, ill, cond, correct, ill_ans, diff, ans, ttok, text) in load(run):
        st = (model, th)
        if cond in ("normal", "reverse"):
            g = "ind"
        elif diff:
            g = "diff"
        else:
            g = "none0"
        t = cells[(st, ill, rung, g)][(image, para)]
        t["n"] += 1
        t["invalid"] += ans == "invalid"
        t["correct"] += ans == correct
        if g == "ind":
            t["ill"] += ans == ill_ans
            t["opp"] += ans not in (ill_ans, correct, "invalid")
        if info[ill]["kind"] == "yes/no":
            t["word"] += ans == info[ill]["word"]
        elif g == "none0" and rung == -1:
            # control images at rung -1: the false fact names a random side. "ill" counts the side NOT named (on the
            # induction images the named side is always the one opposite the illusion), "opp" the named side.
            sides = man[image]["options"].split(" / ")[:2]
            first = text.split(".")[0]
            named = [x for x in sides if re.search(rf"\b{re.escape(x)}\b", first)]
            if len(named) != 1:
                unparsed += 1
            else:
                t["ill"] += ans in sides and ans != named[0]
                t["opp"] += ans == named[0]
        if st == ("claude-sonnet-5-5", "on") and g == "ind" and info[ill]["kind"] == "two-sided":
            k = "thought" if (ttok or 0) > 0 else "no-thought"
            u = thought[(k, rung)][(ill, image, para)]
            u["n"] += 1
            u["ill"] += ans == ill_ans
            u["opp"] += ans not in (ill_ans, correct, "invalid")

    def share(key):
        return lambda cs: (sum(c[key] for c in cs) / sum(c["n"] for c in cs)) if sum(c["n"] for c in cs) else None

    def direction(cs):
        a, b = sum(c["ill"] for c in cs), sum(c["opp"] for c in cs)
        return a / (a + b) if a + b else None

    OUT.mkdir(exist_ok=True)
    per = {}                                                # (setting, illusion, rung) -> dict
    with open(OUT / "by_illusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "thinking", "illusion", "category", "kind", "rung", "n_ind", "p_ill", "p_ill_lo", "p_ill_hi",
                    "p_correct_ind", "side_share", "D", "D_lo", "D_hi", "p_word_none", "delta", "delta_lo", "delta_hi",
                    "p_correct_none", "D_control_rung-1", "n_diff", "p_correct_diff", "invalid_share"])
        for st in SETTINGS:
            for ill in order:
                kind = info[ill]["kind"]
                for rung in RUNGS:
                    ind = list(cells[(st, ill, rung, "ind")].values())
                    non = list(cells[(st, ill, rung, "none0")].values())
                    dif = list(cells[(st, ill, rung, "diff")].values())
                    p, plo, phi = boot(ind, share("ill"), rng)
                    n_ind = sum(c["n"] for c in ind)
                    side = (sum(c["ill"] + c["opp"] for c in ind) / n_ind) if n_ind else None
                    D = Dlo = Dhi = pw = de = delo = dehi = None
                    if kind == "two-sided":
                        D, Dlo, Dhi = boot(ind, direction, rng)
                    else:
                        pw = share("word")(non)
                        # paired bootstrap: resample induction and control cells independently
                        def dstat(_, ind=ind, non=non):
                            return share("ill")(ind) - share("word")(non)
                        de = dstat(None)
                        vals = sorted(share("ill")([ind[rng.randrange(len(ind))] for _ in ind])
                                      - share("word")([non[rng.randrange(len(non))] for _ in non]) for _ in range(B))
                        delo, dehi = vals[int(0.025 * B)], vals[int(0.975 * B) - 1]
                    Dc = direction(non) if (kind == "two-sided" and rung == -1) else None
                    row = dict(p=p, D=D, delta=de, side=side, pw=pw, Dc=Dc, D_lo=Dlo, delta_lo=delo,
                               pc_ind=share("correct")(ind), pc_none=share("correct")(non),
                               pc_diff=share("correct")(dif) if dif else None)
                    per[(st, ill, rung)] = row
                    allc = ind + non + dif
                    inv = sum(c["invalid"] for c in allc) / max(1, sum(c["n"] for c in allc))
                    w.writerow([st[0], st[1], ill, info[ill]["category"], kind, rung, n_ind] +
                               [fmt(x) for x in (p, plo, phi, row["pc_ind"], side, D, Dlo, Dhi, pw, de, delo, dehi,
                                                 row["pc_none"], Dc)] +
                               [sum(c["n"] for c in dif), fmt(row["pc_diff"]), fmt(inv)])

    # means across illusions, bootstrap over illusions
    two = [i for i in order if info[i]["kind"] == "two-sided"]
    yn = [i for i in order if info[i]["kind"] == "yes/no"]
    summary = {}
    with open(OUT / "by_rung.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "thinking", "rung", "p_ill_all", "p_ill_lo", "p_ill_hi", "D_two", "D_lo", "D_hi",
                    "delta_yn", "delta_lo", "delta_hi", "p_correct_none_all", "side_share_two"])
        for st in SETTINGS:
            for rung in RUNGS:
                def mean_of(key, ills):
                    xs = [per[(st, i, rung)][key] for i in ills if per[(st, i, rung)][key] is not None]
                    return lambda sample: (sum(per[(st, i, rung)][key] for i in sample
                                               if per[(st, i, rung)][key] is not None) /
                                           max(1, sum(per[(st, i, rung)][key] is not None for i in sample)))
                res = {}
                for name, key, ills in (("p", "p", order), ("D", "D", two), ("delta", "delta", yn), ("Dc", "Dc", two)):
                    res[name] = boot(ills, mean_of(key, ills), rng)
                pcn = sum(per[(st, i, rung)]["pc_none"] for i in order) / len(order)
                side = sum(per[(st, i, rung)]["side"] for i in two) / len(two)
                summary[(st, rung)] = res
                w.writerow([st[0], st[1], rung] + [fmt(x) for x in (*res["p"], *res["D"], *res["delta"], pcn, side)])

    # resolution: the +-10% images (11 illusions), rungs 0 and 2
    with open(OUT / "resolution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "thinking", "illusion", "rung", "n", "p_correct"])
        for st in SETTINGS:
            for ill in order:
                for rung in RUNGS:
                    dif = list(cells[(st, ill, rung, "diff")].values())
                    if dif:
                        w.writerow([st[0], st[1], ill, rung, sum(c["n"] for c in dif), fmt(share("correct")(dif))])

    # Sonnet with thinking allowed: calls where it thought vs not (two-sided induction images)
    with open(OUT / "sonnet_thought_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "rung", "n", "p_ill", "D"])
        for k in ("thought", "no-thought"):
            for rung in RUNGS:
                cs = list(thought[(k, rung)].values())
                n = sum(c["n"] for c in cs)
                w.writerow([k, rung, n, fmt(share("ill")(cs)) if n else "", fmt(direction(cs)) if n else ""])

    assert unparsed == 0, unparsed
    # illusions with a stimulus-driven effect: two-sided D with its 95% interval above 0.5 (and at least 10% side
    # answers), yes/no delta with its interval above 0
    with open(OUT / "robust.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "thinking", "rung", "n_two_sided", "n_yes_no", "n_total", "illusions"])
        for st in SETTINGS:
            for rung in RUNGS:
                hit = []
                for ill in order:
                    r = per[(st, ill, rung)]
                    if info[ill]["kind"] == "two-sided":
                        ok = r["D"] is not None and r["D_lo"] > 0.5 and r["side"] >= 0.1
                    else:
                        ok = r["delta_lo"] > 0
                    if ok:
                        hit.append(ill)
                n2 = sum(info[i]["kind"] == "two-sided" for i in hit)
                w.writerow([st[0], st[1], rung, n2, len(hit) - n2, len(hit), " ".join(hit)])
    write_report(order, info, per, summary, two, yn)
    write_tex(order, info, per, summary)
    print(f"wrote {OUT}/")


def fmt(x):
    return "" if x is None else f"{x:.4f}"


def f2(x):
    return "     -" if x is None else f"{x:6.2f}"


def write_report(order, info, per, summary, two, yn):
    L = []
    L.append("Mean over illusions by rung, with 95% bootstrap intervals over illusions\n")
    for name, title in (("p", "p_ill, all 25 illusions (share of illusion answers, normal + reversed)"),
                        ("D", "D, 13 two-sided illusions (direction among side answers; 0.5 = none)"),
                        ("delta", "delta, 12 yes/no illusions (p_ill normal+reversed - p_ill-word on control)")):
        L.append(title)
        L.append(" " * 12 + "".join(f"{r:>17}" for r in RUNGS))
        for st in SETTINGS:
            L.append(f"{SHORT[st]:<12}" + "".join(
                f"  {v:.2f} [{lo:.2f},{hi:.2f}]" if v is not None else f"{'-':>17}"
                for v, lo, hi in (summary[(st, r)][name] for r in RUNGS)))
        L.append("")
    L.append("Rung -1, two-sided: D on the illusion images vs the same measure on the control images (share of side")
    L.append("answers NOT on the side the false fact names; on the illusion images that is the illusion side)")
    for st in SETTINGS:
        d, c = summary[(st, -1)]["D"], summary[(st, -1)]["Dc"]
        L.append(f"   {SHORT[st]:<12} illusion {d[0]:.2f} [{d[1]:.2f},{d[2]:.2f}]   control {c[0]:.2f} [{c[1]:.2f},{c[2]:.2f}]")
    L.append("")
    for st in SETTINGS:
        L.append(f"{SHORT[st]}: per illusion, rungs 0 and 6  (two-sided: D; yes/no: delta;  [p_ill])")
        for ill in order:
            k = "D" if info[ill]["kind"] == "two-sided" else "delta"
            a, b = per[(st, ill, 0)], per[(st, ill, 6)]
            L.append(f"   {ill:<22} {info[ill]['kind']:<9} {k:<5} {f2(a[k])} -> {f2(b[k])}    "
                     f"[{a['p']:.2f} -> {b['p']:.2f}]   control correct r0 {a['pc_none']:.2f}"
                     + (f"   +-10% correct r0 {a['pc_diff']:.2f} r2 {per[(st, ill, 2)]['pc_diff']:.2f}"
                        if a["pc_diff"] is not None else ""))
        L.append("")
    (OUT / "report.txt").write_text("\n".join(L) + "\n")


def write_tex(order, info, per, summary):
    """pgfplots data files for the paper."""
    for name in ("p", "D", "delta"):
        with open(OUT / f"curve_{name}.dat", "w") as f:
            f.write("rung " + " ".join(f"{SHORT[st].replace('/', '')} {SHORT[st].replace('/', '')}lo "
                                       f"{SHORT[st].replace('/', '')}hi" for st in SETTINGS) + "\n")
            for r in RUNGS:
                f.write(f"{r} " + " ".join(" ".join(f"{x:.4f}" for x in summary[(st, r)][name]) for st in SETTINGS)
                        + "\n")

    # per-illusion table: main measure at rungs 0, 2 and 6 for the four settings, the control and the +-10% check
    L = [r"\begin{tabular}{llc" + "cc" * len(SETTINGS) + "cc}", r"\toprule",
         r" & & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{SHORT[st]}}}" for st in SETTINGS)
         + r" & control & $\pm$10\% \\",
         " ".join(rf"\cmidrule(lr){{{4 + 2 * k}-{5 + 2 * k}}}" for k in range(len(SETTINGS))),
         r"Illusion & cat. & meas. & " + " & ".join("r0 & r6" for _ in SETTINGS) + r" & r0 & r0 \\", r"\midrule"]
    for kind, meas in (("two-sided", "D"), ("yes/no", "delta")):
        for ill in order:
            if info[ill]["kind"] != kind:
                continue
            cells = []
            for st in SETTINGS:
                for r in (0, 6):
                    v = per[(st, ill, r)][meas]
                    few = meas == "D" and (per[(st, ill, r)]["side"] or 0) < 0.1     # D from under 10% side answers
                    cells.append("--" if v is None else f"{v:.2f}".replace("-", "$-$") + ("$^\\dagger$" if few else ""))
            # control: two-sided -> mean share of the correct "same"; yes/no -> mean share of the illusion word
            ctl = [per[(st, ill, 0)]["pc_none"] if kind == "two-sided" else per[(st, ill, 0)]["pw"] for st in SETTINGS]
            res = [per[(st, ill, 0)]["pc_diff"] for st in SETTINGS]
            res = "--" if res[0] is None else f"{sum(res) / len(res):.2f}"
            L.append(f"{NAMES.get(ill, ill)} & {info[ill]['category']} & {'$D$' if meas == 'D' else r'$\Delta$'} & "
                     + " & ".join(cells) + f" & {sum(ctl) / len(ctl):.2f} & {res} \\\\")
        L.append(r"\midrule" if kind == "two-sided" else r"\bottomrule")
    L.append(r"\end{tabular}")
    (OUT / "table_illusions.tex").write_text("\n".join(L) + "\n")


NAMES = {"muller_lyer": "M\\\"uller-Lyer", "ponzo": "Ponzo", "ebbinghaus": "Ebbinghaus", "delboeuf": "Delboeuf",
         "vertical_horizontal": "Vertical--horizontal", "jastrow": "Jastrow", "poggendorff": "Poggendorff",
         "zollner": "Z\\\"ollner", "cafe_wall": "Caf\\'e wall", "hering": "Hering", "fraser_spiral": "Fraser spiral",
         "simultaneous_contrast": "Simultaneous contrast", "checker_shadow": "Checker shadow",
         "white_effect": "White's effect", "hermann_grid": "Hermann grid", "scintillating_grid": "Scintillating grid",
         "munker_white": "Munker--White", "color_contrast": "Colour contrast", "neon_spreading": "Neon spreading",
         "kanizsa": "Kanizsa triangle", "ehrenstein": "Ehrenstein", "rotating_snakes": "Rotating snakes",
         "pinna_brelstaff": "Pinna--Brelstaff", "ouchi": "Ouchi", "shepard_tables": "Shepard tables"}


if __name__ == "__main__":
    main(*sys.argv[1:2])
