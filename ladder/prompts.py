"""Instruction ladder. English; 8 rungs x 3 paraphrases.

Every prompt = [preamble for the rung] + question + answer line. The question always asks how the picture LOOKS,
except rung 2, which asks for the physical fact. Every call uses one of three paraphrases.

Rungs:
  -1  a false fact, pushing away from the illusion (two-sided items: the wrong side; yes/no items: see FALSE)
   0  neutral
   1  warning not to be fooled
   2  instruction to report the physical fact
   3  told it is an illusion
   4  told the kind of illusion
   5  told the name of the illusion
   6  told the truth, then asked how it looks
Only rung -1 states something false. Rung 6 always states the physical truth of that very image: for the +-10%
resolution checks of `none` this names the larger side (the truth there is not "the same").

python3 -m ladder.prompts [id ...]   prints the prompts (default: one image per illusion and condition)
python3 -m ladder.prompts --md       writes prompts/prompts_all.md (every distinct prompt and which images get it)
"""
import json
import random
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "stimuli" / "manifest.json"
RUNGS = [-1, 0, 1, 2, 3, 4, 5, 6]
# system prompt for every VLM call (minimal; no tools)
SYSTEM = "You will see a picture and a question about it. Answer with exactly one word from the options given, and nothing else."
PARAS = 3

# Per illusion:
#   look   question about appearance (rungs -1, 0, 1, 3-6)
#   real   question about the physical fact (rung 2)
#   kind   rung 4 ("an illusion of ...")
#   name   rung 5
# Two-sided items (options side / side / same):
#   noun   what the sides name ("the left line"); same = truth for d = 0; more = comparative for d != 0 and rung -1
#          (more_d replaces it in the rung-6 truth of the +-10% images when the drawn difference is narrower)
# Yes/no and other items:
#   truth  the physical fact (rung 6); FALSE (below) is the false fact for rung -1
ITEMS = {
    "muller_lyer": dict(look="Which line looks longer?", real="Which line is actually longer?",
                        kind="an illusion of length", name="the Müller-Lyer illusion",
                        noun="line", same="the two lines are exactly the same length", more="longer"),
    "ponzo": dict(look="Which of the two short bars looks longer?", real="Which of the two short bars is actually longer?",
                  kind="an illusion of length", name="the Ponzo illusion",
                  noun="short bar", same="the two short bars are exactly the same length", more="longer"),
    "ebbinghaus": dict(look="Which central circle looks larger?", real="Which central circle is actually larger?",
                       kind="an illusion of size", name="the Ebbinghaus illusion",
                       noun="central circle", same="the two central circles are exactly the same size", more="larger"),
    "delboeuf": dict(look="Which filled circle looks larger?", real="Which filled circle is actually larger?",
                     kind="an illusion of size", name="the Delboeuf illusion",
                     noun="filled circle", same="the two filled circles are exactly the same size", more="larger"),
    "vertical_horizontal": dict(look="Which line looks longer, the vertical or the horizontal one?",
                                real="Which line is actually longer, the vertical or the horizontal one?",
                                kind="an illusion of length", name="the vertical-horizontal illusion",
                                noun="line", same="the vertical and the horizontal line are exactly the same length",
                                more="longer"),
    "jastrow": dict(look="Which of the two shapes looks larger?", real="Which of the two shapes is actually larger?",
                    kind="an illusion of size", name="the Jastrow illusion",
                    noun="shape", same="the two shapes are exactly the same size and shape", more="larger"),
    "poggendorff": dict(look="Compared with where the left part of the slanted line would continue, "
                             "does the right part look higher, lower, or aligned?",
                        real="Compared with where the left part of the slanted line would continue, "
                             "is the right part actually higher, lower, or aligned?",
                        kind="an illusion of alignment", name="the Poggendorff illusion",
                        same="the two parts of the slanted line are exactly aligned: they lie on one straight line",
                        higher="the right part of the slanted line is slightly higher than where the left part would continue",
                        lower="the right part of the slanted line is slightly lower than where the left part would continue"),
    "zollner": dict(look="Do the long lines look parallel to each other?", real="Are the long lines actually parallel to each other?",
                    kind="an illusion of orientation", name="the Zöllner illusion",
                    truth="the long lines are exactly parallel to each other"),
    "cafe_wall": dict(look="Do the horizontal grey lines look parallel to each other?",
                      real="Are the horizontal grey lines actually parallel to each other?",
                      kind="an illusion of orientation", name="the café wall illusion",
                      truth="the horizontal grey lines are exactly parallel to each other"),
    "hering": dict(look="Do the two thick lines look straight?", real="Are the two thick lines actually straight?",
                   kind="an illusion of shape", name="the Hering illusion (or Wundt illusion)",
                   truth="the two thick lines are perfectly straight"),
    "fraser_spiral": dict(look="Does the pattern look like a spiral?", real="Is the pattern actually a spiral?",
                          kind="an illusion of shape", name="the Fraser spiral illusion",
                          truth="the pattern is made of separate concentric circles, not a spiral"),
    "simultaneous_contrast": dict(look="Which small grey square looks lighter?",
                                  real="Which small grey square is actually lighter?",
                                  kind="an illusion of lightness", name="the simultaneous contrast illusion",
                                  noun="small grey square", same="the two small grey squares are exactly the same grey",
                                  more="lighter"),
    "checker_shadow": dict(look="Which square looks lighter, A or B?", real="Which square is actually lighter, A or B?",
                           kind="an illusion of lightness", name="the checker shadow illusion (Adelson)",
                           noun="square", same="squares A and B are exactly the same grey", more="lighter"),
    "white_effect": dict(look="Which group of grey bars looks lighter?", real="Which group of grey bars is actually lighter?",
                         kind="an illusion of lightness", name="White's illusion",
                         noun="group of grey bars", same="the two groups of grey bars are exactly the same grey",
                         more="lighter"),
    "hermann_grid": dict(look="Do you see faint spots where the lines cross?",
                         real="Are there actually spots drawn where the lines cross?",
                         kind="an illusion of lightness", name="the Hermann grid illusion",
                         truth="nothing is drawn where the lines cross: the crossings are the same plain colour as the lines"),
    "scintillating_grid": dict(look="Do you see dark dots inside the white discs?",
                               real="Are there actually dark dots drawn inside the white discs?",
                               kind="an illusion of lightness", name="the scintillating grid illusion",
                               truth="the white discs are plain white: no dark dots are drawn inside them"),
    "munker_white": dict(look="Do the orange stripes on the two sides look the same colour?",
                         real="Are the orange stripes on the two sides actually the same colour?",
                         kind="an illusion of colour", name="the Munker-White illusion",
                         truth="the orange stripes on the two sides are exactly the same colour"),
    "color_contrast": dict(look="Which small grey square looks more reddish?",
                           real="Which small grey square is actually more reddish?",
                           kind="an illusion of colour", name="the simultaneous colour contrast illusion",
                           noun="small grey square", same="the two small squares are exactly the same neutral grey",
                           more="more reddish"),
    "neon_spreading": dict(look="Do you see a faint coloured disc filling the centre, between the lines?",
                           real="Is there actually a faint coloured disc drawn in the centre, between the lines?",
                           kind="an illusion of colour", name="the neon colour spreading illusion",
                           truth="the background between the lines is plain: only the line segments are coloured"),
    "kanizsa": dict(look="Do you see a white triangle with clear edges lying on top of the black shapes?",
                    real="Is a white triangle with edges actually drawn on top of the black shapes?",
                    kind="an illusion of contours", name="the Kanizsa triangle",
                    truth="no triangle is drawn: there are only the separate shapes, and no edges join them"),
    "ehrenstein": dict(look="Do you see a disc in the centre where the lines would meet?",
                       real="Is a disc actually drawn in the centre where the lines would meet?",
                       kind="an illusion of contours", name="the Ehrenstein illusion",
                       truth="no disc is drawn: the centre is the same plain background as the rest"),
    "rotating_snakes": dict(look="Does the disc look like it is rotating? If so, which way?",
                            real="Is the disc actually rotating? If so, which way?",
                            kind="an illusion of motion", name="the rotating snakes illusion (Kitaoka)",
                            truth="the picture is completely still: nothing in it rotates",
                            clockwise="the disc is turning clockwise",
                            counterclockwise="the disc is turning counterclockwise"),
    "pinna_brelstaff": dict(look="While looking at the central cross and moving your head towards and away from the picture, "
                                 "do the rings look like they rotate?",
                            real="Do the rings actually rotate?",
                            kind="an illusion of motion", name="the Pinna-Brelstaff illusion",
                            truth="the picture is completely still: the rings do not rotate"),
    "ouchi": dict(look="Does the central disc look like it moves or floats separately from the background?",
                  real="Does the central disc actually move separately from the background?",
                  kind="an illusion of motion", name="the Ouchi illusion",
                  truth="the picture is completely still: the disc is printed flat on the same page as the background"),
    "shepard_tables": dict(look="Which tabletop looks longer and narrower?", real="Which tabletop is actually longer and narrower?",
                           kind="an illusion of shape and size", name="Shepard's tables illusion",
                           noun="tabletop", same="the two tabletops are exactly the same shape and size",
                           more="longer and narrower", more_d="longer"),   # none +-10%: rectangles, only the length differs
}

# rung -1 for yes/no items: a false fact. Where the correct answer is "yes" (parallel, straight, same colour) the only
# false fact points towards the illusion answer; where it is "no" it claims that the illusory thing is really there.
FALSE = {
    "zollner": "the long lines are not parallel: they slightly converge",
    "cafe_wall": "the horizontal grey lines are not parallel: they are slightly tilted",
    "hering": "the two thick lines are slightly curved",
    "fraser_spiral": "the pattern is a spiral",
    "hermann_grid": "faint grey spots are drawn where the lines cross",
    "scintillating_grid": "dark dots are drawn inside the white discs",
    "munker_white": "the orange stripes on the two sides are different colours",
    "neon_spreading": "a faint coloured disc is drawn in the centre, between the lines",
    "kanizsa": "a white triangle is drawn on top of the other shapes",
    "ehrenstein": "a white disc is drawn in the centre",
    "pinna_brelstaff": "the rings really rotate when you move your head",
    "ouchi": "the central disc is a separate layer that moves over the background",
}

PRE = {
    0: ["", "Here is a picture. ", "Look at the image. "],
    1: ["Be careful not to be fooled by appearances. ",
        "Take care: the picture may mislead you. ",
        "Please do not let the picture trick you. "],
    2: ["Do not answer how it looks; answer about the real, physical picture. ",
        "Ignore how the picture appears and report the physical fact. ",
        "Answer about what is really drawn, not about your impression. "],
    3: ["This picture is an optical illusion. ",
        "The image is a visual illusion. ",
        "Note: this is an optical illusion. "],
    4: ["This picture is {kind}. ",
        "The image shows {kind}. ",
        "Note: this is {kind}. "],
    5: ["This picture is {name}. ",
        "The image shows {name}. ",
        "Note: this is {name}. "],
    6: ["In fact, {fact}. Even so, ",
        "{Fact}. Nevertheless, ",
        "Note that {fact}. All the same, "],
}
PRE[-1] = PRE[6]


def _answer_line(options: list[str], para: int) -> str:
    words = ", ".join(options[:-1]) + " or " + options[-1]
    return [f"Answer with one word: {words}.", f"Reply with just one word ({words}).",
            f"Give a one-word answer: {words}."][para]


def _fact(row: dict, rung: int) -> str:
    it = ITEMS[row["illusion"]]
    opts, correct = row["options"], row["correct"]
    if rung == 6:
        if "truth" in it:
            return it["truth"]
        if correct in ("same", "aligned"):
            return it["same"]
        if correct in ("higher", "lower"):
            return it[correct]
        side = correct                                   # +-10% resolution check: name the larger side
    else:
        if "truth" in it and row["illusion"] not in ("rotating_snakes",):
            return FALSE[row["illusion"]]
        sides = [o for o in opts[:-1]]
        if row["illusion_answer"] in sides:
            side = [o for o in sides if o != row["illusion_answer"]][0]
        elif correct in sides:                           # none, +-10%: the side opposite to physics
            side = [o for o in sides if o != correct][0]
        else:                                            # none, equal: a side drawn at random (fixed per image)
            side = random.Random(f"rung-1-{row['id']}").choice(sides)
        if row["illusion"] == "poggendorff":
            return it[side]
        if row["illusion"] == "rotating_snakes":
            return it[side]
    if row["illusion"] == "checker_shadow":
        return f"square {side} is " + ("10% " if rung == 6 else "slightly ") + "lighter than the other"
    noun = it["noun"]
    the = f"the {side} {noun}" if side not in ("vertical", "horizontal") else f"the {side} line"
    amount = "10% " if rung == 6 else "slightly "
    more = it.get("more_d", it["more"]) if rung == 6 else it["more"]
    if more.startswith("more "):
        more = "slightly " + more if rung != 6 else more
        return f"{the} is {more} than the other"
    return f"{the} is {amount}{more} than the other"


def prompt(row: dict, rung: int, para: int, answer_line: bool = True) -> str:
    it = ITEMS[row["illusion"]]
    q = it["real"] if rung == 2 else it["look"]
    pre = PRE[rung][para]
    if rung in (6, -1):
        f = _fact(row, rung)
        pre = pre.format(fact=f, Fact=f[0].upper() + f[1:])
        q = q[0].lower() + q[1:]
    else:
        pre = pre.format(kind=it["kind"], name=it["name"])
    if not answer_line:                                  # the prompt without the answer line
        return f"{pre}{q}"
    return f"{pre}{q} {_answer_line(row['options'], para)}"


def rows() -> list[dict]:
    return json.load(open(MANIFEST))


def write_md(path: Path) -> None:
    """Every distinct prompt sent to the VLMs, per illusion and rung, with the images that receive it."""
    rs = rows()
    by = {}
    for r in rs:
        for g in RUNGS:
            for p in range(PARAS):
                by.setdefault((r["n"], r["illusion"]), {}).setdefault((g, p, prompt(r, g, p)), []).append(r)
    total = len(rs) * len(RUNGS) * PARAS
    L = ["# All VLM prompts", "",
         f"Generated by `python3 -m ladder.prompts --md` from `ladder/prompts.py` and `stimuli/manifest.json`.", "",
         f"- Calls per model and thinking setting: {len(rs)} images x {len(RUNGS)} rungs x {PARAS} paraphrases = "
         f"**{total:,} prompts** (each repeated 16 times). Many images share the same text, so below are the "
         f"**{sum(len(v) for v in by.values()):,} distinct texts**, each with the images that receive it.",
         "- Each call = this system prompt + one image + one prompt. A fresh conversation every time; no tools.",
         "- Images are listed as `condition_render` (file `NN_illusion_condition_render.png`). A +-10% resolution-check "
         "image is marked with its physically larger side, e.g. `none_2 (top 10% larger)`.",
         "", "## System prompt", "", "```", SYSTEM, "```", ""]
    for (n, ill), d in sorted(by.items()):
        opts = sorted({" / ".join(x["options"]) for v in d.values() for x in v})
        L += [f"## {n:02d} {ill}", "", "Options: " + "; ".join(opts), ""]
        for g in RUNGS:
            L += [f"### rung {g}", ""]
            for (gg, p, text), imgs in sorted(d.items(), key=lambda kv: (kv[0][1], kv[0][2])):
                if gg != g:
                    continue
                names = ", ".join(f"{x['condition']}_{x['render']}" + (f" (true answer: {x['correct']})" if x["diff"] else "")
                                  for x in imgs)
                L += [f"- **{g}.{p}** — {names}", f"  > {text}"]
            L.append("")
    path.write_text("\n".join(L))
    print(f"wrote {path}")


def main(ids: list[str]) -> None:
    if ids == ["--md"]:
        write_md(Path(__file__).resolve().parent.parent / "prompts" / "prompts_all.md")
        return
    rs = rows()
    if ids:
        rs = [r for r in rs if r["id"] in ids]
    else:
        seen, keep = set(), []
        for r in rs:
            k = (r["illusion"], r["condition"], r["diff"] != 0)
            if k not in seen:
                seen.add(k); keep.append(r)
        rs = keep
    for r in rs:
        print(f"## {r['id']}  options={r['options']} correct={r['correct']} illusion={r['illusion_answer']} diff={r['diff']}")
        for rung in RUNGS:
            for p in range(PARAS):
                print(f"  {rung:>2}.{p}  {prompt(r, rung, p)}")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
