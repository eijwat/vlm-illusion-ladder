"""Stimulus figures for the paper (Pillow only).

    python3 -m ladder.figures      # writes results/fig_stimuli.png and results/fig_conditions.png
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
STIM = HERE.parent / "stimuli"
OUT = HERE.parent / "results"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TW, TH = 480, 360                      # tile size (stimuli are 1200 x 900)


def tile(name, label, font):
    im = Image.open(STIM / f"{name}.png").convert("RGB").resize((TW, TH), Image.LANCZOS)
    t = Image.new("RGB", (TW, TH + 44), "white")
    t.paste(im, (0, 0))
    ImageDraw.Draw(t).rectangle((0, 0, TW - 1, TH - 1), outline=(170, 170, 170), width=2)
    d = ImageDraw.Draw(t)
    w = d.textlength(label, font=font)
    d.text(((TW - w) / 2, TH + 8), label, fill="black", font=font)
    return t


def grid(tiles, cols, pad=12):
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    g = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), "white")
    for k, t in enumerate(tiles):
        g.paste(t, (pad + (k % cols) * (w + pad), pad + (k // cols) * (h + pad)))
    return g


def main():
    font = ImageFont.truetype(FONT, 26)
    rows = list(csv.DictReader(open(STIM / "manifest.csv")))
    seen = []
    for r in rows:
        if r["condition"] == "normal" and r["render"] == "0":
            seen.append((r["id"], f"{int(r['n'])}. {r['illusion'].replace('_', ' ')} ({r['category']})"))
    OUT.mkdir(exist_ok=True)
    grid([tile(i, lab, font) for i, lab in seen], 5).save(OUT / "fig_stimuli.png", optimize=True)
    picks = []
    for ill in ("01_muller_lyer", "13_checker_shadow", "20_kanizsa"):
        for cond in ("normal", "reverse", "none"):
            picks.append((f"{ill}_{cond}_0", cond))
    grid([tile(i, lab, font) for i, lab in picks], 3).save(OUT / "fig_conditions.png", optimize=True)
    print(f"wrote {OUT}/fig_stimuli.png, fig_conditions.png")


if __name__ == "__main__":
    main()
