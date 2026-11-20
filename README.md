# vlm-illusion-ladder

Do vision-language models (VLMs) still answer in the direction of a classical visual illusion after they are told
the truth? This repository holds everything behind the paper: 300 self-drawn stimuli of 25 illusions, an
8-step *instruction ladder* of prompts, 460,800 responses of Claude Sonnet 5.5 and Claude Haiku 5.5, and the
analysis.

- Paper: `paper/` (arXiv link to be added)
- Archived release: Zenodo DOI to be added
- Author: Eiji Watanabe, National Institute for Basic Biology, Okazaki, Japan
  ([ORCID 0000-0001-9089-8122](https://orcid.org/0000-0001-9089-8122))

## Design in one paragraph

Each of 25 illusions is drawn in three conditions: **normal** (the illusion), **reversed** (the inducing elements
reversed, so that for two-sided figures the illusion favours the other side) and **none** (the inducing elements
removed; the correct answer is "same" or "no"). Each condition has 4 renders (colours, size, orientation and position
vary), giving 300 PNGs. For 11 illusions, two of the four `none` renders contain a real 10% difference as a check that
the model can resolve the compared property. Every image is asked about at 8 rungs of an instruction ladder and in 3
paraphrases, 16 times each, at temperature 1, with thinking off and on, in a fresh conversation without tools:

| rung | instruction |
|---|---|
| −1 | a false fact pointing away from the illusion ("In fact, the bottom line is slightly longer. Even so, which looks longer?") |
| 0 | neutral ("Which line looks longer?") |
| 1 | warning not to be fooled |
| 2 | report the physical fact ("Which line is actually longer?") |
| 3 | told it is an illusion |
| 4 | told the kind of illusion |
| 5 | told the name of the illusion |
| 6 | told the truth, then asked how it looks ("The two lines are the same length. Even so, which looks longer?") |

Every prompt is listed in `prompts/prompts_all.md` (921 distinct texts).

## Contents

| path | contents |
|---|---|
| `stimuli/` | the 300 PNGs (`NN_illusion_condition_render.png`, 1200×900), `manifest.csv` / `manifest.json` (options, correct answer, illusion answer, drawing parameters), contact sheets |
| `prompts/prompts_all.md` | every prompt text and the images that receive it |
| `data/calls_main.jsonl.gz` | the main run: 460,800 calls (2 models × thinking off/on × 300 images × 8 rungs × 3 paraphrases × 16) |
| `data/calls_trial-1.jsonl.gz` | pilot used to choose the thinking effort (3 illusions, 5 effort settings, 17,280 calls) |
| `data/calls_check-1.jsonl.gz` | check of all 25 illusions at rungs 0 and 6 before the main run (800 calls) |
| `ladder/` | code: `stimuli.py` (draws the stimuli), `prompts.py` (the ladder), `run_vlm.py` (API calls), `analyze.py` (analysis), `figures.py` (stimulus figures) |
| `results/` | output of `analyze.py` (tables, bootstrap intervals, plot data) |
| `paper/` | LaTeX source of the paper (self-contained; `latexmk -pdf main.tex`) |

### One line of `data/calls_*.jsonl.gz`

`run, model, thinking ("off"/"on"), effort, image, rung, para, rep, illusion, condition, correct, illusion_answer,
diff, prompt, request (settings sent), raw (reply text), answer (one option or "invalid"), thinking_text (summary;
the raw thinking is not returned by the API), input_tokens, output_tokens, thinking_tokens, cache_read, cache_write,
stop_reason, seconds, at (UTC)`.

The system prompt of every call was: *You will see a picture and a question about it. Answer with exactly one word
from the options given, and nothing else.*

Model settings (October 2026): `claude-sonnet-5-5` and `claude-haiku-5-5`, temperature 1, `max_tokens` 8,000.
Thinking off = effort `low` with thinking disabled (Haiku: `disabled`; Sonnet: `between_tools`, the lowest setting it
accepts). Thinking on = effort `high` with adaptive thinking (the model decides whether to think). Calls were sent
through the Message Batches API.

## Making new illusion images

`ladder/stimuli.py` draws all 25 illusions, each with its reversed and control version, using only Python and Pillow.
Each illusion has its own drawing function with a few parameters (size, line width, colours, orientation and
illusion-specific values such as the fin length and angle of the Müller-Lyer figure) and a fixed random seed, so new
renders or new stimulus sets can be made by changing these values, e.g. for psychophysics with human observers or for
training and testing other models.

## Reproduce

Python 3.12 or later with `requests` and `Pillow` (no other packages).

```sh
python3 -m ladder.analyze            # results/ from data/calls_main.jsonl.gz (about 1 minute)
python3 -m ladder.prompts --md       # prompts/prompts_all.md
python3 -m ladder.stimuli /tmp/out   # redraw the stimuli
```

Redrawing gives byte-identical files for 297 of the 300 images; in three images (`13_checker_shadow_none_2`,
`_none_3`, `22_rotating_snakes_none_3`) a few dozen pixels around a letter label differ with the Pillow / FreeType
version. The files in `stimuli/` are the ones that were sent to the models.

To query a model again (costs money): put an Anthropic API key in `ANTHROPIC_API_KEY`, then
`python3 -m ladder.run_vlm main claude-sonnet-5-5` (batches; collect with `python3 -m ladder.run_vlm collect`).
Calls are stored in `ladder.db` (SQLite).

## Licence

Code (`ladder/`): MIT (`LICENSE`). Stimuli, prompts, data, results and paper: CC BY 4.0 (`LICENSE-DATA`).
All stimuli were drawn by the code in this repository; no third-party images are included.

## Citation

See `CITATION.cff`.
