# Guitar stem separation

How `neural-tone-mcp` isolates the guitar from a full mix before measuring its tone, why it does it that way, and what
was and was not verified. The code is in `src/ingest/separator.py`; tests are in `tests/test_separation.py`.

## Why separation matters here

Every later stage (feature extraction, the Claude draft, the headless render check, Gemini's listening pass) works from
the guitar stem. Drums, vocals or keys left in the stem skew the measured tone: drum bleed raises transient energy and
makes the guitar look tighter and brighter than it is. A stem that contains no guitar at all makes the tone match
meaningless, so the pipeline also checks for that.

## What the survey says (and what we took from it)

The design follows a survey of guitar stem separation tools (`Guitar Stem Separation Tools.pdf`, not included in this
repository). Points that shaped the code:

| Survey point | Section | What we did |
|---|---|---|
| `htdemucs_6s` is the only mainstream model with an explicit guitar output, but its guitar stem can be noisy | 4.1, 9 | Kept as the guitar model |
| Guitar pick attacks look like drum transients, so models pull them into the drum stem; chord tails leak into piano/vocals | 2.2 | Motivates the cascade and the stem check |
| Highest fidelity comes from a cascade: remove vocals/drums/bass first, then separate the guitar from the cleaned "other" stem | 7.1 | Implemented as the `cascade` strategy (default) |
| Lossless input matters; MP3/Opus degrade separation badly | 3.2 | Warn on lossy input formats; document it |
| Overlap and chunking give smoother, bounded-memory output | 6.3, 7.1 | `NEURAL_TONE_OVERLAP`; Demucs' own overlap-add chunking was already in use |
| BS-RoFormer has the best objective SDR; community fine-tune `htdemucs-6s-guitar-ft` reduces drum bleed | 4.2, 4.1 | **Not implemented** (see Limitations) |

## How it works

```
audio (WAV, any rate)
   │
   ├─ strategy = single ───────────────────────────────┐
   │                                                   ▼
   └─ strategy = cascade                      htdemucs_6s → guitar stem
        htdemucs (4 stems) → "other" ─────────────────▲
        (vocals, drums, bass removed)
                                                       │
                              resample back to the input rate, write <name>_guitar.wav (24-bit)
                                                       │
                                  assess_stem(): warn if silent / far below the mix / mostly gaps
                                                       │
                       extract_part_focus(): rhythm → Side channel, lead → Mid channel
```

Each model gets the signal normalised to unit variance and scaled back afterwards, the same way the Demucs command line
does. Models are cached per name and device, so the cascade loads each only once per process.

### Strategies

| Strategy | Passes | Use when |
|---|---|---|
| `cascade` (default) | 2 | you want the cleanest stem, e.g. a mix with prominent drums |
| `single` | 1 | speed matters, or the section has no drums |

## Configuration

| Setting | Where | Default | Meaning |
|---|---|---|---|
| `NEURAL_TONE_SEPARATION` | env / `.env` | `cascade` | `cascade` or `single` |
| `strategy=` | `separate_guitar(...)` | env, then `cascade` | per-call override |
| `separation=` | `extract_tone`, `extract_tone_hybrid` | env, then `cascade` | per-call override from the pipeline |
| `stage1_model=` | `separate_guitar(...)` | `htdemucs` | the 4-stem model used for the first stage (e.g. `htdemucs_ft`, about 4x slower) |
| `NEURAL_TONE_SHIFTS` / `shifts=` | env / argument | `0` | test-time augmentation; every shift is another full pass of each model |
| `NEURAL_TONE_OVERLAP` / `overlap=` | env / argument | `0.25` | chunk overlap; higher is smoother and slower |

The MCP tools use the environment settings. Restart the MCP server after changing `.env`.

```python
from src.ingest.separator import separate_guitar, assess_stem

stem = separate_guitar("song.wav", out_dir="stems", strategy="cascade")
report = assess_stem("song.wav", stem)       # {"stem_to_mix_db": ..., "active_fraction": ..., "warnings": [...]}
```

## The stem check

There is no ground-truth guitar track for a real recording, so SDR cannot be computed. `assess_stem` only catches the
obvious failures and logs them as `stem check: ...` warnings:

- the stem is almost silent (below -60 dBFS), or more than 25 dB below the mix: there is probably no guitar in this section;
- fewer than 25 % of 50 ms frames are above -50 dBFS: the stem is mostly gaps, so features may be unreliable.

## Measured results

One 25-second slice of a live concert recording (a guitar solo over a full band), CPU only. This is a single clip and a
proxy metric, not a benchmark.

| | `single` | `cascade` |
|---|---|---|
| Separation time | 30.7 s | 63.2 s |
| Correlation of the guitar stem's onset pattern with the drum stem's | 0.172 | 0.091 |
| Attack rise time (drives gate tightness in the tone mapper) | 21.1 ms | 21.1 ms |
| Crest factor | 14.7 dB | 14.3 dB |
| Spectral centroid | 2312 Hz | 2243 Hz |
| Stem level relative to the mix | -2.8 dB | -2.6 dB |

Reading: the cascade roughly halves how much of the drum pattern appears in the guitar stem and costs about twice the
time. The risk that stage 1 would also strip real pick attacks (it can assign them to the drums) did not show up in the
features the tone mapper uses. The drum-correlation number measures onset-envelope similarity against the 4-stem
model's own drum output; it is not an SDR score.

To reproduce on your own audio, run `separate_guitar` with each strategy and compare the stems by ear and with
`extract_features`; if the cascade sounds thinner or loses attack on your material, set `NEURAL_TONE_SEPARATION=single`.

## Limitations

- **No BS-RoFormer first stage.** The survey names it the best model for removing vocals, drums and bass. The widely
  distributed checkpoints split vocals from a single "instrumental" stem, so drums would remain in what reaches the
  guitar model, which defeats the point of the cascade. It would also add heavy dependencies (they could change the
  pinned numpy/torch in the working environment) and large model downloads. Not attempted.
- **No `htdemucs-6s-guitar-ft` fine-tune.** It targets drum and cymbal bleed, but its checkpoint format and loading were
  not verified, so it is not wired in.
- **Source quality is the ceiling.** Audio downloaded from YouTube is lossy. A lossless recording of the same section
  separates better than any strategy change here.
- **Bleed is reduced, not removed.** Synthetic-data training (Slakh2100) means the guitar stem can still carry
  artifacts on dense real-world mixes. Treat the stem as an input to a rule-of-thumb tone match, not a clean recording.
- **Tested with fake models offline.** The unit tests check the routing (which model gets which signal), settings
  and the stem check without Demucs weights. The real-model behaviour was checked once on the clip above.

## Tests

```bash
python -m pytest tests/test_separation.py
```

Covers strategy selection and validation, single vs cascade routing (stage 2 receives stage 1's "other" stem at the
model's sample rate), custom stage-1 model, shifts/overlap from the environment, models without a guitar stem, mono input,
the lossy-format warning, and `assess_stem` on healthy, quiet, silent and sparse stems.
