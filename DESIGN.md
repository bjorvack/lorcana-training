# lorcana-training — Design

> Trains the card encoder, proposal net, and per-step evaluator from a
> pinned `cards-vN` + `tournaments-vN`, gates the result through quality
> thresholds, exports to ONNX, and publishes two GitHub Release streams:
> **`encoder-vN`** (card encoder weights + `card_embeddings.bin`,
> manually triggered when a new card pool lands) and **`model-vN`**
> (proposal + evaluator ONNX, data tables, manifest). `lorcana-web` is
> the sole consumer of `model-vN`.

## Purpose

This repo turns scraped tournament decks into ONNX models that help
`lorcana-web` propose decks. The product goal is **not** to reproduce
tournament decks; it is to produce decks that are:

1. **Plausible** — sensible mana curve, type mix, ink balance.
2. **Functional** — legal, exactly the chosen inks, max-copies respected,
   ≥ 60 cards.
3. **Synergistic** — the chosen cards actually interact (shift chains,
   song/singer pairs, references like "your items", keyword combos).
4. **Novel** — not a verbatim known list. Should explore the design space
   and use cards that the current meta under-uses.

Tournament data is therefore treated as a **quality oracle**, not a
target. A pure maximum-likelihood model on tournament decks collapses
onto a small set of meta archetypes and assigns near-zero probability to
any card that hasn't been played — exactly the behaviour the product
needs to avoid.

The architecture splits the job into three components, each doing
something it's actually suited for:

1. **Card Encoder.** Maps any card (played or not) to an embedding using
   its *intrinsic* features and its card text. The text path is the key
   to (3) and (4): a card with a brand-new keyword still produces a
   meaningful embedding because the text encoder reads the keyword in
   context, and other cards that mention the same keyword in *their*
   text become reachable through cross-card attention.
2. **Proposal Net.** Masked set-completion model over the partial deck,
   trained with entropy regularisation so it does not collapse onto the
   meta. Its output is a *soft proposal distribution*, not the final
   answer.
3. **Per-step Evaluator.** Given a partial deck and a candidate card,
   scores "how plausible is *this card* as the next addition?". Trained
   on real next-cards vs. progressively harder negatives.

At inference (`lorcana-web`) a constrained search combines the proposal
distribution, the evaluator score, a learned-embedding novelty bonus, and
hard rules (legality, ink, max-copies) into a final card choice. The
blend weights are exposed to the user as a *style* control (Safe ↔ Brew),
so the same trained models can produce a competitive meta deck or an
exploratory brew without retraining.

The current project trains an autoregressive LSTM that treats decks as
*sequences*. Decks are unordered, so the LSTM sees the same deck as up to
60! distinct sequences — label noise, mode collapse, and a stack of
sampling hacks downstream to compensate. Set-based masked completion
removes the root cause; reframing the model as a proposal-plus-evaluator
under search removes the "just memorise the meta" failure mode on top.

## Non-goals

- Reinforcement learning from validator reward. The current project's
  `train_rl.js` was a workaround for an autoregressive model that
  collapsed onto the meta; the new architecture (entropy-regularised
  proposal + per-step evaluator + Style-controlled search) addresses
  that directly. Revisit only if v1 evaluation shows it's needed.
- Real-time / online training in the browser.
- Hosting a model server. We ship ONNX + small data tables; the web
  app runs everything locally.
- Hand-coded synergy rules anywhere training can see them. Synergy
  understanding has to come from card-text encoding + co-occurrence;
  the only synergy code in this repo is the diagnostic regexes used
  at *eval* time to measure whether the model learned what we wanted.
- A fine-tuned LLM over card text. The card-text encoder is a small
  ~5M-param Transformer trained from scratch on a few thousand
  cards. It is enough to recognise keywords and their relationships,
  not enough to reason in English.

---

## Stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Best-in-class ML tooling. The whole training pipeline is throwaway compute; cross-language sharing happens via the schemas + the ONNX artifact, not source. |
| Deps | [`uv`](https://github.com/astral-sh/uv) + `pyproject.toml` | Fast, deterministic, lockfile-first. Replaces pip / poetry / conda. |
| ML | PyTorch 2.x | Mature, ONNX export works well, easy custom modules. |
| Tracking | Plain JSON summaries (`run.json`, `samples/`, `eval-report.md`) | No tracking framework. Each subcommand writes its own JSON; CI commits the final summary subset. Easy to inspect by hand, trivial to wrap in Streamlit if a UI is ever wanted. |
| Schema codegen | `datamodel-code-generator` | Generates `pydantic` v2 models from the JSON Schemas published by `lorcana-schemas`. Training code never declares its own dataclasses. |
| Test | `pytest` | Pure-Python pipeline parts; model code unit-tested with tiny tensors. |
| Lint/format | `ruff` + `ruff format` | One tool, no config drift. |
| Type check | `mypy --strict` | Same discipline as the TS side. |
| Tensor IO | `safetensors` for intermediate state, `onnx` for final export | `safetensors` for checkpoints, ONNX for the public artifact. |
| GPU | CUDA when available, CPU fallback | Training the v1 model is small enough to be feasible on CPU; GPU just speeds it up. CI runs are CPU-only. |

Deliberately not chosen:
- TensorFlow / Keras / JAX. PyTorch is what the team already reads, and
  its ONNX export story is the smoothest.
- Hugging Face Transformers. Overkill; we want a ~1 MB ONNX file, not a
  fine-tuned BERT. We hand-write a tiny Transformer encoder (~6 lines).
- Lightning / `accelerate`. Adds abstraction we don't need at this scale.
  We may revisit if the project grows.

---

## Inputs

Three pinned GitHub Releases. The first two come from `lorcana-scraper`;
the third is produced by this repo's own `pretrain-encoder` workflow
(see Q6) and consumed by the proposal/evaluator training stages.

- **`cards-vN`** (a `CardSet` from `lorcana-scraper`) — defines the
  vocabulary. Card index 0 is reserved for `<PAD>`; indices `1..|cards|`
  map 1:1 to `Card.id`. The index→id mapping is canonicalised by
  sorting on `Card.id` so it is reproducible across runs.
- **`tournaments-vN`** (a `Dataset` from `lorcana-scraper`) — the
  labels. Every deck is validated against `Deck` and
  `isTournamentLegal` (from `@bjorvack/lorcana-schemas`'s
  Python-generated counterpart) before it's allowed into training.
  Anything that fails is logged and discarded; if more than 5% of decks
  fail validation the run aborts.
- **`encoder-vN`** (from this repo's `pretrain.yml`) — the
  precomputed card embeddings + tokeniser + encoder weights produced
  by the pretrain stage. The proposal and evaluator stages consume
  `card_embeddings.bin` directly; the encoder weights are kept around
  so we can re-encode without retraining from scratch when a future
  `cards-vN` is a strict superset.

Pinning is by tag name, stored in `config/training.yaml`:

```yaml
cards_release_tag:        cards-v2025.05.01-01
tournaments_release_tag:  tournaments-v1.42.0
encoder_release_tag:      encoder-v2025.06.01-01
```

Bumping any tag is a reviewed PR. The `ModelManifest` records all three
exact tags (`cardsReleaseTag`, `datasetReleaseTag`, `encoderReleaseTag`)
so a published model can always be retraced to the data and encoder it
was trained on.

The same `compute_max_copies` regexes that exist in `lorcana-schemas`
have a Python mirror in `src/lorcana_training/cards/max_copies.py`. Both
implementations run against a shared fixture file (`fixtures/max-copies-cards.json`)
in CI; if they disagree, both repos fail. This is the only place card
logic exists outside JS, and the test makes drift impossible.

---

## Modelling approach

### 1. Card Encoder — the foundation

Every downstream model uses the same card encoder. It is what lets the
system reason about cards it has never seen played, and what lets new
mechanics generalise without retraining hand-coded rules.

```
   per-card structured features            card text
   (cost one-hot, type multi-hot,         (full text including reminder)
    ink multi-hot, lore/strength/willpower         │
    normalised, classification multi-hot)          ▼
              │                       ┌────────────────────────┐
              │                       │ BPE tokeniser          │
              │                       │ (32k vocab,            │
              │                       │  shared across cards)  │
              │                       └──────────┬─────────────┘
              │                                  ▼
              │                       ┌────────────────────────┐
              │                       │ 4-layer Transformer    │
              │                       │ encoder over tokens    │
              │                       │ (d=128, heads=4)       │
              │                       └──────────┬─────────────┘
              │                                  ▼
              │                          mean+max pool over tokens
              │                                  │
              ▼                                  ▼
       MLP (struct → R^64)              MLP (text → R^192)
                  └──────────────┬──────────────┘
                                 ▼
                     concat → MLP → R^256  (card embedding)
```

Why this shape:

- **Text is encoded, not looked up.** A new keyword like "Bounce 3" has
  never appeared in the BPE vocab if it's brand new, but BPE breaks it
  into existing subwords (`Bounce`, `Boun`+`ce`, …) and the Transformer
  reads it in context. The first card that introduces "Bounce" still
  produces a usable embedding; the second card that references "Bounce"
  in its text contributes information toward what the keyword *does*.
- **No hand-coded synergy.** There is deliberately no
  `is_shift_target_of(card_a, card_b)` feature, no song/singer pairing,
  no classification list. The model learns these from card text via the
  deck encoder's cross-attention (next section). If a future set
  introduces "Resonate X" the model can pick up that synergy from data
  without anyone writing a regex.
- **Pretrained, then fine-tuned.** Before any deck data is involved, the
  card encoder is pretrained on a **card-text masked-LM objective**
  (mask 15% of text tokens, predict from context) plus an **autoencoder
  reconstruction** of structured features. This gives every card a
  meaningful embedding even with zero deck examples. Fine-tuning happens
  during the proposal-net training, end-to-end.
- **Pre-encoded at inference.** The exported ONNX bundle includes a
  precomputed `card_embeddings.bin` (one R^256 row per card in the
  pinned `cards-vN`). The Proposal Net and Evaluator just look up by
  card id at runtime; the heavy text Transformer is not part of the
  in-browser execution path. New `cards-vN` releases trigger a
  re-export, gated by the manifest check.

### 2. Proposal Net — masked set completion

The proposal model's job is to give a *soft distribution* over plausible
next cards for a partial deck. Importantly, it is **not** trained to be
right; it is trained to be *informative* (entropy-regularised).

Input: an unordered multiset of card ids representing the partial deck,
plus a 6-dim multi-hot ink vector.

```
                   ink_2hot                  partial deck (multiset)
                       │                              │
                       ▼                              ▼
              ink_embed → R^32           lookup card_embeddings → N × R^256
                       │                              │
                       └────────── broadcast-add ─────┤
                                                      ▼
                                  ┌──────────────────────────────┐
                                  │ 6-layer Transformer encoder  │
                                  │ d=256, heads=8, no positional│
                                  │ encoding                     │
                                  └──────────────┬───────────────┘
                                                 ▼
                                       mean+max pool → R^512
                                                 │
                                                 ▼
                          Linear(512 → |vocab|) + softmax
```

Training objective:

`L = CE(softmax(logits), target) − β · H(softmax(logits))`

The entropy bonus `β · H(·)` rewards the model for keeping mass on more
than one plausible card. Without it the model converges on the meta. We
start with `β = 0.05` and search.

Training data:

- Take every tournament deck. **Do not** apply the per-model 12-month
  recency filter this section originally prescribed. A sweep on
  `tournaments-v0.3.0` (see `scripts/sweep_proposal.py`) showed the
  recency-filtered split (782 decks) produces severe overfitting —
  held-out total loss climbs after epoch 2 while training continues
  down. Widening to the full validated set (~2 780 decks) drops
  held-out total from 4.03 to 2.98 (−26 %) and lets the model
  actually reach epoch 7 before plateauing. The held-out set is
  stratified by `(ink_pair, month)` so month-distribution drift is
  already controlled for. `prepare` still emits a recency-filtered
  `train.proposal.jsonl` artifact for experimentation, but the
  defaults in `ProposalOptions` and the `train-proposal` CLI point
  at `train.evaluator.jsonl`.
- For each deck, generate `k_pos = 12` masked examples by removing one
  card at a time uniformly at random.
- The label is a *distribution*, not a one-hot: if a card appears in 3
  copies, the target distribution puts mass on that card id with weight
  proportional to remaining copies. (Standard label-smoothing variant
  for multisets.) The same sweep confirmed a per-mask one-hot-on-the-
  removed-card alternative is ~10 % *worse* on held-out, so the
  full-deck distribution stays the default.

Outputs are *not* used directly at inference; they are blended with the
Evaluator and a novelty bonus inside `lorcana-web`'s search loop.

### 3. Per-step Evaluator — quality oracle

A pointwise discriminator: `V(partial_deck, candidate_card) → [0, 1]`.
Reads as "how plausible is adding this card next, given this partial
deck?".

```
   partial deck (N × R^256)         candidate card (R^256)
            │                                  │
            ▼                                  ▼
     2-layer Transformer encoder      MLP(R^256 → R^256)
     d=256, heads=4                            │
            │                                  │
            ▼                                  │
        mean+max pool → R^512                  │
            │                                  │
            └──────── concat ──────────────────┤
                                               ▼
                                    MLP(R^768 → R^256 → R^1)
                                               │
                                               ▼
                                            sigmoid
```

Pointwise design (one card scored at a time) means the evaluator can be
applied to *every* legal candidate in the search loop at inference, not
just to a full deck after the fact. That is what enables per-step
correction inside the search.

#### Curriculum training (negatives)

Naive negatives (random in-ink cards) make a trivial evaluator that only
learns "is this card the right ink?". We schedule negatives across
training to force progressively harder discrimination:

| Phase | Negative source | Why |
|---|---|---|
| **1. Warmup** (epochs 1–N₁) | Random in-ink card | Anchors "matches inks / matches cost band". |
| **2. Curve-matching** (N₁–N₂) | Cards drawn from a matched mana-curve distribution within the chosen inks | Forces the model past curve and ink, into card-identity signal. |
| **3. Local swap** (N₂–end) | Real partial decks where the answer card is replaced by another card occupying the analogous slot in a *different* real deck (same inks) | Forces it to learn deck-internal synergies — "this kind of partial deck wants *this* card, not the meta-baseline card." |

`(N₁, N₂)` are config knobs. Mix ratios within each phase get swept in
early experiments and locked.

A fourth "adversarial" phase (negatives mined from a prior model's
proposals) was considered and **deferred** — see open question 3.
Adding it later is purely additive to this phase list.

Loss: BCE on the positive (real next card from a held-out partial)
versus the scheduled negatives. Calibrated post-hoc (isotonic
regression on held-out predictions) so the score that gets blended in
the search is a real probability and the "Realism: 82%" the UI shows is
meaningful.

### Putting it together at inference

The web app's search loop picks each next card as the argmax (or a
nucleus sample) of:

```
score(card) = log P_propose(card | partial, inks)
            + α · V_eval(partial, card)
            + γ · novelty(card, partial)
            − λ · meta_closeness(card | partial)
```

`(α, γ, λ)` are exposed in the web UI as a single **Style** control
running from Safe (high α, low γ, high λ — stick to known-good
combinations) to Brew (low α, high γ, low λ — explore the design
space). Hard constraints (legality, ink, max-copies) are a mask applied
before scoring.

`novelty(card, partial)` is computed in JS from the card embedding's
distance to the centroid of the partial deck (cards "different enough"
from what's already in the deck get a bonus). `meta_closeness(card |
partial)` is the empirical play frequency of `card` conditional on the
partial deck's inks, computed once from the training set and exported
as a per-card scalar table. Both are deterministic at inference; no
extra model runs are needed.

This means the training pipeline must export, beside the two ONNX
models:

- `card_embeddings.bin` — R^256 per card.
- `play_frequency.json` — empirical use frequency per `(card, ink_pair)`.
- `archetype_centroids.json` — small set of known-meta deck centroids
  (e.g. k-means with `k = 20`) used both at inference for the novelty
  term and at training for the archetype-distance quality gate.

### Architecture sizes

| Component | Params | ONNX (int8) | Browser path |
|---|---|---|---|
| Card text encoder | ~5 M | not exported | training-time only, pre-encodes card embeddings |
| Card embedding table | ~|cards| × 256 floats | ~1–2 MB raw | shipped as `card_embeddings.bin` |
| Proposal Net | ~3 M | ~3 MB | runs in worker |
| Evaluator | ~2 M | ~2 MB | runs in worker |
| Total ONNX payload | | < 10 MB | |

### What this fixes vs. today

| Problem today | Solution here |
|---|---|
| Decks treated as sequences → label noise → mode collapse | Set encoder, identical decks → identical examples |
| Top-p + temperature + singleton boost stack of hacks | Single principled search with a learned per-step evaluator |
| Generator just memorises common decks; new ideas are impossible | Entropy-regularised proposal + novelty bonus + Style control |
| Synergies hand-coded in `WeightCalculator` and brittle | Synergies learned from card text via the text encoder; new keywords work without code changes |
| Cards with zero tournament plays get zero probability forever | Card encoder generalises by intrinsic features + text; under-played cards near similar over-played cards remain reachable |
| Validator scores full decks once at the end → no per-step signal | Per-step pointwise evaluator usable inside the search loop |
| Easy negatives (random in-ink) → weak validator | Three-phase curriculum: random → curve-matched → local-swap (phase 4 adversarial is deferred, see Q3) |
| Vocab/embedding drift between train and inference | Single canonical `cards-vN`, `vocabHash` + `cardSetVersion` checked at load and at build time |
| LSTM ops slow in the browser | Small Transformer, ONNX Runtime Web (WASM/WebGPU) in a worker |
| Three training scripts with divergent flags | One CLI, seven subcommands, one config file |
| REINFORCE fine-tune compensating for a misaligned base | Not needed; entropy reg + evaluator + search handle the alignment |

---

## Pipeline

A single CLI (`lorcana-train`) with subcommands. All state passes
through `config/training.yaml`; no flags-only invocations in CI.

```
lorcana-train prepare           # download cards-vN + tournaments-vN, build vocab + features
lorcana-train pretrain-encoder  # card-text MLM + structured AE; emits card_embeddings.bin
lorcana-train train-proposal    # train proposal net (fine-tunes encoder end-to-end)
lorcana-train train-evaluator   # train per-step evaluator with curriculum negatives
lorcana-train build-tables      # play_frequency.json, archetype_centroids.json
lorcana-train eval              # full quality-gate gauntlet on held-out set
lorcana-train export            # PyTorch → ONNX, write manifest + payload
lorcana-train release           # tag + push GitHub Release
```

Default invocation in CI is `lorcana-train all`, which runs:

```
prepare → pretrain-encoder → train-proposal → train-evaluator
        → build-tables → eval → export
```

`release` is gated on PR merge.

### prepare

1. Resolve the three pinned release tags (`cards`, `tournaments`,
   `encoder`) from `config/training.yaml`, download the artifacts,
   verify their hashes.
2. Build the canonical vocabulary: sort `CardSet.cards` by `Card.id`,
   assign `1..N` (0 = PAD).
3. Compute per-card structured features (cost one-hot, type one-hot,
   ink one-hot, keyword multi-hot, lore/strength/willpower normalised).
   Store as `prepared/card_features.safetensors`. Note: this is the
   *structured* half of the features; the text-derived
   `card_embeddings.bin` comes from the pinned `encoder-vN`.
4. Parse the dataset:
   - Validate every deck against `Deck` + `isTournamentLegal`.
   - Drop invalid ones, log counts.
   - Apply **per-model recency cutoffs** from `config/training.yaml`
     (Q2 decision: proposal = trailing 12 months, evaluator = all
     in-vocab decks).
   - Materialise **two** train splits:
     - `prepared/train.proposal.parquet` — decks within the proposal's
       recency window, all cards in current vocab.
     - `prepared/train.evaluator.parquet` — all decks whose cards are
       in current vocab, no time cutoff.
     plus a shared `prepared/heldout.parquet` (90/10 split off the
     evaluator's wider set, stratified by ink pair and
     month-of-tournament so the held-out set is representative).
5. Hash the prepared artifacts and write `prepared/manifest.json`. If
   the same `(cards_tag, tournaments_tag, encoder_tag, config)`
   produces the same hash as a cached run, skip re-computation.

### pretrain-encoder

Two-headed self-supervised pretraining of the card encoder, before any
deck data is involved. Runs **only on `workflow_dispatch`** (Q6), not
on every release.

- **Text head:** BPE-tokenise every card's text. Mask 15% of tokens.
  Predict masked tokens from context (standard MLM loss).
- **Structured head:** mask a random subset of the structured features
  (cost, type, ink, etc.) and predict them back from the unmasked
  parts (denoising autoencoder loss).
- Combined loss `L_pre = L_mlm + L_struct`. AdamW, cosine schedule.
- Stopping: 100k steps or held-out reconstruction plateau, whichever
  is first.
- Emits an **`encoder-vN` release** containing:
  - `card_embeddings.fp32.safetensors` — R^256 per card, primary
    output. The model-vN release bundles a fp16 copy of this as
    `card_embeddings.bin`.
  - `tokeniser.json` — the BPE vocab and merges, versioned so future
    re-encodes of an unchanged `cards-vN` are reproducible.
  - `encoder_weights.safetensors` — full text + structured encoder
    weights, kept around so we can re-encode without retraining
    from scratch if a future `cards-vN` is a strict superset.
  - `encoder-manifest.json` — references `cardsReleaseTag`,
    BPE-vocab size, training hyperparameters, run hash.

This stage is where "the model understands new keywords" actually
happens; the proposal and evaluator stages benefit from it but don't
themselves produce that capability. They pin a specific `encoder-vN`
via `config/training.yaml:encoder_release_tag`.

### train-proposal

Fine-tunes the encoder end-to-end while training the Proposal Net:

- `DataLoader` yields batches of `(partial_deck_ids, ink_2hot, target_distribution)`.
- AdamW, cosine LR schedule with a smaller LR on the encoder than on
  the new layers (10×). Mixed precision when GPU is present.
- Loss: `CE(logits, target) − β · H(softmax(logits))` with β from config.
- Every epoch:
  - Held-out NLL.
  - **Sample 32 decks** under the full inference search recipe (with
    placeholder evaluator from the previous run if available, or
    uniform if not) from a fixed list of ink pairs. Log as JSON.
  - **Diversity metric**: mean pairwise Jaccard distance over the 32
    samples. Reported separately for "low novelty" and "high novelty"
    style settings.
  - **Coverage metric**: number of distinct card ids appearing in
    the 32 samples. Drops if the model is collapsing.
- Best checkpoint by `held-out NLL + λ · (1 - diversity)`; pure NLL
  selection picks meta-collapsed models.

### train-evaluator

Trains the per-step evaluator using the curriculum from the modelling
section.

- Input: held-out partial decks paired with the real next card (positive)
  and curriculum-scheduled negatives.
- AdamW, cosine schedule. Encoder weights are loaded from the
  proposal stage and **frozen** for the evaluator — we don't want the
  two models drifting against each other.
- Per-epoch metrics: AUROC against each negative type separately
  (so we can see "good at rejecting random in-ink, bad at rejecting
  local swaps"), ECE on held-out positives + a balanced negative mix.
- Calibration: isotonic regression on held-out predictions; calibration
  parameters exported alongside the ONNX model as a small JSON.


### build-tables

Builds the deterministic-at-inference data the search uses:

- `play_frequency.json` — empirical play frequency per
  `(card_id, ink_pair)`. Computed from the training split only (not
  held-out) so eval metrics aren't contaminated.
- `archetype_centroids.json` — k-means on deck representations (mean of
  card embeddings per deck), `k = 20`, only over the training split.
  Used at inference for the novelty term and at eval time for the
  archetype-distance gate.

These are tiny (~hundreds of KB total) and ship inside the model
release.

### eval

The quality-gate subcommand. Sampling for eval runs is done under the
full inference recipe (proposal + evaluator + novelty + constraint
mask) at three Style settings: **Safe** (α high, γ low, λ high),
**Balanced** (defaults), **Brew** (α low, γ high, λ low). 200 decks
per ink pair per setting. Metrics are reported per Style.

In addition, an **intermediate Style coherence** diagnostic samples
32 decks at slider positions 0.25 (halfway between Safe and Balanced)
and 0.75 (halfway between Balanced and Brew) using linear `(α, γ, λ)`
interpolation. The diagnostic reports novelty metrics at those points
and sets `style_presets.interpolatable = true` in the manifest iff
both intermediate points sit between their bracketing presets on the
novelty axis. The web app's Advanced Style slider keys off this flag.

Gates fall into three categories: **hard** (must pass, run fails),
**soft** (should pass, release is marked pre-release), and **target
bands** (must land *inside* a range; outside fails — novelty too low
*or* too high is bad).

#### Functionality (hard)

| Metric | What | Threshold |
|---|---|---|
| Legality rate | Samples conform to deck rules: 60+ cards, exactly the chosen inks, copies ≤ `computeMaxCopies` | **= 100%** |
| Manifest validity | `manifest.json` round-trips against `ModelManifest`; `vocabHash` and `cardSetVersion` are present and correct | **must pass** |

#### Plausibility (soft)

| Metric | What | Threshold |
|---|---|---|
| Mana curve KL | KL-divergence of sample curves vs. tournament curve (Balanced setting only) | ≤ 0.10 |
| Type-mix χ² | Type-share χ² vs. tournament distribution (Balanced) | p > 0.05 |
| Inkable share | Fraction of inkable cards (Balanced) | within ±5 pp of tournament mean |

#### Synergy (soft)

The product needs the model to actually pair cards that work together,
without us hand-coding the rules. Measured via auto-derived
*relationship signals* from card text — not used by the model, used by
us to check whether it learned them:

| Metric | What | Threshold |
|---|---|---|
| Shift compliance | When a shifted character is included, ≥ 1 valid shift target is also in the deck | ≥ 95% of cases |
| Song/singer coverage | Total Singer "supply" ≥ total Song "demand", measured by cost-level matching | ≥ 90% of decks |
| Classification reference | When a card text references a classification (e.g. "your Hero characters"), ≥ N cards of that classification are in the deck | ≥ 80% of cases |

These are **diagnostic regexes used only at eval time** — they never
influence training. If a future set introduces a new synergy mechanic
we'll add a new diagnostic regex without changing the model.

#### Novelty (target bands)

The dual-bound design is intentional: too low and we're recommending
the meta verbatim; too high and we're recommending random nonsense.

| Metric | What | Safe band | Balanced band | Brew band |
|---|---|---|---|---|
| Archetype distance | Mean min-distance from sample to nearest training-set archetype centroid | [0.05, 0.20] | [0.15, 0.40] | [0.30, 0.60] |
| Coverage @ Brew | Distinct card ids across all Brew samples / total legal vocab | – | – | ≥ 0.35 |
| Verbatim rate | Fraction of samples whose card multiset exactly matches a training deck | ≤ 5% | ≤ 1% | ≤ 0.1% |
| Underplayed inclusion | Mean fraction of cards per deck below 10th percentile of training play frequency | ≥ 0.05 | ≥ 0.15 | ≥ 0.30 |

`Coverage @ Brew` is the single metric we'd point at to answer the
question "is this finding new ideas?". A model that scores well on
everything else but ships < 0.35 of the vocab even in Brew is
collapsed and should not be released.

#### Evaluator quality (mixed)

| Metric | What | Threshold |
|---|---|---|
| Per-phase AUROC | AUROC against each curriculum phase separately | warmup ≥ 0.95, curve ≥ 0.85, swap ≥ 0.75 (soft) |
| Overall AUROC | Balanced negative mix | **≥ 0.80** (hard) |
| ECE (calibrated) | Held-out | ≤ 0.05 (soft) |

A failing **hard** threshold fails the run. A failing **soft**
threshold or out-of-band **target** marks the release as pre-release;
it gets published but `lorcana-web`'s `model_release_tag: latest`
pointer does not move. Every threshold lives in
`config/quality_gates.yaml` and is versioned with the code.

### export

1. Load the best proposal and evaluator checkpoints.
2. `torch.onnx.export` each, opset 17, dynamic axes on `batch_size` and
   `deck_size`. The card encoder is **not** exported — its output is
   baked into `card_embeddings.bin`.
3. Quantise to int8 with `onnxruntime.quantization.quantize_dynamic`
   (matmul-only, embedding layers kept fp16). Save also a non-quantised
   fp16 variant for fallback.
4. Smoke test: load each ONNX file in `onnxruntime` (Python), feed a
   tiny batch, compare against the PyTorch outputs at fp32 tolerance ≤
   1e-3. Fail on mismatch.
5. **End-to-end search test:** run the full inference recipe (proposal
   + evaluator + novelty + constraint mask, all in Python this time)
   against a handful of seed partial decks, assert the resulting
   complete decks satisfy legality and `computeMaxCopies`. Fails if
   the exported artifacts can't reproduce the in-process behaviour.
6. Write `manifest.json` matching the `ModelManifest` schema:
   `vocabHash`, `cardSetVersion`, eval metrics per Style setting, and
   the default `(α, γ, λ)` Style presets so `lorcana-web` knows what
   "Safe" / "Balanced" / "Brew" mean for this release.

### release

- Tags `model-vX.Y.Z`.
- GitHub Release contains:
  - `proposal.int8.onnx`, `proposal.fp16.onnx`
  - `evaluator.int8.onnx`, `evaluator.fp16.onnx`
  - `card_embeddings.bin` — R^256 per card (fp16; ~1 MB for a few
    thousand cards)
  - `vocab.json` — the card-index → `Card.id` mapping
  - `play_frequency.json` — `{ "<card_id>": { "<ink_pair>": float } }`
  - `archetype_centroids.json` — 20 centroids in R^256, used for the
    novelty term
  - `evaluator_calibration.json` — isotonic regression points
  - `manifest.json`
  - `eval-report.md` — human-readable per-Style summary
- Release notes are auto-generated from the eval report.

---

## File layout

```
lorcana-training/
├── DESIGN.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version              # 3.12
├── config/
│   ├── training.yaml            # cards_release_tag, tournaments_release_tag, hyperparams
│   └── quality_gates.yaml       # eval thresholds
├── schemas/                     # checked-in JSON Schemas, refreshed via bump-schemas PR
│   └── …
├── src/lorcana_training/
│   ├── __init__.py
│   ├── cli.py                   # entry: lorcana-train
│   ├── cards/
│   │   ├── download.py          # fetch + verify cards-vN release
│   │   ├── max_copies.py        # mirror of TS computeMaxCopies
│   │   └── features.py          # card → feature vector
│   ├── dataset/
│   │   ├── download.py          # fetch + verify tournaments-vN release
│   │   ├── validate.py          # uses generated pydantic + isTournamentLegal mirror
│   │   └── splits.py            # train/heldout, recency filters
│   ├── models/
│   │   ├── card_encoder.py      # text + structured, two-headed pretrain
│   │   ├── proposal.py          # set-Transformer over card embeddings
│   │   └── evaluator.py         # pointwise partial+candidate scorer
│   ├── train/
│   │   ├── pretrain_encoder.py
│   │   ├── proposal.py
│   │   ├── evaluator.py
│   │   ├── curriculum.py        # negative-sampling schedule
│   │   └── samplers.py          # nucleus, greedy
│   ├── tables/
│   │   ├── play_frequency.py
│   │   └── archetype_centroids.py
│   ├── inference/                # mirror of lorcana-web's search,
│   │   ├── search.py             # used during eval + export smoke test
│   │   └── styles.py             # Safe/Balanced/Brew presets
│   ├── eval/
│   │   ├── functionality.py
│   │   ├── plausibility.py
│   │   ├── synergy.py            # diagnostic regexes
│   │   ├── novelty.py
│   │   ├── evaluator_quality.py
│   │   ├── gates.py
│   │   └── report.py             # writes eval-report.md
│   ├── export/
│   │   ├── to_onnx.py
│   │   ├── quantize.py
│   │   ├── embeddings.py         # writes card_embeddings.bin
│   │   └── manifest.py
│   ├── release/
│   │   └── github_release.py
│   └── schemas/                 # generated by datamodel-code-generator
│       └── …
├── tests/
│   ├── test_max_copies.py       # parity with TS fixtures
│   ├── test_encoder.py
│   ├── test_gates.py
│   └── fixtures/
│       ├── tiny_cards.json
│       ├── tiny_tournaments.json
│       └── …
├── notebooks/                   # exploratory only, gitignored from CI
└── .github/workflows/
    ├── ci.yml                   # lint, typecheck, test, smoke training
    ├── pretrain.yml             # workflow_dispatch: pretrain encoder
    ├── train.yml                # workflow_dispatch: full release training
    ├── release.yml              # on PR merge: publish model-vN
    └── new-set-reminder.yml     # daily: open issue if cards-vN drifted
```

---

## CI / scheduling

Five workflows.

### `ci.yml` — every PR / push

- `uv sync --frozen`
- `ruff check`, `ruff format --check`
- `mypy --strict src/`
- `pytest -x` (unit tests with small fixtures)
- Schema-pin assertion: the generated `schemas/` matches the major of
  `@bjorvack/lorcana-schemas` pinned in `config/training.yaml`.
  Mismatch → red.
- **Smoke training run:** full CLI end-to-end against tiny fixtures
  (50 decks, tiny model dimensions, one epoch per stage). Verifies
  every stage's plumbing, the ONNX export, and the quality-gate
  machinery (not the metric thresholds). Wall-clock target ≤ 15 min.

### `pretrain.yml` — manual encoder run

- Trigger: `workflow_dispatch` with required input
  `cards_release_tag`.
- Intended to run on a self-hosted GPU runner. The same workflow can
  fall back to a maintainer's machine; the runner label is configured
  via repo settings.
- Steps:
  1. `uv sync --frozen`
  2. `lorcana-train pretrain-encoder --cards <tag>`
  3. Bundle outputs into `encoder-<date>-<seq>.tar.zst`.
  4. Open a PR creating an `encoder-vN` GitHub Release with the bundle
     and updating `RELEASES.md`. The PR body lists which cards are new
     in this encoder vs. the previous.

### `train.yml` — manual release-training run

- Trigger: `workflow_dispatch` only. No cron.
- Intended to run on a self-hosted GPU runner.
- Steps:
  1. `uv sync --frozen`
  2. `lorcana-train all`
  3. Open a PR `train/model-<new-version>` updating
     `RELEASES.md` with the eval-report.md as PR body.

Auto-merge is **off**: a maintainer reviews the eval report and
merges.

### `release.yml` — model publication

- Triggered on merge of a `train/*` PR.
- Re-runs `lorcana-train all` on the merged commit to regenerate
  artifacts.
- Verifies the resulting `manifest.json.metrics` matches what was in
  the PR body (within fp tolerance). Mismatch → fail.
- Publishes `model-vX.Y.Z` GitHub Release.

### `new-set-reminder.yml` — daily check

- Cron: `0 7 * * *` (daily, 07:00 UTC).
- Compares the latest `cards-vN` release tag in `lorcana-scraper`
  against the `cards_release_tag` pinned in `config/training.yaml`.
- If newer: opens (or keeps open) an issue titled `New cards-vN
  available: trigger release training` with the checklist documented
  in the open-questions section.
- Idempotent: only one open issue per "latest tag" value.

---

## Reproducibility

- Seeds: `PYTHONHASHSEED`, `random`, `numpy`, `torch` all seeded from
  `config/training.yaml:seed`. Default 42.
- Determinism flags: `torch.use_deterministic_algorithms(True)` where
  possible; fallback to "best-effort deterministic" with logged
  warnings when not.
- All artifacts are content-hashed. `prepared/manifest.json` and
  `model-vN/manifest.json` together let you re-derive any released
  model from source.
- `uv.lock` is committed; `uv sync --frozen` in CI guarantees the same
  dep tree.
- `run.json` for each training stage is committed alongside the
  release; the model manifest references the git SHA those JSONs
  belong to, so any released model can be retraced to its training
  telemetry without external systems.

---

## Open questions to resolve before implementing

1. **Card-text tokeniser.** *Decided: custom BPE on card texts.* Train
   a BPE vocab on the corpus of all card texts in the pinned
   `cards-vN`. Vocab size starts at 8k and is a swept hyperparameter
   (candidates: 4k, 8k, 16k); the winner is the one that minimises
   pretrain held-out MLM loss without growing the embedding table
   needlessly.

   Re-trained as part of `pretrain-encoder` every time the model
   pipeline runs against a new `cards-vN`. The trained tokeniser is
   versioned and shipped alongside the encoder weights so a future
   re-encode of an unchanged `cards-vN` is reproducible.

   Fallback path if a future set introduces wording that fragments
   poorly under our BPE: swap to a pretrained tokeniser (GPT-2 BPE or
   SentencePiece). The encoder architecture doesn't care which one
   produces the tokens.
2. **Recency cutoff for the proposal net.** *Decided: different
   windows per model.*
   - **Proposal Net:** trailing 12 months. The proposal's job is to
     suggest cards a player would plausibly pick *now*, so it sees
     recent meta.
   - **Evaluator:** all decks containing only cards present in the
     pinned `cards-vN`, no time cutoff. The evaluator's job is "is
     this a plausible Lorcana deck", which is a broader question than
     "is this competitive in this month's meta". A wider positive set
     improves generalisation, and the curriculum's hard negatives
     anchor it to the current card pool.

   In both cases the hard filter is the same: a deck containing **any**
   card not in the pinned `cards-vN` is dropped entirely. We do not
   substitute or interpolate; the example is lost. This means the
   `prepare` stage emits two split files
   (`prepared/train.proposal.parquet`,
   `prepared/train.evaluator.parquet`) with overlapping but distinct
   contents.

   Windows are configurable in `config/training.yaml`:
   ```yaml
   recency:
     proposal: { type: trailing_months, value: 12 }
     evaluator: { type: in_vocab_only }
   ```
   so swapping to "last N set rotations" later is a config change, not
   a code change.
3. **Adversarial phase (phase 4) for the evaluator.** *Decided: drop
   for v1.* The curriculum runs phases 1–3 only (random in-ink →
   curve-matched → local-swap). Phase 4 has the highest "destabilises
   training" risk in the pipeline and is the least demonstrably
   necessary; we ship without it, measure the resulting evaluator
   quality against the gates, and only revisit if eval shows the
   evaluator is failing on subtle synergies.

   This is reflected in `train-evaluator`'s phase config:
   ```yaml
   evaluator_curriculum:
     phases: [warmup, curve_matching, local_swap]
   ```
   Adding phase 4 later is purely additive — a new entry in the
   phases list plus the negative-sampling code path that requires a
   prior model checkpoint.
4. **Style preset weights.** *Decided: auto-calibrate per release.*
   The training pipeline picks the `(α, γ, λ)` tuples for Safe,
   Balanced, and Brew during the `eval` stage by searching the weight
   space and selecting the tuples that land each Style in the
   middle of its target novelty band. The chosen tuples are written
   into `manifest.json` under a `style_presets` key; `lorcana-web`
   reads them at startup and uses them as the labelled-button
   positions and as the bounds of the optional underlying slider.

   Implementation:
   - **Coarse-then-fine sweep**, not full grid. Stage 1: 27-point
     coarse grid (3 levels per dimension). Stage 2: 3-point local
     refinement around the best coarse point for each Style. Total
     ~36 evaluations per Style × 3 Styles ≈ 108 sampling runs.
   - Each sampling run: 50 decks per ink-pair × 6 ink-pairs = 300
     decks. With ~50 search steps per deck this is the dominant cost
     in `eval`; we budget for it in the wall-clock target and accept
     it as a first-class part of release-readiness, not an
     afterthought.
   - The eval report ships a small heatmap per Style (novelty vs.
     evaluator-score on the 27 coarse points) so the calibration is
     auditable in the release PR. If a calibration ever picks
     pathological weights ("Brew is identical to Balanced") it shows
     up visually before merge.

   Implementation note: the search lives in
   `src/lorcana_training/inference/styles.py` alongside the Style
   presets themselves, so the calibration code and the inference
   code can't disagree about what `(α, γ, λ)` mean.
5. **Training telemetry.** *Decided: JSON summaries only, no MLflow.*
   Each training subcommand emits:
   - `out/<stage>/run.json` — per-epoch scalar metrics (loss,
     held-out NLL, AUROC, …) plus run metadata (seed, config hash,
     cards/tournaments tags, git SHA).
   - `out/<stage>/samples/*.json` — generated decks at fixed eval
     checkpoints, useful for visual regression review.
   - `out/eval-report.md` — the human-readable summary that becomes
     the release PR body.

   Drop MLflow from the stack: it adds two mechanisms for capturing
   the same information and at our scale the marginal value is low.
   If someone later wants an interactive UI, a small Streamlit app
   over `run.json` is ~30 lines.

   All `out/*` artifacts are uploaded by CI as workflow artifacts
   and the final subset (the three files above) is committed to the
   release branch alongside `RELEASES.md`.
6. **CPU-only CI training feasibility.** *Decided: tiered, with a
   release-trigger reminder.*

   Three modes of running this pipeline:

   - **Smoke (every PR).** CI runs the full CLI end-to-end against a
     reduced fixture (tiny `cards-vN`, ~50 decks, tiny model
     dimensions, one epoch). Goal: catch code-change breakage,
     schema drift, ONNX-export regressions. Wall-clock target
     ≤ 15 min. Produces artifacts but does **not** publish a
     release.
   - **Pretrain (`workflow_dispatch`).** `pretrain-encoder` runs on
     a maintainer's machine when a new `cards-vN` lands and the
     existing encoder no longer covers the card pool. Output is a
     versioned `encoder-vN` bundle (`card_embeddings.bin`,
     tokeniser, encoder weights) published as its own GitHub
     Release. `config/training.yaml` then pins it like the other
     artifacts:
     ```yaml
     encoder_release_tag: encoder-v2025.06.01-01
     ```
   - **Release run (`workflow_dispatch`).** A maintainer triggers
     the full release pipeline (proposal → evaluator → tables →
     eval with Style calibration → export) on a self-hosted GPU
     runner or a maintainer machine. The smoke run on CI is what
     keeps the code honest between release runs.

   **New-set release reminder.** A small scheduled workflow in this
   repo runs daily and checks whether the latest `cards-vN` release
   tag in `lorcana-scraper` is newer than the one currently pinned
   in `config/training.yaml`. If so, it opens a single GitHub Issue
   titled `New cards-vN available: trigger release training` with
   a checklist:
   - [ ] Decide whether new encoder is needed (run
         `pretrain-encoder` on demand if so).
   - [ ] Bump `cards_release_tag` (and `tournaments_release_tag`)
         in `config/training.yaml`.
   - [ ] Run release pipeline via `workflow_dispatch`.
   - [ ] Review the eval report and open the release PR.

   The issue is auto-closed once `config/training.yaml`'s pinned
   tag matches the latest. The workflow is idempotent: if an issue
   already exists for the current latest tag, no duplicate is
   opened.
7. **Diagnostic synergy regexes — where do they live?** *Decided:
   `eval/synergy.py` plus an enforced import guard.*

   Regexes live exclusively in
   `src/lorcana_training/eval/synergy.py`. To make the discipline
   mechanical rather than aspirational, CI runs an import-guard test:

   ```python
   # tests/test_import_guard.py
   import importlib, pkgutil, lorcana_training as pkg

   FORBIDDEN = "lorcana_training.eval"
   ALLOWED_PARENTS = {"lorcana_training.eval"}  # eval can import its own siblings

   def test_no_training_module_imports_eval():
       for mod_info in pkgutil.walk_packages(pkg.__path__, prefix="lorcana_training."):
           if any(mod_info.name.startswith(p) for p in ALLOWED_PARENTS):
               continue
           mod = importlib.import_module(mod_info.name)
           for attr_name, attr in vars(mod).items():
               source = getattr(attr, "__module__", "") or ""
               assert FORBIDDEN not in source, (
                   f"{mod_info.name} imports from {FORBIDDEN}: {attr_name}"
               )
   ```

   Failing this test is a release-blocking error. If a contributor
   needs a synergy-derived feature inside training, they have to (a)
   add a new failure to the test, (b) document why in the PR, and
   (c) accept that the model no longer satisfies the "learns synergy
   on its own" goal. That cost should make the choice deliberate
   rather than accidental.
