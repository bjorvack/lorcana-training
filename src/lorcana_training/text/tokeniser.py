"""BPE tokeniser for card rules text.

Card text is a narrow, game-specific corpus — a few thousand cards,
maybe 100-200k tokens total. We deliberately pick a large 32k vocab
(matching DESIGN.md) even though the corpus doesn't statistically
warrant it, because:

- Future card pool expansion is likely; an oversized vocab ages
  better than one sized to today's data.
- The vocab table is tiny in absolute terms (~6 MB float16) relative
  to the rest of the model budget.
- A new keyword introduced in a future set gets broken into existing
  BPE pieces on day zero; with a bigger initial vocab the pieces are
  more semantically stable.

We use byte-level BPE (the same family as GPT-2 / RoBERTa), which is
the pragmatic default — it handles stray punctuation, emoji, and any
surprise unicode without ever producing an UNK token.

Special tokens are fixed up-front so downstream modules can hard-code
their ids:

    [PAD] = 0      padding
    [UNK] = 1      never actually emitted by byte-level BPE, kept
                   for compatibility with off-the-shelf training
                   recipes that expect one
    [CLS] = 2      prepended to every text so the encoder has a
                   single "sentence" slot to pool over
    [SEP] = 3      reserved; unused today but lets us concatenate
                   multiple text fields without retraining
    [MASK] = 4     MLM mask token for pretrain_encoder

On top of those we reserve a second tier of *atomic* domain tokens
(card names, classifications, keywords, game glyphs like ``{I}`` /
``{L}``). These never get split by BPE either, so the model always
sees a named card or a classification as one token rather than a
random subword shard. See :func:`collect_reserved_tokens`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from ..cards.logical import LogicalCard


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
SPECIAL_TOKENS: tuple[str, ...] = (
    PAD_TOKEN,
    UNK_TOKEN,
    CLS_TOKEN,
    SEP_TOKEN,
    MASK_TOKEN,
)

DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_MIN_FREQUENCY = 2

# Game-state glyphs that appear in card text. Mirrors what the Lorcana
# rules layer uses for ink / lore / exert / strength / willpower and
# cost-number icons. We reserve them as atomic tokens so a card that
# references "{I}" doesn't get "{", "I", "}" tokenised separately.
GAME_GLYPHS: tuple[str, ...] = (
    "{E}",
    "{I}",
    "{L}",
    "{S}",
    "{W}",
    "{1}",
    "{2}",
    "{IW}",
)


def _build_tokeniser() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
    # Byte-level pre-tokeniser and decoder: matched pair, otherwise
    # decoding produces garbage (see the tokenizers docs for the gotcha).
    tok.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    return tok


def train_tokeniser(
    texts: list[str],
    *,
    out_path: Path,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    reserved_tokens: Iterable[str] = (),
) -> Tokenizer:
    """Train a BPE tokeniser on the supplied texts and save to ``out_path``.

    ``reserved_tokens`` are additional atomic tokens (card names,
    classifications, keywords, glyphs) — BPE will never split them.
    They're deduplicated against :data:`SPECIAL_TOKENS` and each
    other, so passing the same name twice is harmless.

    Returns the fitted tokeniser so callers can use it immediately
    without re-loading from disk.
    """
    if not texts:
        raise ValueError("train_tokeniser: no training texts provided")

    # Preserve insertion order while deduping, so id assignment stays
    # stable across re-runs on the same card pool.
    seen: set[str] = set(SPECIAL_TOKENS)
    extras: list[str] = []
    for token in reserved_tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        extras.append(token)
    all_specials = list(SPECIAL_TOKENS) + extras

    tok = _build_tokeniser()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=all_specials,
        # Seed the vocab with the byte alphabet so every codepoint is
        # guaranteed-representable.
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)

    # After training, pin a post-processor that wraps every input in
    # [CLS] ... [SEP] so the encoder's first token is a consistent
    # "sentence representation" slot.
    tok.post_processor = TemplateProcessing(
        single=f"{CLS_TOKEN} $A {SEP_TOKEN}",
        special_tokens=[
            (CLS_TOKEN, tok.token_to_id(CLS_TOKEN)),
            (SEP_TOKEN, tok.token_to_id(SEP_TOKEN)),
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def load_tokeniser(path: Path) -> Tokenizer:
    """Load a tokeniser previously saved by :func:`train_tokeniser`."""
    return Tokenizer.from_file(str(path))


def special_token_ids(tok: Tokenizer) -> dict[str, int]:
    """Return a ``{name: id}`` map for the special tokens. Useful for
    handing off to torch modules as ``int`` constants."""
    return {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}


def collect_reserved_tokens(logical_cards: Iterable[LogicalCard]) -> list[str]:
    """Extract the atomic-token list for the pinned card pool.

    Includes:

    - Every ``name`` referenced by "named X" constructs in card text
      (e.g. Elsa, Tinker Bell, Mickey Mouse).
    - Every ``classification`` the pool carries (Hero, Princess,
      Villain, …) — referenced by "your Hero characters" etc.
    - Every ``keyword`` defined by the game (Shift, Rush, Evasive, …).
      Both the capitalised form and the lower-case variant are added
      because card text is inconsistent.
    - The :data:`GAME_GLYPHS` tuple.

    Returned in a stable, deduplicated order so the tokeniser's id
    assignment is reproducible from a pinned ``cards-vN``.
    """
    result: list[str] = []
    seen: set[str] = set()

    def _add(value: str | None) -> None:
        if not value or value in seen:
            return
        seen.add(value)
        result.append(value)

    for glyph in GAME_GLYPHS:
        _add(glyph)

    for card in logical_cards:
        _add(card.name)
        if card.canonical.classifications:
            for cls in card.canonical.classifications:
                _add(cls)
        if card.canonical.keywords:
            for kw in card.canonical.keywords:
                _add(kw)
                # Card text is inconsistent about case: "Shift" in some
                # reminder texts becomes "shift" in body text. Reserve
                # both so either form maps to an atomic id.
                _add(kw.lower())

    return result
