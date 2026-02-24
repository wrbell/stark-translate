#!/usr/bin/env python3
"""
test_adaptive_model.py — Adaptive Model Selection Test for P7 Item 6B

Tests the hypothesis: use MarianMT CT2 (~50ms) for simple sentences and
TranslateGemma 4B (~650ms) for complex ones. Measures quality/latency
trade-offs to find the optimal routing threshold.

Usage:
    python test_adaptive_model.py                                # Built-in test sentences
    python test_adaptive_model.py --csv metrics/ab_metrics_*.csv # Use real session data
    python test_adaptive_model.py --quick                        # Fewer sentences, no QE
"""

import argparse
import csv
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# Theological vocabulary (loaded from glossary + hardcoded essentials)
# ---------------------------------------------------------------------------

# Single-word theological terms that need Gemma's nuance for correct
# disambiguation. Loaded from the project glossary at startup.
THEOLOGICAL_TERMS = set()

# Multi-word theological phrases (checked separately since word splitting
# won't catch them).
THEOLOGICAL_PHRASES = set()

# Bible names and places that MarianMT may transliterate incorrectly.
# These overlap with the glossary but are maintained separately for clarity.
BIBLE_NAMES = {
    "jesus",
    "christ",
    "god",
    "lord",
    "moses",
    "abraham",
    "david",
    "solomon",
    "elijah",
    "elisha",
    "peter",
    "paul",
    "james",
    "john",
    "matthew",
    "mark",
    "luke",
    "timothy",
    "stephen",
    "barnabas",
    "nicodemus",
    "lazarus",
    "mary",
    "martha",
    "joseph",
    "jacob",
    "isaac",
    "noah",
    "adam",
    "eve",
    "aaron",
    "samuel",
    "gideon",
    "samson",
    "saul",
    "pharaoh",
    "herod",
    "pilate",
    "satan",
    "jerusalem",
    "bethlehem",
    "nazareth",
    "galilee",
    "jordan",
    "egypt",
    "israel",
    "zion",
    "sinai",
    "calvary",
    "gethsemane",
}


def load_glossary():
    """Load theological terms from bible_data/glossary/glossary_pairs.jsonl."""
    glossary_path = os.path.join(os.path.dirname(__file__), "bible_data", "glossary", "glossary_pairs.jsonl")
    if not os.path.exists(glossary_path):
        print(f"  Glossary not found at {glossary_path}, using hardcoded terms only")
        return

    count = 0
    with open(glossary_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            term = entry["en"].lower()
            # Skip sentence-length examples (only keep actual terms)
            if len(term.split()) > 4:
                continue
            if " " in term:
                THEOLOGICAL_PHRASES.add(term)
            else:
                THEOLOGICAL_TERMS.add(term)
            count += 1

    print(
        f"  Loaded {count} glossary terms ({len(THEOLOGICAL_TERMS)} single-word, {len(THEOLOGICAL_PHRASES)} multi-word)"
    )


# ---------------------------------------------------------------------------
# Complexity Heuristics
# ---------------------------------------------------------------------------

# Conjunctions and relative pronouns that indicate multi-clause sentences
_CLAUSE_MARKERS = re.compile(
    r"\b(who|whom|whose|which|that|where|when|while|although|though|"
    r"because|since|unless|whereas|whereby|wherein|wherever|whenever|"
    r"whoever|whatever|however|moreover|furthermore|nevertheless|"
    r"therefore|accordingly|consequently)\b",
    re.IGNORECASE,
)


def compute_complexity(text):
    """Compute complexity features for a sentence.

    Returns a dict with individual features and a combined score (0-1).
    Higher score = more complex = should use Gemma.
    """
    words = text.split()
    word_count = len(words)
    lower_text = text.lower()

    # Feature 1: Word count (normalized)
    # <8 words = simple, >20 words = complex
    if word_count <= 6:
        wc_score = 0.0
    elif word_count <= 10:
        wc_score = 0.2
    elif word_count <= 16:
        wc_score = 0.5
    elif word_count <= 24:
        wc_score = 0.7
    else:
        wc_score = 1.0

    # Feature 2: Clause complexity
    # Count commas and clause-introducing words
    comma_count = text.count(",")
    semicolons = text.count(";")
    clause_markers = len(_CLAUSE_MARKERS.findall(text))
    clause_signals = comma_count + semicolons * 2 + clause_markers
    if clause_signals == 0:
        clause_score = 0.0
    elif clause_signals <= 1:
        clause_score = 0.2
    elif clause_signals <= 3:
        clause_score = 0.5
    elif clause_signals <= 5:
        clause_score = 0.7
    else:
        clause_score = 1.0

    # Feature 3: Theological vocabulary density
    lower_words = set(w.strip(".,!?;:'\"").lower() for w in words)
    theo_hits = lower_words & THEOLOGICAL_TERMS
    # Also check multi-word phrases
    phrase_hits = [p for p in THEOLOGICAL_PHRASES if p in lower_text]
    theo_count = len(theo_hits) + len(phrase_hits)
    if word_count > 0:
        theo_density = theo_count / word_count
    else:
        theo_density = 0.0
    if theo_count == 0:
        theo_score = 0.0
    elif theo_count == 1 and theo_density < 0.15:
        theo_score = 0.2
    elif theo_count <= 2:
        theo_score = 0.5
    elif theo_count <= 4:
        theo_score = 0.7
    else:
        theo_score = 1.0

    # Feature 4: Named entity presence (Bible names)
    name_hits = lower_words & BIBLE_NAMES
    if len(name_hits) == 0:
        name_score = 0.0
    elif len(name_hits) == 1:
        name_score = 0.3
    else:
        name_score = 0.6

    # Combined score: weighted average
    # Theological terms and clause complexity matter most
    combined = 0.20 * wc_score + 0.30 * clause_score + 0.30 * theo_score + 0.20 * name_score

    return {
        "word_count": word_count,
        "wc_score": round(wc_score, 2),
        "comma_count": comma_count,
        "clause_markers": clause_markers,
        "clause_score": round(clause_score, 2),
        "theo_terms": sorted(theo_hits | set(phrase_hits)),
        "theo_count": theo_count,
        "theo_score": round(theo_score, 2),
        "bible_names": sorted(name_hits),
        "name_score": round(name_score, 2),
        "complexity": round(combined, 3),
    }


def route_model(complexity_score, threshold=0.35):
    """Decide which model to use based on complexity score.

    Returns 'marian' or 'gemma'.
    """
    return "marian" if complexity_score < threshold else "gemma"


# ---------------------------------------------------------------------------
# Test Sentence Corpus
# ---------------------------------------------------------------------------

# Curated sentences spanning easy to very hard.
# Expected routing in parentheses for manual verification.
TEST_SENTENCES = [
    # --- Easy (expect: MarianMT) ---
    ("Thank you for coming today.", "easy"),
    ("Let us pray.", "easy"),
    ("Good morning everyone.", "easy"),
    ("Please be seated.", "easy"),
    ("Welcome to our service.", "easy"),
    ("Amen.", "easy"),
    ("God bless you all.", "easy"),
    ("Please turn to page forty-two.", "easy"),
    ("We will now sing a hymn.", "easy"),
    ("Let us read together.", "easy"),
    # --- Medium (borderline: could go either way) ---
    ("The grace of God is sufficient for all of us.", "medium"),
    ("We are saved by faith and not by works.", "medium"),
    ("Let us remember the cross of Christ today.", "medium"),
    ("The Lord is my shepherd, I shall not want.", "medium"),
    ("Blessed are those who hunger and thirst for righteousness.", "medium"),
    ("We are called to love one another as Christ loved us.", "medium"),
    ("Pray without ceasing and give thanks in all things.", "medium"),
    ("The Word of God is living and powerful.", "medium"),
    ("Jesus said I am the way, the truth, and the life.", "medium"),
    ("Brothers and sisters, let us encourage one another.", "medium"),
    # --- Hard (expect: Gemma) ---
    (
        "For there is one God, and one mediator between God and men, the man Christ Jesus, "
        "who gave himself a ransom for all.",
        "hard",
    ),
    (
        "The apostle Paul wrote to the Romans about justification by faith, "
        "explaining how the righteousness of God is revealed from faith to faith.",
        "hard",
    ),
    (
        "In the Gospel of John, Jesus tells Nicodemus that unless a man is born again, "
        "he cannot see the kingdom of God.",
        "hard",
    ),
    (
        "Abraham believed God, and it was counted unto him for righteousness, "
        "just as David also describes the blessedness of the man to whom God imputes "
        "righteousness without works.",
        "hard",
    ),
    (
        "The prophet Isaiah foretold that the Messiah would be wounded for our "
        "transgressions and bruised for our iniquities.",
        "hard",
    ),
    (
        "Moses led the children of Israel out of Egypt through the Red Sea, "
        "and God provided manna from heaven during their wilderness journey.",
        "hard",
    ),
    (
        "Peter declared at Pentecost that God had made Jesus both Lord and Christ, "
        "and three thousand souls were added to the church that day.",
        "hard",
    ),
    (
        "The epistle to the Hebrews teaches that Jesus is the high priest of our "
        "confession, who passed through the heavens.",
        "hard",
    ),
    # --- Very Hard (definitely Gemma) ---
    (
        "The doctrine of propitiation teaches us that Christ's atoning sacrifice "
        "satisfied the righteous wrath of God against sin.",
        "very_hard",
    ),
    (
        "The substitutionary atonement of Christ on the cross accomplished our "
        "redemption, reconciliation, and justification before a holy God, "
        "fulfilling the covenant promises made to Abraham.",
        "very_hard",
    ),
    (
        "The Westminster Confession teaches that God's eternal decree of election "
        "and predestination is the foundation of our sanctification, "
        "which is the progressive work of the Holy Spirit in the believer.",
        "very_hard",
    ),
    (
        "In Romans chapter eight, Paul writes that there is therefore now no "
        "condemnation to them which are in Christ Jesus, who walk not after the flesh "
        "but after the Spirit, for the law of the Spirit of life in Christ Jesus "
        "hath made me free from the law of sin and death.",
        "very_hard",
    ),
    (
        "The sovereignty of God in providence means that He sustains, directs, "
        "and governs all creatures and all events, from the greatest to the least, "
        "by His most wise and holy counsel, according to His infallible foreknowledge "
        "and the free and immutable counsel of His own will.",
        "very_hard",
    ),
]


# ---------------------------------------------------------------------------
# Lightweight QE (reuse from dry_run_ab.py patterns)
# ---------------------------------------------------------------------------

_EN_WORDS = re.compile(
    r"\b(the|and|of|that|have|for|not|with|you|this|but|his|from|they|"
    r"been|said|each|which|their|will|other|about|many|then|them|these|"
    r"would|could|should|because|into|after|before|between|under|through)\b",
    re.IGNORECASE,
)


def qe_length_ratio(source, translation):
    """Score 0-1 based on length ratio. Spanish typically 15-25% longer."""
    if not source or not translation:
        return 0.0
    ratio = len(translation) / len(source)
    if 0.9 <= ratio <= 1.6:
        return 1.0
    elif 0.7 <= ratio <= 2.0:
        return 0.7
    elif 0.5 <= ratio <= 2.5:
        return 0.4
    return 0.1


def qe_untranslated(source, translation):
    """Score 0-1 detecting untranslated English in Spanish output."""
    if not translation:
        return 0.0
    en_matches = _EN_WORDS.findall(translation)
    words = translation.split()
    if not words:
        return 0.0
    en_ratio = len(en_matches) / len(words)
    if en_ratio < 0.05:
        return 1.0
    elif en_ratio < 0.15:
        return 0.7
    elif en_ratio < 0.30:
        return 0.4
    return 0.1


def qe_combined(source, translation):
    """Combined lightweight QE score (0-1)."""
    lr = qe_length_ratio(source, translation)
    ut = qe_untranslated(source, translation)
    return round((lr + ut) / 2, 2)


# ---------------------------------------------------------------------------
# Model Loading & Translation
# ---------------------------------------------------------------------------


def load_ct2_marian():
    """Load CTranslate2 int8 MarianMT (~76MB)."""
    ct2_path = os.path.join(os.path.dirname(__file__), "ct2_opus_mt_en_es")
    if not os.path.isdir(ct2_path):
        print("ERROR: CT2 model not found. Run the ct2 converter first.", file=sys.stderr)
        print(f"  Expected at: {ct2_path}", file=sys.stderr)
        sys.exit(1)

    import ctranslate2
    from transformers import MarianTokenizer

    print("  Loading CTranslate2 MarianMT (int8)...")
    t0 = time.time()
    translator = ctranslate2.Translator(ct2_path, device="cpu", compute_type="int8")
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    # Warm up
    warm_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello world."))
    translator.translate_batch([warm_tokens], max_decoding_length=32)
    print(f"  CT2 MarianMT ready ({time.time() - t0:.1f}s)")
    return translator, tokenizer


def load_gemma_4b():
    """Load TranslateGemma 4B via MLX (4-bit quantized)."""
    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(256 * 1024 * 1024)

    model_id = "mlx-community/translategemma-4b-it-4bit"
    print(f"  Loading {model_id}...")
    t0 = time.time()
    model, tokenizer = load(model_id)

    # Fix EOS tokens (same as dry_run_ab.py)
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    default_eos = tokenizer.eos_token_id
    if not hasattr(tokenizer, "_eos_token_ids") or eot_id not in tokenizer._eos_token_ids:
        tokenizer._eos_token_ids = {default_eos, eot_id}

    print(f"  Gemma 4B ready ({time.time() - t0:.1f}s)")
    return model, tokenizer


def translate_ct2(text, translator, tokenizer):
    """Translate English->Spanish via CT2 MarianMT. Returns (translation, latency_ms)."""
    t0 = time.perf_counter()
    src_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    ct2_out = translator.translate_batch(
        [src_tokens],
        max_decoding_length=128,
        beam_size=4,
    )
    result = tokenizer.convert_tokens_to_string(ct2_out[0].hypotheses[0])
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms


def translate_gemma(text, model, tokenizer):
    """Translate English->Spanish via TranslateGemma 4B MLX. Returns (translation, latency_ms)."""
    from mlx_lm import generate

    input_words = len(text.split())
    max_tok = max(32, int(input_words * 1.8))

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": text}],
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    t0 = time.perf_counter()
    result = generate(model, tokenizer, prompt=prompt, max_tokens=max_tok, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000

    clean = result.split("<end_of_turn>")[0].strip()
    return clean, latency_ms


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def print_table(results, show_qe=True):
    """Print a formatted comparison table."""
    # Header
    print()
    print(f"{'#':>3}  {'Cplx':>5}  {'Route':>6}  {'Diff':>6}  {'MT ms':>6}  {'TG ms':>6}  {'Save':>6}  ", end="")
    if show_qe:
        print(f"{'QE-MT':>5}  {'QE-TG':>5}  {'QE-D':>5}  ", end="")
    print(f"{'Cat':>9}  Sentence")
    print("-" * (130 if show_qe else 100))

    for i, r in enumerate(results):
        save_ms = r["gemma_ms"] - r["marian_ms"]
        qe_diff = (r.get("qe_marian", 0) or 0) - (r.get("qe_gemma", 0) or 0)

        route = route_model(r["complexity"])
        print(f"{i + 1:>3}  {r['complexity']:>5.3f}  {route:>6}  ", end="")

        # Color-code the quality difference (positive = MarianMT worse)
        print(f"{qe_diff:>+6.2f}  " if show_qe else "        ", end="")
        print(f"{r['marian_ms']:>6.0f}  {r['gemma_ms']:>6.0f}  {save_ms:>+6.0f}  ", end="")

        if show_qe:
            qe_m = r.get("qe_marian")
            qe_g = r.get("qe_gemma")
            print(f"{qe_m:>5.2f}  " if qe_m is not None else "    -  ", end="")
            print(f"{qe_g:>5.2f}  " if qe_g is not None else "    -  ", end="")
            print(f"{qe_diff:>+5.2f}  " if (qe_m is not None and qe_g is not None) else "    -  ", end="")

        print(f"{r['category']:>9}  {r['english'][:55]}")

    print("-" * (130 if show_qe else 100))


def print_translations(results):
    """Print full translations for manual inspection."""
    print(f"\n{'=' * 80}")
    print("FULL TRANSLATIONS (for manual review)")
    print(f"{'=' * 80}")
    for i, r in enumerate(results):
        route = route_model(r["complexity"])
        print(f"\n--- #{i + 1} [{r['category']}] complexity={r['complexity']:.3f} route={route} ---")
        print(f"  EN: {r['english']}")
        print(f"  MT: {r['marian_text']}  ({r['marian_ms']:.0f}ms)")
        print(f"  TG: {r['gemma_text']}  ({r['gemma_ms']:.0f}ms)")
        if r.get("qe_marian") is not None and r.get("qe_gemma") is not None:
            diff = r["qe_marian"] - r["qe_gemma"]
            flag = " <<< QUALITY DROP" if diff < -0.2 else ""
            print(f"  QE: MT={r['qe_marian']:.2f}  TG={r['qe_gemma']:.2f}  diff={diff:+.2f}{flag}")


def print_recommendation(results):
    """Print the recommendation report."""
    print(f"\n{'=' * 80}")
    print("RECOMMENDATION REPORT")
    print(f"{'=' * 80}")

    # Analyze different thresholds
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    has_qe = all(r.get("qe_marian") is not None and r.get("qe_gemma") is not None for r in results)

    print("\n1. THRESHOLD ANALYSIS")
    print(f"   {'Thresh':>6}  {'->MT':>5}  {'->TG':>5}  {'Avg Save':>9}  ", end="")
    if has_qe:
        print(f"{'Avg QE Loss':>11}  {'FP (>0.15)':>10}  {'FP (>0.10)':>10}")
    else:
        print()
    print(f"   {'-' * 70}")

    best_threshold = 0.35
    best_score = -999

    for thresh in thresholds:
        routed_marian = [r for r in results if r["complexity"] < thresh]
        routed_gemma = [r for r in results if r["complexity"] >= thresh]
        n_mt = len(routed_marian)
        n_tg = len(routed_gemma)

        if routed_marian:
            avg_save = sum(r["gemma_ms"] - r["marian_ms"] for r in routed_marian) / n_mt
        else:
            avg_save = 0.0

        print(f"   {thresh:>6.2f}  {n_mt:>5}  {n_tg:>5}  {avg_save:>+8.0f}ms  ", end="")

        if has_qe and routed_marian:
            qe_losses = [r["qe_marian"] - r["qe_gemma"] for r in routed_marian]
            avg_loss = sum(qe_losses) / len(qe_losses)
            fp_15 = sum(1 for l in qe_losses if l < -0.15)
            fp_10 = sum(1 for l in qe_losses if l < -0.10)
            print(f"{avg_loss:>+10.3f}  {fp_15:>5}/{n_mt:<4}  {fp_10:>5}/{n_mt:<4}")

            # Score: maximize savings while minimizing quality loss
            # Penalize heavily for false positives
            score = avg_save * 0.5 - fp_15 * 100 - abs(avg_loss) * 500
            if score > best_score:
                best_score = score
                best_threshold = thresh
        else:
            print()

    # Category breakdown
    print("\n2. CATEGORY BREAKDOWN")
    categories = ["easy", "medium", "hard", "very_hard"]
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        complexities = [r["complexity"] for r in cat_results]
        avg_c = sum(complexities) / len(complexities)
        min_c = min(complexities)
        max_c = max(complexities)
        marian_routed = sum(1 for c in complexities if c < best_threshold)
        avg_save = sum(r["gemma_ms"] - r["marian_ms"] for r in cat_results) / len(cat_results)
        print(
            f"   {cat:>10}: n={len(cat_results):>2}  "
            f"complexity={avg_c:.3f} [{min_c:.3f}-{max_c:.3f}]  "
            f"->MarianMT={marian_routed}/{len(cat_results)}  "
            f"avg_save={avg_save:+.0f}ms"
        )

    # Projected savings
    print("\n3. PROJECTED LATENCY SAVINGS (typical sermon)")
    print("   Assumed distribution: 40% easy, 30% medium, 20% hard, 10% very hard")
    print(f"   Recommended threshold: {best_threshold:.2f}")

    weights = {"easy": 0.40, "medium": 0.30, "hard": 0.20, "very_hard": 0.10}
    total_save_weighted = 0.0
    total_latency_baseline = 0.0
    total_latency_adaptive = 0.0

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        w = weights.get(cat, 0.25)
        for r in cat_results:
            per_sent = 1.0 / len(cat_results)  # equal weight within category
            baseline_ms = r["gemma_ms"]
            if r["complexity"] < best_threshold:
                adaptive_ms = r["marian_ms"]
            else:
                adaptive_ms = r["gemma_ms"]
            total_latency_baseline += baseline_ms * w * per_sent
            total_latency_adaptive += adaptive_ms * w * per_sent
            total_save_weighted += (baseline_ms - adaptive_ms) * w * per_sent

    print(f"   Baseline (all Gemma):    ~{total_latency_baseline:.0f}ms avg per sentence")
    print(f"   Adaptive (mixed):        ~{total_latency_adaptive:.0f}ms avg per sentence")
    print(
        f"   Projected savings:       ~{total_save_weighted:.0f}ms avg per sentence "
        f"({total_save_weighted / total_latency_baseline * 100:.0f}% reduction)"
        if total_latency_baseline > 0
        else ""
    )

    # False positive analysis
    if has_qe:
        print(f"\n4. FALSE POSITIVE ANALYSIS (threshold={best_threshold:.2f})")
        print("   Sentences routed to MarianMT that should have used Gemma")
        print("   (QE loss > 0.15 = meaningful quality drop)")
        print()
        fp_count = 0
        for i, r in enumerate(results):
            if r["complexity"] < best_threshold:
                qe_loss = r["qe_marian"] - r["qe_gemma"]
                if qe_loss < -0.15:
                    fp_count += 1
                    print(f"   FALSE POSITIVE #{fp_count}: [{r['category']}] complexity={r['complexity']:.3f}")
                    print(f"     EN: {r['english'][:70]}")
                    print(f"     QE: MT={r['qe_marian']:.2f} TG={r['qe_gemma']:.2f} loss={qe_loss:+.2f}")
                    print()
        if fp_count == 0:
            print("   None found. All MarianMT-routed sentences maintain quality.")

        marian_count = sum(1 for r in results if r["complexity"] < best_threshold)
        fp_rate = fp_count / marian_count if marian_count > 0 else 0.0
        print(f"   False positive rate: {fp_count}/{marian_count} ({fp_rate:.0%})")

    # Sentences safe for MarianMT
    print(f"\n5. SENTENCES SAFE FOR MARIANMT (threshold={best_threshold:.2f})")
    safe = [(i, r) for i, r in enumerate(results) if r["complexity"] < best_threshold]
    if not safe:
        print("   None at this threshold.")
    else:
        for i, r in safe:
            qe_str = ""
            if r.get("qe_marian") is not None:
                qe_str = f" QE={r['qe_marian']:.2f}"
            print(f"   #{i + 1} [{r['category']}] c={r['complexity']:.3f}{qe_str}: {r['english'][:60]}")


# ---------------------------------------------------------------------------
# CSV Ingestion (for real session data)
# ---------------------------------------------------------------------------


def load_sentences_from_csv(csv_paths):
    """Load English sentences from A/B metrics CSV files."""
    sentences = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  Skipping missing file: {path}", file=sys.stderr)
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                english = row.get("english", "").strip()
                if english and len(english) > 3:
                    sentences.append((english, "csv"))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for text, cat in sentences:
        if text not in seen:
            seen.add(text)
            unique.append((text, cat))
    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_test(sentences, quick=False):
    """Run the full adaptive model selection test."""
    print(f"\n{'=' * 80}")
    print("ADAPTIVE MODEL SELECTION TEST — P7 Item 6B")
    print(f"{'=' * 80}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Quick mode: {quick}")
    print()

    # Load glossary
    load_glossary()

    # Load models
    print("\nLoading models...")
    ct2_translator, marian_tokenizer = load_ct2_marian()

    gemma_model, gemma_tokenizer = None, None
    if not quick:
        gemma_model, gemma_tokenizer = load_gemma_4b()

    # Run translations
    print(f"\nRunning translations on {len(sentences)} sentences...")
    results = []

    for i, (text, category) in enumerate(sentences):
        cx = compute_complexity(text)

        # MarianMT CT2
        mt_text, mt_ms = translate_ct2(text, ct2_translator, marian_tokenizer)

        # Gemma 4B
        if gemma_model is not None:
            tg_text, tg_ms = translate_gemma(text, gemma_model, gemma_tokenizer)
        else:
            tg_text, tg_ms = "(skipped in quick mode)", 0.0

        # QE comparison
        qe_mt = qe_combined(text, mt_text) if mt_text else None
        qe_tg = qe_combined(text, tg_text) if (tg_text and gemma_model is not None) else None

        result = {
            "english": text,
            "category": category,
            "marian_text": mt_text,
            "marian_ms": mt_ms,
            "gemma_text": tg_text,
            "gemma_ms": tg_ms,
            "qe_marian": qe_mt,
            "qe_gemma": qe_tg,
            **cx,
        }
        results.append(result)

        route = route_model(cx["complexity"])
        save = tg_ms - mt_ms if tg_ms > 0 else 0
        print(
            f"  [{i + 1:>2}/{len(sentences)}] c={cx['complexity']:.3f} "
            f"route={route:>6}  MT={mt_ms:>5.0f}ms  TG={tg_ms:>5.0f}ms  "
            f"save={save:>+5.0f}ms  {text[:50]}"
        )

    # Reports
    show_qe = gemma_model is not None
    print_table(results, show_qe=show_qe)
    if not quick:
        print_translations(results)
    print_recommendation(results)

    # Summary stats
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    mt_lats = [r["marian_ms"] for r in results]
    tg_lats = [r["gemma_ms"] for r in results if r["gemma_ms"] > 0]
    print(
        f"  MarianMT CT2:  avg={sum(mt_lats) / len(mt_lats):.0f}ms  min={min(mt_lats):.0f}ms  max={max(mt_lats):.0f}ms"
    )
    if tg_lats:
        print(
            f"  Gemma 4B:      avg={sum(tg_lats) / len(tg_lats):.0f}ms  "
            f"min={min(tg_lats):.0f}ms  max={max(tg_lats):.0f}ms"
        )
        speedup = (sum(tg_lats) / len(tg_lats)) / (sum(mt_lats) / len(mt_lats))
        print(f"  Gemma/MarianMT ratio: {speedup:.1f}x slower")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test adaptive model selection (MarianMT vs Gemma 4B) for P7 item 6B")
    parser.add_argument(
        "--csv", nargs="+", metavar="CSV", help="CSV file(s) from dry_run_ab.py to use as test sentences"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: skip Gemma 4B, only test complexity heuristics + MarianMT"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35, help="Complexity threshold for routing (default: 0.35)"
    )
    args = parser.parse_args()

    if args.csv:
        sentences = load_sentences_from_csv(args.csv)
        if not sentences:
            print("No sentences found in CSV files.", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(sentences)} unique sentences from CSV files")
    else:
        sentences = TEST_SENTENCES

    run_test(sentences, quick=args.quick)


if __name__ == "__main__":
    main()
