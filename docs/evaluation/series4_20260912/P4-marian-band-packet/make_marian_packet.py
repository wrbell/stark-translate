"""P4: reference-free blinded review packet — live Gemma E4B finals vs offline Marian CT2 on the same English (8–12 words).

Sources: the twelve series-3 control replays (ts0911/lb0912 × clips A/B × 3 repeats). Only Gemma-routed finals whose
translated English (post-correction) has 8–12 words; deduplicated by normalized English (repeats replay the same clips).
Hypothesis "gemma_live" = the stored live output; "marian_ct2" = the production Marian CT2 CPU engine run offline here.
No reference exists for this band; the packet asks for meaning preference only. Labels are blinded with a private key.
"""
from __future__ import annotations
import hashlib, hmac, json, os, re, secrets, sys, time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
ROOT = Path("/Users/willem/Code/vibes/SRTranslate"); sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from tools import mac_followup_quality as quality
from tools.mac_bilingual_review import fresh_review, review_schema, text_hash, literal_block
M = ROOT / "metrics"; OUT = ROOT / ".cache/series4-20260912/P4"
CLIPS = {"A": "Gospel_Message__12_14_25__5D2rOMvkwrk", "B": "Gospel_Message__2_8_26__BpIELcuh8O0"}
TAGS = [f"{s}_{c}_ctl_r{r}" for s in ("ts0911", "lb0912") for c in "AB" for r in range(3)]
ENGINES = ("gemma_live", "marian_ct2"); MAX_ITEMS = 120; BAND = (8, 12)

def norm(t): return " ".join(re.sub(r"[^\w\s']", " ", (t or "").casefold()).split())
def blind_labels(seed, item_id, manifest_hash):
    order = ENGINES if hmac.digest(seed.encode(), f"{manifest_hash}:{item_id}".encode(), "sha256")[0] % 2 == 0 else ENGINES[::-1]
    return dict(zip(("A", "B"), order, strict=True))

def collect():
    groups, sources = defaultdict(list), []
    for tag in TAGS:
        clip = tag.split("_")[1]; diag = M / f"diagnostics_{tag}_ctl_{CLIPS[clip]}.jsonl"
        rows = [json.loads(l) for l in open(diag)]
        sources.append({"tag": tag, "file": str(diag.relative_to(ROOT)), "sha256": quality.digest(diag), "model_a": next((r.get("model_a") for r in rows if r.get("model_a")), None)})
        for r in rows:
            if r.get("english") is None or r.get("chunk_id") is None or r.get("final_translation_route") != "gemma": continue
            src = (r.get("corrected_english") or r.get("english") or "").strip(); hyp = (r.get("spanish_gemma") or "").strip()
            wc = len(src.split())
            if not (BAND[0] <= wc <= BAND[1]) or not hyp: continue
            groups[norm(src)].append({"tag": tag, "chunk_id": r["chunk_id"], "utterance_id": r.get("utterance_id"), "source": src, "spanish_gemma": hyp, "stt_confidence": r.get("stt_confidence"), "endpoint_reason": r.get("endpoint_reason"), "clip": clip})
    return groups, sources

def main():
    groups, sources = collect()
    keys = sorted(groups, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    print(f"{len(groups)} unique 8–12-word Gemma-routed finals across {len(TAGS)} runs; taking {min(len(keys), MAX_ITEMS)}")
    keys = keys[:MAX_ITEMS]
    from settings import settings
    from engines.factory import create_translation_engine
    from engines.model_paths import resolve_marian_ct2
    ct2_path = resolve_marian_ct2("en-es", managed_only=True)
    engine = create_translation_engine(ct2_path=ct2_path, backend="cpu", engine_type="marian", model_id=settings.translation.marian_model,
        marian_backend=settings.translation.marian_backend, device="cpu", intra_threads=settings.translation.marian_intra_threads,
        compute_type=settings.translation.marian_compute_type, max_new_tokens=settings.translation.marian_max_new_tokens,
        warmup_passes=settings.translation.marian_warmup_passes, source_lang="en", target_lang="es")
    t0 = time.time(); engine.load(); load_s = time.time() - t0
    marian_files = sorted(Path(ct2_path).rglob("*")) if ct2_path else []
    marian_model = {"ct2_path": str(ct2_path), "backend": getattr(engine, "backend", None), "files": [{"file": p.name, "sha256": quality.digest(p)} for p in marian_files if p.is_file()], "load_s": round(load_s, 2),
                    "settings": {k: getattr(settings.translation, "marian_" + k) for k in ("model", "backend", "compute_type", "intra_threads", "max_new_tokens")}}
    glossary = quality.read_json(ROOT / "bible_data/glossary/tier2_master.json")
    seed = secrets.token_hex(24)
    provenance = {"sources": sources, "marian": {k: v for k, v in marian_model.items() if k != "load_s"}, "band_words": BAND, "route": "gemma", "dedupe": "normalized english", "selection": "sha256 order of normalized english, first %d" % MAX_ITEMS}
    manifest_hash = quality.json_hash(provenance)
    packet_id = hmac.new(seed.encode(), manifest_hash.encode(), "sha256").hexdigest()[:24]
    cases, mapping, auto = [], [], []
    for k in keys:
        occ = groups[k]; first = occ[0]; src = first["source"]
        t0 = time.perf_counter(); res = engine.translate(src); ms = (time.perf_counter() - t0) * 1000
        hyps = {"gemma_live": first["spanish_gemma"], "marian_ct2": res.text.strip()}
        drift = len({o["spanish_gemma"] for o in occ}) > 1
        item_id = "live:" + text_hash(src)[:16]
        labels = blind_labels(seed, item_id, manifest_hash)
        case_id = hmac.new(seed.encode(), f"{packet_id}:{item_id}:0".encode(), "sha256").hexdigest()[:24]
        hypotheses = {label: hyps[e] for label, e in labels.items()}
        case = {"schema_version": 1, "case_id": case_id, "source_id": item_id, "repeat": 0, "source_lang": "en", "target_lang": "es",
                "partition": "series3_control_replays", "category": "live_sermon_final_8_12_words", "source": src, "reference": None,
                "reference_provenance": "live replay final; no human reference", "required_substrings": [], "source_text_sha256": text_hash(src),
                "reference_text_sha256": None, "source_audio_sha256": None, "sentence_id": None, "word_count": len(src.split()),
                "endpoint_reason": first["endpoint_reason"], "occurrences": len(occ),
                "input_sha256": quality.json_hash({"source": src, "item_id": item_id}), "hypotheses": hypotheses,
                "hypothesis_sha256": {l: text_hash(t) for l, t in hypotheses.items()}, "outputs_differ": hypotheses["A"] != hypotheses["B"],
                "repeat_output_drift": {l: (drift if e == "gemma_live" else False) for l, e in labels.items()},
                "human_approved_locally": False, "training_eligible": False}
        case["immutable_case_sha256"] = quality.json_hash(case); case["review"] = fresh_review(); cases.append(case)
        mapping.append({"case_id": case_id, "source_id": item_id, "labels": {l: {"engine": e} for l, e in labels.items()},
                        "live_rows": [{k2: o[k2] for k2 in ("tag", "chunk_id", "utterance_id", "stt_confidence", "spanish_gemma")} for o in occ], "marian_ms": round(ms, 1)})
        terms = quality.terms_in(src, "en", glossary)
        auto.append({"case_id": case_id, "word_count": len(src.split()), "chrf_marian_vs_gemma": round(quality.chrf_score(quality.chrf_counts(hyps["gemma_live"], hyps["marian_ct2"])), 1),
                     "chrf_gemma_vs_marian": round(quality.chrf_score(quality.chrf_counts(hyps["marian_ct2"], hyps["gemma_live"])), 1),
                     "glossary_terms": terms, "glossary_hits": {e: sum(1 for t in terms if quality.contains_term(hyps[e], glossary[t])) for e in ENGINES},
                     "identical": hyps["gemma_live"] == hyps["marian_ct2"], "marian_ms": round(ms, 1), "gemma_len": len(hyps["gemma_live"].split()), "marian_len": len(hyps["marian_ct2"].split())})
    counts = {"cases": len(cases), "unique_sources": len(cases), "changed_cases": sum(c["outputs_differ"] for c in cases), "sources_with_repeat_drift": sum(any(c["repeat_output_drift"].values()) for c in cases),
              "by_word_count": {str(n): sum(1 for c in cases if c["word_count"] == n) for n in range(BAND[0], BAND[1] + 1)}}
    packet = {"schema_version": 1, "packet_id": packet_id, "generated_at": datetime.now(UTC).isoformat(), "kind": "reference_free_meaning_preference", "provenance": provenance, "provenance_sha256": manifest_hash, "counts": counts, "cases": cases,
              "instructions": "Both hypotheses translate the same live English final; no reference exists. Mark meaning errors per hypothesis, then a preference. Automatic scores are withheld from this packet."}
    out = OUT / "packet"; out.mkdir(exist_ok=False)
    quality.save(out / "packet.json", packet, exclusive=True)
    with (out / "annotations.jsonl").open("x", encoding="utf-8") as h:
        for row in cases:
            h.write(json.dumps({k: row[k] for k in ("schema_version", "case_id", "immutable_case_sha256", "human_approved_locally", "training_eligible", "review")}, ensure_ascii=False) + "\n")
    lines = ["# Blinded bilingual translation review — live sermon finals, 8–12 words (reference-free)", "", f"Packet `{packet_id}` — **unreviewed**.", "",
             "A/B labels are fixed per source and may change for another source. Keep the private mapping key separate. No model scores or timing information is shown. No reference translation exists: judge meaning preservation against the English source and record a preference.", "",
             f"{counts['cases']} cases ({counts['changed_cases']} with different A/B text; {counts['sources_with_repeat_drift']} sources whose live output drifted across repeated replays). Word counts: " + ", ".join(f"{k}: {v}" for k, v in counts["by_word_count"].items()) + ".", "",
             "Sources are live replay finals of recorded Stark Road sermons (English STT after correction), not public text. Use `annotations.jsonl` with `review.schema.json`; preserve case IDs and immutable hashes. No training eligibility is granted.", ""]
    for row in cases:
        lines += [f"### {row['case_id']} — en → es", "", f"Source ID: `{row['source_id']}`. Words: {row['word_count']}. A/B text differs: {row['outputs_differ']}.", "", "**Source**", "", literal_block(row["source"])]
        for label, text in row["hypotheses"].items(): lines += ["", f"**{label}**", "", literal_block(text)]
        lines += ["", "Meaning error A / B: unset. Terminology preference: unset. Overall preference: unset. Approval: false.", ""]
    (out / "packet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    quality.save(out / "review.schema.json", review_schema(), exclusive=True)
    inventory = [{"file": p.name, "sha256": quality.digest(p), "size_bytes": p.stat().st_size} for p in sorted(out.iterdir())]
    key = {"packet_id": packet_id, "seed": seed, "provenance_sha256": manifest_hash, "mapping": mapping, "reviewer_artifacts": inventory, "generated_at": packet["generated_at"]}
    fd = os.open(OUT / "private-key.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as h: h.write(json.dumps(key, indent=2, ensure_ascii=False) + "\n")
    quality.save(out / "packet-index.json", {"schema_version": 1, "completed": True, "packet_id": packet_id, "files": inventory, "private_key_sha256": quality.digest(OUT / "private-key.json"), "status": "unreviewed", "human_approved_locally": False, "training_eligible": False}, exclusive=True)
    # automatic tripwire (unblinded aggregate, kept outside the reviewer packet)
    n = len(auto)
    agg = {"n": n, "identical_outputs": sum(a["identical"] for a in auto), "chrf_marian_vs_gemma_p50": quality.distribution([a["chrf_marian_vs_gemma"] for a in auto])["p50"],
           "items_with_glossary_terms": sum(1 for a in auto if a["glossary_terms"]), "glossary_term_instances": sum(len(a["glossary_terms"]) for a in auto),
           "glossary_hit_rate": {e: (round(100 * sum(a["glossary_hits"][e] for a in auto) / max(1, sum(len(a["glossary_terms"]) for a in auto)), 1)) for e in ENGINES},
           "marian_ms": quality.distribution([a["marian_ms"] for a in auto]), "gemma_len_p50": quality.distribution([a["gemma_len"] for a in auto])["p50"], "marian_len_p50": quality.distribution([a["marian_len"] for a in auto])["p50"],
           "marian_model": marian_model, "glossary": {"file": "bible_data/glossary/tier2_master.json", "sha256": quality.digest(ROOT / "bible_data/glossary/tier2_master.json")}}
    quality.save(OUT / "automatic.json", {"aggregate": agg, "items": auto})
    print(json.dumps({k: v for k, v in agg.items() if k != "marian_model"}, indent=1)); print("counts", counts); print("packet", packet_id)

if __name__ == "__main__":
    main()
