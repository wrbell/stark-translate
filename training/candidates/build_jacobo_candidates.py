#!/usr/bin/env python3
"""Build original Jacobo/Santiago preference candidates, never approved training data.

These are twenty original teaching sentences in three original framing variants.
They are not quotations, sermon extracts, model predictions, or human references.
The existing canaries/holdouts are read only to exclude overlaps, never as sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.training_preflight import check_training_row, holdout_index, normalized, sha256

# New prose composed for the candidate pool; not copied/paraphrased from holdouts.
PERSON = [
    (
        "our lesson identifies James by his father Zebedee, so the personal name matters",
        "nuestra lección identifica a Jacobo por su padre Zebedeo, así que el nombre personal importa",
    ),
    (
        "the class will distinguish James the son of Zebedee from the title of an epistle",
        "la clase distinguirá a Jacobo, hijo de Zebedeo, del título de una epístola",
    ),
    (
        "this study card names James as John's brother and asks us to identify the person",
        "esta ficha de estudio nombra a Jacobo como hermano de Juan y nos pide identificar a la persona",
    ),
    (
        "the diagram labels James the son of Zebedee beside his brother",
        "el diagrama sitúa el nombre de Jacobo, hijo de Zebedeo, junto al de su hermano",
    ),
    (
        "the teacher's question concerns James the fisherman, not a Bible book title",
        "la pregunta del maestro trata de Jacobo el pescador, no del título de un libro bíblico",
    ),
    (
        "we are preparing a character study of James the son of Zebedee",
        "estamos preparando un estudio sobre la persona de Jacobo, hijo de Zebedeo",
    ),
    ("the family tree lists James as a son of Zebedee", "el árbol genealógico presenta a Jacobo como hijo de Zebedeo"),
    (
        "this exercise asks which James was the brother of the apostle John",
        "este ejercicio pregunta qué Jacobo era hermano del apóstol Juan",
    ),
    (
        "our discussion follows James the son of Zebedee as a person in the Gospel narrative",
        "nuestra conversación sigue a Jacobo, hijo de Zebedeo, como persona en el relato evangélico",
    ),
    (
        "the glossary example refers to James the apostle whose father was Zebedee",
        "el ejemplo del glosario se refiere al apóstol Jacobo, cuyo padre era Zebedeo",
    ),
]
EPISTLE = [
    (
        "the reading plan assigns a chapter from the epistle of James for next week",
        "el plan de lectura asigna un capítulo de la epístola de Santiago para la próxima semana",
    ),
    (
        "our printed handout uses James as the title of the New Testament letter",
        "nuestro folleto impreso usa Santiago como título de la carta del Nuevo Testamento",
    ),
    (
        "the table of contents places the book of James among the letters",
        "el índice sitúa el libro de Santiago entre las cartas",
    ),
    (
        "the speaker will introduce the epistle of James before reading the first chapter",
        "el orador presentará la epístola de Santiago antes de leer el primer capítulo",
    ),
    (
        "this bookmark marks the beginning of the letter of James",
        "este marcador señala el comienzo de la carta de Santiago",
    ),
    (
        "we need the Spanish book title for James on the study schedule",
        "necesitamos el título español del libro de Santiago en el programa de estudio",
    ),
    (
        "the lesson heading names the epistle of James rather than an individual disciple",
        "el encabezado de la lección nombra la epístola de Santiago en lugar de un discípulo en particular",
    ),
    (
        "the reference list points readers to chapter three of the book of James",
        "la lista de referencias remite a los lectores al capítulo tres del libro de Santiago",
    ),
    (
        "the library catalogue has a commentary on the New Testament letter of James",
        "el catálogo de la biblioteca contiene un comentario sobre la carta de Santiago del Nuevo Testamento",
    ),
    (
        "our bilingual slide must display the title of the epistle of James correctly",
        "nuestra diapositiva bilingüe debe mostrar correctamente el título de la epístola de Santiago",
    ),
]
FRAMES = [
    ("For today's workshop, ", "Para el taller de hoy, "),
    ("Please remember that ", "Por favor, recuerden que "),
    ("In this teaching example, ", "En este ejemplo didáctico, "),
]


def build_rows() -> list[dict]:
    rows = []
    for sense, pairs, chosen_name, rejected_name in (
        ("person_son_of_zebedee", PERSON, "Jacobo", "Santiago"),
        ("epistle_title", EPISTLE, "Santiago", "Jacobo"),
    ):
        for core_id, (en, es) in enumerate(pairs, 1):
            for frame_id, (prefix_en, prefix_es) in enumerate(FRAMES, 1):
                source = prefix_en + en + "."
                chosen = prefix_es + es + "."
                rejected = chosen.replace(chosen_name, rejected_name)
                rows.append(
                    {
                        "id": f"jacobo-candidate-{len(rows) + 1:03d}",
                        "prompt": "Translate the following English text to Spanish. Output only the translation, nothing else.\n\n"
                        + source,
                        "chosen": chosen,
                        "rejected": rejected,
                        "source_text": source,
                        "direction": "en2es",
                        "sense": sense,
                        "split": "train",
                        "usage": "unapproved_training_candidate",
                        "approved_for_training": False,
                        "review_status": "unapproved",
                        "provenance": {
                            "origin": "original_synthetic_teaching_prose",
                            "author": "Codex; requires bilingual human review",
                            "created_on": "2026-09-10",
                            "recipe": "training/candidates/build_jacobo_candidates.py",
                            "core_id": f"{sense}-{core_id:02d}",
                            "frame_id": frame_id,
                            "source_text_sha256": hashlib.sha256(source.encode()).hexdigest(),
                            "rejection_method": "replace only the disambiguated Spanish proper name",
                            "evaluation_material_used_as_source": False,
                            "model_inference_or_external_api_used": False,
                        },
                    }
                )
    return rows


def audit_rows(rows: list[dict], index: dict) -> dict:
    seen = set()
    for row in rows:
        key = normalized(row["source_text"])
        if key in seen:
            raise ValueError("Duplicate candidate source")
        seen.add(key)
        # This is an overlap audit, not a promotion: original approval markers stay false.
        for text in (row["source_text"], row["chosen"], row["rejected"]):
            check_training_row({"source_text": text}, index, row["id"])
        if row["chosen"] == row["rejected"]:
            raise ValueError("Degenerate preference triple")
    return {
        "candidate_count": len(rows),
        "sense_counts": dict(Counter(r["sense"] for r in rows)),
        "unique_core_sentences": len({r["provenance"]["core_id"] for r in rows}),
        "exact_normalized_text_overlaps": 0,
        "duplicate_source_texts": 0,
        "holdout_files_checked": index["files"],
        "required_holdouts_missing": index["missing"],
        "holdout_gate": "pending_missing_inputs" if index["missing"] else "passed_for_listed_inputs",
        "human_review": "pending",
        "approved_for_training": False,
        "semantic_overlap_review": "pending; exact matching does not exclude paraphrase or topic overlap",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, action="append")
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Output must be new; do not overwrite candidates, corpus or holdouts")
    rows = build_rows()
    index = holdout_index(args.holdout, allow_missing=True)
    manifest = audit_rows(rows, index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest["candidate_file_sha256"] = sha256(args.output)
    manifest["generator_sha256"] = sha256(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
