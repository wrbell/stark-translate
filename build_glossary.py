#!/usr/bin/env python3
"""
build_glossary.py — EN→ES Theological Term Glossary Builder

Builds a ~200-500 term glossary for:
1. Soft constraint training (append target terms to source during training)
2. Direct glossary pairs as training examples for TranslateGemma
3. Spot-check evaluation of critical theological terms

Usage:
    python build_glossary.py
    python build_glossary.py --output bible_data/glossary/glossary_pairs.jsonl
    python build_glossary.py --augment bible_data/aligned/verse_pairs_train.jsonl
"""

import argparse
import json
import os

# ---------------------------------------------------------------------------
# Core Glossary (~65 terms — extend to 200-500 as needed)
# ---------------------------------------------------------------------------

THEOLOGICAL_GLOSSARY = {
    # ─── Soteriology (salvation) ───
    "atonement": "expiación",
    "propitiation": "propiciación",
    "redemption": "redención",
    "justification": "justificación",
    "sanctification": "santificación",
    "salvation": "salvación",
    "grace": "gracia",
    "faith": "fe",
    "repentance": "arrepentimiento",
    "forgiveness": "perdón",
    "reconciliation": "reconciliación",
    "regeneration": "regeneración",
    "conversion": "conversión",
    "election": "elección",
    "predestination": "predestinación",
    "original sin": "pecado original",
    "imputation": "imputación",
    "substitutionary atonement": "expiación sustitutoria",
    "born again": "nacido de nuevo",
    "new creation": "nueva creación",
    # ─── Theology proper ───
    "righteousness": "justicia",
    "holiness": "santidad",
    "sovereignty": "soberanía",
    "omnipotence": "omnipotencia",
    "omniscience": "omnisciencia",
    "trinity": "trinidad",
    "incarnation": "encarnación",
    "resurrection": "resurrección",
    "ascension": "ascensión",
    "revelation": "revelación",
    "providence": "providencia",
    "glory": "gloria",
    "mercy": "misericordia",
    "intercession": "intercesión",
    "mediator": "mediador",
    "Savior": "Salvador",
    "Redeemer": "Redentor",
    "Almighty": "Todopoderoso",
    "the Lamb of God": "el Cordero de Dios",
    "the Word of God": "la Palabra de Dios",
    "kingdom of God": "reino de Dios",
    "kingdom of heaven": "reino de los cielos",
    "eternal life": "vida eterna",
    "everlasting": "eterno",
    "sin": "pecado",
    "transgression": "transgresión",
    "iniquity": "iniquidad",
    "wrath": "ira",
    "the cross": "la cruz",
    "crucifixion": "crucifixión",
    "blood of Christ": "sangre de Cristo",
    # ─── Ecclesiology & worship ───
    "covenant": "pacto",  # Protestant convention (vs. "alianza" Catholic)
    "congregation": "congregación",
    "fellowship": "comunión",
    "baptism": "bautismo",
    "communion": "comunión",  # Also "Lord's Supper" = "Cena del Señor"
    "Lord's Supper": "Cena del Señor",
    "breaking of bread": "partimiento del pan",
    "tithe": "diezmo",
    "offering": "ofrenda",
    "worship": "adoración",
    "praise": "alabanza",
    "prayer": "oración",
    "sermon": "sermón",
    "gospel": "evangelio",
    "scripture": "escritura",
    "prophecy": "profecía",
    "parable": "parábola",
    "epistle": "epístola",
    "psalm": "salmo",
    "hymn": "himno",
    "testimony": "testimonio",
    "ministry": "ministerio",
    "disciple": "discípulo",
    "apostle": "apóstol",
    "elder": "anciano",
    "deacon": "diácono",
    "shepherd": "pastor",
    "bishop": "obispo",
    "presbyter": "presbítero",
    "evangelist": "evangelista",
    "missionary": "misionero",
    "preacher": "predicador",
    "teacher": "maestro",
    "prophet": "profeta",
    "priest": "sacerdote",
    "high priest": "sumo sacerdote",
    "Pharisee": "fariseo",
    "Sadducee": "saduceo",
    "scribe": "escriba",
    # ─── Liturgical & worship terms ───
    "benediction": "bendición",
    "doxology": "doxología",
    "amen": "amén",
    "hallelujah": "aleluya",
    "hosanna": "hosanna",
    "invocation": "invocación",
    "supplication": "súplica",
    "thanksgiving": "acción de gracias",
    "confession": "confesión",
    "absolution": "absolución",
    "consecration": "consagración",
    "ordination": "ordenación",
    "laying on of hands": "imposición de manos",
    "anointing": "unción",
    # ─── Common sermon phrases ───
    "let us pray": "oremos",
    "the Word of the Lord": "la Palabra del Señor",
    "praise the Lord": "alaben al Señor",
    "in Jesus' name": "en el nombre de Jesús",
    "Heavenly Father": "Padre celestial",
    "Holy Spirit": "Espíritu Santo",
    "the Son of God": "el Hijo de Dios",
    "beloved brethren": "amados hermanos",
    "brothers and sisters": "hermanos y hermanas",
    "children of God": "hijos de Dios",
    "body of Christ": "cuerpo de Cristo",
    "Word of God": "Palabra de Dios",
    "good news": "buenas nuevas",
    "great commission": "gran comisión",
    "the Lord's prayer": "el Padrenuestro",
    # ─── Eschatology ───
    "rapture": "arrebatamiento",
    "tribulation": "tribulación",
    "millennium": "milenio",
    "judgment": "juicio",
    "second coming": "segunda venida",
    "day of the Lord": "día del Señor",
    "new heaven": "cielo nuevo",
    "new earth": "tierra nueva",
    "throne of God": "trono de Dios",
    "lake of fire": "lago de fuego",
    "Antichrist": "anticristo",
    "abomination of desolation": "abominación desoladora",
    # ─── Bible book names (66 books, EN → ES) ───
    # Old Testament
    "Genesis": "Génesis",
    "Exodus": "Éxodo",
    "Leviticus": "Levítico",
    "Numbers": "Números",
    "Deuteronomy": "Deuteronomio",
    "Joshua": "Josué",
    "Judges": "Jueces",
    "Ruth": "Rut",
    "1 Samuel": "1 Samuel",
    "2 Samuel": "2 Samuel",
    "1 Kings": "1 Reyes",
    "2 Kings": "2 Reyes",
    "1 Chronicles": "1 Crónicas",
    "2 Chronicles": "2 Crónicas",
    "Ezra": "Esdras",
    "Nehemiah": "Nehemías",
    "Esther": "Ester",
    "Job": "Job",
    "Psalms": "Salmos",
    "Proverbs": "Proverbios",
    "Ecclesiastes": "Eclesiastés",
    "Song of Solomon": "Cantares",
    "Isaiah": "Isaías",
    "Jeremiah": "Jeremías",
    "Lamentations": "Lamentaciones",
    "Ezekiel": "Ezequiel",
    "Daniel": "Daniel",
    "Hosea": "Oseas",
    "Joel": "Joel",
    "Amos": "Amós",
    "Obadiah": "Abdías",
    "Jonah": "Jonás",
    "Micah": "Miqueas",
    "Nahum": "Nahúm",
    "Habakkuk": "Habacuc",
    "Zephaniah": "Sofonías",
    "Haggai": "Hageo",
    "Zechariah": "Zacarías",
    "Malachi": "Malaquías",
    # New Testament
    "Matthew": "Mateo",
    "Mark": "Marcos",
    "Luke": "Lucas",
    "John": "Juan",
    "Acts": "Hechos",
    "Romans": "Romanos",
    "1 Corinthians": "1 Corintios",
    "2 Corinthians": "2 Corintios",
    "Galatians": "Gálatas",
    "Ephesians": "Efesios",
    "Philippians": "Filipenses",
    "Colossians": "Colosenses",
    "1 Thessalonians": "1 Tesalonicenses",
    "2 Thessalonians": "2 Tesalonicenses",
    "1 Timothy": "1 Timoteo",
    "2 Timothy": "2 Timoteo",
    "Titus": "Tito",
    "Philemon": "Filemón",
    "Hebrews": "Hebreos",
    "James": "Santiago",
    "1 Peter": "1 Pedro",
    "2 Peter": "2 Pedro",
    "1 John": "1 Juan",
    "2 John": "2 Juan",
    "3 John": "3 Juan",
    "Jude": "Judas",
    "Revelation": "Apocalipsis",
    # ─── Proper names (EN → ES Bible convention) ───
    "James (apostle)": "Jacobo",
    "James (epistle)": "Santiago",
    "Peter": "Pedro",
    "Paul": "Pablo",
    "Moses": "Moisés",
    "Abraham": "Abraham",
    "David": "David",
    "Solomon": "Salomón",
    "Elijah": "Elías",
    "Adam": "Adán",
    "Eve": "Eva",
    "Noah": "Noé",
    "Isaac": "Isaac",
    "Jacob": "Jacob",
    "Joseph": "José",
    "Aaron": "Aarón",
    "Samuel": "Samuel",
    "Elisha": "Eliseo",
    "Gideon": "Gedeón",
    "Samson": "Sansón",
    "Saul": "Saúl",
    "Timothy": "Timoteo",
    "Stephen": "Esteban",
    "Barnabas": "Bernabé",
    "Nicodemus": "Nicodemo",
    "Lazarus": "Lázaro",
    "Mary": "María",
    "Martha": "Marta",
    "Pontius Pilate": "Poncio Pilato",
    "Herod": "Herodes",
    "Pharaoh": "Faraón",
    "Satan": "Satanás",
}


# ---------------------------------------------------------------------------
# Training Pair Generation
# ---------------------------------------------------------------------------


def create_glossary_training_pairs(glossary=None):
    """Convert glossary to short training examples for TranslateGemma.

    Creates two types of pairs per term:
    1. Direct term pair
    2. In-sentence example (contextual usage)
    """
    if glossary is None:
        glossary = THEOLOGICAL_GLOSSARY

    pairs = []
    for en_term, es_term in glossary.items():
        # Direct term pair
        pairs.append({"en": en_term, "es": es_term})
        # In-sentence example
        pairs.append(
            {
                "en": f"The pastor spoke about {en_term} in today's sermon.",
                "es": f"El pastor habló sobre {es_term} en el sermón de hoy.",
            }
        )

    return pairs


# ---------------------------------------------------------------------------
# Soft Constraint Augmentation
# ---------------------------------------------------------------------------


def augment_with_soft_constraints(verse_pairs_path, glossary=None, output_path=None):
    """Soft constraint training (Dinu et al., 2019):

    Append target terminology to source sentences during training.
    Model learns to use glossary terms without hard constraints at inference.

    Format: "source text [GLOSSARY: term1=traducción1, term2=traducción2]"
    """
    if glossary is None:
        glossary = THEOLOGICAL_GLOSSARY

    augmented = []
    with open(verse_pairs_path, encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            en_lower = pair["en"].lower()
            constraints = []

            for en_term, es_term in glossary.items():
                if en_term.lower() in en_lower:
                    constraints.append(f"{en_term}={es_term}")

            if constraints:
                augmented.append(
                    {
                        "en": f"{pair['en']} [GLOSSARY: {', '.join(constraints)}]",
                        "es": pair["es"],
                    }
                )
            else:
                augmented.append(pair)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for p in augmented:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        constrained = sum(1 for p in augmented if "[GLOSSARY:" in p["en"])
        print(f"Augmented {len(augmented)} pairs ({constrained} with glossary constraints)")
        print(f"  Saved to {output_path}")

    return augmented


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_glossary(glossary=None, output_dir="bible_data/glossary"):
    """Export glossary in multiple formats."""
    if glossary is None:
        glossary = THEOLOGICAL_GLOSSARY

    os.makedirs(output_dir, exist_ok=True)

    # 1. Raw glossary JSON
    with open(os.path.join(output_dir, "theological_glossary.json"), "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2, ensure_ascii=False)

    # 2. Training pairs JSONL
    pairs = create_glossary_training_pairs(glossary)
    pairs_path = os.path.join(output_dir, "glossary_pairs.jsonl")
    with open(pairs_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Exported {len(glossary)} terms to {output_dir}/")
    print(f"  theological_glossary.json: {len(glossary)} terms")
    print(f"  glossary_pairs.jsonl: {len(pairs)} training pairs")

    return pairs_path


def main():
    parser = argparse.ArgumentParser(description="Build EN→ES theological glossary for translation fine-tuning")
    parser.add_argument("--output", "-o", default="bible_data/glossary", help="Output directory")
    parser.add_argument("--augment", help="Path to verse pairs JSONL to augment with soft constraints")
    parser.add_argument("--augment-output", help="Output path for augmented pairs (default: <input>_augmented.jsonl)")
    args = parser.parse_args()

    # Export glossary
    export_glossary(output_dir=args.output)

    # Optionally augment verse pairs
    if args.augment:
        aug_output = args.augment_output or args.augment.replace(".jsonl", "_augmented.jsonl")
        augment_with_soft_constraints(args.augment, output_path=aug_output)


if __name__ == "__main__":
    main()
