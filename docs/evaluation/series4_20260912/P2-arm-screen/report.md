# partial_reuse_screen_series4_20260912

Gates cover every declared clip; p95 claim eligibility is separate.

Nearest-rank percentiles; latency in ms. Unknown routes are counted in all-route gates only.

## Clip A

n by route covers all eligible finals; p50/p95 are all-route silence finals.

| Arm | n Gemma / Marian / unknown | Gemma silence n | Silence p50 | Silence p95 | Gemma silence p95 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| ctl | 159 / 24 / 0 | 45 | 1635.1 | 2398.6 | 2398.6 | — | — | — | — | — | — | — |
| reuse100 | 174 / 9 / 0 | 60 | 1530.3 | 2338.2 | 2300.3 | FAIL | FAIL | PASS | FAIL | PASS | FAIL | PASS |
| reuse300 | 176 / 7 / 0 | 62 | 1633.1 | 2242.7 | 2242.7 | FAIL | FAIL | PASS | FAIL | FAIL | PASS | PASS |

### Identity: p2s0912_A_reuse100_r0 vs p2s0912_A_ctl_r0

Translation share: 0.9661016949152542; English share: 0.9672131147540983; aligned rows: 59; chunk-count difference: 0.

Row 42: {"row_index": 42, "fields": ["spanish_a"], "candidate": {"chunk_id": "43", "spanish_a": "Bueno, ahora", "english": "Um so now"}, "control": {"chunk_id": "43", "spanish_a": "Um, ahora", "english": "Um so now"}}

Row 59: {"row_index": 59, "fields": ["spanish_a"], "candidate": {"chunk_id": "61", "spanish_a": "Eso es eso es um", "english": "That is that is um"}, "control": {"chunk_id": "61", "spanish_a": "Eso es lo que es um", "english": "That is that is um"}}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo era un um", "english": "No, I came on I I rem uh I remember it was a um"}, "control": null}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "58", "spanish_a": "Uh, al igual que el versículo detrás de mí, um Juan tres, dieciséis. Bueno, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Well we've all we all know that verse."}, "control": null}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo que fue un um", "english": "No, I came on I I re uh I remember it was a um"}}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "58", "spanish_a": "Uh, justo como el versículo detrás de mí, um Juan tres, dieciséis. Uh, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Uh we've all we all know that verse."}}

### Identity: p2s0912_A_reuse100_r1 vs p2s0912_A_ctl_r1

Translation share: 0.9491525423728814; English share: 0.9672131147540983; aligned rows: 59; chunk-count difference: 0.

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "10", "spanish_a": "Tan bien, uno de ellos", "english": "So well, one of them"}, "control": {"chunk_id": "10", "spanish_a": "Así que, uno de ellos", "english": "So well, one of them"}}

Row 42: {"row_index": 42, "fields": ["spanish_a"], "candidate": {"chunk_id": "43", "spanish_a": "Bueno, ahora", "english": "Um so now"}, "control": {"chunk_id": "43", "spanish_a": "Um, ahora", "english": "Um so now"}}

Row 59: {"row_index": 59, "fields": ["spanish_a"], "candidate": {"chunk_id": "61", "spanish_a": "Eso es eso es um", "english": "That is that is um"}, "control": {"chunk_id": "61", "spanish_a": "Eso es lo que es um", "english": "That is that is um"}}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo era un um", "english": "No, I came on I I rem uh I remember it was a um"}, "control": null}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "58", "spanish_a": "Uh, al igual que el versículo detrás de mí, um Juan tres, dieciséis. Bueno, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Well we've all we all know that verse."}, "control": null}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo que fue un um", "english": "No, I came on I I re uh I remember it was a um"}}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "58", "spanish_a": "Uh, justo como el versículo detrás de mí, um Juan tres, dieciséis. Uh, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Uh we've all we all know that verse."}}

### Identity: p2s0912_A_reuse100_r2 vs p2s0912_A_ctl_r2

Translation share: 0.9661016949152542; English share: 0.9672131147540983; aligned rows: 59; chunk-count difference: 0.

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "10", "spanish_a": "Tan bien, uno de ellos", "english": "So well, one of them"}, "control": {"chunk_id": "10", "spanish_a": "Así que, uno de ellos", "english": "So well, one of them"}}

Row 59: {"row_index": 59, "fields": ["spanish_a"], "candidate": {"chunk_id": "61", "spanish_a": "Eso es eso es um", "english": "That is that is um"}, "control": {"chunk_id": "61", "spanish_a": "Eso es lo que es um", "english": "That is that is um"}}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo era un um", "english": "No, I came on I I rem uh I remember it was a um"}, "control": null}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "58", "spanish_a": "Uh, al igual que el versículo detrás de mí, um Juan tres, dieciséis. Bueno, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Well we've all we all know that verse."}, "control": null}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo que fue un um", "english": "No, I came on I I re uh I remember it was a um"}}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "58", "spanish_a": "Uh, justo como el versículo detrás de mí, um Juan tres, dieciséis. Uh, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Uh we've all we all know that verse."}}

### Identity: p2s0912_A_reuse300_r0 vs p2s0912_A_ctl_r0

Translation share: 0.9482758620689655; English share: 0.9508196721311475; aligned rows: 58; chunk-count difference: 0.

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "10", "spanish_a": "Tan bien, uno de ellos", "english": "So well, one of them"}, "control": {"chunk_id": "10", "spanish_a": "Así que, uno de ellos", "english": "So well, one of them"}}

Row 42: {"row_index": 42, "fields": ["spanish_a"], "candidate": {"chunk_id": "43", "spanish_a": "Bueno, ahora", "english": "Um so now"}, "control": {"chunk_id": "43", "spanish_a": "Um, ahora", "english": "Um so now"}}

Row 59: {"row_index": 59, "fields": ["spanish_a"], "candidate": {"chunk_id": "61", "spanish_a": "Eso es eso es um", "english": "That is that is um"}, "control": {"chunk_id": "61", "spanish_a": "Eso es lo que es um", "english": "That is that is um"}}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "18", "spanish_a": "Solo quiero que pienses en eso porque tienes a estos dos hombres que son similares pero también diferentes y", "english": "I I just want you to to to think about that because you have these two men who are similar but also different and"}, "control": null}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo era un um", "english": "No, I came on I I rem uh I remember it was a um"}, "control": null}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "58", "spanish_a": "Uh, al igual que el versículo detrás de mí, um Juan tres, dieciséis. Bueno, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Well we've all we all know that verse."}, "control": null}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "18", "spanish_a": "Solo quiero que pienses en eso porque tienes a estos dos hombres que son similares pero también diferentes. Y", "english": "I I just want you to to to think about that because you have these two men who are similar but also different. And"}}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo que fue un um", "english": "No, I came on I I re uh I remember it was a um"}}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "58", "spanish_a": "Uh, justo como el versículo detrás de mí, um Juan tres, dieciséis. Uh, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Uh we've all we all know that verse."}}

### Identity: p2s0912_A_reuse300_r1 vs p2s0912_A_ctl_r1

Translation share: 0.9482758620689655; English share: 0.9508196721311475; aligned rows: 58; chunk-count difference: 0.

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "10", "spanish_a": "Tan bien, uno de ellos", "english": "So well, one of them"}, "control": {"chunk_id": "10", "spanish_a": "Así que, uno de ellos", "english": "So well, one of them"}}

Row 42: {"row_index": 42, "fields": ["spanish_a"], "candidate": {"chunk_id": "43", "spanish_a": "Bueno, ahora", "english": "Um so now"}, "control": {"chunk_id": "43", "spanish_a": "Um, ahora", "english": "Um so now"}}

Row 59: {"row_index": 59, "fields": ["spanish_a"], "candidate": {"chunk_id": "61", "spanish_a": "Eso es eso es um", "english": "That is that is um"}, "control": {"chunk_id": "61", "spanish_a": "Eso es lo que es um", "english": "That is that is um"}}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "18", "spanish_a": "Solo quiero que pienses en eso porque tienes a estos dos hombres que son similares pero también diferentes y", "english": "I I just want you to to to think about that because you have these two men who are similar but also different and"}, "control": null}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo era un um", "english": "No, I came on I I rem uh I remember it was a um"}, "control": null}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "58", "spanish_a": "Uh, al igual que el versículo detrás de mí, um Juan tres, dieciséis. Bueno, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Well we've all we all know that verse."}, "control": null}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "18", "spanish_a": "Solo quiero que pienses en eso porque tienes a estos dos hombres que son similares pero también diferentes. Y", "english": "I I just want you to to to think about that because you have these two men who are similar but also different. And"}}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo que fue un um", "english": "No, I came on I I re uh I remember it was a um"}}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "58", "spanish_a": "Uh, justo como el versículo detrás de mí, um Juan tres, dieciséis. Uh, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Uh we've all we all know that verse."}}

### Identity: p2s0912_A_reuse300_r2 vs p2s0912_A_ctl_r2

Translation share: 0.9482758620689655; English share: 0.9508196721311475; aligned rows: 58; chunk-count difference: 0.

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "10", "spanish_a": "Tan bien, uno de ellos", "english": "So well, one of them"}, "control": {"chunk_id": "10", "spanish_a": "Así que, uno de ellos", "english": "So well, one of them"}}

Row 42: {"row_index": 42, "fields": ["spanish_a"], "candidate": {"chunk_id": "43", "spanish_a": "Bueno, ahora", "english": "Um so now"}, "control": {"chunk_id": "43", "spanish_a": "Um, ahora", "english": "Um so now"}}

Row 59: {"row_index": 59, "fields": ["spanish_a"], "candidate": {"chunk_id": "61", "spanish_a": "Eso es eso es um", "english": "That is that is um"}, "control": {"chunk_id": "61", "spanish_a": "Eso es lo que es um", "english": "That is that is um"}}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "18", "spanish_a": "Solo quiero que pienses en eso porque tienes a estos dos hombres que son similares pero también diferentes y", "english": "I I just want you to to to think about that because you have these two men who are similar but also different and"}, "control": null}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo era un um", "english": "No, I came on I I rem uh I remember it was a um"}, "control": null}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "58", "spanish_a": "Uh, al igual que el versículo detrás de mí, um Juan tres, dieciséis. Bueno, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Well we've all we all know that verse."}, "control": null}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "18", "spanish_a": "Solo quiero que pienses en eso porque tienes a estos dos hombres que son similares pero también diferentes. Y", "english": "I I just want you to to to think about that because you have these two men who are similar but also different. And"}}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo que fue un um", "english": "No, I came on I I re uh I remember it was a um"}}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "58", "spanish_a": "Uh, justo como el versículo detrás de mí, um Juan tres, dieciséis. Uh, todos conocemos ese versículo.", "english": "Uh just like the verse behind me, um John three, sixteen. Uh we've all we all know that verse."}}

## Clip B

n by route covers all eligible finals; p50/p95 are all-route silence finals.

| Arm | n Gemma / Marian / unknown | Gemma silence n | Silence p50 | Silence p95 | Gemma silence p95 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| ctl | 162 / 78 / 0 | 129 | 1468.1 | 2298.0 | 2507.6 | — | — | — | — | — | — | — |
| reuse100 | 217 / 28 / 0 | 184 | 1379.1 | 2160.1 | 2366.4 | FAIL | PASS | PASS | FAIL | FAIL | PASS | PASS |
| reuse300 | 243 / 6 / 0 | 210 | 1328.8 | 1745.9 | 1745.9 | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |

### Identity: p2s0912_B_reuse100_r0 vs p2s0912_B_ctl_r0

Translation share: 0.9014084507042254; English share: 0.8658536585365854; aligned rows: 71; chunk-count difference: 2.

Row 0: {"row_index": 0, "fields": ["spanish_a"], "candidate": {"chunk_id": "2", "spanish_a": "Porque él es un salvador.", "english": "Because he is a savior."}, "control": {"chunk_id": "2", "spanish_a": "Porque es un salvador.", "english": "Because he is a savior."}}

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "11", "spanish_a": "Le encanta ahorrar.", "english": "He loves to save."}, "control": {"chunk_id": "11", "spanish_a": "Le encanta salvar.", "english": "He loves to save."}}

Row 10: {"row_index": 10, "fields": ["spanish_a"], "candidate": {"chunk_id": "12", "spanish_a": "No, piensa en un doctor.", "english": "No, think about a doctor."}, "control": {"chunk_id": "12", "spanish_a": "No, piensa en un médico.", "english": "No, think about a doctor."}}

Row 28: {"row_index": 28, "fields": ["spanish_a"], "candidate": {"chunk_id": "30", "spanish_a": "De lo contrario, no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}, "control": {"chunk_id": "30", "spanish_a": "De lo contrario no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}}

Row 48: {"row_index": 48, "fields": ["spanish_a"], "candidate": {"chunk_id": "50", "spanish_a": "Acercándose a él por la fe.", "english": "Coming to him through faith."}, "control": {"chunk_id": "50", "spanish_a": "Venir a él por la fe.", "english": "Coming to him through faith."}}

Row 50: {"row_index": 50, "fields": ["spanish_a"], "candidate": {"chunk_id": "52", "spanish_a": "Está poniendo nuestra fe", "english": "It's putting our faith"}, "control": {"chunk_id": "52", "spanish_a": "Es poner nuestra fe", "english": "It's putting our faith"}}

Row 76: {"row_index": 76, "fields": ["spanish_a"], "candidate": {"chunk_id": "79", "spanish_a": "Sin nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}, "control": {"chunk_id": "79", "spanish_a": "El pecado nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "3", "spanish_a": "¿Qué hace un salvador? Un salvador salva.", "english": "What does a saviour do? A saviour saves."}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "o haremos algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will w will ever do something to earn us the status of God being willing to save us."}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because He loves to save."}, "control": null}

Row 14: {"row_index": 14, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "16", "spanish_a": "lo que está enseñando", "english": "what it is that he's teaching"}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento. Dios", "english": "But he wants everyone to come to repentance. God"}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "Es un salvador y le encanta salvar.", "english": "He is a saviour and he loves to save."}, "control": null}

Row 63: {"row_index": 63, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra.", "english": "When there was one God who created the heavens and the earth."}, "control": null}

Row 66: {"row_index": 66, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "69", "spanish_a": "Después de algún tiempo, el hombre y la mujer, Adán y Eva, pecaron.", "english": "After some time, man and woman, Adam and Eve, they sinned."}, "control": null}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "78", "spanish_a": "A quien todos están separados debido a su pecado.", "english": "To whom everyone is separated from because of their sin."}, "control": null}

Row 77: {"row_index": 77, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "80", "spanish_a": "fuera de su presencia. E Isaías escribe", "english": "out of his presence. And Isaiah writes"}, "control": null}

Row 78: {"row_index": 78, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "eso", "english": "that"}, "control": null}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "3", "spanish_a": "¿Y qué hace un salvador? Un salvador salva.", "english": "And what does a savior do? A savior saves."}}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "7", "spanish_a": "o haremos alguna vez algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will we will ever do something to earn us the status of God being willing to save us."}}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because he loves to save."}}

Row 18: {"row_index": 18, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento.", "english": "But he wants everyone to come to repentance."}}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "22", "spanish_a": "Él es un salvador y le encanta salvar.", "english": "He is a savior and he loves to save."}}

Row 62: {"row_index": 62, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra", "english": "When there was one God who created the heavens and the earth"}}

Row 65: {"row_index": 65, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "69", "spanish_a": "Después de algún tiempo, el hombre y la mujer, Adán y Eva, pecaron", "english": "After some time, man and woman, Adam and Eve, they sinned"}}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "80", "spanish_a": "Y Isaías escribe", "english": "And Isaiah writes"}}

Row 76: {"row_index": 76, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "81", "spanish_a": "Eso", "english": "That"}}

### Identity: p2s0912_B_reuse100_r1 vs p2s0912_B_ctl_r1

Translation share: 0.9027777777777778; English share: 0.8888888888888888; aligned rows: 72; chunk-count difference: 1.

Row 0: {"row_index": 0, "fields": ["spanish_a"], "candidate": {"chunk_id": "2", "spanish_a": "Porque él es un salvador.", "english": "Because he is a savior."}, "control": {"chunk_id": "2", "spanish_a": "Porque es un salvador.", "english": "Because he is a savior."}}

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "11", "spanish_a": "Le encanta ahorrar.", "english": "He loves to save."}, "control": {"chunk_id": "11", "spanish_a": "Le encanta salvar.", "english": "He loves to save."}}

Row 10: {"row_index": 10, "fields": ["spanish_a"], "candidate": {"chunk_id": "12", "spanish_a": "No, piensa en un doctor.", "english": "No, think about a doctor."}, "control": {"chunk_id": "12", "spanish_a": "No, piensa en un médico.", "english": "No, think about a doctor."}}

Row 28: {"row_index": 28, "fields": ["spanish_a"], "candidate": {"chunk_id": "30", "spanish_a": "De lo contrario, no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}, "control": {"chunk_id": "30", "spanish_a": "De lo contrario no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}}

Row 48: {"row_index": 48, "fields": ["spanish_a"], "candidate": {"chunk_id": "50", "spanish_a": "Acercándose a él por la fe.", "english": "Coming to him through faith."}, "control": {"chunk_id": "50", "spanish_a": "Venir a él por la fe.", "english": "Coming to him through faith."}}

Row 50: {"row_index": 50, "fields": ["spanish_a"], "candidate": {"chunk_id": "52", "spanish_a": "Está poniendo nuestra fe", "english": "It's putting our faith"}, "control": {"chunk_id": "52", "spanish_a": "Es poner nuestra fe", "english": "It's putting our faith"}}

Row 75: {"row_index": 75, "fields": ["spanish_a"], "candidate": {"chunk_id": "79", "spanish_a": "Sin nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}, "control": {"chunk_id": "79", "spanish_a": "El pecado nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "3", "spanish_a": "¿Qué hace un salvador? Un salvador salva.", "english": "What does a saviour do? A saviour saves."}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "o haremos algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will w will ever do something to earn us the status of God being willing to save us."}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because He loves to save."}, "control": null}

Row 14: {"row_index": 14, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "16", "spanish_a": "lo que está enseñando", "english": "what it is that he's teaching"}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento. Dios", "english": "But he wants everyone to come to repentance. God"}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "Es un salvador y le encanta salvar.", "english": "He is a saviour and he loves to save."}, "control": null}

Row 63: {"row_index": 63, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra.", "english": "When there was one God who created the heavens and the earth."}, "control": null}

Row 76: {"row_index": 76, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "80", "spanish_a": "fuera de su presencia. E Isaías escribe", "english": "out of his presence. And Isaiah writes"}, "control": null}

Row 77: {"row_index": 77, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "eso", "english": "that"}, "control": null}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "3", "spanish_a": "¿Y qué hace un salvador? Un salvador salva.", "english": "And what does a savior do? A savior saves."}}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "7", "spanish_a": "o haremos alguna vez algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will we will ever do something to earn us the status of God being willing to save us."}}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because he loves to save."}}

Row 18: {"row_index": 18, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento.", "english": "But he wants everyone to come to repentance."}}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "22", "spanish_a": "Él es un salvador y le encanta salvar.", "english": "He is a savior and he loves to save."}}

Row 62: {"row_index": 62, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra", "english": "When there was one God who created the heavens and the earth"}}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "80", "spanish_a": "Y Isaías escribe", "english": "And Isaiah writes"}}

Row 76: {"row_index": 76, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "81", "spanish_a": "Eso", "english": "That"}}

### Identity: p2s0912_B_reuse100_r2 vs p2s0912_B_ctl_r2

Translation share: 0.9014084507042254; English share: 0.8658536585365854; aligned rows: 71; chunk-count difference: 2.

Row 0: {"row_index": 0, "fields": ["spanish_a"], "candidate": {"chunk_id": "2", "spanish_a": "Porque él es un salvador.", "english": "Because he is a savior."}, "control": {"chunk_id": "2", "spanish_a": "Porque es un salvador.", "english": "Because he is a savior."}}

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "11", "spanish_a": "Le encanta ahorrar.", "english": "He loves to save."}, "control": {"chunk_id": "11", "spanish_a": "Le encanta salvar.", "english": "He loves to save."}}

Row 10: {"row_index": 10, "fields": ["spanish_a"], "candidate": {"chunk_id": "12", "spanish_a": "No, piensa en un doctor.", "english": "No, think about a doctor."}, "control": {"chunk_id": "12", "spanish_a": "No, piensa en un médico.", "english": "No, think about a doctor."}}

Row 28: {"row_index": 28, "fields": ["spanish_a"], "candidate": {"chunk_id": "30", "spanish_a": "De lo contrario, no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}, "control": {"chunk_id": "30", "spanish_a": "De lo contrario no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}}

Row 48: {"row_index": 48, "fields": ["spanish_a"], "candidate": {"chunk_id": "50", "spanish_a": "Acercándose a él por la fe.", "english": "Coming to him through faith."}, "control": {"chunk_id": "50", "spanish_a": "Venir a él por la fe.", "english": "Coming to him through faith."}}

Row 50: {"row_index": 50, "fields": ["spanish_a"], "candidate": {"chunk_id": "52", "spanish_a": "Está poniendo nuestra fe", "english": "It's putting our faith"}, "control": {"chunk_id": "52", "spanish_a": "Es poner nuestra fe", "english": "It's putting our faith"}}

Row 76: {"row_index": 76, "fields": ["spanish_a"], "candidate": {"chunk_id": "79", "spanish_a": "Sin nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}, "control": {"chunk_id": "79", "spanish_a": "El pecado nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "3", "spanish_a": "¿Qué hace un salvador? Un salvador salva.", "english": "What does a saviour do? A saviour saves."}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "o haremos algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will w will ever do something to earn us the status of God being willing to save us."}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because He loves to save."}, "control": null}

Row 14: {"row_index": 14, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "16", "spanish_a": "lo que está enseñando", "english": "what it is that he's teaching"}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento. Dios", "english": "But he wants everyone to come to repentance. God"}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "Es un salvador y le encanta salvar.", "english": "He is a saviour and he loves to save."}, "control": null}

Row 63: {"row_index": 63, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra.", "english": "When there was one God who created the heavens and the earth."}, "control": null}

Row 66: {"row_index": 66, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "69", "spanish_a": "Después de algún tiempo, el hombre y la mujer, Adán y Eva, pecaron.", "english": "After some time, man and woman, Adam and Eve, they sinned."}, "control": null}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "78", "spanish_a": "A quien todos están separados debido a su pecado.", "english": "To whom everyone is separated from because of their sin."}, "control": null}

Row 77: {"row_index": 77, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "80", "spanish_a": "fuera de su presencia. E Isaías escribe", "english": "out of his presence. And Isaiah writes"}, "control": null}

Row 78: {"row_index": 78, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "eso", "english": "that"}, "control": null}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "3", "spanish_a": "¿Y qué hace un salvador? Un salvador salva.", "english": "And what does a savior do? A savior saves."}}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "7", "spanish_a": "o haremos alguna vez algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will we will ever do something to earn us the status of God being willing to save us."}}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because he loves to save."}}

Row 18: {"row_index": 18, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento.", "english": "But he wants everyone to come to repentance."}}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "22", "spanish_a": "Él es un salvador y le encanta salvar.", "english": "He is a savior and he loves to save."}}

Row 62: {"row_index": 62, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra", "english": "When there was one God who created the heavens and the earth"}}

Row 65: {"row_index": 65, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "69", "spanish_a": "Después de algún tiempo, el hombre y la mujer, Adán y Eva, pecaron", "english": "After some time, man and woman, Adam and Eve, they sinned"}}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "80", "spanish_a": "Y Isaías escribe", "english": "And Isaiah writes"}}

Row 76: {"row_index": 76, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "81", "spanish_a": "Eso", "english": "That"}}

### Identity: p2s0912_B_reuse300_r0 vs p2s0912_B_ctl_r0

Translation share: 0.8571428571428571; English share: 0.7590361445783133; aligned rows: 63; chunk-count difference: 3.

Row 0: {"row_index": 0, "fields": ["spanish_a"], "candidate": {"chunk_id": "2", "spanish_a": "Porque él es un salvador.", "english": "Because he is a savior."}, "control": {"chunk_id": "2", "spanish_a": "Porque es un salvador.", "english": "Because he is a savior."}}

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "11", "spanish_a": "Le encanta ahorrar.", "english": "He loves to save."}, "control": {"chunk_id": "11", "spanish_a": "Le encanta salvar.", "english": "He loves to save."}}

Row 10: {"row_index": 10, "fields": ["spanish_a"], "candidate": {"chunk_id": "12", "spanish_a": "No, piensa en un doctor.", "english": "No, think about a doctor."}, "control": {"chunk_id": "12", "spanish_a": "No, piensa en un médico.", "english": "No, think about a doctor."}}

Row 28: {"row_index": 28, "fields": ["spanish_a"], "candidate": {"chunk_id": "30", "spanish_a": "De lo contrario, no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}, "control": {"chunk_id": "30", "spanish_a": "De lo contrario no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}}

Row 48: {"row_index": 48, "fields": ["spanish_a"], "candidate": {"chunk_id": "50", "spanish_a": "Acercándose a él por la fe.", "english": "Coming to him through faith."}, "control": {"chunk_id": "50", "spanish_a": "Venir a él por la fe.", "english": "Coming to him through faith."}}

Row 50: {"row_index": 50, "fields": ["spanish_a"], "candidate": {"chunk_id": "52", "spanish_a": "Está poniendo nuestra fe", "english": "It's putting our faith"}, "control": {"chunk_id": "52", "spanish_a": "Es poner nuestra fe", "english": "It's putting our faith"}}

Row 57: {"row_index": 57, "fields": ["spanish_a"], "candidate": {"chunk_id": "59", "spanish_a": "Él dice", "english": "He says"}, "control": {"chunk_id": "59", "spanish_a": "Dice", "english": "He says"}}

Row 70: {"row_index": 70, "fields": ["spanish_a"], "candidate": {"chunk_id": "73", "spanish_a": "Fueron expulsados del jardín.", "english": "They were driven out of the garden."}, "control": {"chunk_id": "73", "spanish_a": "Los echaron del jardín.", "english": "They were driven out of the garden."}}

Row 76: {"row_index": 76, "fields": ["spanish_a"], "candidate": {"chunk_id": "79", "spanish_a": "Sin nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}, "control": {"chunk_id": "79", "spanish_a": "El pecado nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "3", "spanish_a": "¿Qué hace un salvador? Un salvador salva.", "english": "What does a saviour do? A saviour saves."}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "o haremos algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will w will ever do something to earn us the status of God being willing to save us."}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because He loves to save."}, "control": null}

Row 14: {"row_index": 14, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "16", "spanish_a": "lo que está enseñando", "english": "what it is that he's teaching"}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento. Dios", "english": "But he wants everyone to come to repentance. God"}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "Es un salvador y le encanta salvar.", "english": "He is a saviour and he loves to save."}, "control": null}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "25", "spanish_a": "Sabes, no te confundas y pienses que por lo tanto vas y te salvas.", "english": "you know, don't get it twisted and think that therefore you go and save yourself."}, "control": null}

Row 24: {"row_index": 24, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "26", "spanish_a": "Dios es el Salvador. Él hace toda la salvación.", "english": "God is the Saviour. He does all the saving."}, "control": null}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "31", "spanish_a": "Si la salvación fuera algo que pudiéramos hacer, entonces no necesitaríamos descanso.", "english": "If salvation was something we could do, then we wouldn't be in need of rest."}, "control": null}

Row 30: {"row_index": 30, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "32", "spanish_a": "Así que para mi tercer y último punto,", "english": "So for my third and final point,"}, "control": null}

Row 47: {"row_index": 47, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "49", "spanish_a": "En realidad está llegando a la fe.", "english": "It's actually coming to faith."}, "control": null}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "54", "spanish_a": "Hace años y años", "english": "Years and years ago"}, "control": null}

Row 63: {"row_index": 63, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra.", "english": "When there was one God who created the heavens and the earth."}, "control": null}

Row 66: {"row_index": 66, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "69", "spanish_a": "Después de algún tiempo, el hombre y la mujer, Adán y Eva, pecaron.", "english": "After some time, man and woman, Adam and Eve, they sinned."}, "control": null}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "78", "spanish_a": "A quien todos están separados debido a su pecado.", "english": "To whom everyone is separated from because of their sin."}, "control": null}

Row 77: {"row_index": 77, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "80", "spanish_a": "fuera de su presencia. E Isaías escribe", "english": "out of his presence. And Isaiah writes"}, "control": null}

Row 78: {"row_index": 78, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "eso", "english": "that"}, "control": null}

Row 79: {"row_index": 79, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "82", "spanish_a": "No dice que sea", "english": "It doesn't say it's"}, "control": null}

Row 81: {"row_index": 81, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "84", "spanish_a": "Él dice tu pecado", "english": "He says your sin"}, "control": null}

Row 82: {"row_index": 82, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "85", "spanish_a": "Separar", "english": "Separate"}, "control": null}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "3", "spanish_a": "¿Y qué hace un salvador? Un salvador salva.", "english": "And what does a savior do? A savior saves."}}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "7", "spanish_a": "o haremos alguna vez algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will we will ever do something to earn us the status of God being willing to save us."}}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because he loves to save."}}

Row 18: {"row_index": 18, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento.", "english": "But he wants everyone to come to repentance."}}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "22", "spanish_a": "Él es un salvador y le encanta salvar.", "english": "He is a savior and he loves to save."}}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "25", "spanish_a": "Sabes, no te confundas y pienses que por lo tanto vas y te salvas.", "english": "You know, don't get it twisted and think that therefore you go and save yourself."}}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "26", "spanish_a": "Dios es el salvador. Él hace toda la salvación.", "english": "God is the savior. He does all the saving."}}

Row 28: {"row_index": 28, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "31", "spanish_a": "Si la salvación fuera algo que pudiéramos hacer, entonces no necesitaríamos ser rescatados.", "english": "If salvation was something we could do, then we wouldn't be in need of rescue."}}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "32", "spanish_a": "Y así, para mi tercer y último punto,", "english": "And so for my third and final point,"}}

Row 46: {"row_index": 46, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "49", "spanish_a": "Pero en realidad está llegando a la fe", "english": "But it's actually coming to faith"}}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "54", "spanish_a": "Hace años y años.", "english": "Years and years ago."}}

Row 62: {"row_index": 62, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra", "english": "When there was one God who created the heavens and the earth"}}

Row 65: {"row_index": 65, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "69", "spanish_a": "Después de algún tiempo, el hombre y la mujer, Adán y Eva, pecaron", "english": "After some time, man and woman, Adam and Eve, they sinned"}}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "80", "spanish_a": "Y Isaías escribe", "english": "And Isaiah writes"}}

Row 76: {"row_index": 76, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "81", "spanish_a": "Eso", "english": "That"}}

Row 77: {"row_index": 77, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "82", "spanish_a": "No dice que sea una buena idea.", "english": "He doesn't say it's a good idea."}}

Row 79: {"row_index": 79, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "84", "spanish_a": "Dice,", "english": "He says,"}}

### Identity: p2s0912_B_reuse300_r1 vs p2s0912_B_ctl_r1

Translation share: 0.8615384615384616; English share: 0.7831325301204819; aligned rows: 65; chunk-count difference: 3.

Row 0: {"row_index": 0, "fields": ["spanish_a"], "candidate": {"chunk_id": "2", "spanish_a": "Porque él es un salvador.", "english": "Because he is a savior."}, "control": {"chunk_id": "2", "spanish_a": "Porque es un salvador.", "english": "Because he is a savior."}}

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "11", "spanish_a": "Le encanta ahorrar.", "english": "He loves to save."}, "control": {"chunk_id": "11", "spanish_a": "Le encanta salvar.", "english": "He loves to save."}}

Row 10: {"row_index": 10, "fields": ["spanish_a"], "candidate": {"chunk_id": "12", "spanish_a": "No, piensa en un doctor.", "english": "No, think about a doctor."}, "control": {"chunk_id": "12", "spanish_a": "No, piensa en un médico.", "english": "No, think about a doctor."}}

Row 28: {"row_index": 28, "fields": ["spanish_a"], "candidate": {"chunk_id": "30", "spanish_a": "De lo contrario, no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}, "control": {"chunk_id": "30", "spanish_a": "De lo contrario no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}}

Row 48: {"row_index": 48, "fields": ["spanish_a"], "candidate": {"chunk_id": "50", "spanish_a": "Acercándose a él por la fe.", "english": "Coming to him through faith."}, "control": {"chunk_id": "50", "spanish_a": "Venir a él por la fe.", "english": "Coming to him through faith."}}

Row 50: {"row_index": 50, "fields": ["spanish_a"], "candidate": {"chunk_id": "52", "spanish_a": "Está poniendo nuestra fe", "english": "It's putting our faith"}, "control": {"chunk_id": "52", "spanish_a": "Es poner nuestra fe", "english": "It's putting our faith"}}

Row 57: {"row_index": 57, "fields": ["spanish_a"], "candidate": {"chunk_id": "59", "spanish_a": "Él dice", "english": "He says"}, "control": {"chunk_id": "59", "spanish_a": "Dice", "english": "He says"}}

Row 70: {"row_index": 70, "fields": ["spanish_a"], "candidate": {"chunk_id": "73", "spanish_a": "Fueron expulsados del jardín.", "english": "They were driven out of the garden."}, "control": {"chunk_id": "73", "spanish_a": "Los echaron del jardín.", "english": "They were driven out of the garden."}}

Row 76: {"row_index": 76, "fields": ["spanish_a"], "candidate": {"chunk_id": "79", "spanish_a": "Sin nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}, "control": {"chunk_id": "79", "spanish_a": "El pecado nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "3", "spanish_a": "¿Qué hace un salvador? Un salvador salva.", "english": "What does a saviour do? A saviour saves."}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "o haremos algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will w will ever do something to earn us the status of God being willing to save us."}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because He loves to save."}, "control": null}

Row 14: {"row_index": 14, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "16", "spanish_a": "lo que está enseñando", "english": "what it is that he's teaching"}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento. Dios", "english": "But he wants everyone to come to repentance. God"}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "Es un salvador y le encanta salvar.", "english": "He is a saviour and he loves to save."}, "control": null}

Row 24: {"row_index": 24, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "26", "spanish_a": "Dios es el Salvador. Él hace toda la salvación.", "english": "God is the Saviour. He does all the saving."}, "control": null}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "31", "spanish_a": "Si la salvación fuera algo que pudiéramos hacer, entonces no necesitaríamos descanso.", "english": "If salvation was something we could do, then we wouldn't be in need of rest."}, "control": null}

Row 30: {"row_index": 30, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "32", "spanish_a": "Así que para mi tercer y último punto,", "english": "So for my third and final point,"}, "control": null}

Row 47: {"row_index": 47, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "49", "spanish_a": "En realidad está llegando a la fe.", "english": "It's actually coming to faith."}, "control": null}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "54", "spanish_a": "Hace años y años", "english": "Years and years ago"}, "control": null}

Row 63: {"row_index": 63, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra.", "english": "When there was one God who created the heavens and the earth."}, "control": null}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "78", "spanish_a": "A quien todos están separados debido a su pecado.", "english": "To whom everyone is separated from because of their sin."}, "control": null}

Row 77: {"row_index": 77, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "80", "spanish_a": "fuera de su presencia. E Isaías escribe", "english": "out of his presence. And Isaiah writes"}, "control": null}

Row 78: {"row_index": 78, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "eso", "english": "that"}, "control": null}

Row 79: {"row_index": 79, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "82", "spanish_a": "No dice que sea", "english": "It doesn't say it's"}, "control": null}

Row 81: {"row_index": 81, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "84", "spanish_a": "Él dice tu pecado", "english": "He says your sin"}, "control": null}

Row 82: {"row_index": 82, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "85", "spanish_a": "Separar", "english": "Separate"}, "control": null}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "3", "spanish_a": "¿Y qué hace un salvador? Un salvador salva.", "english": "And what does a savior do? A savior saves."}}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "7", "spanish_a": "o haremos alguna vez algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will we will ever do something to earn us the status of God being willing to save us."}}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because he loves to save."}}

Row 18: {"row_index": 18, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento.", "english": "But he wants everyone to come to repentance."}}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "22", "spanish_a": "Él es un salvador y le encanta salvar.", "english": "He is a savior and he loves to save."}}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "26", "spanish_a": "Dios es el salvador. Él hace toda la salvación.", "english": "God is the savior. He does all the saving."}}

Row 28: {"row_index": 28, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "31", "spanish_a": "Si la salvación fuera algo que pudiéramos hacer, entonces no necesitaríamos ser rescatados.", "english": "If salvation was something we could do, then we wouldn't be in need of rescue."}}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "32", "spanish_a": "Y así, para mi tercer y último punto,", "english": "And so for my third and final point,"}}

Row 46: {"row_index": 46, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "49", "spanish_a": "Pero en realidad está llegando a la fe", "english": "But it's actually coming to faith"}}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "54", "spanish_a": "Hace años y años.", "english": "Years and years ago."}}

Row 62: {"row_index": 62, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra", "english": "When there was one God who created the heavens and the earth"}}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "80", "spanish_a": "Y Isaías escribe", "english": "And Isaiah writes"}}

Row 76: {"row_index": 76, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "81", "spanish_a": "Eso", "english": "That"}}

Row 77: {"row_index": 77, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "82", "spanish_a": "No dice que sea una buena idea.", "english": "He doesn't say it's a good idea."}}

Row 79: {"row_index": 79, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "84", "spanish_a": "Dice,", "english": "He says,"}}

### Identity: p2s0912_B_reuse300_r2 vs p2s0912_B_ctl_r2

Translation share: 0.8571428571428571; English share: 0.7590361445783133; aligned rows: 63; chunk-count difference: 3.

Row 0: {"row_index": 0, "fields": ["spanish_a"], "candidate": {"chunk_id": "2", "spanish_a": "Porque él es un salvador.", "english": "Because he is a savior."}, "control": {"chunk_id": "2", "spanish_a": "Porque es un salvador.", "english": "Because he is a savior."}}

Row 9: {"row_index": 9, "fields": ["spanish_a"], "candidate": {"chunk_id": "11", "spanish_a": "Le encanta ahorrar.", "english": "He loves to save."}, "control": {"chunk_id": "11", "spanish_a": "Le encanta salvar.", "english": "He loves to save."}}

Row 10: {"row_index": 10, "fields": ["spanish_a"], "candidate": {"chunk_id": "12", "spanish_a": "No, piensa en un doctor.", "english": "No, think about a doctor."}, "control": {"chunk_id": "12", "spanish_a": "No, piensa en un médico.", "english": "No, think about a doctor."}}

Row 28: {"row_index": 28, "fields": ["spanish_a"], "candidate": {"chunk_id": "30", "spanish_a": "De lo contrario, no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}, "control": {"chunk_id": "30", "spanish_a": "De lo contrario no necesitaríamos ser salvados.", "english": "Otherwise we wouldn't need to be saved."}}

Row 48: {"row_index": 48, "fields": ["spanish_a"], "candidate": {"chunk_id": "50", "spanish_a": "Acercándose a él por la fe.", "english": "Coming to him through faith."}, "control": {"chunk_id": "50", "spanish_a": "Venir a él por la fe.", "english": "Coming to him through faith."}}

Row 50: {"row_index": 50, "fields": ["spanish_a"], "candidate": {"chunk_id": "52", "spanish_a": "Está poniendo nuestra fe", "english": "It's putting our faith"}, "control": {"chunk_id": "52", "spanish_a": "Es poner nuestra fe", "english": "It's putting our faith"}}

Row 57: {"row_index": 57, "fields": ["spanish_a"], "candidate": {"chunk_id": "59", "spanish_a": "Él dice", "english": "He says"}, "control": {"chunk_id": "59", "spanish_a": "Dice", "english": "He says"}}

Row 70: {"row_index": 70, "fields": ["spanish_a"], "candidate": {"chunk_id": "73", "spanish_a": "Fueron expulsados del jardín.", "english": "They were driven out of the garden."}, "control": {"chunk_id": "73", "spanish_a": "Los echaron del jardín.", "english": "They were driven out of the garden."}}

Row 76: {"row_index": 76, "fields": ["spanish_a"], "candidate": {"chunk_id": "79", "spanish_a": "Sin nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}, "control": {"chunk_id": "79", "spanish_a": "El pecado nos separó, nos alejó.", "english": "Sin separated us, it drove us away."}}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "3", "spanish_a": "¿Qué hace un salvador? Un salvador salva.", "english": "What does a saviour do? A saviour saves."}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "o haremos algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will w will ever do something to earn us the status of God being willing to save us."}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because He loves to save."}, "control": null}

Row 14: {"row_index": 14, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "16", "spanish_a": "lo que está enseñando", "english": "what it is that he's teaching"}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento. Dios", "english": "But he wants everyone to come to repentance. God"}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "Es un salvador y le encanta salvar.", "english": "He is a saviour and he loves to save."}, "control": null}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "25", "spanish_a": "Sabes, no te confundas y pienses que por lo tanto vas y te salvas.", "english": "you know, don't get it twisted and think that therefore you go and save yourself."}, "control": null}

Row 24: {"row_index": 24, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "26", "spanish_a": "Dios es el Salvador. Él hace toda la salvación.", "english": "God is the Saviour. He does all the saving."}, "control": null}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "31", "spanish_a": "Si la salvación fuera algo que pudiéramos hacer, entonces no necesitaríamos descanso.", "english": "If salvation was something we could do, then we wouldn't be in need of rest."}, "control": null}

Row 30: {"row_index": 30, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "32", "spanish_a": "Así que para mi tercer y último punto,", "english": "So for my third and final point,"}, "control": null}

Row 47: {"row_index": 47, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "49", "spanish_a": "En realidad está llegando a la fe.", "english": "It's actually coming to faith."}, "control": null}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "54", "spanish_a": "Hace años y años", "english": "Years and years ago"}, "control": null}

Row 63: {"row_index": 63, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra.", "english": "When there was one God who created the heavens and the earth."}, "control": null}

Row 66: {"row_index": 66, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "69", "spanish_a": "Después de algún tiempo, el hombre y la mujer, Adán y Eva, pecaron.", "english": "After some time, man and woman, Adam and Eve, they sinned."}, "control": null}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "78", "spanish_a": "A quien todos están separados debido a su pecado.", "english": "To whom everyone is separated from because of their sin."}, "control": null}

Row 77: {"row_index": 77, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "80", "spanish_a": "fuera de su presencia. E Isaías escribe", "english": "out of his presence. And Isaiah writes"}, "control": null}

Row 78: {"row_index": 78, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "eso", "english": "that"}, "control": null}

Row 79: {"row_index": 79, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "82", "spanish_a": "No dice que sea", "english": "It doesn't say it's"}, "control": null}

Row 81: {"row_index": 81, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "84", "spanish_a": "Él dice tu pecado", "english": "He says your sin"}, "control": null}

Row 82: {"row_index": 82, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "85", "spanish_a": "Separar", "english": "Separate"}, "control": null}

Row 1: {"row_index": 1, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "3", "spanish_a": "¿Y qué hace un salvador? Un salvador salva.", "english": "And what does a savior do? A savior saves."}}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "7", "spanish_a": "o haremos alguna vez algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will we will ever do something to earn us the status of God being willing to save us."}}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "8", "spanish_a": "No, Dios salva porque ama salvar.", "english": "No, God saves because he loves to save."}}

Row 18: {"row_index": 18, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "21", "spanish_a": "Pero él quiere que todos vengan al arrepentimiento.", "english": "But he wants everyone to come to repentance."}}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "22", "spanish_a": "Él es un salvador y le encanta salvar.", "english": "He is a savior and he loves to save."}}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "25", "spanish_a": "Sabes, no te confundas y pienses que por lo tanto vas y te salvas.", "english": "You know, don't get it twisted and think that therefore you go and save yourself."}}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "26", "spanish_a": "Dios es el salvador. Él hace toda la salvación.", "english": "God is the savior. He does all the saving."}}

Row 28: {"row_index": 28, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "31", "spanish_a": "Si la salvación fuera algo que pudiéramos hacer, entonces no necesitaríamos ser rescatados.", "english": "If salvation was something we could do, then we wouldn't be in need of rescue."}}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "32", "spanish_a": "Y así, para mi tercer y último punto,", "english": "And so for my third and final point,"}}

Row 46: {"row_index": 46, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "49", "spanish_a": "Pero en realidad está llegando a la fe", "english": "But it's actually coming to faith"}}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "54", "spanish_a": "Hace años y años.", "english": "Years and years ago."}}

Row 62: {"row_index": 62, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "66", "spanish_a": "Cuando había un solo Dios que creó los cielos y la tierra", "english": "When there was one God who created the heavens and the earth"}}

Row 65: {"row_index": 65, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "69", "spanish_a": "Después de algún tiempo, el hombre y la mujer, Adán y Eva, pecaron", "english": "After some time, man and woman, Adam and Eve, they sinned"}}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "80", "spanish_a": "Y Isaías escribe", "english": "And Isaiah writes"}}

Row 76: {"row_index": 76, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "81", "spanish_a": "Eso", "english": "That"}}

Row 77: {"row_index": 77, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "82", "spanish_a": "No dice que sea una buena idea.", "english": "He doesn't say it's a good idea."}}

Row 79: {"row_index": 79, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "84", "spanish_a": "Dice,", "english": "He says,"}}

## Experiment counters

| Run | Source | Counters |
| --- | --- | --- |
| p2s0912_A_ctl_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 5, "partial_suppressed_translation_running": 0} |
| p2s0912_A_reuse100_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 1, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 18, "partial_suppressed_translation_running": 0} |
| p2s0912_A_reuse300_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 3, "partial_suppressed_translation_running": 0} |
| p2s0912_A_reuse300_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 12, "partial_suppressed_translation_running": 0} |
| p2s0912_A_reuse100_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 16, "partial_suppressed_translation_running": 0} |
| p2s0912_A_ctl_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 6, "partial_suppressed_translation_running": 0} |
| p2s0912_A_ctl_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 1, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 16, "partial_suppressed_translation_running": 0} |
| p2s0912_A_reuse300_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 26, "partial_suppressed_translation_running": 0} |
| p2s0912_A_reuse100_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 10, "partial_suppressed_translation_running": 0} |
| p2s0912_B_ctl_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 9, "partial_suppressed_translation_running": 0} |
| p2s0912_B_reuse100_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 5, "partial_suppressed_translation_running": 0} |
| p2s0912_B_reuse300_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 6, "partial_suppressed_translation_running": 0} |
| p2s0912_B_reuse300_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 1, "partial_suppressed_in_flight": 4, "partial_suppressed_translation_running": 0} |
| p2s0912_B_reuse100_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 15, "partial_suppressed_translation_running": 0} |
| p2s0912_B_ctl_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 2, "partial_suppressed_in_flight": 12, "partial_suppressed_translation_running": 0} |
| p2s0912_B_ctl_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 1, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 1, "partial_suppressed_in_flight": 13, "partial_suppressed_translation_running": 0} |
| p2s0912_B_reuse300_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_discarded_utterance": 1, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 3, "partial_suppressed_translation_running": 0} |
| p2s0912_B_reuse100_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_discarded_utterance": 1, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 6, "partial_suppressed_translation_running": 0} |

## Outcomes

ctl p95_claim_eligible: false — screen without p95 claim.

reuse100: REJECTED

Failing gates: A:G1, A:G2, A:G4, A:G6, B:G1, B:G4, B:G5.

reuse100 p95_claim_eligible: true.

reuse300: REJECTED

Failing gates: A:G1, A:G2, A:G4, A:G5, B:G4.

reuse300 p95_claim_eligible: true.
