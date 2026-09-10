# E4B and E2B: fixed-reference development comparison

All 12 runs completed: three alternating repeats per model and direction, 50 public development references per direction, plus all 18 existing EN→ES lexical canaries in each English run. There were no empty outputs, failed items or exhausted output budgets. Texts were unchanged across repeats.

These are isolated translation calls with identical source transcripts and prompt settings, including thinking and drafting disabled. They exclude STT, VAD, capture, live queues and browser rendering. No model default or human approval changes follow.

| Direction | E4B public-text p50 range | E2B public-text p50 range | E4B chrF | E2B chrF | E2B − E4B chrF |
|---|---:|---:|---:|---:|---:|
| EN→ES | 1214–1243 ms | 721–746 ms | 57.027 | 56.799 | -0.227 |
| ES→EN | 993–1030 ms | 623–654 ms | 61.343 | 60.252 | -1.090 |

chrF measures character overlap with one upstream reference, not a percentage of correct meaning. Its implementation and normalization are frozen in the [JSON report](quality/translation-development-report.json). Model load time, individual call p95, separate RSS/Metal peaks, revisions and complete source hashes remain in the per-run files. E2B is faster here, with lower reference overlap and two fewer passing lexical canaries. Bilingual meaning review is pending.

## All 18 canaries

Exact first-repeat outputs follow; all three repeats matched. Pass/fail uses the existing case-insensitive substring contract, not a new semantic judgment.

| ID | Source | Required substrings | E4B output / lexical result | E2B output / lexical result |
|---|---|---|---|---|
| canary_00 | The atonement of Christ reconciles us to God. | expiación, cristo, dios | La expiación de Cristo nos reconcilia con Dios. — pass | La expiación de Cristo nos reconcilia con Dios. — pass |
| canary_01 | James wrote about faith and works. | santiago, fe, obras | James escribió sobre fe y obras. — fail | James escribió sobre la fe y las obras. — fail |
| canary_02 | The propitiation for our sins was the blood of Christ. | propiciación, pecados, sangre, cristo | La propiciación por nuestros pecados fue la sangre de Cristo. — pass | La propiciación por nuestros pecados fue la sangre de Cristo. — pass |
| canary_03 | The breaking of bread is a solemn remembrance. | partimiento, pan | El partir del pan es un solemne recuerdo. — fail | Romper el pan es un solemne recuerdo. — fail |
| canary_04 | Paul wrote to the Corinthians about the resurrection. | pablo, corintios, resurrección | Pablo escribió a los corintios sobre la resurrección. — pass | Pablo escribió a los corintios sobre la resurrección. — pass |
| canary_05 | Justification is by faith alone, not by works of the law. | justificación, fe | La justificación es por la fe solamente, no por las obras de la ley. — pass | La justificación es por fe solamente, no por obras de la ley. — pass |
| canary_06 | The Holy Spirit convicts the world of sin and righteousness. | espíritu, pecado | El Espíritu Santo convence al mundo de pecado y de justicia. — pass | El Espíritu Santo convence al mundo del pecado y de la justicia. — pass |
| canary_07 | Grace and mercy meet at the cross of our Lord Jesus Christ. | gracia, misericordia, cristo | La gracia y la misericordia se encuentran en la cruz de nuestro Señor Jesucristo. — pass | La gracia y la misericordia se encuentran en la cruz de nuestro Señor Jesucristo. — pass |
| canary_08 | We gather before the mercy seat in prayer. | propiciatorio | Nos reunimos ante el trono de la misericordia en oración. — fail | Nos reunimos ante el asiento de la misericordia en oración. — fail |
| canary_09 | The table of the Lord is spread for His people. | mesa, señor | La mesa del Señor está extendida para Su pueblo. — pass | La mesa del Señor está extendida para su pueblo. — pass |
| canary_10 | We are gathered to Thy Name on the first day of the week. | nombre | Nos reunimos en Tu Nombre el primer día de la semana. — pass | Nos reunimos en Tu Nombre el primer día de la semana. — pass |
| canary_11 | A little while, and we shall see the Lord. | poco, señor | En poco tiempo, veremos al Señor. — pass | Un poco más, y veremos al Señor. — pass |
| canary_12 | Nothing but the blood of Jesus can wash away my sin. | sangre, jesús | Nada sino la sangre de Jesús puede lavar mi pecado. — pass | Nada sino la sangre de Jesús puede lavar mi pecado. — pass |
| canary_13 | Christ is the Surety of a better covenant. | fiador, pacto | Cristo es la garantía de un pacto mejor. — fail | Cristo es la garantía de un mejor pacto. — fail |
| canary_14 | This do in remembrance of Me at His table. | memoria, mesa | Esto se hace en memoria de Mí en su mesa. — pass | Esto es en recuerdo de Mí en Su mesa. — fail |
| canary_15 | At the mercy seat we remember His propitiation for our sins. | propiciatorio, propiciación | En el asiento de la misericordia recordamos Su propiciación por nuestros pecados. — fail | En el asiento de la misericordia recordamos su propiciación por nuestros pecados. — fail |
| canary_16 | Yet there is room at the gospel feast for the weary soul. | lugar, evangelio | Sin embargo, hay lugar en el banquete del evangelio para el alma cansada. — pass | Sin embargo, hay espacio en el banquete del evangelio para el alma cansada. — fail |
| canary_17 | We come as pilgrims to Calvary and the cross. | peregrin, calvario | Venimos como peregrinos a Calvario y a la cruz. — pass | Venimos como peregrinos a Calvario y a la cruz. — pass |

Totals: E4B 13/18; E2B 11/18. These reused canaries are development probes, not untouched confirmation.

## Changed public examples

The first four differing examples by source ID in each direction are shown deterministically, without selecting for either model. References are upstream public annotations, not local bilingual approval.

### EN→ES

37/50 public outputs differ between models; this is a text-difference count, not an error count.

**fleurs-en-development-1514-1250175024747413359**

- Source: When the fighting ceased after the wounded were transported to the hospital, about 40 of the other remaining inmates stayed in the yard and refused to return to their cells.
- Public reference: Una vez que cesaron los enfrentamientos, luego de que se trasladaran al hospital a los heridos, cerca de 40 de los prisioneros restantes permanecieron en el patio, resistiéndose a ir a sus celdas.
- E4B: Cuando los combates cesaron después de que los heridos fueran transportados al hospital, unos 40 de los otros reclusos restantes se quedaron en el patio y se negaron a volver a sus celdas.
- E2B: Cuando el combate cesó después de que los heridos fueran transportados al hospital, alrededor de 40 de los demás reclusos restantes se quedaron en el patio y se negaron a regresar a sus celdas.

**fleurs-en-development-1518-10807778671969097912**

- Source: A former Filipino policeman has kept Hong Kong tourists hostage by hijacking their bus in Manila, the capital of the Philippines.
- Public reference: En Manila, capital de Filipinas, un ex agente policial de ese país secuestró un autobús y mantuvo cautivos a turistas que venían de Hong Kong.
- E4B: Un ex policía filipino ha tomado como rehenes a turistas de Hong Kong al secuestrar su autobús en Manila, la capital de Filipinas.
- E2B: Un expolicía filipino ha mantenido como rehenes a turistas de Hong Kong secuestrando su autobús en Manila, la capital de Filipinas.

**fleurs-en-development-1519-786465049904960526**

- Source: In many cases, enrolling on a gap-year course abroad can actually improve your chances of moving into higher education back in your home country.
- Public reference: En muchos casos, inscribirse en un programa de estudios en el extranjero durante un año sabático puede, en efecto, incrementar las posibilidades de acceder a la educación superior una vez se regresa al país de origen.
- E4B: En muchos casos, inscribirse en un curso de año sabático en el extranjero puede mejorar en realidad tus posibilidades de ingresar a la educación superior en tu país de origen.
- E2B: En muchos casos, inscribirse en un curso de año sabático en el extranjero puede mejorar realmente tus posibilidades de pasar a la educación superior en tu país de origen.

**fleurs-en-development-1522-1307190059726935196**

- Source: Technology offers the solution with virtual field trips. Students can look at museum artifacts, visit an aquarium, or admire beautiful art while sitting with their class.
- Public reference: La solución viene de la mano de la tecnología, a través de las excursiones virtuales. Los alumnos pueden observar los objetos exhibidos en un museo, visitar un acuario o contemplar hermosas obras de arte sentados con su clase.
- E4B: La tecnología ofrece la solución con viajes virtuales. Los estudiantes pueden ver artefactos de museos, visitar un acuario o admirar bellas obras de arte mientras están sentados en clase.
- E2B: La tecnología ofrece la solución con visitas virtuales a campo. Los estudiantes pueden ver artefactos de museos, visitar un acuario o admirar arte hermoso mientras están sentados con su clase.

### ES→EN

39/50 public outputs differ between models; this is a text-difference count, not an error count.

**fleurs-es-development-1515-16255658476875547265**

- Source: En cuatro años, es el quinto CEP de Martelly.
- Public reference: It is Martelly's fifth CEP in four years.
- E4B: In four years, it is Martelly's fifth CEP.
- E2B: In four years, it is the fifth CEP of Martelly.

**fleurs-es-development-1518-18251544358446811373**

- Source: En Manila, capital de Filipinas, un ex agente policial de ese país secuestró un autobús y mantuvo cautivos a turistas que venían de Hong Kong.
- Public reference: A former Filipino policeman has kept Hong Kong tourists hostage by hijacking their bus in Manila, the capital of the Philippines.
- E4B: In Manila, the capital of the Philippines, a former police agent from that country hijacked a bus and held captive tourists from Hong Kong.
- E2B: In Manila, the capital of the Philippines, a former police agent from that country kidnapped a bus and held hostage tourists who were coming from Hong Kong.

**fleurs-es-development-1524-9449585527367687175**

- Source: Son muchas sus consecuencias políticas y sociales, tales como el uso del sistema métrico, el desplazamiento del absolutismo hacia el republicanismo, el nacionalismo y la convicción de que el país pertenece al pueblo, y no a un solo soberano.
- Public reference: There are a lot of social and political effects such as the use of metric system, a shift from absolutism to republicanism, nationalism and the belief the country belongs to the people not to one sole ruler.
- E4B: Its political and social consequences are many, such as the use of the metric system, the shift from absolutism to republicanism, nationalism, and the conviction that the country belongs to the people, and not to a single sovereign.
- E2B: They have many political and social consequences, such as the use of the metric system, the shift from absolutism to republicanism, nationalism, and the conviction that the country belongs to the people and not to a single sovereign.

**fleurs-es-development-1528-14516235058824705075**

- Source: Algunos festivales cuentan con áreas especiales de acampe, destinadas a los grupos familiares con hijos pequeños.
- Public reference: Some festivals have special camping areas for families with young children.
- E4B: Some festivals have special camping areas designated for families with small children.
- E2B: Some festivals have special camping areas, intended for families with young children.

## Blinded review preparation

A separate packet contains 354 cases (118 sources × three repeats), including 261 cases with changed model text. No source changed across repeats. Ratings and approvals are blank/false, and training eligibility is false. The shuffled model key is kept outside the reviewer directory and outside version control. No reviewer has been contacted or approval inferred.

The portable reviewer ZIP is retained locally under `.cache/mac-en-es-closeout/quality/blinded-development-reviewer.zip`; supply only this ZIP to reviewers. The model-labeled report above must be withheld during a blinded review.
