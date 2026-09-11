# tail_screen_series3_20260912

Gates cover every declared clip; p95 claim eligibility is separate.

Nearest-rank percentiles; latency in ms. Unknown routes are counted in all-route gates only.

## Clip A

n by route covers all eligible finals; p50/p95 are all-route silence finals.

| Arm | n Gemma / Marian / unknown | Gemma silence n | Silence p50 | Silence p95 | Gemma silence p95 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| ctl | 159 / 24 / 0 | 45 | 1650.1 | 2477.3 | 2481.5 | — | — | — | — | — | — | — |
| marian_threads_2 | 159 / 24 / 0 | 45 | 1618.5 | 2779.1 | 2875.0 | FAIL | FAIL | PASS | PASS | FAIL | PASS | PASS |
| max_utterance_6 | 183 / 48 / 0 | 36 | 1476.2 | 2684.7 | 3182.8 | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL |
| partial_recheck_translation | 159 / 24 / 0 | 45 | 1592.2 | 2247.8 | 2263.0 | FAIL | PASS | PASS | PASS | FAIL | PASS | PASS |

### Identity: lb0912_A_marian_threads_2_r0 vs lb0912_A_ctl_r0

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

### Identity: lb0912_A_marian_threads_2_r1 vs lb0912_A_ctl_r1

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

### Identity: lb0912_A_marian_threads_2_r2 vs lb0912_A_ctl_r2

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

### Identity: lb0912_A_max_utterance_6_r0 vs lb0912_A_ctl_r0

Translation share: 1.0; English share: 0.24675324675324675; aligned rows: 19; chunk-count difference: 16.

Row 2: {"row_index": 2, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "3", "spanish_a": "uh al menos hasta este momento sus vidas uh", "english": "uh at least up until this moment their their lives uh"}, "control": null}

Row 3: {"row_index": 3, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "4", "spanish_a": "terminaron igual, pero cada uno tenía un destino diferente.", "english": "uh ended the same, but each one had a different destination."}, "control": null}

Row 4: {"row_index": 4, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "5", "spanish_a": "Um y podemos y podemos ver eso. Eh uno de ellos eh e el el el criminal a la derecha incluso dice", "english": "Um and you can and we can see that. Uh one of them uh e the the the criminal on on the right even says"}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "6", "spanish_a": "uh este hombre no ha hecho nada malo. Y uh él sabía que este hombre era", "english": "uh this man has done nothing wrong. And uh he knew that this man was"}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "fue Cristo. Él sabía que este hombre estaba allí", "english": "was Christ. He knew that this man was there"}, "control": null}

Row 7: {"row_index": 7, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "para morir por nuestros pecados, tus pecados y tus pecados. Y así esta noche uno de", "english": "to uh die for our sins, your s um my sins and your sins. And so tonight uh one of"}, "control": null}

Row 10: {"row_index": 10, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "12", "spanish_a": "Pero esta noche nuestro eh si no estás salvado, eh", "english": "But tonight our uh if you're not saved, uh"}, "control": null}

Row 11: {"row_index": 11, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "13", "spanish_a": "cuál donde dónde será tu destino. Así que uno uno de ellos", "english": "which where w where will your destination be. So one one of them"}, "control": null}

Row 13: {"row_index": 13, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "15", "spanish_a": "Ninguno de estos tipos respetó la ley, pero uno de ellos se despertó con Dios y el otro se despertó con", "english": "Um d neither of these guys kept the law, but one of them uh woke up with God and the other uh woke up with"}, "control": null}

Row 14: {"row_index": 14, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "16", "spanish_a": "el diablo. Quiero preguntar de nuevo, ¿estás salvado? Um", "english": "the devil. I want to ask again, are you saved? Um"}, "control": null}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "19", "spanish_a": "uh la forma más brutal uh en que un hombre uh cualquier persona puede uh puede ser ejecutado. Y sin embargo, estos tipos eran ambos", "english": "uh the most brutal way uh a man uh an any person can uh can be put to death. And yet these these guys were both"}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Solo quiero que pienses en eso, porque tienes a estos dos hombres que son", "english": "I I just want you to to to think about that, because you have these two men who are"}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "similar pero también diferente. Y", "english": "similar but also different. And"}, "control": null}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "24", "spanish_a": "¿Estás salvado? Y quizás estés pensando, Bueno, yo no soy como estos dos hombres. Bueno, tú tú lo eres. Lo eres. Um", "english": "Are you saved? And you might be thinking, Well, I'm not like these two men. Well you you are. You are. Um"}, "control": null}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "25", "spanish_a": "Tengas ganas o no, todos somos como estos dos hombres. Obviamente, sabes, espero que nadie haya", "english": "Whether you want to, uh admit it or not, we we are all like these two men. Obviously, you know, I hope anyone hasn't no"}, "control": null}

Row 25: {"row_index": 25, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "28", "spanish_a": "Pero espero que nadie haya, ya sabes, cometido asesinato. Ya sabes. De nuevo, como dije, no importa qué tipo de pecado hayas cometido, todos hemos dicho", "english": "But I hope no one has, you know, committed murder. You know. Again, like I said, it doesn't matter what kind of sin you've committed, we've all said"}, "control": null}

Row 26: {"row_index": 26, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "29", "spanish_a": "Ya sea tomar algo que no deberías o, ya sabes, tal vez robar dulces", "english": "Whether that's taking something that you shouldn't have or uh you know maybe stealing candy"}, "control": null}

Row 27: {"row_index": 27, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "30", "spanish_a": "O um eh mentirle a tu mamá cada vez que te dice que hagas algo y tú dices que lo haces. Um eso es que nosotros nosotros", "english": "Or um uh lying to your mom whenever she tells you to do something and you say you do it. Um that is we we"}, "control": null}

Row 28: {"row_index": 28, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "32", "spanish_a": "Um y estos dos y y y estos dos estaban pagando por los pecados que eh habían cometido.", "english": "Um and these two and and and these two were repaying for the sins that uh they had committed."}, "control": null}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "33", "spanish_a": "Pero", "english": "But"}, "control": null}

Row 30: {"row_index": 30, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "34", "spanish_a": "Um ahora", "english": "Um now"}, "control": null}

Row 31: {"row_index": 31, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "35", "spanish_a": "Eh, déjame preguntarte esto. ¿Deberías morir por los pecados que hemos cometido?", "english": "Uh let me ask you this. Should you die for the should we should me and you should we die for the sins that we have uh committed?"}, "control": null}

Row 32: {"row_index": 32, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "36", "spanish_a": "Y la respuesta a eso es sí. Pero", "english": "And the answer to that is is yes. But"}, "control": null}

Row 36: {"row_index": 36, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "40", "spanish_a": "Vamos a vamos a sacar uno con eh eh bueno, quiero comparar a nosotros ahora con estos dos criminales. Entonces, ¿eres", "english": "let's let's take out one w uh uh well I wanna compare us now to to these two criminals. So are you"}, "control": null}

Row 37: {"row_index": 37, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "41", "spanish_a": "Así que", "english": "So"}, "control": null}

Row 38: {"row_index": 38, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "42", "spanish_a": "Bueno, puedes, ya sabes, con el crecimiento, ya sabes, puedes, puedes, puedes decir que he venido a la iglesia consistentemente.", "english": "Uh so you can you know, with gro growing up, you know, you can you you you can you know say that well I've I've come to church consistently. I"}, "control": null}

Row 39: {"row_index": 39, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "43", "spanish_a": "Sabes, he estado he estado bajo el evangelio eh yo me senté bajo el evangelio. No soy tan malo. Pero", "english": "You know, I've been o I've been underneath the gospel uh I I sat underneath the gospel. I'm not all that bad. But"}, "control": null}

Row 40: {"row_index": 40, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "44", "spanish_a": "Sí, eso es bueno, pero a menos que no conozcas a Cristo como tu Salvador, no entrarás al cielo.", "english": "Yes, that's good, but unless you don't know Christ as your Savior, you're not getting into heaven."}, "control": null}

Row 41: {"row_index": 41, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "45", "spanish_a": "Um y yo no quiero sonar", "english": "Um and I don't mean to sound"}, "control": null}

Row 42: {"row_index": 42, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "46", "spanish_a": "Ya sabes, um", "english": "You know, um"}, "control": null}

Row 43: {"row_index": 43, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "47", "spanish_a": "directo cuando yo lo veo, no quiero sonar frívolo cuando digo eso, pero esa es la realidad de", "english": "blunt when I s oh I I don't mean to sound flippant when I say that, but that is the reality of"}, "control": null}

Row 44: {"row_index": 44, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "48", "spanish_a": "de qué um de", "english": "of what um of"}, "control": null}

Row 45: {"row_index": 45, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "49", "spanish_a": "de nuestra de tu situación esta noche si no estás salvo. Um si no conoces a Cristo como tu Salvador, entonces eh", "english": "of our of your situation tonight if you aren't saved. Um if you do not know Christ as your Savior, then uh"}, "control": null}

Row 46: {"row_index": 46, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "50", "spanish_a": "Tú, eh, como estos dos hombres, eh, tendrás dos opciones. Puedes o bien eh", "english": "You uh just like these two men, uh you will have two options. You can either uh"}, "control": null}

Row 47: {"row_index": 47, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "51", "spanish_a": "O despiertas en el cielo o puedes despertar en el infierno. Um y", "english": "Uh wake up in heaven or you can wake up in hell. Um and"}, "control": null}

Row 48: {"row_index": 48, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "52", "spanish_a": "Tienes esa elección de libre albedrío esta noche, pero eh", "english": "You have that free will choice tonight, but uh"}, "control": null}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "53", "spanish_a": "uh esa es una decisión que tienes que tomar.", "english": "uh that is a decision that you have to make."}, "control": null}

Row 50: {"row_index": 50, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "54", "spanish_a": "y eh Dios no tomará esa decisión por ti. Él quiere estar en tu vida, pero tienes que aceptarlo y", "english": "and uh God won't make that decision for you. He wants to be in your life, but you have to accept him and"}, "control": null}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "55", "spanish_a": "te das cuenta de que sin él, uh tú", "english": "realize that without him, uh you"}, "control": null}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "56", "spanish_a": "no irás al cielo y tú estás en y estás y tienes que darte cuenta de que estás perdido en tus pecados esta noche.", "english": "uh won't be getting into heaven and you are in and you're s and and you have to realize that you're lost in your sins tonight."}, "control": null}

Row 54: {"row_index": 54, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "58", "spanish_a": "Um", "english": "Um"}, "control": null}

Row 55: {"row_index": 55, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "59", "spanish_a": "uh como estos dos criminales um tú tú tendrás uh tienes dos opciones cuando cuando se trata", "english": "uh just like these two criminals um you you will uh you have two options when when it comes"}, "control": null}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "60", "spanish_a": "Sí.", "english": "Yeah."}, "control": null}

Row 57: {"row_index": 57, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "61", "spanish_a": "donde estará tu destino eterno. Uno de estos escuchó eh uno de estos eh criminales escuchó um verdaderamente", "english": "where your eternal destination will will be. One of the one of these heard uh one of these uh criminals heard um truly"}, "control": null}

Row 58: {"row_index": 58, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "63", "spanish_a": "Uh lo siento, de verdad te digo, hoy estarás conmigo en el paraíso y el otro escuchó, departamento apártate de mí porque nunca supe.", "english": "Uh sorry, truly I say to you, today you'll be with me in paradise and the other one heard, Department depart from me for I never knew."}, "control": null}

Row 59: {"row_index": 59, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "64", "spanish_a": "Así que, para mí personalmente, crecer eso me asustó.", "english": "So um for me personally growing up that that did scare me."}, "control": null}

Row 60: {"row_index": 60, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "65", "spanish_a": "También tuve que darme cuenta de que no había nada que pudiera hacer. No había nada que no", "english": "I also had to come to a realization that there was nothing that I could do. There was nothing that there was no"}, "control": null}

Row 61: {"row_index": 61, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "66", "spanish_a": "una ecuación. No había, ya sabes, un algoritmo largo que tuviera que descifrar.", "english": "uh equation. There was no you know uh long there's there was no long algorithm that I had to figure out"}, "control": null}

Row 62: {"row_index": 62, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "67", "spanish_a": "que tuve que hacer esto, esto y aquello.", "english": "that I had to do this, this and that."}, "control": null}

Row 63: {"row_index": 63, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "68", "spanish_a": "Para poder ser salvado. No, yo vine en yo yo recuerdo que era un", "english": "In order to get saved. No, I came on I I rem uh I remember it was a um"}, "control": null}

Row 64: {"row_index": 64, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "69", "spanish_a": "una tarde de verano en agosto, me di cuenta de que", "english": "uh summer evening in August, I I came to a realisation that"}, "control": null}

Row 65: {"row_index": 65, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "70", "spanish_a": "Estoy perdido y voy al infierno. Y me doy cuenta", "english": "I am lost and I am going to hell. And I realize"}, "control": null}

Row 66: {"row_index": 66, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "71", "spanish_a": "Y me di cuenta y clamé al Señor para que me salvara, y lo hizo. Y esta noche tienes la opción de caminar en el mundo.", "english": "And I realized that and I cried out to the Lord to save me, and he did. And tonight you have the very option to walk in the world."}, "control": null}

Row 67: {"row_index": 67, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "73", "spanish_a": "y estar con él para siempre.", "english": "um and be with him forever. Or"}, "control": null}

Row 73: {"row_index": 73, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "80", "spanish_a": "Esa última esa última sección no debería no debería perecer, sino tener vida eterna. Eso no es solo una afirmación.", "english": "That last that last section there should not have should not perish, but have eternal life. That isn't just a statement."}, "control": null}

Row 74: {"row_index": 74, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "Eso es una promesa. Eso es eso es um", "english": "That is a promise. That is that is um"}, "control": null}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "82", "spanish_a": "Uh eso es Dios diciendo que si crees y si confías en lo que mi hijo hizo en esa cruz que", "english": "Uh th that is God saying that if you believe and you tr uh if if you trust on what my son did on that cross that"}, "control": null}

Row 76: {"row_index": 76, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "83", "spanish_a": "Ese día.", "english": "uh that that uh that day."}, "control": null}

Row 2: {"row_index": 2, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "3", "spanish_a": "uh al menos hasta este momento sus vidas uh terminaron igual pero cada uno tenía un destino diferente.", "english": "uh at least up until this moment their their lives souh uh ended the same but each one had a different destination."}}

Row 3: {"row_index": 3, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "4", "spanish_a": "Um y podemos y podemos ver que eh uno de ellos eh e el el criminal de la derecha incluso dice eh este hombre no ha hecho nada malo.", "english": "Um and you can and we can see that uh one of them uh e th the the criminal on on the right even says uh this man has done nothing wrong."}}

Row 4: {"row_index": 4, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "5", "spanish_a": "Y eh él sabía que este hombre era Cristo. Sabía que este hombre estaba allí", "english": "And uh he knew that this man was was Christ. He knew that this man was there"}}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "6", "spanish_a": "para morir por nuestros pecados, tus pecados y mis pecados. Y así esta noche, uno de estos dos criminales es", "english": "to uh die for our sins, your s um uh my sins and your sins. And so tonight, uh one of these two criminals is"}}

Row 8: {"row_index": 8, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "9", "spanish_a": "Pero esta noche, eh, si no estás salvado, eh, ¿cuál será tu destino?", "english": "But tonight are uh if you're not saved, uh which where w where will your destination be?"}}

Row 9: {"row_index": 9, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "10", "spanish_a": "Así que, uno de ellos", "english": "So well, one of them"}}

Row 11: {"row_index": 11, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "12", "spanish_a": "Ninguno de estos tipos respetó la ley, pero uno de ellos se despertó con Dios y el otro se despertó con el diablo.", "english": "Um th neither of these guys kept the law, but one of them uh woke up with God and the other one woke up with the devil."}}

Row 12: {"row_index": 12, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "13", "spanish_a": "Quiero preguntar de nuevo, ¿estás salvado? Um", "english": "I want to ask again, are you saved? Um"}}

Row 15: {"row_index": 15, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "16", "spanish_a": "uh la forma más brutal uh en que un hombre uh cualquier persona puede ser puto a muerte. Y sin embargo, estos tipos eran ambos pecadores", "english": "uh the most brutal way uh a man uh an any person can uh can be put to death. And yet these were these guys were both sinners"}}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "18", "spanish_a": "Solo quiero que pienses en eso porque tienes a estos dos hombres que son similares pero también diferentes. Y", "english": "I I just want you to to to think about that because you have these two men who are similar but also different. And"}}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "20", "spanish_a": "¿Estás salvado? Y quizás estés pensando, bueno, yo no soy como estos dos hombres. Bueno, tú lo eres. Tú lo eres. Um, ya sea que quieras admitirlo o no.", "english": "Are you saved? And you might be thinking, well, I'm not like these two men. Well you you are. You are. Um whether you want to uh admit it or not."}}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "21", "spanish_a": "Todos somos como estos dos hombres. Obviamente, ya sabes, espero que nadie haya cometido asesinato. Um y si lo haces, entonces eh", "english": "We we are all like these two men. Obviously, you know, I hope anyone hasn't I hope no one's committed murder. Um and if you do then uh"}}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "23", "spanish_a": "Pero espero que nadie haya, ya sabes, cometido asesinato. Ya sabes. De nuevo, como dije, no importa qué tipo de pecado hayas cometido, todos hemos pecado, ya sea que sea", "english": "But I hope no one has, you know, committed murder. You know. Again, like I said, it doesn't matter what kind of sin you've committed, we've all sinned, whether that's"}}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "24", "spanish_a": "tomar algo que no deberías o eh ya sabes, tal vez robar dulces", "english": "taking something that you shouldn't have or uh you know maybe stealing candy"}}

Row 24: {"row_index": 24, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "25", "spanish_a": "O um eh mentirle a tu mamá cada vez que te dice que hagas algo y dices que lo haces. Um eso es nosotros, todos somos pecadores.", "english": "Or um uh lying to your mom whenever she tells you to do something and you say you do it. Um that is we we are all we are all sinners."}}

Row 25: {"row_index": 25, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "26", "spanish_a": "Um y estos dos y y y estos dos estaban pagando por los pecados que eh habían cometido. Pero", "english": "Um and these two and and and these two were repaying for the sins that uh they had committed. But"}}

Row 26: {"row_index": 26, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "27", "spanish_a": "Ahora, ahora déjame preguntarte esto. ¿Deberías morir por los pecados que hemos cometido?", "english": "Um now now let me ask you this. Should you die for the should we should me and you should we die for the sins that we have uh committed?"}}

Row 27: {"row_index": 27, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "28", "spanish_a": "Y la respuesta a eso es que es sí. Um pero", "english": "And the answer to that is it's yes. Um but"}}

Row 31: {"row_index": 31, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "32", "spanish_a": "Vamos a sacar uno, eh, eh, bueno, quiero comparar a nosotros ahora con estos dos criminales. Entonces, ¿eres", "english": "Let's let's take out one w uh uh well I wanna compare us now to to these two criminals. So are you"}}

Row 32: {"row_index": 32, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "33", "spanish_a": "Bueno, sabes, con el crecer, sabes, puedes decir que he venido a la iglesia consistentemente. He, sabes, he estado, he estado bajo el evangelio.", "english": "Uh so you can you know with grow growing up, you know, you can you you you can you know say that well I've I've come to church consistently. I've you know I've been I've been underneath the gospel."}}

Row 33: {"row_index": 33, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "34", "spanish_a": "uh Yo yo me senté debajo del evangelio. No soy tan malo. Pero", "english": "uh I I I sat underneath the gospel. I'm not all that bad. But"}}

Row 34: {"row_index": 34, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "35", "spanish_a": "Sí, eso es bueno, pero a menos que no conozcas a Cristo como tu Salvador, no entrarás al cielo. Um y no quiero sonar", "english": "Yes, that's good, but unless you don't know Christ as your Savior, you're not getting into heaven. Um and I don't mean to sound"}}

Row 35: {"row_index": 35, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "36", "spanish_a": "sabes, um, franco cuando yo s oh no quiero sonar frívolo cuando digo eso, pero esa es la realidad de lo que um", "english": "you know, um, blunt when I s oh I I don't mean to sound flimpent when I say that, but that is the reality of of what um"}}

Row 36: {"row_index": 36, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "37", "spanish_a": "De", "english": "Of"}}

Row 37: {"row_index": 37, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "38", "spanish_a": "de nuestra de tu situación esta noche si no estás salvado. Um si no conoces a Cristo como tu Salvador, entonces um tú eh como estos dos hombres", "english": "of our of your situation tonight if you aren't saved. Um if you do not know Christ as your Savior, then um you uh just like these two men"}}

Row 38: {"row_index": 38, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "39", "spanish_a": "uh tendrás dos opciones. Puedes o bien despertar en el cielo o puedes despertar en el infierno. Um y", "english": "uh you will have two options. You can either uh wake up in heaven or you can wake up in hell. Um and"}}

Row 39: {"row_index": 39, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "40", "spanish_a": "Tienes esa elección de libre albedrío esta noche, pero eh eh esa es una decisión que tienes que tomar", "english": "You have that free will choice tonight, but uh uh that is a decision that you have to make"}}

Row 40: {"row_index": 40, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "41", "spanish_a": "y eh Dios no tomará esa decisión por ti. Él quiere estar en tu vida, pero tienes que aceptarlo y darte cuenta de que sin él,", "english": "and uh God won't make that decision for you. He wants to be in your life, but you have to accept him and realize that without him,"}}

Row 41: {"row_index": 41, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "42", "spanish_a": "uh tú uh no vas a entrar al cielo y estás en y estás y tienes que darte cuenta de que estás perdido en tus pecados esta noche.", "english": "uh you uh won't be getting into heaven and you are in and you're s and and you have to realize that you're lost in your sins tonight."}}

Row 43: {"row_index": 43, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "44", "spanish_a": "Um, al igual que estos dos criminales, um, tú tendrás dos opciones cuando se trata de", "english": "Um just like these two criminals, um you you will uh you have two options when when it comes to"}}

Row 44: {"row_index": 44, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "45", "spanish_a": "donde estará tu destino eterno. Uno de estos escuchó eh uno de estos eh criminales escuchó um verdaderamente tú eh verdaderamente eh", "english": "where your eternal destination will will be. One of the one of these heard uh one one of these uh criminals heard um truly you uh truly uh"}}

Row 45: {"row_index": 45, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "46", "spanish_a": "Uh lo siento, de verdad te digo, hoy estarás conmigo en el paraíso. Y el otro escuchó, Apártate de mí, apártate de mí, porque nunca te conocí.", "english": "Uh sorry, truly I say to you, today you'll be with me in paradise. And the other one heard, Depart from me Depart from me, for I never knew you."}}

Row 46: {"row_index": 46, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "47", "spanish_a": "Entonces, um, para mí personalmente, crecer eso me asustó, pero n yo luego, pero también tuve que llegar a la comprensión de que", "english": "So, um, for me personally growing up that that did scare me but n I then but I also had to come to a realization that"}}

Row 47: {"row_index": 47, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "48", "spanish_a": "No había nada que pudiera hacer. No había nada que no hubiera", "english": "there was nothing that I could do. There was nothing that there was no"}}

Row 48: {"row_index": 48, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "49", "spanish_a": "una ecuación, no había, ya sabes, eh largo, no había un algoritmo largo que tuviera que descifrar, que tuviera que hacer esto, esto y aquello", "english": "uh equation, there was no you know, uh long there's there was no long algorithm that I had to figure out, that I had to do this, this, and that"}}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo que fue un um", "english": "No, I came on I I re uh I remember it was a um"}}

Row 50: {"row_index": 50, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "51", "spanish_a": "una tarde de verano en agosto, me di cuenta de que estoy perdido y que voy al infierno. Y me di cuenta", "english": "uh summer evening in August, I I came to a realization that I am lost and I am going to hell. And I realized"}}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "52", "spanish_a": "Y me di cuenta y clamé al Señor para que me salvara, y él lo hizo. Y esta noche tienes la opción de salir de aquí sabiendo que tus pecados son perdonados.", "english": "And I realized that and I cried out to the Lord to save me, and he did. And tonight you have the very option to walk out of here knowing that your sins are forgiven."}}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "53", "spanish_a": "que puedas tener un lugar en el cielo y um y estar con él para siempre. O", "english": "that you can have a a place in heaven and um and be with him forever. Or"}}

Row 58: {"row_index": 58, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "60", "spanish_a": "Esa última esa última sección ahí. No debería perecer, sino tener vida eterna. Eso no es solo una declaración, es una promesa.", "english": "That last that last section there. Should not have should not perish, but have eternal life. That isn't just a statement, that is a promise."}}

Row 59: {"row_index": 59, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "61", "spanish_a": "Eso es lo que es um", "english": "That is that is um"}}

Row 60: {"row_index": 60, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "62", "spanish_a": "Uh eso es Dios diciendo que si crees y si confías en lo que mi hijo hizo en esa cruz ese día.", "english": "Uh that is God saying that if you believe and you tr uh if you trust on what my son did on that cross that uh that that uh that day."}}

### Identity: lb0912_A_max_utterance_6_r1 vs lb0912_A_ctl_r1

Translation share: 1.0; English share: 0.24675324675324675; aligned rows: 19; chunk-count difference: 16.

Row 2: {"row_index": 2, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "3", "spanish_a": "uh al menos hasta este momento sus vidas uh", "english": "uh at least up until this moment their their lives uh"}, "control": null}

Row 3: {"row_index": 3, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "4", "spanish_a": "terminaron igual, pero cada uno tenía un destino diferente.", "english": "uh ended the same, but each one had a different destination."}, "control": null}

Row 4: {"row_index": 4, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "5", "spanish_a": "Um y podemos y podemos ver eso. Eh uno de ellos eh e el el el criminal a la derecha incluso dice", "english": "Um and you can and we can see that. Uh one of them uh e the the the criminal on on the right even says"}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "6", "spanish_a": "uh este hombre no ha hecho nada malo. Y uh él sabía que este hombre era", "english": "uh this man has done nothing wrong. And uh he knew that this man was"}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "fue Cristo. Él sabía que este hombre estaba allí", "english": "was Christ. He knew that this man was there"}, "control": null}

Row 7: {"row_index": 7, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "para morir por nuestros pecados, tus pecados y tus pecados. Y así esta noche uno de", "english": "to uh die for our sins, your s um my sins and your sins. And so tonight uh one of"}, "control": null}

Row 10: {"row_index": 10, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "12", "spanish_a": "Pero esta noche nuestro eh si no estás salvado, eh", "english": "But tonight our uh if you're not saved, uh"}, "control": null}

Row 11: {"row_index": 11, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "13", "spanish_a": "cuál donde dónde será tu destino. Así que uno uno de ellos", "english": "which where w where will your destination be. So one one of them"}, "control": null}

Row 13: {"row_index": 13, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "15", "spanish_a": "Ninguno de estos tipos respetó la ley, pero uno de ellos se despertó con Dios y el otro se despertó con", "english": "Um d neither of these guys kept the law, but one of them uh woke up with God and the other uh woke up with"}, "control": null}

Row 14: {"row_index": 14, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "16", "spanish_a": "el diablo. Quiero preguntar de nuevo, ¿estás salvado? Um", "english": "the devil. I want to ask again, are you saved? Um"}, "control": null}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "19", "spanish_a": "uh la forma más brutal uh en que un hombre uh cualquier persona puede uh puede ser ejecutado. Y sin embargo, estos tipos eran ambos", "english": "uh the most brutal way uh a man uh an any person can uh can be put to death. And yet these these guys were both"}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Solo quiero que pienses en eso, porque tienes a estos dos hombres que son", "english": "I I just want you to to to think about that, because you have these two men who are"}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "similar pero también diferente. Y", "english": "similar but also different. And"}, "control": null}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "24", "spanish_a": "¿Estás salvado? Y quizás estés pensando, Bueno, yo no soy como estos dos hombres. Bueno, tú tú lo eres. Lo eres. Um", "english": "Are you saved? And you might be thinking, Well, I'm not like these two men. Well you you are. You are. Um"}, "control": null}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "25", "spanish_a": "Tengas ganas o no, todos somos como estos dos hombres. Obviamente, sabes, espero que nadie haya", "english": "Whether you want to, uh admit it or not, we we are all like these two men. Obviously, you know, I hope anyone hasn't no"}, "control": null}

Row 25: {"row_index": 25, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "28", "spanish_a": "Pero espero que nadie haya, ya sabes, cometido asesinato. Ya sabes. De nuevo, como dije, no importa qué tipo de pecado hayas cometido, todos hemos dicho", "english": "But I hope no one has, you know, committed murder. You know. Again, like I said, it doesn't matter what kind of sin you've committed, we've all said"}, "control": null}

Row 26: {"row_index": 26, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "29", "spanish_a": "Ya sea tomar algo que no deberías o, ya sabes, tal vez robar dulces", "english": "Whether that's taking something that you shouldn't have or uh you know maybe stealing candy"}, "control": null}

Row 27: {"row_index": 27, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "30", "spanish_a": "O um eh mentirle a tu mamá cada vez que te dice que hagas algo y tú dices que lo haces. Um eso es que nosotros nosotros", "english": "Or um uh lying to your mom whenever she tells you to do something and you say you do it. Um that is we we"}, "control": null}

Row 28: {"row_index": 28, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "32", "spanish_a": "Um y estos dos y y y estos dos estaban pagando por los pecados que eh habían cometido.", "english": "Um and these two and and and these two were repaying for the sins that uh they had committed."}, "control": null}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "33", "spanish_a": "Pero", "english": "But"}, "control": null}

Row 30: {"row_index": 30, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "34", "spanish_a": "Um ahora", "english": "Um now"}, "control": null}

Row 31: {"row_index": 31, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "35", "spanish_a": "Eh, déjame preguntarte esto. ¿Deberías morir por los pecados que hemos cometido?", "english": "Uh let me ask you this. Should you die for the should we should me and you should we die for the sins that we have uh committed?"}, "control": null}

Row 32: {"row_index": 32, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "36", "spanish_a": "Y la respuesta a eso es sí. Pero", "english": "And the answer to that is is yes. But"}, "control": null}

Row 36: {"row_index": 36, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "40", "spanish_a": "Vamos a vamos a sacar uno con eh eh bueno, quiero comparar a nosotros ahora con estos dos criminales. Entonces, ¿eres", "english": "let's let's take out one w uh uh well I wanna compare us now to to these two criminals. So are you"}, "control": null}

Row 37: {"row_index": 37, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "41", "spanish_a": "Así que", "english": "So"}, "control": null}

Row 38: {"row_index": 38, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "42", "spanish_a": "Bueno, puedes, ya sabes, con el crecimiento, ya sabes, puedes, puedes, puedes decir que he venido a la iglesia consistentemente.", "english": "Uh so you can you know, with gro growing up, you know, you can you you you can you know say that well I've I've come to church consistently. I"}, "control": null}

Row 39: {"row_index": 39, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "43", "spanish_a": "Sabes, he estado he estado bajo el evangelio eh yo me senté bajo el evangelio. No soy tan malo. Pero", "english": "You know, I've been o I've been underneath the gospel uh I I sat underneath the gospel. I'm not all that bad. But"}, "control": null}

Row 40: {"row_index": 40, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "44", "spanish_a": "Sí, eso es bueno, pero a menos que no conozcas a Cristo como tu Salvador, no entrarás al cielo.", "english": "Yes, that's good, but unless you don't know Christ as your Savior, you're not getting into heaven."}, "control": null}

Row 41: {"row_index": 41, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "45", "spanish_a": "Um y yo no quiero sonar", "english": "Um and I don't mean to sound"}, "control": null}

Row 42: {"row_index": 42, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "46", "spanish_a": "Ya sabes, um", "english": "You know, um"}, "control": null}

Row 43: {"row_index": 43, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "47", "spanish_a": "directo cuando yo lo veo, no quiero sonar frívolo cuando digo eso, pero esa es la realidad de", "english": "blunt when I s oh I I don't mean to sound flippant when I say that, but that is the reality of"}, "control": null}

Row 44: {"row_index": 44, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "48", "spanish_a": "de qué um de", "english": "of what um of"}, "control": null}

Row 45: {"row_index": 45, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "49", "spanish_a": "de nuestra de tu situación esta noche si no estás salvo. Um si no conoces a Cristo como tu Salvador, entonces eh", "english": "of our of your situation tonight if you aren't saved. Um if you do not know Christ as your Savior, then uh"}, "control": null}

Row 46: {"row_index": 46, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "50", "spanish_a": "Tú, eh, como estos dos hombres, eh, tendrás dos opciones. Puedes o bien eh", "english": "You uh just like these two men, uh you will have two options. You can either uh"}, "control": null}

Row 47: {"row_index": 47, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "51", "spanish_a": "O despiertas en el cielo o puedes despertar en el infierno. Um y", "english": "Uh wake up in heaven or you can wake up in hell. Um and"}, "control": null}

Row 48: {"row_index": 48, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "52", "spanish_a": "Tienes esa elección de libre albedrío esta noche, pero eh", "english": "You have that free will choice tonight, but uh"}, "control": null}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "53", "spanish_a": "uh esa es una decisión que tienes que tomar.", "english": "uh that is a decision that you have to make."}, "control": null}

Row 50: {"row_index": 50, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "54", "spanish_a": "y eh Dios no tomará esa decisión por ti. Él quiere estar en tu vida, pero tienes que aceptarlo y", "english": "and uh God won't make that decision for you. He wants to be in your life, but you have to accept him and"}, "control": null}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "55", "spanish_a": "te das cuenta de que sin él, uh tú", "english": "realize that without him, uh you"}, "control": null}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "56", "spanish_a": "no irás al cielo y tú estás en y estás y tienes que darte cuenta de que estás perdido en tus pecados esta noche.", "english": "uh won't be getting into heaven and you are in and you're s and and you have to realize that you're lost in your sins tonight."}, "control": null}

Row 54: {"row_index": 54, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "58", "spanish_a": "Um", "english": "Um"}, "control": null}

Row 55: {"row_index": 55, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "59", "spanish_a": "uh como estos dos criminales um tú tú tendrás uh tienes dos opciones cuando cuando se trata", "english": "uh just like these two criminals um you you will uh you have two options when when it comes"}, "control": null}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "60", "spanish_a": "Sí.", "english": "Yeah."}, "control": null}

Row 57: {"row_index": 57, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "61", "spanish_a": "donde estará tu destino eterno. Uno de estos escuchó eh uno de estos eh criminales escuchó um verdaderamente", "english": "where your eternal destination will will be. One of the one of these heard uh one of these uh criminals heard um truly"}, "control": null}

Row 58: {"row_index": 58, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "63", "spanish_a": "Uh lo siento, de verdad te digo, hoy estarás conmigo en el paraíso y el otro escuchó, departamento apártate de mí porque nunca supe.", "english": "Uh sorry, truly I say to you, today you'll be with me in paradise and the other one heard, Department depart from me for I never knew."}, "control": null}

Row 59: {"row_index": 59, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "64", "spanish_a": "Así que, para mí personalmente, crecer eso me asustó.", "english": "So um for me personally growing up that that did scare me."}, "control": null}

Row 60: {"row_index": 60, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "65", "spanish_a": "También tuve que darme cuenta de que no había nada que pudiera hacer. No había nada que no", "english": "I also had to come to a realization that there was nothing that I could do. There was nothing that there was no"}, "control": null}

Row 61: {"row_index": 61, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "66", "spanish_a": "una ecuación. No había, ya sabes, un algoritmo largo que tuviera que descifrar.", "english": "uh equation. There was no you know uh long there's there was no long algorithm that I had to figure out"}, "control": null}

Row 62: {"row_index": 62, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "67", "spanish_a": "que tuve que hacer esto, esto y aquello.", "english": "that I had to do this, this and that."}, "control": null}

Row 63: {"row_index": 63, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "68", "spanish_a": "Para poder ser salvado. No, yo vine en yo yo recuerdo que era un", "english": "In order to get saved. No, I came on I I rem uh I remember it was a um"}, "control": null}

Row 64: {"row_index": 64, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "69", "spanish_a": "una tarde de verano en agosto, me di cuenta de que", "english": "uh summer evening in August, I I came to a realisation that"}, "control": null}

Row 65: {"row_index": 65, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "70", "spanish_a": "Estoy perdido y voy al infierno. Y me doy cuenta", "english": "I am lost and I am going to hell. And I realize"}, "control": null}

Row 66: {"row_index": 66, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "71", "spanish_a": "Y me di cuenta y clamé al Señor para que me salvara, y lo hizo. Y esta noche tienes la opción de caminar en el mundo.", "english": "And I realized that and I cried out to the Lord to save me, and he did. And tonight you have the very option to walk in the world."}, "control": null}

Row 67: {"row_index": 67, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "73", "spanish_a": "y estar con él para siempre.", "english": "um and be with him forever. Or"}, "control": null}

Row 73: {"row_index": 73, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "80", "spanish_a": "Esa última esa última sección no debería no debería perecer, sino tener vida eterna. Eso no es solo una afirmación.", "english": "That last that last section there should not have should not perish, but have eternal life. That isn't just a statement."}, "control": null}

Row 74: {"row_index": 74, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "Eso es una promesa. Eso es eso es um", "english": "That is a promise. That is that is um"}, "control": null}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "82", "spanish_a": "Uh eso es Dios diciendo que si crees y si confías en lo que mi hijo hizo en esa cruz que", "english": "Uh th that is God saying that if you believe and you tr uh if if you trust on what my son did on that cross that"}, "control": null}

Row 76: {"row_index": 76, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "83", "spanish_a": "Ese día.", "english": "uh that that uh that day."}, "control": null}

Row 2: {"row_index": 2, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "3", "spanish_a": "uh al menos hasta este momento sus vidas uh terminaron igual pero cada uno tenía un destino diferente.", "english": "uh at least up until this moment their their lives souh uh ended the same but each one had a different destination."}}

Row 3: {"row_index": 3, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "4", "spanish_a": "Um y podemos y podemos ver que eh uno de ellos eh e el el criminal de la derecha incluso dice eh este hombre no ha hecho nada malo.", "english": "Um and you can and we can see that uh one of them uh e th the the criminal on on the right even says uh this man has done nothing wrong."}}

Row 4: {"row_index": 4, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "5", "spanish_a": "Y eh él sabía que este hombre era Cristo. Sabía que este hombre estaba allí", "english": "And uh he knew that this man was was Christ. He knew that this man was there"}}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "6", "spanish_a": "para morir por nuestros pecados, tus pecados y mis pecados. Y así esta noche, uno de estos dos criminales es", "english": "to uh die for our sins, your s um uh my sins and your sins. And so tonight, uh one of these two criminals is"}}

Row 8: {"row_index": 8, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "9", "spanish_a": "Pero esta noche, eh, si no estás salvado, eh, ¿cuál será tu destino?", "english": "But tonight are uh if you're not saved, uh which where w where will your destination be?"}}

Row 9: {"row_index": 9, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "10", "spanish_a": "Así que, uno de ellos", "english": "So well, one of them"}}

Row 11: {"row_index": 11, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "12", "spanish_a": "Ninguno de estos tipos respetó la ley, pero uno de ellos se despertó con Dios y el otro se despertó con el diablo.", "english": "Um th neither of these guys kept the law, but one of them uh woke up with God and the other one woke up with the devil."}}

Row 12: {"row_index": 12, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "13", "spanish_a": "Quiero preguntar de nuevo, ¿estás salvado? Um", "english": "I want to ask again, are you saved? Um"}}

Row 15: {"row_index": 15, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "16", "spanish_a": "uh la forma más brutal uh en que un hombre uh cualquier persona puede ser puto a muerte. Y sin embargo, estos tipos eran ambos pecadores", "english": "uh the most brutal way uh a man uh an any person can uh can be put to death. And yet these were these guys were both sinners"}}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "18", "spanish_a": "Solo quiero que pienses en eso porque tienes a estos dos hombres que son similares pero también diferentes. Y", "english": "I I just want you to to to think about that because you have these two men who are similar but also different. And"}}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "20", "spanish_a": "¿Estás salvado? Y quizás estés pensando, bueno, yo no soy como estos dos hombres. Bueno, tú lo eres. Tú lo eres. Um, ya sea que quieras admitirlo o no.", "english": "Are you saved? And you might be thinking, well, I'm not like these two men. Well you you are. You are. Um whether you want to uh admit it or not."}}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "21", "spanish_a": "Todos somos como estos dos hombres. Obviamente, ya sabes, espero que nadie haya cometido asesinato. Um y si lo haces, entonces eh", "english": "We we are all like these two men. Obviously, you know, I hope anyone hasn't I hope no one's committed murder. Um and if you do then uh"}}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "23", "spanish_a": "Pero espero que nadie haya, ya sabes, cometido asesinato. Ya sabes. De nuevo, como dije, no importa qué tipo de pecado hayas cometido, todos hemos pecado, ya sea que sea", "english": "But I hope no one has, you know, committed murder. You know. Again, like I said, it doesn't matter what kind of sin you've committed, we've all sinned, whether that's"}}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "24", "spanish_a": "tomar algo que no deberías o eh ya sabes, tal vez robar dulces", "english": "taking something that you shouldn't have or uh you know maybe stealing candy"}}

Row 24: {"row_index": 24, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "25", "spanish_a": "O um eh mentirle a tu mamá cada vez que te dice que hagas algo y dices que lo haces. Um eso es nosotros, todos somos pecadores.", "english": "Or um uh lying to your mom whenever she tells you to do something and you say you do it. Um that is we we are all we are all sinners."}}

Row 25: {"row_index": 25, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "26", "spanish_a": "Um y estos dos y y y estos dos estaban pagando por los pecados que eh habían cometido. Pero", "english": "Um and these two and and and these two were repaying for the sins that uh they had committed. But"}}

Row 26: {"row_index": 26, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "27", "spanish_a": "Ahora, ahora déjame preguntarte esto. ¿Deberías morir por los pecados que hemos cometido?", "english": "Um now now let me ask you this. Should you die for the should we should me and you should we die for the sins that we have uh committed?"}}

Row 27: {"row_index": 27, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "28", "spanish_a": "Y la respuesta a eso es que es sí. Um pero", "english": "And the answer to that is it's yes. Um but"}}

Row 31: {"row_index": 31, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "32", "spanish_a": "Vamos a sacar uno, eh, eh, bueno, quiero comparar a nosotros ahora con estos dos criminales. Entonces, ¿eres", "english": "Let's let's take out one w uh uh well I wanna compare us now to to these two criminals. So are you"}}

Row 32: {"row_index": 32, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "33", "spanish_a": "Bueno, sabes, con el crecer, sabes, puedes decir que he venido a la iglesia consistentemente. He, sabes, he estado, he estado bajo el evangelio.", "english": "Uh so you can you know with grow growing up, you know, you can you you you can you know say that well I've I've come to church consistently. I've you know I've been I've been underneath the gospel."}}

Row 33: {"row_index": 33, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "34", "spanish_a": "uh Yo yo me senté debajo del evangelio. No soy tan malo. Pero", "english": "uh I I I sat underneath the gospel. I'm not all that bad. But"}}

Row 34: {"row_index": 34, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "35", "spanish_a": "Sí, eso es bueno, pero a menos que no conozcas a Cristo como tu Salvador, no entrarás al cielo. Um y no quiero sonar", "english": "Yes, that's good, but unless you don't know Christ as your Savior, you're not getting into heaven. Um and I don't mean to sound"}}

Row 35: {"row_index": 35, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "36", "spanish_a": "sabes, um, franco cuando yo s oh no quiero sonar frívolo cuando digo eso, pero esa es la realidad de lo que um", "english": "you know, um, blunt when I s oh I I don't mean to sound flimpent when I say that, but that is the reality of of what um"}}

Row 36: {"row_index": 36, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "37", "spanish_a": "De", "english": "Of"}}

Row 37: {"row_index": 37, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "38", "spanish_a": "de nuestra de tu situación esta noche si no estás salvado. Um si no conoces a Cristo como tu Salvador, entonces um tú eh como estos dos hombres", "english": "of our of your situation tonight if you aren't saved. Um if you do not know Christ as your Savior, then um you uh just like these two men"}}

Row 38: {"row_index": 38, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "39", "spanish_a": "uh tendrás dos opciones. Puedes o bien despertar en el cielo o puedes despertar en el infierno. Um y", "english": "uh you will have two options. You can either uh wake up in heaven or you can wake up in hell. Um and"}}

Row 39: {"row_index": 39, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "40", "spanish_a": "Tienes esa elección de libre albedrío esta noche, pero eh eh esa es una decisión que tienes que tomar", "english": "You have that free will choice tonight, but uh uh that is a decision that you have to make"}}

Row 40: {"row_index": 40, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "41", "spanish_a": "y eh Dios no tomará esa decisión por ti. Él quiere estar en tu vida, pero tienes que aceptarlo y darte cuenta de que sin él,", "english": "and uh God won't make that decision for you. He wants to be in your life, but you have to accept him and realize that without him,"}}

Row 41: {"row_index": 41, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "42", "spanish_a": "uh tú uh no vas a entrar al cielo y estás en y estás y tienes que darte cuenta de que estás perdido en tus pecados esta noche.", "english": "uh you uh won't be getting into heaven and you are in and you're s and and you have to realize that you're lost in your sins tonight."}}

Row 43: {"row_index": 43, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "44", "spanish_a": "Um, al igual que estos dos criminales, um, tú tendrás dos opciones cuando se trata de", "english": "Um just like these two criminals, um you you will uh you have two options when when it comes to"}}

Row 44: {"row_index": 44, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "45", "spanish_a": "donde estará tu destino eterno. Uno de estos escuchó eh uno de estos eh criminales escuchó um verdaderamente tú eh verdaderamente eh", "english": "where your eternal destination will will be. One of the one of these heard uh one one of these uh criminals heard um truly you uh truly uh"}}

Row 45: {"row_index": 45, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "46", "spanish_a": "Uh lo siento, de verdad te digo, hoy estarás conmigo en el paraíso. Y el otro escuchó, Apártate de mí, apártate de mí, porque nunca te conocí.", "english": "Uh sorry, truly I say to you, today you'll be with me in paradise. And the other one heard, Depart from me Depart from me, for I never knew you."}}

Row 46: {"row_index": 46, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "47", "spanish_a": "Entonces, um, para mí personalmente, crecer eso me asustó, pero n yo luego, pero también tuve que llegar a la comprensión de que", "english": "So, um, for me personally growing up that that did scare me but n I then but I also had to come to a realization that"}}

Row 47: {"row_index": 47, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "48", "spanish_a": "No había nada que pudiera hacer. No había nada que no hubiera", "english": "there was nothing that I could do. There was nothing that there was no"}}

Row 48: {"row_index": 48, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "49", "spanish_a": "una ecuación, no había, ya sabes, eh largo, no había un algoritmo largo que tuviera que descifrar, que tuviera que hacer esto, esto y aquello", "english": "uh equation, there was no you know, uh long there's there was no long algorithm that I had to figure out, that I had to do this, this, and that"}}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo que fue un um", "english": "No, I came on I I re uh I remember it was a um"}}

Row 50: {"row_index": 50, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "51", "spanish_a": "una tarde de verano en agosto, me di cuenta de que estoy perdido y que voy al infierno. Y me di cuenta", "english": "uh summer evening in August, I I came to a realization that I am lost and I am going to hell. And I realized"}}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "52", "spanish_a": "Y me di cuenta y clamé al Señor para que me salvara, y él lo hizo. Y esta noche tienes la opción de salir de aquí sabiendo que tus pecados son perdonados.", "english": "And I realized that and I cried out to the Lord to save me, and he did. And tonight you have the very option to walk out of here knowing that your sins are forgiven."}}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "53", "spanish_a": "que puedas tener un lugar en el cielo y um y estar con él para siempre. O", "english": "that you can have a a place in heaven and um and be with him forever. Or"}}

Row 58: {"row_index": 58, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "60", "spanish_a": "Esa última esa última sección ahí. No debería perecer, sino tener vida eterna. Eso no es solo una declaración, es una promesa.", "english": "That last that last section there. Should not have should not perish, but have eternal life. That isn't just a statement, that is a promise."}}

Row 59: {"row_index": 59, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "61", "spanish_a": "Eso es lo que es um", "english": "That is that is um"}}

Row 60: {"row_index": 60, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "62", "spanish_a": "Uh eso es Dios diciendo que si crees y si confías en lo que mi hijo hizo en esa cruz ese día.", "english": "Uh that is God saying that if you believe and you tr uh if you trust on what my son did on that cross that uh that that uh that day."}}

### Identity: lb0912_A_max_utterance_6_r2 vs lb0912_A_ctl_r2

Translation share: 1.0; English share: 0.24675324675324675; aligned rows: 19; chunk-count difference: 16.

Row 2: {"row_index": 2, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "3", "spanish_a": "uh al menos hasta este momento sus vidas uh", "english": "uh at least up until this moment their their lives uh"}, "control": null}

Row 3: {"row_index": 3, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "4", "spanish_a": "terminaron igual, pero cada uno tenía un destino diferente.", "english": "uh ended the same, but each one had a different destination."}, "control": null}

Row 4: {"row_index": 4, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "5", "spanish_a": "Um y podemos y podemos ver eso. Eh uno de ellos eh e el el el criminal a la derecha incluso dice", "english": "Um and you can and we can see that. Uh one of them uh e the the the criminal on on the right even says"}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "6", "spanish_a": "uh este hombre no ha hecho nada malo. Y uh él sabía que este hombre era", "english": "uh this man has done nothing wrong. And uh he knew that this man was"}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "fue Cristo. Él sabía que este hombre estaba allí", "english": "was Christ. He knew that this man was there"}, "control": null}

Row 7: {"row_index": 7, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "para morir por nuestros pecados, tus pecados y tus pecados. Y así esta noche uno de", "english": "to uh die for our sins, your s um my sins and your sins. And so tonight uh one of"}, "control": null}

Row 10: {"row_index": 10, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "12", "spanish_a": "Pero esta noche nuestro eh si no estás salvado, eh", "english": "But tonight our uh if you're not saved, uh"}, "control": null}

Row 11: {"row_index": 11, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "13", "spanish_a": "cuál donde dónde será tu destino. Así que uno uno de ellos", "english": "which where w where will your destination be. So one one of them"}, "control": null}

Row 13: {"row_index": 13, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "15", "spanish_a": "Ninguno de estos tipos respetó la ley, pero uno de ellos se despertó con Dios y el otro se despertó con", "english": "Um d neither of these guys kept the law, but one of them uh woke up with God and the other uh woke up with"}, "control": null}

Row 14: {"row_index": 14, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "16", "spanish_a": "el diablo. Quiero preguntar de nuevo, ¿estás salvado? Um", "english": "the devil. I want to ask again, are you saved? Um"}, "control": null}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "19", "spanish_a": "uh la forma más brutal uh en que un hombre uh cualquier persona puede uh puede ser ejecutado. Y sin embargo, estos tipos eran ambos", "english": "uh the most brutal way uh a man uh an any person can uh can be put to death. And yet these these guys were both"}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Solo quiero que pienses en eso, porque tienes a estos dos hombres que son", "english": "I I just want you to to to think about that, because you have these two men who are"}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "similar pero también diferente. Y", "english": "similar but also different. And"}, "control": null}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "24", "spanish_a": "¿Estás salvado? Y quizás estés pensando, Bueno, yo no soy como estos dos hombres. Bueno, tú tú lo eres. Lo eres. Um", "english": "Are you saved? And you might be thinking, Well, I'm not like these two men. Well you you are. You are. Um"}, "control": null}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "25", "spanish_a": "Tengas ganas o no, todos somos como estos dos hombres. Obviamente, sabes, espero que nadie haya", "english": "Whether you want to, uh admit it or not, we we are all like these two men. Obviously, you know, I hope anyone hasn't no"}, "control": null}

Row 25: {"row_index": 25, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "28", "spanish_a": "Pero espero que nadie haya, ya sabes, cometido asesinato. Ya sabes. De nuevo, como dije, no importa qué tipo de pecado hayas cometido, todos hemos dicho", "english": "But I hope no one has, you know, committed murder. You know. Again, like I said, it doesn't matter what kind of sin you've committed, we've all said"}, "control": null}

Row 26: {"row_index": 26, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "29", "spanish_a": "Ya sea tomar algo que no deberías o, ya sabes, tal vez robar dulces", "english": "Whether that's taking something that you shouldn't have or uh you know maybe stealing candy"}, "control": null}

Row 27: {"row_index": 27, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "30", "spanish_a": "O um eh mentirle a tu mamá cada vez que te dice que hagas algo y tú dices que lo haces. Um eso es que nosotros nosotros", "english": "Or um uh lying to your mom whenever she tells you to do something and you say you do it. Um that is we we"}, "control": null}

Row 28: {"row_index": 28, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "32", "spanish_a": "Um y estos dos y y y estos dos estaban pagando por los pecados que eh habían cometido.", "english": "Um and these two and and and these two were repaying for the sins that uh they had committed."}, "control": null}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "33", "spanish_a": "Pero", "english": "But"}, "control": null}

Row 30: {"row_index": 30, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "34", "spanish_a": "Um ahora", "english": "Um now"}, "control": null}

Row 31: {"row_index": 31, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "35", "spanish_a": "Eh, déjame preguntarte esto. ¿Deberías morir por los pecados que hemos cometido?", "english": "Uh let me ask you this. Should you die for the should we should me and you should we die for the sins that we have uh committed?"}, "control": null}

Row 32: {"row_index": 32, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "36", "spanish_a": "Y la respuesta a eso es sí. Pero", "english": "And the answer to that is is yes. But"}, "control": null}

Row 36: {"row_index": 36, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "40", "spanish_a": "Vamos a vamos a sacar uno con eh eh bueno, quiero comparar a nosotros ahora con estos dos criminales. Entonces, ¿eres", "english": "let's let's take out one w uh uh well I wanna compare us now to to these two criminals. So are you"}, "control": null}

Row 37: {"row_index": 37, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "41", "spanish_a": "Así que", "english": "So"}, "control": null}

Row 38: {"row_index": 38, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "42", "spanish_a": "Bueno, puedes, ya sabes, con el crecimiento, ya sabes, puedes, puedes, puedes decir que he venido a la iglesia consistentemente.", "english": "Uh so you can you know, with gro growing up, you know, you can you you you can you know say that well I've I've come to church consistently. I"}, "control": null}

Row 39: {"row_index": 39, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "43", "spanish_a": "Sabes, he estado he estado bajo el evangelio eh yo me senté bajo el evangelio. No soy tan malo. Pero", "english": "You know, I've been o I've been underneath the gospel uh I I sat underneath the gospel. I'm not all that bad. But"}, "control": null}

Row 40: {"row_index": 40, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "44", "spanish_a": "Sí, eso es bueno, pero a menos que no conozcas a Cristo como tu Salvador, no entrarás al cielo.", "english": "Yes, that's good, but unless you don't know Christ as your Savior, you're not getting into heaven."}, "control": null}

Row 41: {"row_index": 41, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "45", "spanish_a": "Um y yo no quiero sonar", "english": "Um and I don't mean to sound"}, "control": null}

Row 42: {"row_index": 42, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "46", "spanish_a": "Ya sabes, um", "english": "You know, um"}, "control": null}

Row 43: {"row_index": 43, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "47", "spanish_a": "directo cuando yo lo veo, no quiero sonar frívolo cuando digo eso, pero esa es la realidad de", "english": "blunt when I s oh I I don't mean to sound flippant when I say that, but that is the reality of"}, "control": null}

Row 44: {"row_index": 44, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "48", "spanish_a": "de qué um de", "english": "of what um of"}, "control": null}

Row 45: {"row_index": 45, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "49", "spanish_a": "de nuestra de tu situación esta noche si no estás salvo. Um si no conoces a Cristo como tu Salvador, entonces eh", "english": "of our of your situation tonight if you aren't saved. Um if you do not know Christ as your Savior, then uh"}, "control": null}

Row 46: {"row_index": 46, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "50", "spanish_a": "Tú, eh, como estos dos hombres, eh, tendrás dos opciones. Puedes o bien eh", "english": "You uh just like these two men, uh you will have two options. You can either uh"}, "control": null}

Row 47: {"row_index": 47, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "51", "spanish_a": "O despiertas en el cielo o puedes despertar en el infierno. Um y", "english": "Uh wake up in heaven or you can wake up in hell. Um and"}, "control": null}

Row 48: {"row_index": 48, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "52", "spanish_a": "Tienes esa elección de libre albedrío esta noche, pero eh", "english": "You have that free will choice tonight, but uh"}, "control": null}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "53", "spanish_a": "uh esa es una decisión que tienes que tomar.", "english": "uh that is a decision that you have to make."}, "control": null}

Row 50: {"row_index": 50, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "54", "spanish_a": "y eh Dios no tomará esa decisión por ti. Él quiere estar en tu vida, pero tienes que aceptarlo y", "english": "and uh God won't make that decision for you. He wants to be in your life, but you have to accept him and"}, "control": null}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "55", "spanish_a": "te das cuenta de que sin él, uh tú", "english": "realize that without him, uh you"}, "control": null}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "56", "spanish_a": "no irás al cielo y tú estás en y estás y tienes que darte cuenta de que estás perdido en tus pecados esta noche.", "english": "uh won't be getting into heaven and you are in and you're s and and you have to realize that you're lost in your sins tonight."}, "control": null}

Row 54: {"row_index": 54, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "58", "spanish_a": "Um", "english": "Um"}, "control": null}

Row 55: {"row_index": 55, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "59", "spanish_a": "uh como estos dos criminales um tú tú tendrás uh tienes dos opciones cuando cuando se trata", "english": "uh just like these two criminals um you you will uh you have two options when when it comes"}, "control": null}

Row 56: {"row_index": 56, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "60", "spanish_a": "Sí.", "english": "Yeah."}, "control": null}

Row 57: {"row_index": 57, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "61", "spanish_a": "donde estará tu destino eterno. Uno de estos escuchó eh uno de estos eh criminales escuchó um verdaderamente", "english": "where your eternal destination will will be. One of the one of these heard uh one of these uh criminals heard um truly"}, "control": null}

Row 58: {"row_index": 58, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "63", "spanish_a": "Uh lo siento, de verdad te digo, hoy estarás conmigo en el paraíso y el otro escuchó, departamento apártate de mí porque nunca supe.", "english": "Uh sorry, truly I say to you, today you'll be with me in paradise and the other one heard, Department depart from me for I never knew."}, "control": null}

Row 59: {"row_index": 59, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "64", "spanish_a": "Así que, para mí personalmente, crecer eso me asustó.", "english": "So um for me personally growing up that that did scare me."}, "control": null}

Row 60: {"row_index": 60, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "65", "spanish_a": "También tuve que darme cuenta de que no había nada que pudiera hacer. No había nada que no", "english": "I also had to come to a realization that there was nothing that I could do. There was nothing that there was no"}, "control": null}

Row 61: {"row_index": 61, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "66", "spanish_a": "una ecuación. No había, ya sabes, un algoritmo largo que tuviera que descifrar.", "english": "uh equation. There was no you know uh long there's there was no long algorithm that I had to figure out"}, "control": null}

Row 62: {"row_index": 62, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "67", "spanish_a": "que tuve que hacer esto, esto y aquello.", "english": "that I had to do this, this and that."}, "control": null}

Row 63: {"row_index": 63, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "68", "spanish_a": "Para poder ser salvado. No, yo vine en yo yo recuerdo que era un", "english": "In order to get saved. No, I came on I I rem uh I remember it was a um"}, "control": null}

Row 64: {"row_index": 64, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "69", "spanish_a": "una tarde de verano en agosto, me di cuenta de que", "english": "uh summer evening in August, I I came to a realisation that"}, "control": null}

Row 65: {"row_index": 65, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "70", "spanish_a": "Estoy perdido y voy al infierno. Y me doy cuenta", "english": "I am lost and I am going to hell. And I realize"}, "control": null}

Row 66: {"row_index": 66, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "71", "spanish_a": "Y me di cuenta y clamé al Señor para que me salvara, y lo hizo. Y esta noche tienes la opción de caminar en el mundo.", "english": "And I realized that and I cried out to the Lord to save me, and he did. And tonight you have the very option to walk in the world."}, "control": null}

Row 67: {"row_index": 67, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "73", "spanish_a": "y estar con él para siempre.", "english": "um and be with him forever. Or"}, "control": null}

Row 73: {"row_index": 73, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "80", "spanish_a": "Esa última esa última sección no debería no debería perecer, sino tener vida eterna. Eso no es solo una afirmación.", "english": "That last that last section there should not have should not perish, but have eternal life. That isn't just a statement."}, "control": null}

Row 74: {"row_index": 74, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "Eso es una promesa. Eso es eso es um", "english": "That is a promise. That is that is um"}, "control": null}

Row 75: {"row_index": 75, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "82", "spanish_a": "Uh eso es Dios diciendo que si crees y si confías en lo que mi hijo hizo en esa cruz que", "english": "Uh th that is God saying that if you believe and you tr uh if if you trust on what my son did on that cross that"}, "control": null}

Row 76: {"row_index": 76, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "83", "spanish_a": "Ese día.", "english": "uh that that uh that day."}, "control": null}

Row 2: {"row_index": 2, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "3", "spanish_a": "uh al menos hasta este momento sus vidas uh terminaron igual pero cada uno tenía un destino diferente.", "english": "uh at least up until this moment their their lives souh uh ended the same but each one had a different destination."}}

Row 3: {"row_index": 3, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "4", "spanish_a": "Um y podemos y podemos ver que eh uno de ellos eh e el el criminal de la derecha incluso dice eh este hombre no ha hecho nada malo.", "english": "Um and you can and we can see that uh one of them uh e th the the criminal on on the right even says uh this man has done nothing wrong."}}

Row 4: {"row_index": 4, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "5", "spanish_a": "Y eh él sabía que este hombre era Cristo. Sabía que este hombre estaba allí", "english": "And uh he knew that this man was was Christ. He knew that this man was there"}}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "6", "spanish_a": "para morir por nuestros pecados, tus pecados y mis pecados. Y así esta noche, uno de estos dos criminales es", "english": "to uh die for our sins, your s um uh my sins and your sins. And so tonight, uh one of these two criminals is"}}

Row 8: {"row_index": 8, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "9", "spanish_a": "Pero esta noche, eh, si no estás salvado, eh, ¿cuál será tu destino?", "english": "But tonight are uh if you're not saved, uh which where w where will your destination be?"}}

Row 9: {"row_index": 9, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "10", "spanish_a": "Así que, uno de ellos", "english": "So well, one of them"}}

Row 11: {"row_index": 11, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "12", "spanish_a": "Ninguno de estos tipos respetó la ley, pero uno de ellos se despertó con Dios y el otro se despertó con el diablo.", "english": "Um th neither of these guys kept the law, but one of them uh woke up with God and the other one woke up with the devil."}}

Row 12: {"row_index": 12, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "13", "spanish_a": "Quiero preguntar de nuevo, ¿estás salvado? Um", "english": "I want to ask again, are you saved? Um"}}

Row 15: {"row_index": 15, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "16", "spanish_a": "uh la forma más brutal uh en que un hombre uh cualquier persona puede ser puto a muerte. Y sin embargo, estos tipos eran ambos pecadores", "english": "uh the most brutal way uh a man uh an any person can uh can be put to death. And yet these were these guys were both sinners"}}

Row 17: {"row_index": 17, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "18", "spanish_a": "Solo quiero que pienses en eso porque tienes a estos dos hombres que son similares pero también diferentes. Y", "english": "I I just want you to to to think about that because you have these two men who are similar but also different. And"}}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "20", "spanish_a": "¿Estás salvado? Y quizás estés pensando, bueno, yo no soy como estos dos hombres. Bueno, tú lo eres. Tú lo eres. Um, ya sea que quieras admitirlo o no.", "english": "Are you saved? And you might be thinking, well, I'm not like these two men. Well you you are. You are. Um whether you want to uh admit it or not."}}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "21", "spanish_a": "Todos somos como estos dos hombres. Obviamente, ya sabes, espero que nadie haya cometido asesinato. Um y si lo haces, entonces eh", "english": "We we are all like these two men. Obviously, you know, I hope anyone hasn't I hope no one's committed murder. Um and if you do then uh"}}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "23", "spanish_a": "Pero espero que nadie haya, ya sabes, cometido asesinato. Ya sabes. De nuevo, como dije, no importa qué tipo de pecado hayas cometido, todos hemos pecado, ya sea que sea", "english": "But I hope no one has, you know, committed murder. You know. Again, like I said, it doesn't matter what kind of sin you've committed, we've all sinned, whether that's"}}

Row 23: {"row_index": 23, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "24", "spanish_a": "tomar algo que no deberías o eh ya sabes, tal vez robar dulces", "english": "taking something that you shouldn't have or uh you know maybe stealing candy"}}

Row 24: {"row_index": 24, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "25", "spanish_a": "O um eh mentirle a tu mamá cada vez que te dice que hagas algo y dices que lo haces. Um eso es nosotros, todos somos pecadores.", "english": "Or um uh lying to your mom whenever she tells you to do something and you say you do it. Um that is we we are all we are all sinners."}}

Row 25: {"row_index": 25, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "26", "spanish_a": "Um y estos dos y y y estos dos estaban pagando por los pecados que eh habían cometido. Pero", "english": "Um and these two and and and these two were repaying for the sins that uh they had committed. But"}}

Row 26: {"row_index": 26, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "27", "spanish_a": "Ahora, ahora déjame preguntarte esto. ¿Deberías morir por los pecados que hemos cometido?", "english": "Um now now let me ask you this. Should you die for the should we should me and you should we die for the sins that we have uh committed?"}}

Row 27: {"row_index": 27, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "28", "spanish_a": "Y la respuesta a eso es que es sí. Um pero", "english": "And the answer to that is it's yes. Um but"}}

Row 31: {"row_index": 31, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "32", "spanish_a": "Vamos a sacar uno, eh, eh, bueno, quiero comparar a nosotros ahora con estos dos criminales. Entonces, ¿eres", "english": "Let's let's take out one w uh uh well I wanna compare us now to to these two criminals. So are you"}}

Row 32: {"row_index": 32, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "33", "spanish_a": "Bueno, sabes, con el crecer, sabes, puedes decir que he venido a la iglesia consistentemente. He, sabes, he estado, he estado bajo el evangelio.", "english": "Uh so you can you know with grow growing up, you know, you can you you you can you know say that well I've I've come to church consistently. I've you know I've been I've been underneath the gospel."}}

Row 33: {"row_index": 33, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "34", "spanish_a": "uh Yo yo me senté debajo del evangelio. No soy tan malo. Pero", "english": "uh I I I sat underneath the gospel. I'm not all that bad. But"}}

Row 34: {"row_index": 34, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "35", "spanish_a": "Sí, eso es bueno, pero a menos que no conozcas a Cristo como tu Salvador, no entrarás al cielo. Um y no quiero sonar", "english": "Yes, that's good, but unless you don't know Christ as your Savior, you're not getting into heaven. Um and I don't mean to sound"}}

Row 35: {"row_index": 35, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "36", "spanish_a": "sabes, um, franco cuando yo s oh no quiero sonar frívolo cuando digo eso, pero esa es la realidad de lo que um", "english": "you know, um, blunt when I s oh I I don't mean to sound flimpent when I say that, but that is the reality of of what um"}}

Row 36: {"row_index": 36, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "37", "spanish_a": "De", "english": "Of"}}

Row 37: {"row_index": 37, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "38", "spanish_a": "de nuestra de tu situación esta noche si no estás salvado. Um si no conoces a Cristo como tu Salvador, entonces um tú eh como estos dos hombres", "english": "of our of your situation tonight if you aren't saved. Um if you do not know Christ as your Savior, then um you uh just like these two men"}}

Row 38: {"row_index": 38, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "39", "spanish_a": "uh tendrás dos opciones. Puedes o bien despertar en el cielo o puedes despertar en el infierno. Um y", "english": "uh you will have two options. You can either uh wake up in heaven or you can wake up in hell. Um and"}}

Row 39: {"row_index": 39, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "40", "spanish_a": "Tienes esa elección de libre albedrío esta noche, pero eh eh esa es una decisión que tienes que tomar", "english": "You have that free will choice tonight, but uh uh that is a decision that you have to make"}}

Row 40: {"row_index": 40, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "41", "spanish_a": "y eh Dios no tomará esa decisión por ti. Él quiere estar en tu vida, pero tienes que aceptarlo y darte cuenta de que sin él,", "english": "and uh God won't make that decision for you. He wants to be in your life, but you have to accept him and realize that without him,"}}

Row 41: {"row_index": 41, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "42", "spanish_a": "uh tú uh no vas a entrar al cielo y estás en y estás y tienes que darte cuenta de que estás perdido en tus pecados esta noche.", "english": "uh you uh won't be getting into heaven and you are in and you're s and and you have to realize that you're lost in your sins tonight."}}

Row 43: {"row_index": 43, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "44", "spanish_a": "Um, al igual que estos dos criminales, um, tú tendrás dos opciones cuando se trata de", "english": "Um just like these two criminals, um you you will uh you have two options when when it comes to"}}

Row 44: {"row_index": 44, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "45", "spanish_a": "donde estará tu destino eterno. Uno de estos escuchó eh uno de estos eh criminales escuchó um verdaderamente tú eh verdaderamente eh", "english": "where your eternal destination will will be. One of the one of these heard uh one one of these uh criminals heard um truly you uh truly uh"}}

Row 45: {"row_index": 45, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "46", "spanish_a": "Uh lo siento, de verdad te digo, hoy estarás conmigo en el paraíso. Y el otro escuchó, Apártate de mí, apártate de mí, porque nunca te conocí.", "english": "Uh sorry, truly I say to you, today you'll be with me in paradise. And the other one heard, Depart from me Depart from me, for I never knew you."}}

Row 46: {"row_index": 46, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "47", "spanish_a": "Entonces, um, para mí personalmente, crecer eso me asustó, pero n yo luego, pero también tuve que llegar a la comprensión de que", "english": "So, um, for me personally growing up that that did scare me but n I then but I also had to come to a realization that"}}

Row 47: {"row_index": 47, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "48", "spanish_a": "No había nada que pudiera hacer. No había nada que no hubiera", "english": "there was nothing that I could do. There was nothing that there was no"}}

Row 48: {"row_index": 48, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "49", "spanish_a": "una ecuación, no había, ya sabes, eh largo, no había un algoritmo largo que tuviera que descifrar, que tuviera que hacer esto, esto y aquello", "english": "uh equation, there was no you know, uh long there's there was no long algorithm that I had to figure out, that I had to do this, this, and that"}}

Row 49: {"row_index": 49, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "50", "spanish_a": "No, yo vine en yo yo recuerdo que fue un um", "english": "No, I came on I I re uh I remember it was a um"}}

Row 50: {"row_index": 50, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "51", "spanish_a": "una tarde de verano en agosto, me di cuenta de que estoy perdido y que voy al infierno. Y me di cuenta", "english": "uh summer evening in August, I I came to a realization that I am lost and I am going to hell. And I realized"}}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "52", "spanish_a": "Y me di cuenta y clamé al Señor para que me salvara, y él lo hizo. Y esta noche tienes la opción de salir de aquí sabiendo que tus pecados son perdonados.", "english": "And I realized that and I cried out to the Lord to save me, and he did. And tonight you have the very option to walk out of here knowing that your sins are forgiven."}}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "53", "spanish_a": "que puedas tener un lugar en el cielo y um y estar con él para siempre. O", "english": "that you can have a a place in heaven and um and be with him forever. Or"}}

Row 58: {"row_index": 58, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "60", "spanish_a": "Esa última esa última sección ahí. No debería perecer, sino tener vida eterna. Eso no es solo una declaración, es una promesa.", "english": "That last that last section there. Should not have should not perish, but have eternal life. That isn't just a statement, that is a promise."}}

Row 59: {"row_index": 59, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "61", "spanish_a": "Eso es lo que es um", "english": "That is that is um"}}

Row 60: {"row_index": 60, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "62", "spanish_a": "Uh eso es Dios diciendo que si crees y si confías en lo que mi hijo hizo en esa cruz ese día.", "english": "Uh that is God saying that if you believe and you tr uh if you trust on what my son did on that cross that uh that that uh that day."}}

### Identity: lb0912_A_partial_recheck_translation_r0 vs lb0912_A_ctl_r0

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

### Identity: lb0912_A_partial_recheck_translation_r1 vs lb0912_A_ctl_r1

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

### Identity: lb0912_A_partial_recheck_translation_r2 vs lb0912_A_ctl_r2

Translation share: 1.0; English share: 1.0; aligned rows: 61; chunk-count difference: 0.

## Clip B

n by route covers all eligible finals; p50/p95 are all-route silence finals.

| Arm | n Gemma / Marian / unknown | Gemma silence n | Silence p50 | Silence p95 | Gemma silence p95 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| ctl | 162 / 78 / 0 | 129 | 1467.9 | 1843.2 | 1886.5 | — | — | — | — | — | — | — |
| marian_threads_2 | 162 / 78 / 0 | 129 | 1496.8 | 1942.8 | 2008.8 | FAIL | PASS | PASS | PASS | PASS | PASS | PASS |
| max_utterance_6 | 171 / 96 / 0 | 114 | 1384.7 | 1795.3 | 1841.9 | FAIL | PASS | PASS | PASS | PASS | PASS | FAIL |
| partial_recheck_translation | 162 / 78 / 0 | 129 | 1469.4 | 1887.9 | 1892.1 | FAIL | PASS | PASS | PASS | FAIL | PASS | PASS |

### Identity: lb0912_B_marian_threads_2_r0 vs lb0912_B_ctl_r0

Translation share: 1.0; English share: 1.0; aligned rows: 80; chunk-count difference: 0.

### Identity: lb0912_B_marian_threads_2_r1 vs lb0912_B_ctl_r1

Translation share: 1.0; English share: 1.0; aligned rows: 80; chunk-count difference: 0.

### Identity: lb0912_B_marian_threads_2_r2 vs lb0912_B_ctl_r2

Translation share: 1.0; English share: 1.0; aligned rows: 80; chunk-count difference: 0.

### Identity: lb0912_B_max_utterance_6_r0 vs lb0912_B_ctl_r0

Translation share: 1.0; English share: 0.7865168539325843; aligned rows: 70; chunk-count difference: 9.

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "o hará algo para merecernos el estatus de Dios", "english": "or will will will ever do something to earn us the status of God"}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "estar dispuestos a salvarnos.", "english": "being willing to save us."}, "control": null}

Row 8: {"row_index": 8, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "10", "spanish_a": "Porque de las riquezas", "english": "Because out of the the riches"}, "control": null}

Row 9: {"row_index": 9, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "11", "spanish_a": "de su gracia de la que estábamos escuchando hoy. Él ama y anhela salvar", "english": "of his grace that we were hearing about today. He loves and longs to save"}, "control": null}

Row 15: {"row_index": 15, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "17", "spanish_a": "¿Y qué le encanta hacer a un profesor? A un profesor le encanta enseñar.", "english": "And what does a teacher love to do? A teacher loves to teach."}, "control": null}

Row 16: {"row_index": 16, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "18", "spanish_a": "y hacer que la gente entienda lo que está enseñando.", "english": "and bring people to understand what it is that he's teaching."}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Y Dios aquí es descrito como el Salvador. Le encanta salvar. Él está lleno de misericordia.", "english": "And God here is described as the Saviour. He loves to save. He he's full of mercy."}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "Es amable.", "english": "He's kind."}, "control": null}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "24", "spanish_a": "Sí.", "english": "Yeah."}, "control": null}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "31", "spanish_a": "Y vamos a ver aquí en un segundo lo que hace Dios", "english": "And we're going to look at here in a second what it is that God does"}, "control": null}

Row 30: {"row_index": 30, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "32", "spanish_a": "Lo que ha hecho para lograr la salvación para ti y para mí.", "english": "What he has done to accomplish salvation for you and for me."}, "control": null}

Row 31: {"row_index": 31, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "33", "spanish_a": "La salvación realmente solo ocurre cuando Dios hace algo que es completamente", "english": "Salvation really only occurs when when God does something that is completely"}, "control": null}

Row 32: {"row_index": 32, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "34", "spanish_a": "fuera de nuestra capacidad de hacer.", "english": "outside of our ability to do."}, "control": null}

Row 50: {"row_index": 50, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "52", "spanish_a": "Al final del verso cuatro dice que él", "english": "At the end of verse four he says that he"}, "control": null}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "53", "spanish_a": "quiere que todos lleguen al conocimiento de la verdad.", "english": "wants all to come to the knowledge of the truth."}, "control": null}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "54", "spanish_a": "Así que la salvación no es algo que tú y yo logramos.", "english": "So salvation is not something that you and I accomplish."}, "control": null}

Row 53: {"row_index": 53, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "55", "spanish_a": "algo que hacemos o producimos", "english": "something that we do or produce"}, "control": null}

Row 78: {"row_index": 78, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "Y a partir de ese momento, toda la humanidad ha sido separada", "english": "And from that moment on, all of humanity has been separated"}, "control": null}

Row 79: {"row_index": 79, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "82", "spanish_a": "De Dios.", "english": "From God."}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "7", "spanish_a": "o haremos alguna vez algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will we will ever do something to earn us the status of God being willing to save us."}}

Row 7: {"row_index": 7, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "9", "spanish_a": "Porque de las riquezas de su gracia de las que estábamos escuchando hoy, él ama y anhela salvar", "english": "Because out of the the riches of his grace that we were hearing about today, he loves and longs to save"}}

Row 13: {"row_index": 13, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "15", "spanish_a": "¿Y qué le encanta hacer a un profesor? A un profesor le encanta enseñar y hacer que la gente entienda", "english": "And what does a teacher love to do? A teacher loves to teach and bring people to understand"}}

Row 16: {"row_index": 16, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "19", "spanish_a": "Y Dios aquí es descrito como el Salvador. Le encanta salvar. Él está lleno de misericordia. Es amable.", "english": "And God here is described as the Savior. He loves to save. He he's full of mercy. He's kind."}}

Row 24: {"row_index": 24, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "27", "spanish_a": "Y vamos a ver aquí en un segundo lo que es lo que Dios hace, lo que ha hecho", "english": "And we're gonna look at here in a second what it is that God does, what he has done"}}

Row 25: {"row_index": 25, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "28", "spanish_a": "lograr la salvación para ti y para mí.", "english": "to accomplish salvation for you and for me."}}

Row 26: {"row_index": 26, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "29", "spanish_a": "La salvación realmente solo ocurre cuando Dios hace algo que está completamente fuera de nuestra capacidad de hacer.", "english": "Salvation really only occurs when when God does something that is completely outside of our ability to do."}}

Row 44: {"row_index": 44, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "47", "spanish_a": "Al final del versículo cuatro dice que quiere que todos lleguen al conocimiento de la verdad.", "english": "At the end of verse four he says that he wants all to come to the knowledge of the truth."}}

Row 45: {"row_index": 45, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "48", "spanish_a": "Así que la salvación no es algo que tú y yo logramos, algo que hacemos o producimos.", "english": "So salvation is not something that you and I accomplish, something that we do or produce."}}

Row 70: {"row_index": 70, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "74", "spanish_a": "Y a partir de ese momento, toda la humanidad ha sido separada de Dios.", "english": "And from that moment on, all of humanity has been separated from God."}}

### Identity: lb0912_B_max_utterance_6_r1 vs lb0912_B_ctl_r1

Translation share: 1.0; English share: 0.7865168539325843; aligned rows: 70; chunk-count difference: 9.

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "o hará algo para merecernos el estatus de Dios", "english": "or will will will ever do something to earn us the status of God"}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "estar dispuestos a salvarnos.", "english": "being willing to save us."}, "control": null}

Row 8: {"row_index": 8, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "10", "spanish_a": "Porque de las riquezas", "english": "Because out of the the riches"}, "control": null}

Row 9: {"row_index": 9, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "11", "spanish_a": "de su gracia de la que estábamos escuchando hoy. Él ama y anhela salvar", "english": "of his grace that we were hearing about today. He loves and longs to save"}, "control": null}

Row 15: {"row_index": 15, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "17", "spanish_a": "¿Y qué le encanta hacer a un profesor? A un profesor le encanta enseñar.", "english": "And what does a teacher love to do? A teacher loves to teach."}, "control": null}

Row 16: {"row_index": 16, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "18", "spanish_a": "y hacer que la gente entienda lo que está enseñando.", "english": "and bring people to understand what it is that he's teaching."}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Y Dios aquí es descrito como el Salvador. Le encanta salvar. Él está lleno de misericordia.", "english": "And God here is described as the Saviour. He loves to save. He he's full of mercy."}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "Es amable.", "english": "He's kind."}, "control": null}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "24", "spanish_a": "Sí.", "english": "Yeah."}, "control": null}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "31", "spanish_a": "Y vamos a ver aquí en un segundo lo que hace Dios", "english": "And we're going to look at here in a second what it is that God does"}, "control": null}

Row 30: {"row_index": 30, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "32", "spanish_a": "Lo que ha hecho para lograr la salvación para ti y para mí.", "english": "What he has done to accomplish salvation for you and for me."}, "control": null}

Row 31: {"row_index": 31, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "33", "spanish_a": "La salvación realmente solo ocurre cuando Dios hace algo que es completamente", "english": "Salvation really only occurs when when God does something that is completely"}, "control": null}

Row 32: {"row_index": 32, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "34", "spanish_a": "fuera de nuestra capacidad de hacer.", "english": "outside of our ability to do."}, "control": null}

Row 50: {"row_index": 50, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "52", "spanish_a": "Al final del verso cuatro dice que él", "english": "At the end of verse four he says that he"}, "control": null}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "53", "spanish_a": "quiere que todos lleguen al conocimiento de la verdad.", "english": "wants all to come to the knowledge of the truth."}, "control": null}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "54", "spanish_a": "Así que la salvación no es algo que tú y yo logramos.", "english": "So salvation is not something that you and I accomplish."}, "control": null}

Row 53: {"row_index": 53, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "55", "spanish_a": "algo que hacemos o producimos", "english": "something that we do or produce"}, "control": null}

Row 78: {"row_index": 78, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "Y a partir de ese momento, toda la humanidad ha sido separada", "english": "And from that moment on, all of humanity has been separated"}, "control": null}

Row 79: {"row_index": 79, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "82", "spanish_a": "De Dios.", "english": "From God."}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "7", "spanish_a": "o haremos alguna vez algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will we will ever do something to earn us the status of God being willing to save us."}}

Row 7: {"row_index": 7, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "9", "spanish_a": "Porque de las riquezas de su gracia de las que estábamos escuchando hoy, él ama y anhela salvar", "english": "Because out of the the riches of his grace that we were hearing about today, he loves and longs to save"}}

Row 13: {"row_index": 13, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "15", "spanish_a": "¿Y qué le encanta hacer a un profesor? A un profesor le encanta enseñar y hacer que la gente entienda", "english": "And what does a teacher love to do? A teacher loves to teach and bring people to understand"}}

Row 16: {"row_index": 16, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "19", "spanish_a": "Y Dios aquí es descrito como el Salvador. Le encanta salvar. Él está lleno de misericordia. Es amable.", "english": "And God here is described as the Savior. He loves to save. He he's full of mercy. He's kind."}}

Row 24: {"row_index": 24, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "27", "spanish_a": "Y vamos a ver aquí en un segundo lo que es lo que Dios hace, lo que ha hecho", "english": "And we're gonna look at here in a second what it is that God does, what he has done"}}

Row 25: {"row_index": 25, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "28", "spanish_a": "lograr la salvación para ti y para mí.", "english": "to accomplish salvation for you and for me."}}

Row 26: {"row_index": 26, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "29", "spanish_a": "La salvación realmente solo ocurre cuando Dios hace algo que está completamente fuera de nuestra capacidad de hacer.", "english": "Salvation really only occurs when when God does something that is completely outside of our ability to do."}}

Row 44: {"row_index": 44, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "47", "spanish_a": "Al final del versículo cuatro dice que quiere que todos lleguen al conocimiento de la verdad.", "english": "At the end of verse four he says that he wants all to come to the knowledge of the truth."}}

Row 45: {"row_index": 45, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "48", "spanish_a": "Así que la salvación no es algo que tú y yo logramos, algo que hacemos o producimos.", "english": "So salvation is not something that you and I accomplish, something that we do or produce."}}

Row 70: {"row_index": 70, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "74", "spanish_a": "Y a partir de ese momento, toda la humanidad ha sido separada de Dios.", "english": "And from that moment on, all of humanity has been separated from God."}}

### Identity: lb0912_B_max_utterance_6_r2 vs lb0912_B_ctl_r2

Translation share: 1.0; English share: 0.7865168539325843; aligned rows: 70; chunk-count difference: 9.

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "7", "spanish_a": "o hará algo para merecernos el estatus de Dios", "english": "or will will will ever do something to earn us the status of God"}, "control": null}

Row 6: {"row_index": 6, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "8", "spanish_a": "estar dispuestos a salvarnos.", "english": "being willing to save us."}, "control": null}

Row 8: {"row_index": 8, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "10", "spanish_a": "Porque de las riquezas", "english": "Because out of the the riches"}, "control": null}

Row 9: {"row_index": 9, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "11", "spanish_a": "de su gracia de la que estábamos escuchando hoy. Él ama y anhela salvar", "english": "of his grace that we were hearing about today. He loves and longs to save"}, "control": null}

Row 15: {"row_index": 15, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "17", "spanish_a": "¿Y qué le encanta hacer a un profesor? A un profesor le encanta enseñar.", "english": "And what does a teacher love to do? A teacher loves to teach."}, "control": null}

Row 16: {"row_index": 16, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "18", "spanish_a": "y hacer que la gente entienda lo que está enseñando.", "english": "and bring people to understand what it is that he's teaching."}, "control": null}

Row 19: {"row_index": 19, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "21", "spanish_a": "Y Dios aquí es descrito como el Salvador. Le encanta salvar. Él está lleno de misericordia.", "english": "And God here is described as the Saviour. He loves to save. He he's full of mercy."}, "control": null}

Row 20: {"row_index": 20, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "22", "spanish_a": "Es amable.", "english": "He's kind."}, "control": null}

Row 22: {"row_index": 22, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "24", "spanish_a": "Sí.", "english": "Yeah."}, "control": null}

Row 29: {"row_index": 29, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "31", "spanish_a": "Y vamos a ver aquí en un segundo lo que hace Dios", "english": "And we're going to look at here in a second what it is that God does"}, "control": null}

Row 30: {"row_index": 30, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "32", "spanish_a": "Lo que ha hecho para lograr la salvación para ti y para mí.", "english": "What he has done to accomplish salvation for you and for me."}, "control": null}

Row 31: {"row_index": 31, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "33", "spanish_a": "La salvación realmente solo ocurre cuando Dios hace algo que es completamente", "english": "Salvation really only occurs when when God does something that is completely"}, "control": null}

Row 32: {"row_index": 32, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "34", "spanish_a": "fuera de nuestra capacidad de hacer.", "english": "outside of our ability to do."}, "control": null}

Row 50: {"row_index": 50, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "52", "spanish_a": "Al final del verso cuatro dice que él", "english": "At the end of verse four he says that he"}, "control": null}

Row 51: {"row_index": 51, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "53", "spanish_a": "quiere que todos lleguen al conocimiento de la verdad.", "english": "wants all to come to the knowledge of the truth."}, "control": null}

Row 52: {"row_index": 52, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "54", "spanish_a": "Así que la salvación no es algo que tú y yo logramos.", "english": "So salvation is not something that you and I accomplish."}, "control": null}

Row 53: {"row_index": 53, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "55", "spanish_a": "algo que hacemos o producimos", "english": "something that we do or produce"}, "control": null}

Row 78: {"row_index": 78, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "81", "spanish_a": "Y a partir de ese momento, toda la humanidad ha sido separada", "english": "And from that moment on, all of humanity has been separated"}, "control": null}

Row 79: {"row_index": 79, "fields": ["spanish_a", "english"], "candidate": {"chunk_id": "82", "spanish_a": "De Dios.", "english": "From God."}, "control": null}

Row 5: {"row_index": 5, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "7", "spanish_a": "o haremos alguna vez algo para merecernos el estatus de que Dios está dispuesto a salvarnos.", "english": "or will we will ever do something to earn us the status of God being willing to save us."}}

Row 7: {"row_index": 7, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "9", "spanish_a": "Porque de las riquezas de su gracia de las que estábamos escuchando hoy, él ama y anhela salvar", "english": "Because out of the the riches of his grace that we were hearing about today, he loves and longs to save"}}

Row 13: {"row_index": 13, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "15", "spanish_a": "¿Y qué le encanta hacer a un profesor? A un profesor le encanta enseñar y hacer que la gente entienda", "english": "And what does a teacher love to do? A teacher loves to teach and bring people to understand"}}

Row 16: {"row_index": 16, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "19", "spanish_a": "Y Dios aquí es descrito como el Salvador. Le encanta salvar. Él está lleno de misericordia. Es amable.", "english": "And God here is described as the Savior. He loves to save. He he's full of mercy. He's kind."}}

Row 24: {"row_index": 24, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "27", "spanish_a": "Y vamos a ver aquí en un segundo lo que es lo que Dios hace, lo que ha hecho", "english": "And we're gonna look at here in a second what it is that God does, what he has done"}}

Row 25: {"row_index": 25, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "28", "spanish_a": "lograr la salvación para ti y para mí.", "english": "to accomplish salvation for you and for me."}}

Row 26: {"row_index": 26, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "29", "spanish_a": "La salvación realmente solo ocurre cuando Dios hace algo que está completamente fuera de nuestra capacidad de hacer.", "english": "Salvation really only occurs when when God does something that is completely outside of our ability to do."}}

Row 44: {"row_index": 44, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "47", "spanish_a": "Al final del versículo cuatro dice que quiere que todos lleguen al conocimiento de la verdad.", "english": "At the end of verse four he says that he wants all to come to the knowledge of the truth."}}

Row 45: {"row_index": 45, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "48", "spanish_a": "Así que la salvación no es algo que tú y yo logramos, algo que hacemos o producimos.", "english": "So salvation is not something that you and I accomplish, something that we do or produce."}}

Row 70: {"row_index": 70, "fields": ["spanish_a", "english"], "candidate": null, "control": {"chunk_id": "74", "spanish_a": "Y a partir de ese momento, toda la humanidad ha sido separada de Dios.", "english": "And from that moment on, all of humanity has been separated from God."}}

### Identity: lb0912_B_partial_recheck_translation_r0 vs lb0912_B_ctl_r0

Translation share: 1.0; English share: 1.0; aligned rows: 80; chunk-count difference: 0.

### Identity: lb0912_B_partial_recheck_translation_r1 vs lb0912_B_ctl_r1

Translation share: 1.0; English share: 1.0; aligned rows: 80; chunk-count difference: 0.

### Identity: lb0912_B_partial_recheck_translation_r2 vs lb0912_B_ctl_r2

Translation share: 1.0; English share: 1.0; aligned rows: 80; chunk-count difference: 0.

## Experiment counters

| Run | Source | Counters |
| --- | --- | --- |
| lb0912_A_ctl_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 9, "partial_suppressed_translation_running": 0} |
| lb0912_A_marian_threads_2_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 5, "partial_suppressed_translation_running": 0} |
| lb0912_A_max_utterance_6_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 1, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 1, "partial_suppressed_translation_running": 0} |
| lb0912_A_partial_recheck_translation_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 9, "partial_suppressed_translation_running": 82} |
| lb0912_A_partial_recheck_translation_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 2, "partial_suppressed_translation_running": 79} |
| lb0912_A_max_utterance_6_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 5, "partial_suppressed_translation_running": 0} |
| lb0912_A_marian_threads_2_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 9, "partial_suppressed_translation_running": 0} |
| lb0912_A_ctl_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 8, "partial_suppressed_translation_running": 0} |
| lb0912_A_ctl_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 13, "partial_suppressed_translation_running": 0} |
| lb0912_A_partial_recheck_translation_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 3, "partial_suppressed_translation_running": 78} |
| lb0912_A_marian_threads_2_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 2, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 1, "partial_suppressed_in_flight": 25, "partial_suppressed_translation_running": 0} |
| lb0912_A_max_utterance_6_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 19, "partial_suppressed_translation_running": 0} |
| lb0912_B_ctl_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 3, "partial_suppressed_translation_running": 0} |
| lb0912_B_marian_threads_2_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 1, "partial_suppressed_translation_running": 0} |
| lb0912_B_max_utterance_6_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 1, "partial_suppressed_translation_running": 0} |
| lb0912_B_partial_recheck_translation_r0 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 4, "partial_suppressed_translation_running": 32} |
| lb0912_B_partial_recheck_translation_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 1, "partial_suppressed_translation_running": 34} |
| lb0912_B_max_utterance_6_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 1, "partial_suppressed_translation_running": 0} |
| lb0912_B_marian_threads_2_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 1, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 5, "partial_suppressed_translation_running": 0} |
| lb0912_B_ctl_r1 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 8, "partial_suppressed_published_final": 1, "partial_suppressed_translation_running": 0} |
| lb0912_B_ctl_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 1, "partial_suppressed_translation_running": 0} |
| lb0912_B_partial_recheck_translation_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 3, "partial_suppressed_translation_running": 31} |
| lb0912_B_marian_threads_2_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 1, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 1, "partial_suppressed_in_flight": 3, "partial_suppressed_translation_running": 0} |
| lb0912_B_max_utterance_6_r2 | diagnostics_jsonl.session_summary | {"final_stt_waited_for_translation": 0, "partial_suppressed_after_stt": 0, "partial_suppressed_backlog": 0, "partial_suppressed_empty_translation": 0, "partial_suppressed_final_decode": 0, "partial_suppressed_final_pending": 0, "partial_suppressed_in_flight": 1, "partial_suppressed_translation_running": 0} |

## Outcomes

ctl p95_claim_eligible: false — screen without p95 claim.

marian_threads_2: REJECTED

Failing gates: A:G1, A:G2, A:G5, B:G1.

marian_threads_2 p95_claim_eligible: false — screen without p95 claim.

max_utterance_6: REJECTED

Failing gates: A:G1, A:G2, A:G5, A:G7, B:G1, B:G7.

max_utterance_6 p95_claim_eligible: false — screen without p95 claim.

partial_recheck_translation: REJECTED

Failing gates: A:G1, A:G5, B:G1, B:G5.

partial_recheck_translation p95_claim_eligible: false — screen without p95 claim.
