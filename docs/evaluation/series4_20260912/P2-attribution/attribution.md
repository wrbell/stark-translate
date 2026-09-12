# P2 attribution — last emitted partial vs final STT (silence finals)

| cohort | silence finals | with partial | gap ≤100 ms | ≤300 ms | ≤600 ms | gap p50/p95 ms | partial STT done before finalize | lead p50 ms (min) | text equal (norm) | prefix | differs | final STT p50/p95 ms | Gemma-routed | reuse@100 / @300 (Gemma) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lb0912_A_ctl_r0 | 21 | 20 | 70.0 % | 90.0 % | 100.0 % | -160/352 | 100.0 % | 208 (29) | 85.0 % | 5.0 % | 10.0 % | 439/1165 | 15 | 10 / 14 |
| lb0912_A_ctl_r1 | 21 | 20 | 80.0 % | 100.0 % | 100.0 % | -160/192 | 100.0 % | 160 (9) | 90.0 % | 0.0 % | 10.0 % | 361/502 | 15 | 10 / 14 |
| lb0912_A_ctl_r2 | 21 | 20 | 75.0 % | 95.0 % | 95.0 % | -160/224 | 100.0 % | 166 (29) | 85.0 % | 5.0 % | 10.0 % | 352/546 | 15 | 9 / 13 |
| lb0912_B_ctl_r0 | 68 | 67 | 74.6 % | 98.5 % | 98.5 % | 0/192 | 100.0 % | 339 (9) | 82.1 % | 1.5 % | 16.4 % | 366/484 | 43 | 33 / 42 |
| lb0912_B_ctl_r1 | 68 | 67 | 76.1 % | 100.0 % | 100.0 % | 0/192 | 100.0 % | 352 (1) | 83.6 % | 0.0 % | 16.4 % | 355/441 | 43 | 34 / 43 |
| lb0912_B_ctl_r2 | 68 | 67 | 76.1 % | 100.0 % | 100.0 % | 0/192 | 100.0 % | 355 (3) | 83.6 % | 0.0 % | 16.4 % | 355/447 | 43 | 34 / 43 |
| clip A pooled | 63 | 60 | 75.0 % | 95.0 % | 98.3 % | -160/224 | 100.0 % | 166 (9) | 86.7 % | 3.3 % | 10.0 % | 392/565 | 45 | 29 / 41 |
| clip B pooled | 204 | 201 | 75.6 % | 99.5 % | 99.5 % | 0/192 | 100.0 % | 346 (1) | 83.1 % | 0.5 % | 16.4 % | 360/447 | 129 | 101 / 128 |
| all pooled | 267 | 261 | 75.5 % | 98.5 % | 99.2 % | -32/192 | 100.0 % | 318 (1) | 83.9 % | 1.1 % | 14.9 % | 360/489 | 174 | 130 / 169 |

Differing silence finals (partial text vs raw final STT, not equal and not a prefix): 42

- `lb0912_A_ctl_r0` u45 gap 384 ms — partial: “Um” / final: “Um so now”
- `lb0912_A_ctl_r0` u52 gap -192 ms — partial: “No, I came on I I rem uh I remember it was a um” / final: “No, I came on I I re uh I remember it was a um”
- `lb0912_A_ctl_r0` u63 gap -224 ms — partial: “Uh just like the verse behind me, um John three, sixteen. Well we've all we all know that verse.” / final: “Uh just like the verse behind me, um John three, sixteen. Uh we've all we all know that verse.”
- `lb0912_A_ctl_r1` u52 gap -192 ms — partial: “No, I came on I I rem uh I remember it was a um” / final: “No, I came on I I re uh I remember it was a um”
- `lb0912_A_ctl_r1` u63 gap -224 ms — partial: “Uh just like the verse behind me, um John three, sixteen. Well we've all we all know that verse.” / final: “Uh just like the verse behind me, um John three, sixteen. Uh we've all we all know that verse.”
- `lb0912_A_ctl_r2` u13 gap 1504 ms — partial: “I want to ask again, are you” / final: “I want to ask again, are you saved? Um”
- `lb0912_A_ctl_r2` u52 gap -192 ms — partial: “No, I came on I I rem uh I remember it was a um” / final: “No, I came on I I re uh I remember it was a um”
- `lb0912_A_ctl_r2` u63 gap -224 ms — partial: “Uh just like the verse behind me, um John three, sixteen. Well we've all we all know that verse.” / final: “Uh just like the verse behind me, um John three, sixteen. Uh we've all we all know that verse.”
- `lb0912_B_ctl_r0` u3 gap 32 ms — partial: “What does a saviour do? A saviour saves.” / final: “And what does a savior do? A savior saves.”
- `lb0912_B_ctl_r0` u4 gap 704 ms — partial: “God is a God who loves to” / final: “God is a God who loves to save.”
- `lb0912_B_ctl_r0` u7 gap 96 ms — partial: “or will w will ever do something to earn us the status of God being willing to save us.” / final: “or will we will ever do something to earn us the status of God being willing to save us.”
- `lb0912_B_ctl_r0` u21 gap -320 ms — partial: “But he wants everyone to come to repentance. God” / final: “But he wants everyone to come to repentance.”
- `lb0912_B_ctl_r0` u22 gap -224 ms — partial: “He is a saviour and he loves to save.” / final: “He is a savior and he loves to save.”
- `lb0912_B_ctl_r0` u26 gap 160 ms — partial: “God is the Saviour. He does all the saving.” / final: “God is the savior. He does all the saving.”
- `lb0912_B_ctl_r0` u32 gap -320 ms — partial: “If if salvation was something we could do, then we wouldn't be in need of rescue.” / final: “If salvation was something we could do, then we wouldn't be in need of rescue.”
- `lb0912_B_ctl_r0` u33 gap 192 ms — partial: “So for my third and final point,” / final: “And so for my third and final point,”
- `lb0912_B_ctl_r0` u51 gap 160 ms — partial: “It's actually coming to faith.” / final: “But it's actually coming to faith”
- `lb0912_B_ctl_r0` u82 gap 32 ms — partial: “out of his presence. And Isaiah writes” / final: “And Isaiah writes”
- `lb0912_B_ctl_r0` u84 gap 224 ms — partial: “It doesn't say it's” / final: “He doesn't say it's a good idea.”
- `lb0912_B_ctl_r0` u86 gap 256 ms — partial: “He says your sin” / final: “He says,”
- `lb0912_B_ctl_r1` u3 gap 32 ms — partial: “What does a saviour do? A saviour saves.” / final: “And what does a savior do? A savior saves.”
- `lb0912_B_ctl_r1` u7 gap 96 ms — partial: “or will w will ever do something to earn us the status of God being willing to save us.” / final: “or will we will ever do something to earn us the status of God being willing to save us.”
- `lb0912_B_ctl_r1` u21 gap -320 ms — partial: “But he wants everyone to come to repentance. God” / final: “But he wants everyone to come to repentance.”
- `lb0912_B_ctl_r1` u22 gap -224 ms — partial: “He is a saviour and he loves to save.” / final: “He is a savior and he loves to save.”
- `lb0912_B_ctl_r1` u26 gap 160 ms — partial: “God is the Saviour. He does all the saving.” / final: “God is the savior. He does all the saving.”
- `lb0912_B_ctl_r1` u32 gap -320 ms — partial: “If if salvation was something we could do, then we wouldn't be in need of rescue.” / final: “If salvation was something we could do, then we wouldn't be in need of rescue.”
- `lb0912_B_ctl_r1` u33 gap 192 ms — partial: “So for my third and final point,” / final: “And so for my third and final point,”
- `lb0912_B_ctl_r1` u51 gap 160 ms — partial: “It's actually coming to faith.” / final: “But it's actually coming to faith”
- `lb0912_B_ctl_r1` u82 gap 32 ms — partial: “out of his presence. And Isaiah writes” / final: “And Isaiah writes”
- `lb0912_B_ctl_r1` u84 gap 224 ms — partial: “It doesn't say it's” / final: “He doesn't say it's a good idea.”
- `lb0912_B_ctl_r1` u86 gap 256 ms — partial: “He says your sin” / final: “He says,”
- `lb0912_B_ctl_r2` u3 gap 32 ms — partial: “What does a saviour do? A saviour saves.” / final: “And what does a savior do? A savior saves.”
- `lb0912_B_ctl_r2` u7 gap 96 ms — partial: “or will w will ever do something to earn us the status of God being willing to save us.” / final: “or will we will ever do something to earn us the status of God being willing to save us.”
- `lb0912_B_ctl_r2` u21 gap -320 ms — partial: “But he wants everyone to come to repentance. God” / final: “But he wants everyone to come to repentance.”
- `lb0912_B_ctl_r2` u22 gap -224 ms — partial: “He is a saviour and he loves to save.” / final: “He is a savior and he loves to save.”
- `lb0912_B_ctl_r2` u26 gap 160 ms — partial: “God is the Saviour. He does all the saving.” / final: “God is the savior. He does all the saving.”
- `lb0912_B_ctl_r2` u32 gap -320 ms — partial: “If if salvation was something we could do, then we wouldn't be in need of rescue.” / final: “If salvation was something we could do, then we wouldn't be in need of rescue.”
- `lb0912_B_ctl_r2` u33 gap 192 ms — partial: “So for my third and final point,” / final: “And so for my third and final point,”
- `lb0912_B_ctl_r2` u51 gap 160 ms — partial: “It's actually coming to faith.” / final: “But it's actually coming to faith”
- `lb0912_B_ctl_r2` u82 gap 32 ms — partial: “out of his presence. And Isaiah writes” / final: “And Isaiah writes”
- `lb0912_B_ctl_r2` u84 gap 224 ms — partial: “It doesn't say it's” / final: “He doesn't say it's a good idea.”
- `lb0912_B_ctl_r2` u86 gap 256 ms — partial: “He says your sin” / final: “He says,”
