# Mac E4B / E2B comparison

E4B remains the default. No model-selection UI has been added.

Human-reference audio coverage: {"approved_natural_utterances": {"en": 0, "es": 0}, "natural_audio_gate": false}

Natural Spanish / two-speaker / bilingual review gates remain pending until explicitly validated.

## Translation on identical input

chrF++ measures agreement with public-domain verse wording, not overall translation accuracy. References use unique book/chapter/verse alignment to KJV and RVR1909; ambiguous source text has no score. Existing model outputs are preserved when references are repaired. Canary checks require every listed term; older benchmarks checked only the first term.

| Model | Prompt | Direction | Items | References | p50 ms | p95 ms | Canaries | chrF++ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| e2b | none | en→hi | 43 | 0 | 552.6 | 922.2 | 0/0 | — |
| e4b | none | en→hi | 43 | 0 | 891.9 | 1618.5 | 0/0 | — |

### All theological canaries

| Input | Model / prompt | Result | Required terms | Output |
|---|---|---|---|---|

## E2B tradeoff relative to E4B

| Prompt | Direction | Median latency reduction | Canary pass difference | chrF++ difference |
|---|---|---:|---:|---:|
| none | en→hi | 38.0% | +0 | — |

## STT on saved session audio

Unreviewed machine transcripts are not WER references. Unconfirmed session recordings cannot pass the natural-speech gate.

| Engine | Language | Audio items | Approved references | WER | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|

## Real-time server latency

Speech end is estimated from captured VAD-positive frames. Server-final latency ends at payload readiness; it is not browser display latency. Clips, inference code/config cohorts, source types and endpoints are kept separate. Cohort IDs bind recorded source hashes, package versions and effective settings. A matching lifecycle source hash observed at startup takes precedence over later pipeline-file snapshots; this does not prove imported engine bytecode, and older snapshot timing may be ambiguous.

| Experiment | Model | Clip / cohort | Language/source | Endpoint | n | p50 ms | p95 ms |
|---|---|---|---|---|---:|---:|---:|

## Partial delivery and visible browser timing

First-partial delay starts at the first captured speech frame; one earliest delay is counted per known utterance. Update gaps use chronological emissions, including speaking pauses. Browser receipt-to-render is measured per visible client acknowledgment; speech-end-to-ack includes return-network time. Duplicate or non-v2 acknowledgments are excluded.

| Experiment | Model | Clip / cohort | Source | Metric | n | p50 ms | p95 ms |
|---|---|---|---|---|---:|---:|---:|

### Visible final acknowledgment coverage

Coverage counts each finalized chunk once when any visible client acknowledged it. Missing acknowledgments are missing evidence, not proof of display failure; replay shutdown can race the final browser acknowledgment.

| Experiment | Model | Clip / cohort | Received final chunks | Finalized chunks |
|---|---|---|---:|---:|

## Changed translation examples

**none_hi_ed39cc872669cf26_canary_00** — The atonement of Christ reconciles us to God.

- E4B: मसीह का प्रायश्चित हमें परमेश्वर से मिलाता है।
- E2B: मसीह का प्रायश्चित हमें परमेश्वर से मेल कराता है।

**none_hi_ed39cc872669cf26_canary_01** — James wrote about faith and works.

- E4B: जेम्स ने विश्वास और कर्मों के बारे में लिखा।
- E2B: जेम्स ने विश्वास और कर्म के बारे में लिखा।

**none_hi_ed39cc872669cf26_canary_03** — The breaking of bread is a solemn remembrance.

- E4B: रोटी का टूटना एक गंभीर स्मरण है।
- E2B: रोटी तोड़ना एक गंभीर स्मरण है।

**none_hi_ed39cc872669cf26_canary_04** — Paul wrote to the Corinthians about the resurrection.

- E4B: पौलुस ने कुरिन्थियों को पुनरुत्थान के बारे में लिखा।
- E2B: पौल ने कोरिंथियों को पुनरुत्थान के बारे में लिखा।

**none_hi_ed39cc872669cf26_canary_05** — Justification is by faith alone, not by works of the law.

- E4B: औचित्य केवल विश्वास से है, न कि नियम के कामों से।
- E2B: न्याय केवल विश्वास से है, विधि के कर्मों से नहीं।

**none_hi_ed39cc872669cf26_canary_06** — The Holy Spirit convicts the world of sin and righteousness.

- E4B: पवित्र आत्मा संसार को पाप और धार्मिकता का बोध कराती है।
- E2B: पवित्र आत्मा दुनिया को पाप और धार्मिकता के लिए दोषी ठहराता है।

**none_hi_ed39cc872669cf26_canary_07** — Grace and mercy meet at the cross of our Lord Jesus Christ.

- E4B: कृपा और दया हमारे प्रभु यीशु मसीह के क्रूस पर मिलते हैं।
- E2B: कृपा और दया हमारे प्रभु यीशु मसीह के क्रूस पर मिलती है।

**none_hi_ed39cc872669cf26_canary_08** — We gather before the mercy seat in prayer.

- E4B: हम प्रार्थना में दया के आसन के सामने एकत्रित होते हैं।
- E2B: हम प्रार्थना के लिए मरसी सीट से पहले इकट्ठा होते हैं।

**none_hi_ed39cc872669cf26_canary_09** — The table of the Lord is spread for His people.

- E4B: प्रभु की मेज उसके लोगों के लिए बिछाई गई है।
- E2B: प्रभु की तालिका उसके लोगों के लिए बिछाई गई है।

**none_hi_ed39cc872669cf26_canary_10** — We are gathered to Thy Name on the first day of the week.

- E4B: हम सप्ताह के पहले दिन तेरे नाम पर एकत्रित हुए हैं।
- E2B: हम सप्ताह के पहले दिन तुम्हारे नाम पर एकत्रित हुए हैं।

**none_hi_ed39cc872669cf26_canary_11** — A little while, and we shall see the Lord.

- E4B: थोड़ी देर में, हम प्रभु को देखेंगे।
- E2B: थोड़ी देर में, और हम प्रभु को देखेंगे।

**none_hi_ed39cc872669cf26_canary_12** — Nothing but the blood of Jesus can wash away my sin.

- E4B: यीशु का लहू ही मेरे पापों को धो सकता है।
- E2B: केवल यीशु का लहू ही मेरे पापों को धो सकता है।

**none_hi_ed39cc872669cf26_canary_13** — Christ is the Surety of a better covenant.

- E4B: मसीह एक बेहतर वाचा का surety है।
- E2B: मसीह एक बेहतर वाचा का साखकर्ता है।

**none_hi_ed39cc872669cf26_canary_14** — This do in remembrance of Me at His table.

- E4B: यह मेरे स्मरण में उसकी मेज पर किया जाता है।
- E2B: यह मेरे स्मरण में उनके मेज पर किया जाता है।

**none_hi_ed39cc872669cf26_canary_15** — At the mercy seat we remember His propitiation for our sins.

- E4B: दया की सीट पर हम अपने पापों के लिए उनकी क्षमा को याद करते हैं।
- E2B: मर्सी सीट पर हम हमारे पापों के लिए उनकी मध्यस्थता याद करते हैं।

**none_hi_ed39cc872669cf26_canary_16** — Yet there is room at the gospel feast for the weary soul.

- E4B: फिर भी, थकी हुई आत्मा के लिए सुसमाचार भोज में जगह है।
- E2B: फिर भी सुसमाचार के भोज में थके हुए आत्मा के लिए जगह है।

**none_hi_ed39cc872669cf26_canary_17** — We come as pilgrims to Calvary and the cross.

- E4B: हम कैल्वरी और क्रूस पर तीर्थयात्री बनकर आते हैं।
- E2B: हम कलवरी और क्रॉस के तीर्थयात्री के रूप में आते हैं।

**none_hi_ed39cc872669cf26_verse_00_en** — They look at themselves, then go on their way,

- E4B: वे खुद को देखते हैं, फिर अपने रास्ते चले जाते हैं,
- E2B: वे खुद को देखते हैं, फिर चले जाते हैं।

**none_hi_ed39cc872669cf26_verse_01_en** — The trees of the Lord drink their fill the cedars he planted on Lebanon,

- E4B: प्रभु के वृक्ष तृप्त होते हैं, लेबनान में लगाए गए देवदार।
- E2B: प्रभु के वृक्ष लेबनान में लगाए गए देवदार को तृप्त करते हैं।

**none_hi_ed39cc872669cf26_verse_02_en** — From Kibroth-hattaavah the people journeyed unto Hazeroth; and they abode at Hazeroth.

- E4B: किब्रोट-हत्तावा से लोग हाज़ेरोथ की ओर चले और वे हाज़ेरोथ में ठहरे।
- E2B: किब्रोथ-हतावा से लोग हाज़रथ की ओर यात्रा करते थे; और वे हाज़रथ में निवास करते थे।

**none_hi_ed39cc872669cf26_verse_03_en** — and from the Nethinim, whom David and the heads gave for the service of the Levites, two hundred and twenty Nethinim, all of them defined by name.

- E4B: और नेतिमिनों से, जिन्हें दाऊद और प्रमुखों ने लेवियों की सेवा के लिए दिया था, दो सौ बीस नेतिमिन, सभी के नाम बताए गए थे।
- E2B: और नेथिनिमों से, जिन्हें दाऊद और प्रमुखों ने लेवीयों की सेवा के लिए दिया था, दो सौ बीस नेथिनिम, सभी का नाम से परिभाषित किया गया था।

**none_hi_ed39cc872669cf26_verse_04_en** — And the name of the daughter of Asher <FI>is<Fi> Sarah.

- E4B: और आशेर की बेटी का नाम सारा है।
- E2B: और Asher की बेटी का नाम सारा है।

**none_hi_ed39cc872669cf26_verse_05_en** — If thou lend money to any of my people with thee that is poor, thou shalt not be to him as a creditor; neither shall ye lay upon him interest.

- E4B: यदि तू मेरे लोगों में से किसी गरीब को धन उधार देता है, तो तू उसका लेनदार नहीं होगा; न ही तू उस पर ब्याज लगाएगा।
- E2B: यदि तुम अपने किसी भी व्यक्ति को धन उधार देते हो जो गरीब है, तो तुम उसके लिए लेनदार नहीं होगे; न ही तुम उस पर ब्याज लगाओगे।

**none_hi_ed39cc872669cf26_verse_06_en** — But I counted it necessary to send to you Epaphroditus, my brother and fellow-worker and fellow-soldier, and your messenger and minister to my need;

- E4B: लेकिन मैंने तुम्हारे पास मेरे भाई और सहकर्मी और साथी सैनिक, और मेरी आवश्यकता के लिए तुम्हारा दूत और सेवक, एपैफ्रोडिटस को भेजना आवश्यक समझा;
- E2B: परन्तु मैंने आपको एपैफ़्रोडिटस, मेरे भाई और सहकर्मी और सह-सैनिक, और मेरी आवश्यकता के लिए आपके दूत और मंत्री को भेजने को आवश्यक समझा;

**none_hi_ed39cc872669cf26_verse_07_en** — And this is the testimony of John, when the Jews sent out of Jerusalem priests and Levites, that they might question him, `Who art thou?'

- E4B: और यह यूहन्ना की गवाही है, जब यहूदियों ने यरूशलेम से पुरोहितों और लेवियों को भेजा, ताकि वे उससे पूछ सकें, 'तू कौन है?'
- E2B: और यह जॉन की गवाही है, जब यहूदी यरूशलेम से पुजारी और लेवीय भेजते थे, ताकि वे उससे पूछताछ कर सकें, "तुम कौन हो?"

**none_hi_ed39cc872669cf26_verse_08_en** — And thou shalt make holy garments for Aaron thy brother, for glory and for beauty.

- E4B: और तू अपने भाई हारून के लिए महिमा और सुंदरता के लिए पवित्र वस्त्र बनाएगा।
- E2B: और तुम हारून के लिए पवित्र वस्त्र बनाओ, महिमा और सुंदरता के लिए।

**none_hi_ed39cc872669cf26_verse_09_en** — and they said “I will come quickly.”Blessed will he be who lays to heart the words of the prophecy contained in this book.’

- E4B: और उन्होंने कहा, "मैं जल्दी आऊंगा।" धन्य होगा वह जो इस पुस्तक में निहित भविष्यवाणी के शब्दों को हृदय में रखेगा।
- E2B: और उन्होंने कहा "मैं जल्दी आऊंगा।" वह धन्य होगा जो इस पुस्तक में निहित भविष्यवाणियों के शब्दों को हृदय में रखेगा।

**none_hi_ed39cc872669cf26_verse_10_en** — and again it saith, `Rejoice ye nations, with His people;'

- E4B: और फिर यह कहता है, "हे राष्ट्रों, उसके लोगों के साथ आनन्दित होओ;"
- E2B: और फिर वह कहता है, "राष्ट्रों, अपने लोगों के साथ आनन्द मनाओ;"

**none_hi_ed39cc872669cf26_verse_11_en** — So two or three cities wandered unto one city, to drink water; but they were not satisfied: yet have ye not returned unto me, saith the Lord.

- E4B: तो दो या तीन नगर पानी पीने एक नगर में भटक गए; परन्तु वे तृप्त नहीं हुए: फिर भी तुम मेरे पास नहीं लौटे, कहता है प्रभु।
- E2B: तो दो या तीन शहर पानी पीने के लिए एक शहर की ओर भटक गए; लेकिन वे संतुष्ट नहीं हुए: परन्तु तू मेरे पास वापस नहीं आया है, यहोवा कहता है।

**none_hi_ed39cc872669cf26_verse_12_en** — And the God of peace shall bruise Satan under your feet shortly. The grace of our Lord Jesus Christ be with you. Amen.

- E4B: और शांति का परमेश्वर शीघ्र ही तुम्हारे पैरों के नीचे शैतान को कुचलेगा। हमारे प्रभु यीशु मसीह की अनुग्रह तुम्हारे साथ रहे। आमीन।
- E2B: और शांति के परमेश्वर शीघ्र ही तुम्हारे पैरों के नीचे शैतान को कुचल देंगे। हमारे प्रभु यीशु मसीह की कृपा तुम पर बनी रहे। आमीन।

**none_hi_ed39cc872669cf26_verse_13_en** — Take my teaching, and not silver; get knowledge in place of the best gold.

- E4B: मेरा शिक्षण लो, और चाँदी नहीं; सर्वोत्तम सोने के बदले ज्ञान प्राप्त करो।
- E2B: मेरी शिक्षा लो, और चाँदी नहीं; सर्वोत्तम सोने के स्थान पर ज्ञान प्राप्त करो।

**none_hi_ed39cc872669cf26_verse_14_en** — O death, where is your power? O death, where are your pains?

- E4B: हे मृत्यु, तुम्हारी शक्ति कहाँ है? हे मृत्यु, तुम्हारा दर्द कहाँ है?
- E2B: हे मृत्यु, तुम्हारी शक्ति कहाँ है? हे मृत्यु, तुम्हारे दर्द कहाँ हैं?

**none_hi_ed39cc872669cf26_verse_15_en** — And by the desire of all the people, the feast went on for another seven days, and they kept the seven days with joy.

- E4B: और सभी लोगों की इच्छा से, दावत सात और दिनों तक चली, और उन्होंने सात दिनों को खुशी से मनाया।
- E2B: और सभी लोगों की इच्छा से, दावत सात और दिनों तक चली और उन्होंने उन सात दिनों को खुशी से मनाया।

**none_hi_ed39cc872669cf26_verse_16_en** — `Remember that which Amalek hath done to thee in the way, in your going out from Egypt,

- E4B: याद रखो कि अमालेक ने तुम्हारे साथ रास्ते में क्या किया था, जब तुम मिस्र से बाहर जा रहे थे,
- E2B: याद रखना कि अमलेक ने तुम्हें मिस्र से निकलते समय रास्ते में क्या किया था

**none_hi_ed39cc872669cf26_verse_17_en** — Building up Zion with blood, And Jerusalem with iniquity.

- E4B: रक्त से सियोन का निर्माण करना, और अधर्म से यरूशलेम का।
- E2B: ज़ियोन को खून से और यरूशलेम को पाप से बनाना।

**none_hi_ed39cc872669cf26_verse_18_en** — He took them and sent them over the stream with all he had.

- E4B: उसने उन्हें लिया और अपने पास जो कुछ भी था, सब कुछ के साथ उन्हें धारा के पार भेज दिया।
- E2B: उसने उन्हें लिया और अपने पास जो कुछ था, उससे उन्हें धारा में भेज दिया।

**none_hi_ed39cc872669cf26_verse_19_en** — And the Lord your God will give to you and to your seed a circumcision of the heart, so that, loving him with all your heart and all your soul, you may have life.

- E4B: और तुम्हारा परमेश्वर यहोवा तुम्हारे और तुम्हारे वंश को हृदय की परिचमदाह (या हृदय की कटाई) देगा, ताकि तुम अपने पूरे हृदय और अपनी पूरी आत्मा से उससे प्रेम करो और जीवन पाओ।
- E2B: और तुम्हारा परमेश्वर तुम्हारे और तुम्हारे वंश को हृदय का अनुपालन देगा, ताकि तुम उससे अपने पूरे हृदय और अपनी पूरी आत्मा से प्रेम करो, तो तुम्हें जीवन मिलेगा।

**none_hi_ed39cc872669cf26_verse_20_en** — And he arose, and departed to his house.

- E4B: और वह उठा और अपने घर चला गया।
- E2B: और वह उठा, और अपने घर चला गया।

**none_hi_ed39cc872669cf26_verse_21_en** — And he arose, and departed to his house.

- E4B: और वह उठा और अपने घर चला गया।
- E2B: और वह उठा, और अपने घर चला गया।

**none_hi_ed39cc872669cf26_verse_22_en** — And he stayeth yet other seven days, and addeth to send forth the dove from the ark;

- E4B: और वह सात और दिन ठहरता है, और कबूतर को नाव से भेजने के लिए भेजता है;
- E2B: और वह सात और दिन रहता है, और वह जहाज से कबूतर भेजने में जोड़ता है;

**none_hi_ed39cc872669cf26_verse_23_en** — And in the evening, at sundown, they took to him all who were diseased, and those who had evil spirits.

- E4B: और शाम को, सूर्यास्त पर, वे सभी बीमारों और जिन पर दुष्ट आत्माओं का साया था, उन्हें उसके पास ले आए।
- E2B: और शाम को, सूर्यास्त के समय, वे उन सभी को उसके पास ले गए जो बीमार थे, और जिनके साथ बुरी आत्माएं थीं।

**none_hi_ed39cc872669cf26_verse_24_en** — Thou hast turned my mourning to dancing for me, Thou hast loosed my sackcloth, And girdest me <FI>with<Fi> joy.

- E4B: तूने मेरे शोक को मेरे लिए नृत्य में बदल दिया है, तूने मेरे शोक वस्त्रों को हटा दिया है, और मुझे आनंद से लपेट दिया है।
- E2B: तूने मेरे शोक को मेरे लिए नृत्य में बदल दिया, तूने मेरे लंगोट को खोल दिया, और मुझे आनंद से परिधान पहना दिया।

## Pending gates

- ≥50 approved natural-speech reference utterances per language.
- Bilingual review of meaning errors and terminology preferences (blind_review.jsonl).
- Visible-browser render acknowledgments for the sub-second caption-delivery gate.
- Physical second-output / hotplug and real two-speaker validation.

Recorded failed runs: 0. Excluded incompatible/duplicate runs: 0. Details are retained in comparison.json.
