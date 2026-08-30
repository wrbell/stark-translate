# License report — hymn corpus

We train on public-domain hymn *texts*, not on the New Believers Hymn Book (2019) / John Ritchie compilation.

## Excluded and why

- **O soul, are you weary and troubled?** (`pd-turn-your-eyes-upon-jesus`): Helen Howarth Lemmel d.1961 — not public domain. Zero lyrics in stanza/pair files.
- **Great is Thy faithfulness, O God my Father** (`pd-great-is-thy-faithfulness`): Thomas O. Chisholm d.1960; 1923 publication still under copyright in many jurisdictions. Index only.

## Refused sources (see sources.json)

- New BHB 2019 full text / numbering / 2019-only hymns
- BHB+ app database / Gospel Folio music edition
- Wholesale gospelriver.com scrape as 'the book'
- Himnos y Cánticos del Evangelio full dump without LEC permission
- JW.org, CCLI-only modern worship

No hymn-singing audio is included for Whisper training.
