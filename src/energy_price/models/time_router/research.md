# Verfahren für Issue #25

Gelesen am 07.11.2026; die Entscheidung unten verwendet keine neuen Testresultate.

| Quelle | Aussage und Verwendung |
|---|---|
| [Nowotarski und Weron, online 2014 / 2015, Abschnitte 3.2–3.3](https://link.springer.com/article/10.1007/s00180-014-0523-0) | Gleichgewichte sind ein einfacher Vergleich. Original-QRA regressiert die Beobachtung auf **Punktprognosen** und schätzt quantilspezifische Koeffizienten mit Pinball-Loss. Unsere bereits vorhandenen Quantile zu kombinieren ist eine andere Variante; sie wird deshalb nicht als unverändertes QRA bezeichnet. |
| [Hansen, Lunde und Nason, 2011; Working Paper, Abschnitte 1–2](https://pure.au.dk/ws/files/34269366/rp10_76.pdf) | Ein Model Confidence Set lässt mehrere statistisch ununterscheidbare Modelle zu. Eine Reihe unbereinigter Paarvergleiche ist kein MCS. Hier wird kein MCS implementiert und der kleine Pool nicht nachträglich ausgedünnt. Die Auswahl verwendet konservativ Bonferroni-Intervalle. |
| [Samuels und Sekkel, 2013, Einleitung und Schluss](https://www.oar-rao.bank-banque-canada.ca/record/1636/files/wp2013-11.pdf) | Die Arbeit untersucht MCS-basiertes Ausdünnen vor der Kombination in makroökonomischen Prognosen. Das ist kein Beleg für Gewinne in unseren Viertelstundenpreisen. Bei vier Experten und kurzer Historie behalten wir alle zugelassenen Experten im Pool. Der Original-PDF wurde direkt geladen, da der Browserzugriff fehlschlug. |
| [Berrisch und Ziel, Version 2, 2023; Journalfassung 2024, Abschnitt 2](https://arxiv.org/pdf/2303.10019v2) | Horizontale Kombination kann Gewichte nach Quantil und Prognosehorizont schätzen; viele freie Gewichte verlangen Stabilisierung. Wir übernehmen die Batch-Idee auf groben Blöcken. Der BOA-Algorithmus und seine asymptotischen Aussagen werden nicht übernommen. Drei Quantile sind auch keine vollständige CRPS-Auswertung. |

**Festgelegtes Gewichtsverfahren:** je Block und Quantil empirischen Pinball-Loss
auf den letzten 56 Kalendertagen verfügbarer OOS-Prognosen minimieren. Gewichte
sind nichtnegativ und summieren sich zu eins; kein Intercept. Ein dünn besetztes
lineares Programm löst das Problem exakt. Die Lösung wird zu 50 % zum
Standardmodell geschrumpft. 56 Tage und 50 % sind vorab festgelegte
Stabilisierungen, keine auf Juni–August optimierten Parameter. Gewichte werden
wöchentlich eingefroren. Bei weniger als 28 beobachteten Tagen gilt das
Standardmodell. Mögliche Quantilkreuzungen werden sortiert; Gewichte und Werte
vor der Sortierung werden mitgespeichert. Anschliessend kommt einmal die
asymmetrische Kalibrierung aus #19 (56 Tage, 6-Stunden-Gruppen).

**Vorab festgelegte Power-Planung:** fachliche Blöcke 00–08, 08–16 und 16–24 Uhr;
zusätzlich 4×6 Stunden als reine Robustheitsdiagnostik. Keine Wochentagstrennung.
Planungseffekt: 1 % des Standardmodell-Loss im jeweiligen Block, Power 80 %.
Die langfristige Varianz wird aus einem 7-Kalendertage-Bootstrap mit 5000
Wiederholungen und Seed 0 geschätzt. Die Normalapproximation für die benötigten
Tage berücksichtigt `(Poolgrösse−1) × Blockzahl` Vergleiche. Alle Vergleiche
müssen den Effekt mit der verfügbaren Zahl April/Mai-Tage unterstützen. Sonst
wird die Blockzahl reduziert; wenn auch ein globaler Block nicht ausreichend
Power hat, bleibt dieser als ausdrücklich unzureichend abgesicherter Fallback.
Vier 6-Stunden-Blöcke dürfen im Test keinen vierten Kandidaten erzeugen.

Das ist eine Planungsabschätzung, keine garantierte Power bei Extrempreisen oder
Verteilungsänderung. Die Auswahl verlangt unabhängig davon ein vollständig
negatives 95-%-Intervall, zusätzlich über die gleichzeitigen Blockvergleiche
bereinigt. Die wöchentliche Wiederholung ist keine zeituniforme Signifikanzgarantie.
Für den Architekturentscheid zählt allein der spätere einzelne Holdout-Vergleich.
