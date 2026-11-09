# Verfahren für Issue #25

Literatur gelesen am 07.11.2026. Präzisierung am 09.11.2026: Die Hauptvariante
trifft 24 getrennte Entscheidungen je lokaler Uhrzeit. Der wöchentliche Takt
bestimmt die Aktualisierung dieser Tabelle, nicht eine einzige Wahl für die
ganze Woche. Die bereits ausgewerteten Juni–August-Daten bleiben Entwicklungsdaten.

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

**Power-Diagnostik:** 24 einzelne Stunden als Hauptvariante; globale, fachliche
Blöcke 00–08, 08–16 und 16–24 Uhr sowie 4×6 Stunden zur Einordnung.
Keine Wochentagstrennung.
Planungseffekt: 1 % des Standardmodell-Loss im jeweiligen Block, Power 80 %.
Die langfristige Varianz wird aus einem 7-Kalendertage-Bootstrap mit 5000
Wiederholungen und Seed 0 geschätzt. Die Normalapproximation für die benötigten
Tage berücksichtigt `(Poolgrösse−1) × Blockzahl` Vergleiche. Das Ergebnis wird
als Einschränkung ausgewiesen und darf die Stundenauflösung nicht automatisch
auf einen ganzen Tag reduzieren. Die frühere Reduktion auf einen Tagesblock
bildete das präzisierte Ziel des Nutzers nicht ab. Der historische Stand ist
unter `docs/exploration/time_router/history/global_v1_2026-11-07/` erhalten.
Vier 6-Stunden-Blöcke bleiben diagnostisch und ersetzen nicht nachträglich die
Stundenvariante.

Das ist eine Planungsabschätzung, keine garantierte Power bei Extrempreisen oder
Verteilungsänderung. Die bisherige Wechselregel `significant` verlangt ein
vollständig negatives 95-%-Intervall, zusätzlich über die gleichzeitigen
Stundenvergleiche bereinigt. Alternativ implementiert `empirical` den direkten
historischen Gewinner je Stunde; das ist keine statistische Überlegenheitsbehauptung.
Die wöchentliche Wiederholung ist keine zeituniforme Signifikanzgarantie.
Für den Architekturentscheid zählt allein der spätere einzelne Holdout-Vergleich.
