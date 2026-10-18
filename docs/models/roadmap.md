# Modell-Roadmap nach der Baseline

> Issue [#14](https://github.com/ibidreli/energy_price_prediction/issues/14) · Stand 18.10.2026 · Grundlage: [Baseline](baseline_model.md), [EDA, PR #5](https://github.com/ibidreli/energy_price_prediction/pull/5), [Datengrundlage](../data.md), Review mit dem Coach vom 15.10.2026

## Kurzfassung

- **Ausgangslage:** Die Baseline ist gleich gut wie die Referenz, aber nicht besser. Die Sonne hilft mittags, abends fehlt Information, die Bänder sind zu schmal (Coverage 74 % statt 80 %).
- **Review vom 15.10.:** Quantilregression und MAE passen. Den Nutzen der Datenquellen müssen wir selbst über Feature Importance zeigen. Prüfen, wo das Modell gut ist und wo Daten fehlen, und die Nichtlinearität bei unterdeckter Regelzone (short) anschauen. Am Ende ableiten, wer was zahlt.
- **Plan:** vier Experimente, dazu zwei optionale mit neuen Daten. Feiertage und Extremwerte werden in der Fehleranalyse und in den Modellen ausgewertet, nicht als eigene Modelle, weil es dafür zu wenige Tage gibt.
- **Massstab:** besser als die Referenz im Test Juni bis August, also MAE von P50 unter 9.92 ct/kWh, bei Coverage möglichst nahe an 80 %.

## 1. Voraussetzungen

| Punkt | Stand | Wofür |
|---|---|---|
| Grenze D−2 bei Swissgrid bestätigen | täglicher Abruf bis ca. 18.10. ([data.md](../data.md#prüfung-des-aktualisierungsrhythmus)) | Features aus der Regelzonenbilanz (M3, M4) |
| Hauptkennzahl festlegen | Coach: MAE passt | Vorschlag: MAE von P50 als Hauptkennzahl, Pinball Loss und Coverage für das Band |
| Provisorische und finale Preise vergleichen | ab ca. 21.10. | Grösse des Restrisikos in allen Backtests |

## 2. Experimente

Jedes Experiment bekommt ein EXP-Issue und einen Branch `exp/<issue>-<name>`. Die Modelle M2 bis M4 haben schon einen Ordner mit Steckbrief: `calibrated_bands/`, `gbm_quantile/` und `regime_mixture/` ([Anleitung](README.md#3-modell-starten)).

| ID | Experiment | Hypothese | Vorgehen | Aufwand |
|---|---|---|---|---|
| M1 | **Fehleranalyse** | Die Fehler häufen sich abends, bei short, an Feiertagen und bei Extremwerten | Prognosen der Baseline nach Stunde, Monat, Tagtyp, Richtung und Extremwerten aufteilen. Zeigt, wo das Modell gut ist und wo Daten fehlen | klein |
| M2 | **Kalibrierte Bänder** | Eine Korrektur aus den Fehlern der letzten Wochen hebt die Coverage auf rund 80 % | Conformal Quantile Regression auf B2, getrennt nach Tageszeit. Lässt sich danach auf jedes Modell anwenden | klein |
| M3 | **Gradient Boosting** | Ein Baummodell erkennt Interaktionen und den Sprung zwischen short und long besser als ein lineares Modell | `HistGradientBoostingRegressor` mit Quantil-Loss auf `PerQuantileModel`. Darauf **Feature Importance pro Datenquelle** (Kalender, Feiertage, Strahlung, Bewölkung, Regelzonenbilanz bis D−2) und Auswertung an Feiertagen | mittel |
| M4 | **Regime-Modell short/long** | Der Preis ist eine Mischung aus zwei Verteilungen; getrennt modelliert wird die Nichtlinearität bei short besser getroffen | Wahrscheinlichkeit für short schätzen, Preise je Richtung modellieren, zu P10, P50 und P90 zusammensetzen. Liefert auch die Richtung für „wer zahlt was“ | gross |
| M5 | Lastprognose ENTSO-E, optional | Die Lastprognose erklärt die Abendstunden | nur wenn Token und Verfügbarkeit um 11:00 belegt sind ([data.md](../data.md#7-nicht-erhoben-entso-e-lastprognose)) | mittel |
| M6 | Stauseen, optional | Der Füllungsgrad bildet die Saison ab | nur wenn die historische Verfügbarkeit belegt ist ([data.md](../data.md#6-stauseen-bfe)) | klein |

## 3. Abschluss

- **Modellwahl:** bestes Modell im Test nach der Hauptkennzahl, unter Berücksichtigung von Coverage und Erklärbarkeit. Danach läuft der Holdout (September und Oktober) genau einmal für Referenz, Baseline und das gewählte Modell.
- **Wer zahlt was:** Aus Preisprognose und Richtung (M4) ableiten, ob eine Bilanzgruppe mit einer bestimmten Abweichung zahlt oder vergütet wird und in welcher Grössenordnung (Preis × Abweichungsmenge), gestützt auf das Anreizschema in [overview.md](../overview.md#5-wer-wem-bezahlt).

## 4. Zeitplan

| Sprint bis | Inhalt |
|---|---|
| 30.10.2026 | Voraussetzungen, M1, M2 |
| 13.11.2026 | M3 mit Feature Importance |
| 27.11.2026 | M4, optional M5 und M6 |
| 11.12.2026 | Modellwahl, Holdout, wer zahlt was |
| 20.12.2026 | Abgabe Repository |

Die Verantwortlichen legt das Team in der Sprintplanung fest und trägt sie im jeweiligen Issue ein. Der Status jedes Modells steht in der [Übersicht](README.md#1-übersicht).
