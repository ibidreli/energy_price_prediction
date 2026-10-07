# Bewertung der Prognose: Metrik und Unsicherheitsmass

> Entschieden am 2026-10-06 (Luca Manna), auf Basis der EDA ([`docs/eda.md`](eda.md)) und der unten genannten Quellen. Umgesetzt in `src/energy_price/metrics.py`, getestet in `tests/test_metrics.py`.

## Entscheid

| Was | Festlegung |
|---|---|
| **Ausgabe pro Viertelstunde** | drei Quantile des Preises: **P10, P50 und P90** in ct/kWh |
| **Punktprognose** | P50 (Median) |
| **Konfidenzmass** | das Intervall von P10 bis P90, also ein 80-%-Intervall |
| **Hauptmetrik** (entscheidet zwischen Modellen) | **mittlerer Pinball-Loss** über P10, P50 und P90, tiefer ist besser |
| **immer mit ausgewiesen** | **MAE des Medians**; **Abdeckung** des Intervalls P10 bis P90 (Ziel 80 %); **mittlere Intervallbreite** |

Im Modul zählt laut Flavio Müller vor allem, dass wir die Wahl von Modell und Metrik **erklären und begründen** können (Rückblick Kick-off, 18.09.2026). Darum steht unten die Begründung zu jedem Punkt.

## Warum Quantile und nicht Mittelwert ± Standardabweichung

Die EDA zeigt, dass der Preis nicht normalverteilt ist:

- **zwei getrennte Verteilungen:** bei short Median 18.9 ct/kWh und nie negativ, bei long Median 5.6 und bis −688 ct/kWh (`docs/data.md`, Abschnitt 3);
- **stark linksschief** mit langen Rändern (1 %-Quantil −38, Minimum −688 ct/kWh);
- **Streuung abhängig von Uhrzeit und Saison:** P10 bis P90 ist um 17:00 rund dreimal so breit wie um 03:15 (H7).

Eine Normalverteilung würde ein symmetrisches Band um einen Mittelwert legen, der oft zwischen den beiden Verteilungen liegt, wo kaum echte Preise vorkommen. Quantile brauchen keine Verteilungsannahme und dürfen pro Viertelstunde unterschiedlich weit und asymmetrisch sein.

**Warum genau P10 und P90:** Ein 80-%-Intervall lässt sich mit acht Monaten Daten noch verlässlich prüfen: Ein Monat hat rund 2'900 Viertelstunden, davon sollen etwa 580 ausserhalb liegen. Bei 95 % wären es nur etwa 150 pro Monat, und die äusseren Quantile hingen stark an wenigen Extremereignissen (H4: 153 Extrem-Viertelstunden in acht Monaten).

## Warum der Pinball-Loss als Hauptmetrik

Für das Quantil τ und eine Prognose q gilt pro Viertelstunde mit dem echten Preis y:

```
Pinball(y, q, τ) = τ · (y − q)        wenn y ≥ q   (Prognose zu tief)
                   (1 − τ) · (q − y)  wenn y < q   (Prognose zu hoch)
```

Der mittlere Pinball-Loss ist das Mittel über alle Viertelstunden und die drei Quantile.

- **Er ist eine „proper scoring rule“:** Das beste Ergebnis erreicht, wer die echten Quantile vorhersagt. Ein Modell kann sich nicht verbessern, indem es das Band künstlich schmal oder breit macht.
- **Er bewertet Treffsicherheit und Schärfe zugleich:** Ein zu breites Band kostet über die Abstände, ein zu schmales über die Treffer ausserhalb.
- **Er wächst linear mit dem Fehler, nicht quadratisch:** Ein einzelner Preis von −688 ct/kWh dominiert ihn nicht so wie einen quadratischen Fehler (RMSE).
- **Er ist ein übliches Mass** in der Literatur zur probabilistischen Strompreisprognose, zusammen mit dem CRPS (Übersicht: Nowotarski und Weron 2018). Das Mittel über viele Quantile nähert den CRPS an; mit drei Quantilen ist es eine grobe, aber gut erklärbare Version davon.
- **Für τ = 0.5 ist er die Hälfte des MAE.** Der Median-Teil ist also direkt mit der Punktgenauigkeit verbunden.

**Verworfen:** CRPS (braucht viele Quantile oder eine Verteilungsannahme), Winkler- bzw. Interval-Score (bewertet nur das Intervall, nicht den Median), MAE allein (bewertet die Unsicherheit gar nicht).

## Warum die Zusatzmetriken

| Metrik | Frage, die sie beantwortet | gut ist |
|---|---|---|
| **MAE des Medians** | Wie weit liegt die Punktprognose im Mittel daneben, in ct/kWh? | tief; vergleichbar mit naiven Baselines |
| **Abdeckung P10 bis P90** | Ist das Band ehrlich? Liegen wirklich etwa 80 % der Preise darin? | nahe 80 %. Deutlich darunter: Band zu schmal, das Modell ist zu sicher. Deutlich darüber: Band unnötig breit |
| **Mittlere Intervallbreite** | Wie nützlich ist das Band? | so schmal wie möglich bei Abdeckung nahe 80 % |

Abdeckung und Breite trennen, was der Pinball-Loss in einer Zahl zusammenfasst. Für die Präsentation sind sie anschaulicher.

## Wie gemessen wird

- **Rollierender Backtest:** pro Liefertag einmal um 11:00 an D−1 prognostizieren, nur mit dem damaligen Wissensstand (Meeting vom 01.10.2026, `known_at_issue()`), kein zufälliger Split.
- **Alle Metriken zusätzlich aufgeschlüsselt** nach Uhrzeit, Monat und Richtung (short/long), weil sich der Preis dort stark unterscheidet (H1, H7, H8 in `docs/eda.md`).
- **Vergleich mit einer naiven Baseline:** Quantile der Vergangenheit pro Viertelstunde des Tages. Ein Modell muss sie im mittleren Pinball-Loss schlagen.
- **Kreuzende Quantile** (P10 über P90) lehnt `metrics.py` ab. Das Modell muss sie vorher sortieren oder korrigieren.

```python
from energy_price.metrics import score

score(actual, p10=p10, p50=p50, p90=p90)
# {"mean_pinball": ..., "mae_p50": ..., "coverage_p10_p90": ..., "width_p10_p90": ...}
```

## Quellen

- J. Nowotarski, R. Weron (2018): *Recent advances in electricity price forecasting: A review of probabilistic forecasting.* Renewable and Sustainable Energy Reviews 81(1), S. 1548 bis 1568. [EconPapers](https://econpapers.repec.org/article/eeerensus/v_3a81_3ay_3a2018_3ai_3ap1_3ap_3a1548-1568.htm)
- Energy Data Hackdays 2025, Challenge „Predict Imbalance Price“ (HEIG-VD): verlangt Backtesting und eine klar begründete Metrik. [Archiv](https://www.energydatahackdays.ch/archiv/predict-imbalance-price)
- Modul CDS1, Rückblick Kick-off vom 18.09.2026 (FHNW Space, Abzug in `.idea/03_aufgaben.md`): Fokus der Bewertung auf der Begründung von Modell und Metrik.
