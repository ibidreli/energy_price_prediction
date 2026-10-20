# Kalibrierte Bänder

**Status:** geplant

## Hypothese

Die Bänder der Baseline sind zu schmal (Coverage 74 % statt 80 %). Eine Korrektur aus den Fehlern der letzten Wochen bis D−2 hebt die Coverage auf rund 80 %, ohne MAE und Pinball Loss deutlich zu verschlechtern.

## Modell und Features

Conformal Quantile Regression auf der Baseline B2 (`models/shared/linear.py`, Feature-Set B2): P10 und P90 werden um den beobachteten Fehler der vorangehenden Wochen verschoben, getrennt nach Tageszeit. Die Methode lässt sich danach auf jedes Modell anwenden.

## Ergebnis (Test Juni bis August)

| Modell | MAE P50 | Pinball Loss | Coverage |
|---|---|---|---|
| Referenz | 9.92 | 3.20 | 75 % |
| Baseline B2 | 9.81 | 3.23 | 74 % |
| Kalibrierte Bänder | | | |

**Entscheid:** offen

## Starten

`model.py` und `__init__.py` aus `../_template/` kopieren, Anleitung in [docs/models/README.md](../../../../docs/models/README.md).
