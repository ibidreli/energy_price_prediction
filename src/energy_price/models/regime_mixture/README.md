# Regime-Modell short/long (M4)

**Issue:** offen · **Status:** geplant · **Roadmap:** [M4](../../../../docs/models/roadmap.md#2-experimente)

## Hypothese

Der Preis ist eine Mischung aus zwei Verteilungen, short um 19 ct/kWh und long um 6 ct/kWh. Wer die Wahrscheinlichkeit für short schätzt und die Preise je Richtung getrennt modelliert, trifft die Nichtlinearität bei unterdeckter Regelzone besser.

## Modell und Features

Klassifikator für die Wahrscheinlichkeit short, dazu Quantile des Preises je Richtung; die Mischung ergibt P10, P50 und P90. Features wie M3, dazu Muster der Regelzonenbilanz bis D−2, sobald die Grenze D−2 bestätigt ist. Liefert auch die Richtung für „wer zahlt was“.

## Ergebnis (Test Juni bis August)

| Modell | MAE P50 | Pinball Loss | Coverage |
|---|---|---|---|
| Referenz | 9.92 | 3.20 | 75 % |
| Baseline B2 | 9.81 | 3.23 | 74 % |
| Regime-Modell | | | |

**Entscheid:** offen

## Starten

`model.py` und `__init__.py` aus `../_template/` kopieren, Anleitung in [docs/models/README.md](../../../../docs/models/README.md#3-modell-starten).
