# Gradient Boosting mit Quantil-Loss (M3)

**Issue:** offen · **Status:** geplant · **Roadmap:** [M3](../../../../docs/models/roadmap.md#2-experimente)

## Hypothese

Ein Baummodell erkennt Interaktionen wie Uhrzeit × Position in der Stunde × Steigung der Strahlung und den Sprung zwischen short und long besser als ein lineares Modell.

## Modell und Features

`HistGradientBoostingRegressor(loss="quantile")` aus scikit-learn als `make_estimator` von `PerQuantileModel` (`models/shared/base.py`), Feature-Set B2, ergänzt um Bewölkung und Tagtyp. Darauf Feature Importance pro Datenquelle (Kalender, Feiertage, Strahlung, Bewölkung, Regelzonenbilanz bis D−2) und Auswertung an Feiertagen.

## Ergebnis (Test Juni bis August)

| Modell | MAE P50 | Pinball Loss | Coverage |
|---|---|---|---|
| Referenz | 9.92 | 3.20 | 75 % |
| Baseline B2 | 9.81 | 3.23 | 74 % |
| Gradient Boosting | | | |

**Entscheid:** offen

## Starten

`model.py` und `__init__.py` aus `../_template/` kopieren, Anleitung in [docs/models/README.md](../../../../docs/models/README.md#3-modell-starten).
