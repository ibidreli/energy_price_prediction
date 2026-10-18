# Modelle: Aufbau und neues Modell anlegen

> Issues [#13](https://github.com/ibidreli/energy_price_prediction/issues/13) und [#14](https://github.com/ibidreli/energy_price_prediction/issues/14) · Stand 18.10.2026

Jedes Modell ist ein Ordner mit zwei Dateien: `model.py` (Modell und Einstellungen) und `README.md` (Hypothese und Ergebnis). Alles andere wird von allen Modellen gemeinsam genutzt und ist nur einmal geschrieben. So werden alle Modelle gleich trainiert, gleich bewertet und sind direkt vergleichbar.

## 1. Übersicht

| Modell | Ordner | Issue | Status | Pinball Loss Test | MAE P50 Test |
|---|---|---|---|---|---|
| Referenz (letzte 56 Tage) | `models/shared/reference.py` | – | Massstab | 3.20 | 9.92 |
| Baseline B2 | [`baseline_model/`](../../src/energy_price/models/baseline_model/) | [#6](https://github.com/ibidreli/energy_price_prediction/issues/6) | abgeschlossen | 3.23 | 9.81 |
| M2 kalibrierte Bänder | [`calibrated_bands/`](../../src/energy_price/models/calibrated_bands/) | offen | geplant | | |
| M3 Gradient Boosting | [`gbm_quantile/`](../../src/energy_price/models/gbm_quantile/) | offen | geplant | | |
| M4 Regime-Modell short/long | [`regime_mixture/`](../../src/energy_price/models/regime_mixture/) | offen | geplant | | |

Geplante Experimente, Reihenfolge und Zeitplan: [Modell-Roadmap](roadmap.md). Jedes umgesetzte Modell bekommt hier eine Zeile.

## 2. Gemeinsame Bausteine

Diese Teile wurden für die Baseline gebaut und stehen jedem Modell zur Verfügung:

| Baustein | Datei | Wofür |
|---|---|---|
| Feature-Sets `B1`, `B2` | `features/design.py` | `design_matrix(table, "B2")`; neue Features als Block dort ergänzen |
| `PerQuantileModel` | `models/shared/base.py` | Grundgerüst: ein Schätzer pro Quantil, Standardisierung nur auf Trainingsdaten, fehlende Features, sortierte Quantile |
| `LinearQuantileModel` | `models/shared/linear.py` | lineare Quantilregression der Baseline, mit Koeffizienten in ct/kWh |
| Referenz | `models/shared/reference.py` | Massstab für jedes Modell |
| Backtest | `evaluation/backtest.py` | Training bis D−2, wöchentlich neu, Holdout-Sperre |
| Kennzahlen | `evaluation/scores.py` | MAE, Pinball Loss, Coverage, Skill Score |
| Ablauf und Befehl | `evaluation/experiment.py` | `make backtest MODEL=<name> STEP=tune\|test` für jeden Modellordner |
| Grafiken | `evaluation/figures.py` | Fehler pro Stunde, Coverage pro Monat, Koeffizienten |
| W&B | `evaluation/tracking.py` | mit `TRACK=1`, gruppiert nach der Issue-Nummer im Branchnamen |

Ein neues Modell braucht deshalb oft nur wenige Zeilen. Ein Gradient-Boosting-Modell auf B2 sähe zum Beispiel so aus:

```python
class Model(PerQuantileModel):
    def make_estimator(self, tau):
        return HistGradientBoostingRegressor(loss="quantile", quantile=tau)
```

## 3. Modell starten

Geplante Modelle haben schon einen Ordner mit `README.md` (Hypothese aus der [Roadmap](roadmap.md)). Zum Start kommt `model.py` aus der Vorlage dazu, für eine neue Idee wird der ganze Ordner kopiert:

```bash
git checkout -b exp/15-gbm-quantile
cp src/energy_price/models/_template/__init__.py src/energy_price/models/gbm_quantile/   # geplantes Modell
cp src/energy_price/models/_template/model.py src/energy_price/models/gbm_quantile/
cp -r src/energy_price/models/_template src/energy_price/models/<neue_idee>                 # neue Idee
```

1. **`model.py`:** das Modell schreiben (Platzhalter ersetzen), `GRID` und `CHOSEN` anpassen.
2. **Abstimmen:** `make backtest MODEL=gbm_quantile STEP=tune`, dann die beste Zeile aus `reports/gbm_quantile/tune_scores.csv` als `CHOSEN` eintragen.
3. **Testen:** `make backtest MODEL=gbm_quantile STEP=test`. Verglichen wird automatisch mit Referenz und Baseline B2, Grafiken landen in `docs/figures/gbm_quantile/`.
4. **`README.md`:** Ergebnis und Entscheid eintragen, Zeile in der Übersicht oben nachführen.

Der Platzhalter im Template ist lauffähig. Eine frische Kopie lässt sich also sofort backtesten. Eigene Tests sind optional; `tests/test_model_template.py` zeigt, wie.

## 4. Regeln

- **Kein Data Leakage:** Der Backtest gibt `fit` nur Preise bis D−2. Neue Features gehören in `features/`, mit einem Test, dass sie um D−1 11:00 bekannt sind ([data.md](../data.md#1-die-regel-gegen-datenlecks)).
- **Einstellungen vor dem Test festlegen:** April und Mai zum Abstimmen, Juni bis August als Test. `CHOSEN` danach nicht mehr ändern. Der Holdout (September und Oktober) läuft erst ganz am Schluss.
- **Reproduzierbar:** Modelle mit Zufall setzen einen festen Seed (`random_state`).
