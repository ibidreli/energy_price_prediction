# Modelle: neues Modell erstellen

Jedes Modell ist ein eigener Ordner in `src/energy_price/models/`. Daten, Backtest, Kennzahlen und Grafiken sind für alle Modelle gleich und schon fertig. Für ein neues Modell schreibt man deshalb nur das Modell selbst und seine Einstellungen. Wie ausgewählt, verglichen und getestet wird, steht in der [Trainingspipeline](../training_pipeline.md).

## 1. Übersicht

Jeder Steckbrief folgt derselben Vorlage ([`_template/README.md`](../../src/energy_price/models/_template/README.md)). Kandidaten = Settings in `GRID` × Fenster + je Setting ein `window_mean`, jeder auf 153 Validierungstagen. Pinball (Mittel der täglichen Pinball-Loss) und MAE von P50 in ct/kWh aus `make compare` (#51): Validierung, alle 152 Tage (der 24.06. fehlt), dieselben Viertelstunden für alle. Details und Intervalle in der [Trainingspipeline](../training_pipeline.md#7-ergebnisse-auf-den-validierungsblöcken), die alten Zahlen in der [Historie](#historie-alte-pipeline).

| Modell | Verantwortlich | Warum dieses Modell | Fenster | Kandidaten | Pinball | MAE P50 |
|---|---|---|---|---|---|---|
| Referenz (letzte 56 Tage, [`reference.py`](../../src/energy_price/models/shared/reference.py)) | Team | einfachste Prognose ohne Modell, Fenster aus EDA H11 | 56 Tage, fest | Massstab | 3.301 | 9.943 |
| [Baseline B1, B2](../../src/energy_price/models/baseline_model/README.md) | Rodrigo Urech | Kalender und Sonne (H1, H3, H8), lineare Quantilregression ist erklärbar | `expanding`, 28, 56, 90 | 6 × 4 + 6 = 30 | B1 3.309, B2 3.248 | B1 9.973, B2 9.793 |
| [Gradient Boosting](../../src/energy_price/models/gbm_quantile/README.md) | Elias Pulver | findet Wechselwirkungen von Sonne, Uhrzeit und Saison selbst (H3, H7, H8) | `expanding`, 56, 90 | 4 × 3 + 4 = 16 | 3.202 | **9.773** |
| [Regime-Modell short/long](../../src/energy_price/models/regime_mixture/README.md) | Rodrigo Urech | zwei Preiswolken je Richtung, Richtung zuerst schätzen | `expanding`, 56, 90 | 2 × 3 + 2 = 8 | 3.252 | 9.817 |
| [Kalibrierte Bänder](../../src/energy_price/models/calibrated_bands/README.md) | Luca Manna | Bänder zu schmal, Korrektur von P10 und P90 aus vergangenen Fehlern bis D−2, auf B2 und Boosting | nur `expanding`, Basis mit eigenem Fenster | 14 | auf B2 3.238, auf Boosting **3.190** | auf B2 9.793, auf Boosting **9.773** |
| [Quantil-LASSO](../../src/energy_price/models/quantile_lasso/README.md) | Luca Manna | L1-Strafe wählt die Merkmale selbst, prüft ob Daten bis D−2 helfen (`B3`, `B4`) | `expanding`, 56, 90 | 12 × 3 + 12 = 48 | `B2` 3.256, `B3` 3.264 | `B2` 9.779, `B3` 9.879 |

Ein neues Modell muss die Referenz schlagen.

**Getestet und verworfen, kein Modellkandidat:** Ein [Bandit über die Modelle](../exploration/bandit/README.md) (Luca Manna, #21) wählt pro Viertelstunde laufend ein Modell. Er lernt praktisch nichts und wurde nicht weiterverfolgt.

## 2. Aufbau eines Modellordners

```text
src/energy_price/models/<name>/
├── __init__.py   leer, damit Python den Ordner findet
├── README.md     Steckbrief nach der Vorlage
└── model.py      das Modell und seine Einstellungen
```

| Variable in `model.py` | Inhalt | Pflicht |
|---|---|---|
| `MODEL` | Klasse (oder Funktion), die aus den Settings ein neues Modell baut | ja |
| `GRID` | Liste der Settings für `select`, **ohne** `window` | ja |
| `CHOSEN` | `{Variante: Settings inklusive window}`, erst **nach** `select` von Hand eintragen; `window` ist ein Fenster oder `window_mean` | ja |
| `WINDOWS` | Fenster, Standard `expanding`, `rolling_56d`, `rolling_90d` | nein |
| `REFIT_EVERY` | Neutraining alle k Tage, Standard 1; nur wenn `select` länger als 8 h dauert, Entscheid Luca | nein |
| `ISSUE`, `RUN_INFO` | W&B-Gruppe `issue-<n>`, Notizen in `run_info.json` | nein |
| `select_candidates(scores, settings)` | eigene Auswahlregel, landet unter `chosen` in `select_selection.json` (Beispiel `calibrated_bands`) | nein |
| `COEFFICIENTS_SHOWN`, `IMPORTANCE_GROUPS` | Koeffizienten-Grafik, gruppierte Permutation Importance | nein |

`model.py` wird nie direkt ausgeführt. Das übernimmt `make select`: Der Befehl liest die Datei, trainiert jeden Kandidaten an jedem Validierungstag neu und bewertet ihn.

## 3. Schritt für Schritt

Beispiel: ein neues Boosting-Modell `my_gbm`.

**1. Vorlage kopieren:** `cp -r src/energy_price/models/_template src/energy_price/models/my_gbm`. Die Vorlage ist sofort lauffähig.

**2. `model.py` schreiben**

```python
from energy_price.models.shared.boosting import GradientBoostingQuantileModel

MODEL = GradientBoostingQuantileModel

GRID = [{"feature_set": fs, "learning_rate": lr} for fs in ("B1", "B2") for lr in (0.05, 0.1)]   # ohne window

CHOSEN = {"my_gbm": {"feature_set": "B2", "learning_rate": 0.05, "window": "expanding"}}   # Platzhalter bis nach select
```

Das ergibt 4 Settings × 3 Fenster + 4 `window_mean` = 16 Kandidaten. Eigene Modelle erben von `PerQuantileModel` (`models/shared/base.py`): ein Schätzer pro Quantil, sortierte Quantile.

**3. Probelauf, dann Auswahl**

```bash
python -m energy_price.evaluation.experiment my_gbm select --days 2026-04-01 2026-04-03   # wenige Tage, ohne Grafiken
make select MODEL=my_gbm TRACK=1 JOBS=4
```

`select` schreibt nach `reports/my_gbm/` (nicht versioniert):

| Datei | Inhalt |
|---|---|
| `select_scores.csv` | jeder Kandidat je Monatsblock 2026-04 bis 2026-08 und `all`, mit Rang |
| `select_selection.json` | gewählter Kandidat (kleinster mittlerer täglicher Pinball-Loss über alle Tage), Fenster, `refit_every`, `dropped_days` |
| `select_runtime.csv` | Sekunden je Fit, Grundlage für die 8-h-Grenze |
| `select_forecasts.parquet`, `select_breakdowns.csv` | Prognosen aller Kandidaten, Referenz und B2; Auswertung nach Stunde, Monat, Tagtyp und Richtung |

Grafiken (Pinball-Loss je Stunde, Coverage je Monat) landen in `docs/figures/my_gbm/select_*.png`.

**4. `CHOSEN` festhalten:** den gewählten Kandidaten aus `select_selection.json` inklusive `window` in `model.py` eintragen und committen. Danach wird `CHOSEN` nicht mehr geändert.

**5. Vergleich:** `make compare` rechnet `CHOSEN` aller Modelle, Referenz und B2 auf denselben Viertelstunden der Validierung und schreibt `reports/compare/validation_*`.

**6. Test:** nur einmal nach dem Freeze aller Modelle mit `make final-test UNLOCK_TEST=1` (#53). Ohne Freigabe wirft `test_days()` `TestLockedError`.

**7. Ergebnis dokumentieren:** im `README.md` des Modellordners und in der Übersicht oben.

## 4. Was alle Modelle gemeinsam nutzen

| Baustein | Datei | Wofür |
|---|---|---|
| Feature-Sets `B1` bis `B4` | `features/design.py` | B1 Kalender, B2 plus Sonnenprognose, B3 plus Preisniveau bis D−2, B4 plus Systembilanz bis D−2 und Bewölkung ([Mindesthistorie](../data.md#mindesthistorie-der-preis--und-bilanzmerkmale)); neue Features als Block dort ergänzen |
| `PerQuantileModel`, `LinearQuantileModel`, `GradientBoostingQuantileModel` | `models/shared/base.py`, `linear.py`, `boosting.py` | Grundgerüst, lineare Quantilregression, Quantile Boosting mit festem Seed |
| Kalibrierung, Importance, Referenz | `models/shared/calibration.py`, `importance.py`, `reference.py` | Hülle um eine Basis, Bedeutung einer Datenquelle, Massstab |
| Design | `evaluation/design.py` | Perioden, Fenster, `refit_every`, Testsperre |
| Backtest und Ablauf | `evaluation/backtest.py`, `evaluation/experiment.py` | tägliches Walk-forward bis D−2, `select`, `compare`, `final-test` |
| Kennzahlen, Diagnostik, Grafiken, W&B | `evaluation/scores.py`, `diagnostics.py`, `figures.py`, `tracking.py` | gleich für jedes Modell ([Bewertung](../evaluation.md), [Tracking](../experiment_tracking.md)) |

## 5. Regeln

- **Kein Data Leakage:** Der Backtest gibt dem Modell nur Zeilen bis D−2. Neue Features gehören in `features/`, mit einem Test, dass sie um D−1 11:00 bekannt sind ([data.md](../data.md#1-die-regel-gegen-datenlecks)).
- **Auswahl nur auf der Validierung:** `select` und `compare` sehen nur 01.04. bis 31.08.; die Testperiode ist gesperrt und läuft einmal ganz am Schluss ([Trainingspipeline](../training_pipeline.md#5-regeln-gegen-leakage)).
- **Reproduzierbar:** Modelle mit Zufall setzen einen festen Seed (`random_state=0`).
- **Tests sind optional:** `tests/test_model_template.py` zeigt, wie ein Modell getestet wird.


## 6. In Arbeit

Diese Modelle sind noch nicht in `main`. Ihre Steckbriefe übernehmen die Vorlage im jeweiligen Pull Request.

| Modell | Verantwortlich | Issue und PR | Idee |
|---|---|---|---|
| Zeitabhängige Modellwahl (`time_router`) | Elias Pulver | #25 | pro Zeitfenster des Liefertags das Modell nutzen, das dort verlässlich am besten ist. Boosting ist vormittags bis mittags besser, abends oft die Referenz |

## Historie (alte Pipeline)

> Gewählt auf April und Mai, verglichen auf Juni bis August, wöchentlich neu trainiert, Fenster `expanding` oder 56 Tage. Juni bis August waren nicht mehr blind (#21). Ersetzt durch #36.

Stand der Zahlen: Test Juni bis August, 91 Tage. Preise und Fehler in ct/kWh.

| Modell | Verantwortlich | Warum dieses Modell | Pinball-Loss | MAE P50 | Coverage | Entscheid |
|---|---|---|---|---|---|---|
| Referenz (letzte 56 Tage, [`reference.py`](../../src/energy_price/models/shared/reference.py)) | Team | einfachste Prognose ohne Modell, Fenster aus EDA H11 | 3.20 | 9.92 | 75.2 % | Massstab |
| [Baseline B2](../../src/energy_price/models/baseline_model/README.md) | Rodrigo Urech | Kalender und Sonne (H1, H3, H8), lineare Quantilregression ist erklärbar | 3.23 | 9.81 | 73.6 % | bleibt als erklärbarer Massstab |
| [Gradient Boosting](../../src/energy_price/models/gbm_quantile/README.md) | Elias Pulver | findet Wechselwirkungen von Sonne, Uhrzeit und Saison selbst (H3, H7, H8) | 3.16 | **9.79** | 74.1 % | Kandidat für den Holdout |
| [Regime-Modell short/long](../../src/energy_price/models/regime_mixture/README.md) | Rodrigo Urech | zwei Preiswolken je Richtung, Richtung zuerst schätzen | 3.23 | 9.80 | 74.9 % | nicht übernommen, Richtung am Vortag zu schwach vorhersagbar |
| [Kalibrierte Bänder auf B2](../../src/energy_price/models/calibrated_bands/README.md) | Luca Manna | Bänder aller Modelle zu schmal (Coverage rund 74 % statt 80 %), Korrektur aus vergangenen Fehlern bis D−2 | 3.17 | 9.81 | 77.7 % | besser als B2, Coverage-Ziel 78 % knapp verfehlt |
| [Kalibrierte Bänder auf Boosting](../../src/energy_price/models/calibrated_bands/README.md) | Luca Manna | gleiche Korrektur auf Gradient Boosting | **3.13** | **9.79** | 77.4 % | bester Pinball-Loss, Coverage-Ziel 78 % knapp verfehlt |
| [Quantil-LASSO `B2`](../../src/energy_price/models/quantile_lasso/README.md) | Luca Manna | L1-Strafe wählt die Merkmale selbst, prüft ob Daten bis D−2 helfen | 3.22 | 9.80 | 74.5 % | nicht übernommen, schlägt die Referenz nicht |
| [Quantil-LASSO mit Daten bis D−2 (`B4`)](../../src/energy_price/models/quantile_lasso/README.md) | Luca Manna | zusätzlich Preisniveau, Systembilanz und Bewölkung bis D−2 | 3.27 | 9.95 | 75.2 % | nicht übernommen, Daten bis D−2 helfen nicht |
