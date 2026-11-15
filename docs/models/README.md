# Modelle: neues Modell erstellen

Jedes Modell ist ein eigener Ordner in `src/energy_price/models/`. Daten, Backtest, Kennzahlen und Grafiken sind für alle Modelle gleich und schon fertig. Für ein neues Modell schreibt man deshalb nur das Modell selbst und seine Einstellungen.

## 1. Übersicht

Stand der Zahlen: Test Juni bis August, 91 Tage. Preise und Fehler in ct/kWh. Jeder Steckbrief folgt derselben Vorlage mit Hypothese, Herleitung, Vorgehen, Erkenntnissen, Outcome, Entscheid und Belegen ([`_template/README.md`](../../src/energy_price/models/_template/README.md)).

| Modell | Verantwortlich | Warum dieses Modell | Pinball-Loss | MAE P50 | Coverage | Entscheid |
|---|---|---|---|---|---|---|
| Referenz (letzte 56 Tage, [`reference.py`](../../src/energy_price/models/shared/reference.py)) | Team | einfachste Prognose ohne Modell, Fenster aus EDA H11 | 3.20 | 9.92 | 75.2 % | Massstab |
| [Baseline B2](../../src/energy_price/models/baseline_model/README.md) | Rodrigo Urech | Kalender und Sonne (H1, H3, H8), lineare Quantilregression ist erklärbar | 3.23 | 9.81 | 73.6 % | bleibt als erklärbarer Massstab |
| [Gradient Boosting](../../src/energy_price/models/gbm_quantile/README.md) | Elias Pulver | findet Wechselwirkungen von Sonne, Uhrzeit und Saison selbst (H3, H7, H8) | **3.16** | **9.79** | 74.1 % | Kandidat für den Holdout |
| [Regime-Modell short/long](../../src/energy_price/models/regime_mixture/README.md) | Rodrigo Urech | zwei Preiswolken je Richtung, Richtung zuerst schätzen | 3.23 | 9.80 | 74.9 % | nicht übernommen, Richtung am Vortag zu schwach vorhersagbar |

Ein neues Modell muss die Referenz schlagen. Jedes Modell wird hier nachgeführt, sobald es Ergebnisse hat. Laufende Modelle stehen am Schluss unter `6. In Arbeit`.

**Explorativ geprüft, kein Modellkandidat:** Ein [Bandit über die Modelle](../exploration/bandit/README.md) (Luca Manna, #21) wählt pro Viertelstunde laufend ein Modell. Er lernt praktisch nichts und wurde nicht weiterverfolgt.

## 2. Aufbau eines Modellordners

```text
src/energy_price/models/<name>/
├── __init__.py   leer, damit Python den Ordner findet
├── README.md     Steckbrief nach der Vorlage, sieben Abschnitte von Hypothese bis Belege
└── model.py      das Modell und seine Einstellungen
```

In `model.py` stehen drei Variablen:

| Variable | Inhalt | Wann |
|---|---|---|
| `MODEL` | das Modell, das trainiert wird | beim Schreiben festlegen |
| `GRID` | alle Settings, die beim Tuning ausprobiert werden | beim Schreiben festlegen |
| `CHOSEN` | das beste Setting aus dem Tuning, das im Test verwendet wird | **erst nach dem Tuning** eintragen |

`model.py` wird nie direkt ausgeführt. Das übernimmt `make backtest`: Der Befehl liest die Datei, trainiert das Modell und bewertet es.

## 3. Schritt für Schritt

Beispiel: `gbm_quantile`. Dieses Modell ist inzwischen implementiert; die Kopierschritte
gelten für einen neuen Modellordner, nicht zum Überschreiben der bestehenden Implementierung.

**1. Vorlage kopieren**

Geplante Modelle haben schon einen Ordner mit `README.md`. Dazu kommen `__init__.py` und `model.py` aus der Vorlage:

```bash
cp src/energy_price/models/_template/__init__.py src/energy_price/models/gbm_quantile/
cp src/energy_price/models/_template/model.py src/energy_price/models/gbm_quantile/
```

Für eine ganz neue Idee wird der ganze Ordner kopiert: `cp -r src/energy_price/models/_template src/energy_price/models/<name>`. Die Vorlage ist sofort lauffähig.

**2. `model.py` schreiben (`MODEL` und `GRID`)**

```python
from sklearn.ensemble import HistGradientBoostingRegressor

from energy_price.models.shared.base import PerQuantileModel


class Model(PerQuantileModel):
    def __init__(self, feature_set="B2", learning_rate=0.1):
        super().__init__(feature_set)
        self.learning_rate = learning_rate

    def make_estimator(self, tau):
        return HistGradientBoostingRegressor(loss="quantile", quantile=tau, learning_rate=self.learning_rate, random_state=0)


MODEL = Model

GRID = [
    {"feature_set": fs, "window": "expanding", "learning_rate": lr}
    for fs in ("B1", "B2")
    for lr in (0.05, 0.1)
]

CHOSEN = {"gbm": {"feature_set": "B2", "window": "expanding", "learning_rate": 0.1}}   # Platzhalter bis nach dem Tuning
```

- Ein Setting enthält die Parameter von `Model` und dazu `window`: `"expanding"` trainiert mit allen Daten seit 01.01., `"rolling_56d"` nur mit den letzten 56 Tagen.
- `PerQuantileModel` erledigt den Rest: ein Modell pro Quantil P10, P50, P90, Features aus dem Feature-Set, keine Prognose bei fehlenden Werten und sortierte Quantile.

**3. Tuning**

```bash
make backtest MODEL=gbm_quantile STEP=tune
```

Probiert alle Settings aus `GRID` auf April und Mai aus. Das Ergebnis steht in `reports/gbm_quantile/tune_scores.csv`, eine Zeile pro Setting. `tune_selection.json` enthält das Setting mit dem tiefsten Pinball-Loss. Dazu werden Prognosen, tägliche Scores, Auswertungen nach Uhrzeit, Monat, Tagtyp und tatsächlicher Richtung sowie die Code-/Datenversion gespeichert. Andere Modelle werden im Tuning zusätzlich mit der frisch berechneten Baseline B2 verglichen.

Mit `TRACK=1` werden auch die Tuning-Varianten, die Referenz und die Baseline B2 in W&B geloggt.
`JOBS=2` begrenzt die parallelen Wochen-Fits, zum Beispiel:

```bash
make backtest MODEL=gbm_quantile STEP=tune TRACK=1 JOBS=2
```

Wenn ein Modell `IMPORTANCE_GROUPS` definiert, wird die beste Tuning-Variante zusätzlich mit gruppierter Permutation Importance erklärt. Die Modelle werden dafür nur auf früheren Daten bis D−2 trainiert; die Merkmale werden ausschliesslich in den folgenden Validierungswochen permutiert. Ergebnisse stehen in `tune_feature_importance.csv` und `tune_feature_importance_repeats.csv`, bei Tracking zusätzlich als Tabelle und Balkengrafik in W&B.

**4. Bestes Setting als `CHOSEN` eintragen**

Die Zeile mit dem tiefsten Pinball Loss (Spalte `pinball`) in `model.py` als `CHOSEN` eintragen. Der Pinball Loss ist die Hauptkennzahl, weil er P10, P50 und P90 zusammen bewertet; MAE (`mae_q50`) und Coverage (`coverage_80`) werden ergänzend angeschaut. Der Name links (`"gbm"`) erscheint in den Ergebnissen und Grafiken. Danach wird `CHOSEN` nicht mehr geändert.

**5. Test**

Die tägliche Swissgrid-Verfügbarkeit bis D−2 wurde am 22.10.2026 vom Nutzer im Projekt
bestätigt. Damit ist der finale Test für `gbm_quantile` gemäss Issue #17 freigegeben.
Der Holdout bleibt gesperrt.

```bash
make backtest MODEL=gbm_quantile STEP=test
```

Rechnet Juni bis August und vergleicht automatisch mit der Referenz und der Baseline B2. Ergebnisse in `reports/gbm_quantile/`, Grafiken in `docs/figures/gbm_quantile/`. Mit `TRACK=1` werden die Läufe zusätzlich in W&B gespeichert: die Kennzahlen des ganzen Zeitraums in der Summary (Runs-Tabelle) und pro Liefertag `test/daily_pinball` sowie `test/daily_pinball_vs_reference` (Modell minus Referenz, unter 0 ist besser) als Liniendiagramm im Workspace. Step 0 ist der erste Tag des Zeitraums.

**6. Ergebnis dokumentieren**

Ergebnis und Entscheid im `README.md` des Modellordners eintragen und die Zeile in der Übersicht oben nachführen.

## 4. Was alle Modelle gemeinsam nutzen

| Baustein | Datei | Wofür |
|---|---|---|
| Feature-Sets `B1` bis `B4` | `features/design.py` | B1 Kalender, B2 Kalender plus Sonnenprognose, B3 zusätzlich Preisniveau bis D−2, B4 zusätzlich Systembilanz bis D−2 und Bewölkung ([Zeitumstellung](../data.md#zeitumstellung-uhrzeiten-die-fehlen-oder-doppelt-vorkommen), [Mindesthistorie](../data.md#mindesthistorie-der-preis--und-bilanzmerkmale)); neue Features als Block dort ergänzen |
| `PerQuantileModel` | `models/shared/base.py` | Grundgerüst für jedes Modell |
| `LinearQuantileModel` | `models/shared/linear.py` | lineare Quantilregression der Baseline |
| `GradientBoostingQuantileModel` | `models/shared/boosting.py` | regularisiertes Quantile Boosting mit festem Seed und ohne zufälliges Early Stopping |
| Gruppierte Permutation Importance | `models/shared/importance.py` | wiederverwendbare Bedeutung einer Datenquelle, gemessen am Anstieg des Pinball-Loss |
| Referenz | `models/shared/reference.py` | Massstab für jedes Modell |
| Backtest und Ablauf | `evaluation/backtest.py`, `evaluation/experiment.py` | Training bis D−2, wöchentlich neu, `make backtest` |
| Kennzahlen | `evaluation/scores.py` | MAE, Pinball Loss, Coverage, Skill Score |
| Diagnostik | `evaluation/diagnostics.py` | zeitlich getrennte Feature Importance und Auswertung nach Stunde, Monat, Tagtyp und tatsächlichem short/long |
| Grafiken, W&B | `evaluation/figures.py`, `evaluation/tracking.py` | gleiche Grafiken und Tracking für jedes Modell |

## 5. Regeln

- **Kein Data Leakage:** Der Backtest gibt dem Modell nur Preise bis D−2. Neue Features gehören in `features/`, mit einem Test, dass sie um D−1 11:00 bekannt sind ([data.md](../data.md#1-die-regel-gegen-datenlecks)).
- **Erst abstimmen, dann testen:** April und Mai zum Abstimmen, Juni bis August als Test. Der Holdout (September und Oktober) läuft genau einmal ganz am Schluss: `make backtest MODEL=<name> STEP=holdout UNLOCK=1`.
- **Reproduzierbar:** Modelle mit Zufall setzen einen festen Seed (`random_state=0`).
- **Tests sind optional:** `tests/test_model_template.py` zeigt, wie ein Modell getestet wird.

## 6. In Arbeit

Diese Modelle sind noch nicht in `main`. Ihre Steckbriefe übernehmen die Vorlage im jeweiligen Pull Request.

| Modell | Verantwortlich | Issue und PR | Idee |
|---|---|---|---|
| Kalibrierte Bänder ([`calibrated_bands/`](../../src/energy_price/models/calibrated_bands/)) | Luca Manna | #19, PR #24 | Die Bänder aller Modelle sind zu schmal (Coverage rund 74 % statt 80 %). Eine Korrektur aus den Fehlern der letzten Wochen bis D−2 (Conformal Prediction) soll die Abdeckung auf 80 % heben, ohne den Pinball-Loss deutlich zu verschlechtern |
| Quantil-LASSO (`quantile_lasso`) | Luca Manna | #27, PR #35 | Quantilregression mit allen um D−1 11:00 bekannten Daten, auch Preisniveau und Systembilanz bis D−2. Das LASSO behält nur die Merkmale, die tragen. Beantwortet, ob vergangene Preise und Systembilanz überhaupt helfen |
| Zeitabhängige Modellwahl (`time_router`) | Elias Pulver | #25 | pro Zeitfenster des Liefertags das Modell nutzen, das dort verlässlich am besten ist. Boosting ist vormittags bis mittags besser, abends oft die Referenz |
