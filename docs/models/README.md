# Modelle: neues Modell erstellen

Jedes Modell ist ein eigener Ordner in `src/energy_price/models/`. Daten, Backtest, Kennzahlen und Grafiken sind für alle Modelle gleich und schon fertig. Für ein neues Modell schreibt man deshalb nur das Modell selbst und seine Einstellungen.

## 1. Übersicht

| Modell | Ordner | Status | MAE P50 Test | Pinball Loss Test |
|---|---|---|---|---|
| Referenz (letzte 56 Tage) | `models/shared/reference.py` | Massstab | 9.92 | 3.20 |
| Baseline B2 | [`baseline_model/`](../../src/energy_price/models/baseline_model/) | abgeschlossen | 9.81 | 3.23 |
| Quantile Boosting | [`gbm_quantile/`](../../src/energy_price/models/gbm_quantile/) | abgestimmt; finaler Test freigegeben | ausstehend | ausstehend |

Ein neues Modell muss die Referenz schlagen. Jedes Modell wird hier nachgeführt, sobald es Ergebnisse hat. Geplante Modelle stehen am Schluss unter To-do.

Quantile Boosting erreicht im **Tuning April/Mai** einen Pinball-Loss von 3.3087
(Referenz 3.4485; Baseline B2 3.3112), bei 76.35 % Coverage. Das sind 4.05 % weniger
Loss als die Referenz, aber praktisch gleich viel wie bei der linearen Baseline.
Diese Tuning-Zahlen sind nicht mit den Testzahlen der Tabelle vergleichbar.

## 2. Aufbau eines Modellordners

```text
src/energy_price/models/<name>/
├── __init__.py   leer, damit Python den Ordner findet
├── README.md     Steckbrief: Hypothese, Modell und Features, Ergebnis, Entscheid
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
| Feature-Sets `B1`, `B2` | `features/design.py` | B1 Kalender, B2 Kalender plus Sonnenprognose; neue Features als Block dort ergänzen |
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

## 6. To-do: geplante Modelle

Die Baseline erreicht die Referenz, schlägt sie aber nicht. Kalibrierte Bänder und das Regime-Modell sind noch geplant. Quantile Boosting ist implementiert; den aktuellen Stand und die Ergebnisse beschreibt sein Modell-README.

**Kalibrierte Bänder** · [`calibrated_bands/`](../../src/energy_price/models/calibrated_bands/)

Korrigiert die Bänder der Baseline anhand der Fehler der letzten Wochen, damit sie so oft treffen wie vorgesehen (Conformal Prediction).

- Die Bänder der Baseline sind zu schmal, der echte Preis liegt häufiger ausserhalb als vorgesehen.
- Ein verlässliches Band ist für den Use Case zentral, weil eine Bilanzgruppe damit ihr Risiko abschätzt.
- Einfach umzusetzen, und der Korrekturschritt lässt sich später auf jedes Modell anwenden.

**Gradient Boosting (implementiert)** · [`gbm_quantile/`](../../src/energy_price/models/gbm_quantile/)

Viele kleine Entscheidungsbäume, die nacheinander die Fehler der vorherigen korrigieren und direkt P10, P50 und P90 vorhersagen.

- Der Preis hängt von Kombinationen ab, zum Beispiel wirkt die Sonne je nach Uhrzeit und Jahreszeit anders. Ein lineares Modell bildet solche Wechselwirkungen nur ab, wenn man sie von Hand vorgibt; ein Baummodell findet sie selbst.
- Die gruppierte Permutation Importance zeigt, wie stark das Modell auf Kalender, Feiertage und Strahlung angewiesen ist. Sie beschreibt die Nutzung der Merkmale durch das Modell, keinen kausalen Beitrag. Das wurde im Review ausdrücklich verlangt.

**Regime-Modell short/long** · [`regime_mixture/`](../../src/energy_price/models/regime_mixture/)

Schätzt zuerst, ob im Netz Strom fehlt (short) oder zu viel da ist (long), und sagt den Preis dann für beide Situationen getrennt voraus.

- Fehlt Strom, ist der Preis meist hoch; gibt es zu viel Strom, ist er tief und teils negativ. Ein einzelnes Modell mittelt die beiden Situationen und landet dazwischen, wo kaum echte Preise liegen.
- Im Review wurde dieser Sprung bei unterdeckter Regelzone als zentraler Punkt genannt.
- Die geschätzte Richtung zeigt zusätzlich, ob eine Bilanzgruppe bei einer Abweichung zahlt oder vergütet wird.
