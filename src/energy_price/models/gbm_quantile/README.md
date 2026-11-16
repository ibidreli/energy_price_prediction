# Gradient Boosting mit Quantil-Loss

**Status:** abgeschlossen, bestes Modell bisher, Kandidat für den Holdout · **Verantwortlich:** Elias Pulver · **Issue:** #17 · **Stand der Zahlen:** 22.10.2026

## 1. Hypothese

Der Einfluss der Sonnenstrahlung auf den Preis hängt von Uhrzeit und Position innerhalb der Stunde ab. Gradient Boosting bildet diese nichtlinearen Zusammenhänge flexibler ab als die lineare Baseline B2 und schlägt die Referenz im Pinball-Loss.

## 2. Herleitung

- **Ausgangspunkt aus der EDA:**
    - Die Sonne wirkt je nach Uhrzeit und Jahreszeit anders, das Mittagsloch des Preises gibt es erst ab April (H3, H7)
    - Die Richtung des Kippens innerhalb der Stunde hängt von der Uhrzeit ab (H8)
- **Erwarteter Vorteil:** Ein lineares Modell bildet solche Wechselwirkungen nur ab, wenn man sie von Hand vorgibt. Ein Baummodell findet sie selbst
- **Gleiche Features wie die Baseline:** Damit zeigt der Vergleich nur den Effekt der Modellform, nicht neuer Daten
- **Erklärbarkeit:** Die gruppierte Permutation Importance zeigt, wie stark das Modell auf Kalender, Feiertage und Strahlung angewiesen ist. Das wurde im Review verlangt

## 3. Vorgehen

- **Modell:** `GradientBoostingQuantileModel` in `models/shared/boosting.py`, je ein `HistGradientBoostingRegressor(loss="quantile")` für P10, P50 und P90
    - `PerQuantileModel` übernimmt fehlende Werte, die Feature-Auswahl und sortierte Quantile
    - keine Standardisierung nötig, Early Stopping deaktiviert, damit keine zufällige interne Validierung stattfindet, Seed 0
- **Features:** exakt die **B2-Features** der Baseline
    - Uhrzeit, Position innerhalb der Stunde, Wochenende, PV-gewichteter Feiertagsanteil
    - Strahlungsprognose und Strahlungsänderung × Position
    - keine weiteren Wettervariablen und keine Merkmale der Regelzonenbilanz. Die tatsächliche Richtung short/long wird nur nach der Prognose für die Auswertung verknüpft
- **Tuning:** April und Mai, 61 Tage bzw. 5'856 Viertelstunden, neun wöchentliche Fits, Training nur mit Preisen bis D−2

| Konfiguration | Lernrate | maximale Blätter | Mindestwerte pro Blatt |
|---|---|---|---|
| Ausgangspunkt | 0.05 | 7 | 50 |
| Höhere Lernrate | 0.10 | 7 | 50 |
| Grössere Bäume | 0.05 | 15 | 50 |
| Mehr Werte pro Blatt | 0.05 | 7 | 100 |

- **Für alle Varianten:** 150 Boosting-Iterationen, L2-Regularisierung 1.0, Trainingsfenster alle Daten seit Januar (`expanding`) oder die letzten 56 Tage
- **Gewählt:** `expanding`, Lernrate 0.05, 7 Blätter, mindestens 50 Werte pro Blatt. Tiefster Tuning-Pinball-Loss, vor dem Test als `CHOSEN` festgelegt und danach nicht mehr geändert
    - mehr Blätter und das kurze Trainingsfenster bringen im untersuchten Grid keine Verbesserung
- **Verfügbarkeit:** Die tägliche Swissgrid-Verfügbarkeit bis D−2 wurde am 22.10.2026 im Projekt bestätigt

**Tuning April und Mai** (nur April und Mai, nicht direkt mit dem Test vergleichbar):

| Modell | Pinball-Loss | MAE P50 | Coverage | Intervallbreite |
|---|---|---|---|---|
| Referenz | 3.4485 | 9.9821 | 77.60 % | 26.0844 |
| Baseline B2 | 3.3112 | 9.7192 | 78.26 % | 25.8234 |
| Quantile Boosting | **3.3087** | 9.7887 | 76.35 % | 25.6868 |

## 4. Erkenntnisse

**Kleiner Vorteil, vor allem gegenüber der Referenz**
- Im Tuning 4.05 % weniger Loss als die Referenz, aber nur 0.08 % weniger als die lineare Baseline
- Im Test 1.20 % weniger als die Referenz und 1.93 % weniger als B2, an 58.24 % der Testtage besser als die Referenz
- Die Testhypothese ist im Mittel erfüllt, die MAE-Verbesserung gegenüber B2 ist mit 0.0140 ct/kWh aber sehr gering
- Ein statistisch abgesicherter Vorteil wurde nicht nachgewiesen

**Die Wirkung hängt von der Richtung ab**
- Bei short sinkt der Test-Loss gegenüber B2 von 3.3041 auf 3.0836
- Bei long steigt er von 3.1360 auf 3.2539
- Im Tuning dasselbe Bild, bei short 2.1935 gegenüber 2.3131 der Baseline, bei long 4.6908 gegenüber 4.5482
- Das Modell löst die Unsicherheit über die Richtung im Netz nicht auf

**Strahlung und Kalender tragen das Modell** (Permutation Importance, Tuning)

| Quelle | Anstieg Pinball-Loss | Streuung über Wochen und Wiederholungen |
|---|---|---|
| Strahlung | 0.4318 | 0.1616 |
| Kalender | 0.2926 | 0.1016 |
| Feiertage | 0.0263 | 0.0589 |

- Fünf Permutationen je Quelle und Woche, 45 Messungen pro Quelle, nur auf den folgenden Validierungswochen bewertet
- Die Werte zeigen die Abhängigkeit dieses Modells von seinen Eingaben, keine Kausalität und keine additiven Beiträge zum Preis
- Berechnet auf dem Tuning-Zeitraum, der auch zur Auswahl diente, deshalb explorativ

**Die Bänder bleiben zu schmal**
- Coverage im Test 74.08 % statt 80 %, im August nur 68.51 %
- Der August-Loss ist mit 3.3153 praktisch gleich wie bei der Referenz (3.3136)
- Boosting löst das Problem zu schmaler Bänder nicht, eine Kalibrierung bleibt nötig (#19)

**Feiertage und Abende bleiben schwierig**
- Nur fünf Testtage fallen unter „PV-Feiertagsanteil > 0“, dort beträgt die Coverage 61.67 %. Im Tuning waren es sechs Tage mit 63.72 %
- Eine belastbare Aussage über einzelne Feiertagseffekte lässt sich daraus nicht ableiten
- Vormittags bis mittags bringt Boosting Vorteile, abends ist die Referenz häufig besser

## 5. Outcome

Test Juni bis August, 91 Tage bzw. 8'736 Viertelstunden. Der 24.06. fehlt wegen unvollständiger Wetterdaten für alle Modelle. Finaler Lauf vom 22.10.2026 mit unverändertem `CHOSEN`, 14 wöchentlichen Fits und Code-Commit `161b317`. Die gespeicherten B2-Features und Zielwerte der Baseline wurden vor dem Lauf gegen die aktuellen Daten geprüft und stimmen überein. Die Run-Metadaten vermerken einen sauberen Git-Stand und die D−2-Bestätigung samt Datum und Quelle. Der Holdout wurde nicht geöffnet. Preise, Fehler und Breiten in ct/kWh.

| Modell | Pinball-Loss | MAE P50 | Coverage | Intervallbreite |
|---|---|---|---|---|
| Referenz | 3.2018 | 9.9171 | 75.21 % | 26.3184 |
| Baseline B2 | 3.2254 | 9.8063 | 73.60 % | 25.5145 |
| Gradient Boosting | **3.1633** | **9.7923** | 74.08 % | 24.8447 |

![Pinball-Loss pro Stunde](../../../../docs/figures/gbm_quantile/test_pinball_by_hour.png)

![Coverage pro Monat](../../../../docs/figures/gbm_quantile/test_coverage_by_month.png)

## 6. Entscheid

**Weiterverfolgen als Kandidat für den gemeinsamen Holdout**, weil es die Referenz im vorab definierten Test leicht schlägt. Eine verlässliche Abdeckung von 80 % erreicht es noch nicht, die Kalibrierung wird separat untersucht. `CHOSEN` wird anhand dieser Testauswertung nicht geändert. Das Restrisiko nachträglich revidierter Swissgrid-Preise bleibt bestehen, siehe [Datengrundlage](../../../../docs/data.md).

## 7. Belege

- **Issue und PR:** #17, PR #18
- **Code:** `src/energy_price/models/gbm_quantile/model.py`, `src/energy_price/models/shared/boosting.py`, `src/energy_price/models/shared/importance.py`
- **Tests:** Leakage, Reproduzierbarkeit, Importance und Integration in `tests/test_gbm_quantile.py`, `tests/test_model_importance.py` und `tests/test_evaluation_experiment.py`
- **Ergebnisse und Auswahlprotokoll:** `reports/gbm_quantile/` (nicht versioniert), Grafiken in `docs/figures/gbm_quantile/`
- **W&B:** Gruppe `issue-17` mit zehn Tuning-Läufen und drei finalen Testläufen (Referenz, Boosting, Baseline B2)
    - jeweils mit Summary-Metriken und täglichen Pinball-Kurven, die gewählte Tuning-Variante zusätzlich mit der Importance-Grafik
    - der finale Boosting-Lauf mit Auswertungstabellen und einem Artefakt aus aggregierten Ergebnissen und dem verwendeten Quellcode
    - die Run-Links stehen nur lokal in `reports/gbm_quantile/tune_wandb_urls.json` und `test_wandb_urls.json`

```bash
make backtest MODEL=gbm_quantile STEP=tune TRACK=1 JOBS=2
make backtest MODEL=gbm_quantile STEP=test TRACK=1 JOBS=2
```

Unter PowerShell direkt mit dem bestehenden Environment:

```powershell
.venv/Scripts/python.exe -m energy_price.evaluation.experiment gbm_quantile tune --jobs 2 --track
```
