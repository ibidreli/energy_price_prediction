# Kalibrierte Bänder

**Status:** abgeschlossen, Erfolgskriterium verfehlt · **Verantwortlich:** Luca Manna · **Issue:** #19 · **Stand der Zahlen:** 10.11.2026

## 1. Hypothese

Getrennte Korrekturen von P10 und P90 aus vergangenen Prognosefehlern heben die Coverage auf 78 bis 82 %. Der mittlere Pinball-Loss steigt gegenüber dem jeweiligen Basismodell höchstens um 1 %, P50 bleibt unverändert.

## 2. Herleitung

- **Ausgangspunkt:**
    - Die Bänder der Baseline B2 und des Gradient Boosting sind zu schmal, im Test liegen nur rund 74 % der Preise zwischen P10 und P90 statt 80 %
    - Ein verlässliches Band ist für den Use Case zentral, weil eine Bilanzgruppe damit ihr Risiko abschätzt
- **Erwarteter Vorteil:** Die Korrektur ist eine Hülle um ein bestehendes Modell. Sie verbessert die Bänder, ohne P50 und damit den MAE zu verändern, und lässt sich auf jedes Basismodell anwenden
- **Verworfene Alternative:** eine gemeinsame, symmetrische Verbreiterung beider Ränder. Fehler unter P10 und über P90 können unterschiedlich gross sein und werden deshalb getrennt korrigiert

## 3. Vorgehen

- **Modell:** Hülle `CalibratedQuantileModel` in `models/shared/calibration.py` um die festgelegte Baseline B2 (expanding) und das festgelegte `gbm_quantile` aus #17, beide mit B2-Features. Keine neuen Datenquellen
- **Korrektur:**
    - Einseitige Fehler `q10 - y` und `y - q90` im gewählten Fehlerfenster, je Rang `ceil((n+1)*0.9)`
    - P10 wird um den ersten Wert verringert, P90 um den zweiten erhöht. Negative Korrekturen machen das Band enger, die Grenzen werden an P50 begrenzt
    - Gruppen mit weniger als 20 Fehlern verwenden die gepoolte Korrektur. Fehlende Prognosen bleiben fehlend
- **Zeitliche Trennung:**
    - Die äusseren Fits erfolgen wöchentlich wie im gemeinsamen Backtest und sehen nur Preise bis D−2 des ersten Liefertags ihrer Woche. Die Korrektur bleibt für diese Woche fest
    - Für die Fehlerhistorie werden eigene wöchentliche Prognosen ab 30. Januar berechnet. Jeder historische Fit sieht nur Preise bis D−2 seines ersten Liefertags und mindestens 28 Tage Training. Keine Kalibrierungszeile liegt im Training ihres Prognosemodells
    - Wegen der Zeitabhängigkeit der Fehler gibt es keine verteilungsfreie Garantie für 80 % Coverage, die Abdeckung wird deshalb empirisch und je Monat geprüft
- **Tuning:** April und Mai, 61 gemeinsame Tage. Raster aus 28 oder 56 Tagen Fehlerhistorie, jeweils gepoolt, nach vier 6-Stunden-Blöcken oder nach 24 Stunden. Je Basismodell gewinnt die Variante mit dem niedrigsten Pinball-Loss unter denen, die beide Kriterien erfüllen, sonst die mit dem niedrigsten Pinball-Loss

| Variante | Pinball-Loss | MAE P50 | Coverage | Intervallbreite |
|---|---|---|---|---|
| Baseline B2 | 3.311 | 9.72 | 78.3 % | 25.8 |
| B2 kalibriert, 56 Tage, gepoolt | 3.327 | 9.72 | 83.5 % | 29.4 |
| Gradient Boosting | 3.309 | 9.79 | 76.4 % | 25.7 |
| **Boosting kalibriert, 56 Tage, 6-Stunden-Blöcke** | **3.318** | 9.79 | 81.8 % | 30.1 |

- **Gewählt:**
    - B2 mit 56 Tagen, gepoolt. Keine B2-Variante erfüllt beide Kriterien, gewählt wurde der niedrigste Pinball-Loss
    - Boosting mit 56 Tagen und 6-Stunden-Blöcken, beide Kriterien im Tuning erfüllt
    - Beide vor dem Test als `CHOSEN` festgelegt und danach nicht mehr geändert

## 4. Erkenntnisse

**Die Kalibrierung verbessert den Pinball-Loss**
- Gegenüber der jeweiligen Basis B2 −1.6 %, Boosting −1.2 %
- Gegenüber der Referenz B2 −0.9 %, Boosting −2.4 %. Das kalibrierte Boosting hat den niedrigsten Test-Loss aller bisher verglichenen Modelle
- P50 ist über alle Prognosezeilen exakt identisch zum Basismodell

**Das Coverage-Ziel wird knapp verfehlt**
- Im Tuning lag die Coverage mit 83.5 % und 81.8 % über dem Ziel, im Test mit 77.7 % und 77.4 % knapp darunter
- Die Korrektur aus den Wochen davor überträgt sich also nicht vollständig auf die Testmonate

**Die Prognosequalität nimmt über den Sommer ab**
- Ohne Kalibrierung fällt die Coverage von Juni bis August von rund 79 % auf 67 bis 69 %
- Die Kalibrierung fängt einen grossen Teil davon auf, im August liegt sie bei 76 bis 77 %
- Auch der monatliche Pinball-Loss steigt, bei B2 von 3.06 über 3.14 auf 3.32, bei Boosting von 3.02 über 3.14 auf 3.22. Die Ursache klärt dieses Experiment nicht

## 5. Outcome

Test Juni bis August, 91 Tage (8736 Viertelstunden). Der 24. Juni entfällt bei allen Modellen wegen fehlender Wettermerkmale. Preise, Fehler und Breiten in ct/kWh.

| Modell | Pinball-Loss | MAE P50 | Coverage | Intervallbreite |
|---|---|---|---|---|
| Referenz | 3.20 | 9.92 | 75.2 % | 26.3 |
| Baseline B2 | 3.23 | 9.81 | 73.6 % | 25.5 |
| B2 kalibriert | 3.17 | 9.81 | **77.7 %** | 26.2 |
| Gradient Boosting | 3.16 | **9.79** | 74.1 % | 24.8 |
| Boosting kalibriert | **3.13** | **9.79** | 77.4 % | 26.6 |

| Variante | Pinball gegenüber Basis | Coverage 78 bis 82 % | Loss höchstens +1 % | Beide erfüllt |
|---|---|---|---|---|
| B2 | −1.58 % | nein | ja | nein |
| Boosting | −1.20 % | nein | ja | nein |

**Coverage nach Monat**

| Modell | Juni | Juli | August |
|---|---|---|---|
| Baseline B2 | 79.6 % | 75.1 % | 66.5 % |
| B2 kalibriert | 78.9 % | 78.0 % | 76.3 % |
| Gradient Boosting | 78.4 % | 75.6 % | 68.5 % |
| Boosting kalibriert | 78.8 % | 77.0 % | 76.7 % |

![Coverage nach Monat](../../../../docs/figures/calibrated_bands/test_coverage_by_month.png)

![Pinball-Loss nach Stunde](../../../../docs/figures/calibrated_bands/test_pinball_by_hour.png)

## 6. Entscheid

**Erfolgskriterium verfehlt, Ansatz weiterverfolgen**, weil beide Varianten das vorab festgelegte Coverage-Ziel von 78 bis 82 % knapp verfehlen, den Pinball-Loss aber verbessern und die Coverage im August deutlich stabilisieren. Das kalibrierte Boosting hat den niedrigsten Test-Loss, gilt aber nicht als erfolgreich kalibriert. Die Settings werden nach diesem Test nicht geändert. Weitere Kalibrierungsansätze brauchen ein eigenes Experiment mit sauberem Validierungsprotokoll. Der Holdout September und Oktober bleibt gesperrt.

## 7. Belege

- **Issue und PR:** #19, PR #24
- **Code:** `src/energy_price/models/calibrated_bands/model.py`, `src/energy_price/models/shared/calibration.py`
- **Tests:** `tests/test_calibrated_bands.py` prüft innere und äussere D−2-Grenzen, Unabhängigkeit von späteren Preisen, getrennte Korrekturen, unverändertes P50, Quantilordnung, fehlende Werte, kleine Gruppen und die Auswahl je Basismodell
- **Ergebnisse:** `reports/calibrated_bands/` (nicht versioniert), Grafiken in `docs/figures/calibrated_bands/`, W&B-Projekt `fhnw-cds1/energy-price-prediction`, Gruppe `issue-19`

```bash
make backtest MODEL=calibrated_bands STEP=tune JOBS=1          # Raster auf April und Mai
# chosen aus reports/calibrated_bands/tune_selection.json als CHOSEN festschreiben
make backtest MODEL=calibrated_bands STEP=test JOBS=1 TRACK=1  # CHOSEN auf Juni bis August
```

Der lokale Joblib-Cache unter `reports/calibrated_bands/cache/` speichert historische Basisprognosen anhand von Daten und Basisparametern. Änderungen am Quellcode oder an den numerischen Paketversionen erzeugen automatisch einen neuen Cache-Namensraum.
