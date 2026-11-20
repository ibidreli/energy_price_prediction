# Trainingspipeline: tägliches Walk-forward

> Konzept zu #36, Stand 17.11.2026. Umgesetzt in #48 (Kern) und #49 (Modelle), Zahlen aus #51 in [Abschnitt 6](#6-neutraining-und-rechenzeit) und [7](#7-ergebnisse-auf-den-validierungsblöcken). Metriken und Referenz stehen in [`evaluation.md`](evaluation.md), die Datengrenze D−2 in [`data.md`](data.md#1-die-regel-gegen-datenlecks).

## 1. Überblick

Jeder Liefertag D ist ein eigener Prognoseursprung: Das Modell wird mit allen Daten bis Ende D−2 neu trainiert und sagt um 11 Uhr an D−1 die Viertelstunden von D voraus. Januar bis März dienen nur als Training. April bis August sind die Validierung: Dort wählt jedes Modell seine Einstellungen und dort werden die Modelle verglichen, ausgewiesen je Monat. September und Oktober sind die Testperiode, im Code gesperrt und erst nach dem Freeze genau einmal ausgewertet (#53).

```mermaid
gantt
    title Perioden 2026
    dateFormat YYYY-MM-DD
    todayMarker off
    axisFormat %b
    tickInterval 1month
    section Entwicklung
    Nur Training         :done, 2026-01-01, 2026-04-01
    Validierung April    :active, 2026-04-01, 2026-05-01
    Validierung Mai      :active, 2026-05-01, 2026-06-01
    Validierung Juni     :active, 2026-06-01, 2026-07-01
    Validierung Juli     :active, 2026-07-01, 2026-08-01
    Validierung August   :active, 2026-08-01, 2026-09-01
    section Test
    Testperiode mit Vorbehalt, gesperrt :crit, 2026-09-01, 2026-11-01
```

„Mit Vorbehalt“, weil die EDA TSI und Wetter aus September und Oktober betrachtet hat (H2, H3; TSI 26.10. bis 31.10. in Kapitel 1.5.4), nie aber Preise oder Prognosen (siehe [Einschränkungen](#8-einschränkungen)).

## 2. Wie ein Ursprung funktioniert

Beispiel Liefertag **D = 03.06.2026**: Prognose am 02.06. um 11 Uhr, Training mit Daten bis Ende 01.06. (D−2). Die Preise von D−1 sind um 11 Uhr noch nicht publiziert und fehlen deshalb.

```mermaid
gantt
    title Ursprung D = 03.06.2026
    dateFormat YYYY-MM-DD HH:mm
    todayMarker off
    axisFormat %d.%m.
    tickInterval 1day
    section Daten
    … Training bis Ende D−2 (01.06.)   :done, 2026-05-29 00:00, 2026-06-02 00:00
    D−1 (02.06.) noch unbekannt        :crit, 2026-06-02 00:00, 24h
    section Prognose
    Ausgabe 02.06. um 11 Uhr           :milestone, 2026-06-02 11:00, 0m
    Liefertag D mit 96 Viertelstunden  :active, 2026-06-03 00:00, 24h
```

Am nächsten Tag rückt alles um einen Tag weiter: neuer Ursprung, ein Tag mehr Trainingsdaten, neues Modell.

## 3. Rolling Origin und Rolling Window

- **Rolling Origin:** Der Ursprung wandert Tag für Tag durch Validierung (und später Test).
- **Fenster:** wie viel Vergangenheit ein Ursprung zum Training nutzt. `expanding` nimmt alles ab 01.01., `rolling_<w>d` nur die letzten w Tage (D−1−w bis D−2).

Drei aufeinanderfolgende Ursprünge:

| Ursprung D | Prognose am | `expanding` | `rolling_56d` |
|---|---|---|---|
| 01.06. | 31.05., 11 Uhr | 01.01. bis 30.05. (150 Tage) | 05.04. bis 30.05. (56 Tage) |
| 02.06. | 01.06., 11 Uhr | 01.01. bis 31.05. (151 Tage) | 06.04. bis 31.05. (56 Tage) |
| 03.06. | 02.06., 11 Uhr | 01.01. bis 01.06. (152 Tage) | 07.04. bis 01.06. (56 Tage) |

`expanding` wächst, `rolling` gleitet. Dasselbe gilt im Test: Das Training läuft bis D−2 in die Testperiode hinein. Das ist kein Leakage, denn diese Preise sind um 11 Uhr an D−1 publiziert. Beispiel Ursprung 15.10.:

```mermaid
gantt
    title Ursprung 15.10.2026 im Test, Training bis D−2 = 13.10.
    dateFormat YYYY-MM-DD
    todayMarker off
    axisFormat %b
    tickInterval 1month
    section Perioden
    Entwicklung                         :done, 2026-01-01, 2026-09-01
    Test                                :crit, 2026-09-01, 2026-11-01
    section Fenster
    expanding 01.01. bis 13.10.         :active, 2026-01-01, 2026-10-14
    rolling 90 16.07. bis 13.10.        :active, 2026-07-16, 2026-10-14
    rolling 56 19.08. bis 13.10.        :active, 2026-08-19, 2026-10-14
    rolling 28 nur Baseline 16.09. bis 13.10. :active, 2026-09-16, 2026-10-14
    section Prognose
    Liefertag 15.10.                    :milestone, 2026-10-15, 0d
```

| Fenster | wer | Hinweis |
|---|---|---|
| `expanding`, `rolling_56d`, `rolling_90d` | alle Modelle (Standard) | `calibrated_bands` nur `expanding` |
| `rolling_28d` | nur `baseline_model` (sparsam, B1 und B2) | unter 8 Wochen, siehe [Herleitung](#9-herleitung) |
| `window_mean` („Mittel über die Fenster“) | zusätzlicher Kandidat je Setting | Mittel von P10, P50 und P90 über die Fenster des Modells; bleibt sortiert |

Früh in der Validierung sind Fenster gleich: `rolling_90d` entspricht `expanding` bis Ursprung 02.04., solange die Merkmale ab 01.01. vorliegen (B1, B2), und bis 01.05. bei Merkmalen ab 30.01. (B3, B4, siehe [Mindesthistorie](data.md#mindesthistorie-der-preis--und-bilanzmerkmale)).

## 4. Ablauf mit Befehlen

```mermaid
flowchart LR
    S["make select MODEL=name<br/>je Modell"] --> C["CHOSEN in model.py<br/>festhalten und committen"]
    C --> K["make compare<br/>alle Modelle"]
    K --> F["Freeze<br/>sauberer git-Stand"]
    F --> T["make final-test UNLOCK_TEST=1<br/>einmal, Folge-Issue"]
```

| Schritt | schreibt |
|---|---|
| `make select MODEL=<name>` | `reports/<name>/select_*`: Prognosen, `scores.csv` je Kandidat und Monat (mit Rang), `selection.json`, `runtime.csv`. Gewählt wird der kleinste mittlere Pinball über alle Validierungstage (152, der 24.06. fehlt), jeder Tag gleich gewichtet |
| `CHOSEN` festhalten | `src/energy_price/models/<name>/model.py`, inklusive Fenster; per Commit nachvollziehbar |
| `make compare` | `reports/compare/validation_*` und Tabelle: alle `CHOSEN`, Referenz und B2 auf denselben Viertelstunden, Skill und 95-%-Intervall der täglichen Differenz |
| Freeze | nichts: ab hier ändert sich kein Modell und kein `CHOSEN` mehr |
| `make final-test UNLOCK_TEST=1` | zuerst `docs/test_protocol.json` (Commit, alle `CHOSEN`, Design, Zeitpunkt), dann die Scores auf dem Test. In #36 nur Gerüst, der Lauf folgt in #53 |

Optionen: `JOBS=<n>` (parallel), `TRACK=1` (W&B, siehe [`experiment_tracking.md`](experiment_tracking.md)), `REFIT=<k>`.

## 5. Regeln gegen Leakage

- **Grenze D−2:** Trainingszeilen und Merkmale entstehen nur über `known_at_issue` (Preise und Bilanz bis Ende D−2), [Details](data.md#1-die-regel-gegen-datenlecks).
- **Test abgeschnitten:** In `select` und `compare` schneidet `development_inputs` die Rohdaten am 01.09. ab, bevor `build_table` Merkmale bildet. Auch rollierende Merkmale sehen so keinen Testwert.
- **Anpassungen nur im Fit:** Skalierung, Merkmalsauswahl und Kalibrierung lernen nur aus den Trainingszeilen des jeweiligen Ursprungs. Bei `calibrated_bands` gelten Fenster und Refit-Frequenz der Basis in `fit` und in `predict`.
- **Identische Viertelstunden:** `common_rows` bewertet alle Modelle, Referenz und B2 auf denselben Viertelstunden. Ein Tag zählt nur, wenn jedes Modell alle 92, 96 oder 100 Viertelstunden liefert.
- **Testsperre:** `test_days()` wirft ohne Freigabe `TestLockedError`. `final-test` verlangt einen sauberen git-Stand, schreibt `docs/test_protocol.json` vor der Auswertung und verweigert, wenn die Datei schon existiert.

## 6. Neutraining und Rechenzeit

- **Täglich:** Innerhalb eines Laufs wird je Ursprung neu trainiert (`refit_every = 1`), also 153 Fits je Fenster und Setting in `select`.
- **Ausnahme:** Dauert ein `select`-Lauf eines Modells länger als 8 h, setzt das Modell `REFIT_EVERY = k` (Entscheid je Modell durch Luca). Beispiel k = 7: Das Modell vom 01.06. (Training bis 30.05.) prognostiziert 01.06. bis 07.06.
- **Kein Betrieb:** Läufe startet nur das Team. Im Betrieb wäre tägliches automatisches Training optimal, wird aber mangels Ressourcen nicht betrieben.

**Laufzeit je Modell** (#51, `select_runtime.csv` und Laufprotokolle, 12 Kerne mit `JOBS=-1`)

| Modell | Kandidaten | Fits | Sekunden je Fit (Mittel) | Dauer `select` | `REFIT_EVERY` |
|---|---|---|---|---|---|
| `baseline_model` | 30 | 3'672 | 8.1 | 2'847 s (47 min) | 1 |
| `quantile_lasso` | 48 | 5'661 | 12.5 | 6'497 s (1 h 48 min) | 1 |
| `gbm_quantile` | 16 | 1'989 | 4.7 | 938 s (16 min) | 1 |
| `regime_mixture` | 8 | 1'071 | 39.7 | 3'772 s (1 h 3 min) | 1 |
| `calibrated_bands` | 14 | 2'295 | 15.9 | 3'223 s (54 min) | 1 |

Kandidaten inklusive `window_mean`, das kein eigenes Training braucht. Fits = Zeilen in `select_runtime.csv`, ausser bei `baseline_model` inklusive der 153 Fits von B2 als Vergleich. `make compare` dauerte 6'390 s. **Alle Läufe bleiben unter 8 h, deshalb wird jedes Modell täglich neu trainiert (k = 1).**

## 7. Ergebnisse auf den Validierungsblöcken

> #51, `reports/compare/validation_scores.csv` (`compare` auf Commit `90aec86`). `CHOSEN` nach der vor den Läufen festgelegten Regel (Kommentar in #51 vom 19.11.2026). 152 Validierungstage, 01.04. bis 31.08.2026: Der 24.06. fehlt für alle, weil die Strahlungsprognose fehlt. Alle Modelle auf denselben 14'592 Viertelstunden. Werte in ct/kWh, tiefer ist besser. Differenz = mittlere tägliche Pinball-Differenz mit 95-%-Intervall (Block-Bootstrap), unter 0 ist das Modell besser. B2 = `baseline_model` B2 mit `rolling_90d`. Monatswerte je Modell stehen im README des Modells.

| Modell (`CHOSEN`) | Pinball (Tagesmittel) | MAE P50 | Abdeckung 80 % | Bandbreite | Differenz zur Referenz [95 %] | Differenz zu B2 [95 %] |
|---|---|---|---|---|---|---|
| `calibrated_bands/calibrated_gbm_quantile` | **3.190** | **9.773** | 80.3 % | 27.5 | −0.111 (−0.172 bis −0.050) | −0.058 (−0.092 bis −0.025) |
| `gbm_quantile` (= `calibrated_bands/raw_gbm_quantile`) | 3.202 | **9.773** | 76.1 % | 25.6 | −0.099 (−0.152 bis −0.044) | −0.046 (−0.069 bis −0.023) |
| `calibrated_bands/calibrated_B2` | 3.238 | 9.793 | 80.5 % | 28.7 | −0.063 (−0.120 bis −0.006) | −0.010 (−0.037 bis +0.011) |
| B2 (= `calibrated_bands/raw_B2`) | 3.248 | 9.793 | 76.9 % | 26.9 | −0.053 (−0.105 bis +0.003) | Massstab |
| `regime_mixture` | 3.252 | 9.817 | 77.8 % | 27.1 | −0.049 (−0.108 bis +0.025) | +0.004 (−0.017 bis +0.032) |
| `quantile_lasso` | 3.256 | 9.779 | 76.3 % | 25.5 | −0.045 (−0.098 bis +0.019) | +0.008 (−0.018 bis +0.045) |
| `quantile_lasso_d2` | 3.264 | 9.879 | 78.2 % | 26.5 | −0.037 (−0.090 bis +0.014) | +0.016 (−0.022 bis +0.049) |
| Referenz (56 Tage) | 3.301 | 9.943 | 76.2 % | 26.2 | Massstab | +0.053 (−0.003 bis +0.105) |
| `baseline_model/B1` | 3.309 | 9.973 | 77.1 % | 27.3 | +0.008 (−0.044 bis +0.083) | +0.061 (+0.012 bis +0.112) |

- **Bestes Modell ist das kalibrierte `gbm_quantile`:** Pinball 3.190, gegenüber Referenz und B2 liegen beide Intervalle ganz unter 0. An 69.7 % der Tage besser als die Referenz.
- **Der MAE ist bei allen Modellen mit Strahlung fast gleich** (9.773 bis 9.879, Referenz 9.943). Der Gewinn liegt im Band: `gbm_quantile` gegenüber B2 hat bei P50 einen Pinball von 4.887 statt 4.897, bei P10 2.678 statt 2.766 und bei P90 2.040 statt 2.081 (`validation_model_scores.csv`).
- **Die Kalibrierung lässt P50 und damit den MAE unverändert** (9.773 und 9.793 wie die Basis). Sie hebt die Abdeckung von 76.1 % auf 80.3 % (Boosting) und von 76.9 % auf 80.5 % (B2) und senkt den Pinball um 0.011 und 0.010.

**Vorbehalt:** Die Validierungszahlen sind optimistisch, weil jedes `CHOSEN` das beste von vielen Settings auf denselben Tagen ist ([Einschränkungen](#8-einschränkungen)). Welches Modell besser ist, entscheidet der Test in #53.

## 8. Einschränkungen

- **Test zwei statt mindestens zwölf Monate:** Lago et al. empfehlen mindestens ein Jahr (Abschn. 5.1). Das Einpreissystem gilt erst seit 01.01.2026, der Test zeigt also nur den Herbst.
- **Finale statt provisorische Preise:** Um 11 Uhr wären nur provisorische Preise bekannt gewesen. Der Rückblick ist dadurch leicht zu optimistisch, die Grösse ist nicht gemessen ([Restrisiko](data.md#bekanntes-restrisiko-swissgrid-überschreibt-werte)).
- **EDA hat in den Test geschaut:** TSI und Wetter von September und Oktober (H2, H3; TSI 26.10. bis 31.10. für die Umstellung auf Viertelstunden), keine Preise. Darum „Testperiode mit Vorbehalt“.
- **Validierungszahl ist optimistisch:** Das gewählte Setting ist das beste von vielen auf denselben Tagen. Eine unverzerrte Zahl liefert erst der Test.

## 9. Herleitung

| Entscheid | Begründung | Quelle |
|---|---|---|
| Rolling Origin statt eines festen Splits | Das Training „consists only of observations that occurred prior to the observation that forms the test set“, die Genauigkeit ist das Mittel über alle Ursprünge: „The forecast accuracy is computed by averaging over the test sets.“ Die Lücke D−1 entspricht der Mehrschritt-Variante. Rolling-Origin-Auswertungen machen Out-of-sample-Tests effizienter und verlässlicher. Bei nichtstationären Reihen schätzen zeitlich geordnete Out-of-sample-Verfahren am besten | FPP3 5.10; Tashman 2000; Cerqueira et al. 2020 |
| Tägliches Neutraining | „a forecasting model should be retrained on a daily basis as new information is available“ | Lago et al. 5.6 |
| Grenze 8 h, sonst `REFIT_EVERY` | Lago et al. verlangen, die Rechenzeit auszuweisen, weil sie über die Tauglichkeit eines Modells mitentscheidet. Die 8 h selbst sind ein Projektentscheid (Rechenbudget), keine Zahl aus der Literatur | Lago et al. 5.8 |
| Auswahl nur auf der Validierung | Hyperparameter „always performed ex-ante using a validation dataset“, sonst droht „overfitting the test dataset“ | Lago et al. 5.7 |
| Test am Schluss, einmal, gesperrt | Der Test „does never overlap with the training or validation datasets“ und „is always selected as the last segment of the full dataset“ | Lago et al. 5.10 |
| Fünf Monatsblöcke, Kriterium über alle Tage | Mittel über viele Ursprünge statt eines kurzen Blocks; die Ränge je Monat zeigen, ob ein Kandidat nur in einem Block gewinnt | FPP3 5.10 |
| Fenster als Einstellung | Das Kalibrierungsfenster ist „The amount of past data employed for estimation“; nur jüngere Daten, „to avoid estimating effects that no longer play a role“. Ihr Beispiel mit täglichem Neutraining nutzt für 15.02.2017 die Daten „between 18.02.2015 and 14.02.2017“, also unser Schema mit D−1 statt D−2 | Lago et al. 3.7 |
| Mindestens 56 Tage, 28 nur für B1 und B2 | „a minimum of 8 weeks“, weil kürzere Fenster für parameterreiche Modelle zu wenig Information enthalten. Die sparsame Baseline darf darunter | Lago et al. 4.2 |
| `window_mean` als Kandidat | Kombinationen kurzer und langer Fenster können „significantly outperform predictions obtained for the best ex-post selected calibration window“; das LEAR-Ensemble ist das „arithmetic average of forecasts across four calibration window lengths“. Das Mittel über Fenster ist eine robuste Alternative dazu, ein Fenster vorab zu wählen | Lago et al. 2.1 und 4.4; Marcjasz et al. 2018 |
| Vergleich über die tägliche Differenz | Lago et al. verlangen einen statistischen Test und beschreiben den DM-Test auf der täglichen Loss-Differenz, mit Pinball als zulässigem Loss. Wir nutzen dieselbe tägliche Pinball-Differenz, aber mit einem Block-Bootstrap-Intervall statt des DM-Tests | Lago et al. 5.5 |
| Identische Viertelstunden für alle | Projektregel: Unterschiede sollen aus dem Modell kommen, nicht aus anderen bewerteten Tagen | eigener Entscheid |

### Quellen

- J. Lago, G. Marcjasz, B. De Schutter, R. Weron (2021): *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark.* Applied Energy 293, 116983. [doi:10.1016/j.apenergy.2021.116983](https://doi.org/10.1016/j.apenergy.2021.116983), [arXiv 2008.08004](https://arxiv.org/abs/2008.08004). Volltext gelesen.
- R. J. Hyndman, G. Athanasopoulos (2021): *Forecasting: Principles and Practice*, 3. Auflage, Abschn. 5.10 „Time series cross-validation“. OTexts. [otexts.com/fpp3/tscv.html](https://otexts.com/fpp3/tscv.html). Gelesen.
- L. J. Tashman (2000): *Out-of-sample tests of forecasting accuracy: an analysis and review.* International Journal of Forecasting 16(4), S. 437 bis 450. [doi:10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0). Nur Abstract gelesen.
- G. Marcjasz, T. Serafin, R. Weron (2018): *Selection of Calibration Windows for Day-Ahead Electricity Price Forecasting.* Energies 11(9), 2364. [doi:10.3390/en11092364](https://doi.org/10.3390/en11092364). Nur Abstract gelesen.
- V. Cerqueira, L. Torgo, I. Mozetič (2020): *Evaluating time series forecasting models: an empirical study on performance estimation methods.* Machine Learning 109, S. 1997 bis 2028. [doi:10.1007/s10994-020-05910-7](https://doi.org/10.1007/s10994-020-05910-7). Nur Abstract gelesen.
