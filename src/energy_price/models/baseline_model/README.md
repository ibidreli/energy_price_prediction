# Baseline: lineare Quantilregression

**Status:** Massstab für alle Modelle, Auswahl in der neuen Pipeline offen (#51) · **Verantwortlich:** Rodrigo Urech · **Issue:** #6 · **Ablauf:** [`docs/training_pipeline.md`](../../../../docs/training_pipeline.md) · **Ausführliche Doku:** [`docs/models/baseline_model.md`](../../../../docs/models/baseline_model.md)

## 1. Hypothese

Kalendermerkmale und die Sonnenprognose bringen mehr als der Blick in die Vergangenheit, also die Referenz aus den Preisen derselben Uhrzeit der letzten 56 Tage.

## 2. Herleitung

- **Ausgangspunkt aus der EDA:** Die Systembilanz kippt innerhalb jeder Stunde, die Position der Viertelstunde ist deshalb ein starkes Kalendermerkmal (H1, H8). Die Strahlungsprognose hängt mit dem Preisniveau zusammen, vor allem mittags (H3, H7). Feiertage drücken den Preis, gewichtet nach Solarleistung im Kanton (H6)
- **Warum Quantilregression:** Der Preis hat zwei typische Bereiche, short um 19 ct/kWh und long um 6 ct/kWh. Ein normales lineares Modell sagt einen Mittelwert dazwischen voraus, wo kaum echte Preise liegen. Die Quantilregression lernt für P10, P50 und P90 je eine eigene Gleichung
- **Warum linear:** erklärbar, jeder Koeffizient ist eine Verschiebung in ct/kWh. Lineare Modelle sind das übliche Vergleichsmodell für Strompreise (Lago et al. 2021)
- **Erwartung aus der Literatur:** Ausgleichspreise sind am Vortag kaum vorhersagbar (Klæboe et al. 2015), ein grosser Vorsprung ist deshalb nicht zu erwarten

## 3. Vorgehen

- **Modell:** lineare Quantilregression (`models/shared/linear.py`) mit L1-Strafe `alpha`, eine Gleichung je Quantil
- **Features:** **B1** Uhrzeit, Position der Viertelstunde in der Stunde, Wochenende, Feiertage gewichtet nach PV-Leistung. **B2** zusätzlich PV-gewichtete Strahlung und ihre Änderung in der Stunde (ECMWF-Lauf D−2 18 UTC)
- **Auswahl:** `GRID` = B1 und B2 × `alpha` 0, 0.01, 0.1. Als einziges Modell auch mit `rolling_28d`, weil es sparsam ist ([Herleitung](../../../../docs/training_pipeline.md#9-herleitung)): 6 Settings × 4 Fenster + 6 `window_mean` = **30 Kandidaten**, jeder mit 153 täglichen Ursprüngen
- **Neutraining:** täglich (`REFIT_EVERY` nicht gesetzt)
- **Gewählt:** `CHOSEN` hält die Varianten `B1` und `B2` fest, je der Kandidat des Feature-Sets mit dem kleinsten Wert in `select_scores.csv` (Block `all`). `B2` ist in `select` und `compare` aller anderen Modelle der Massstab. Bis #51 provisorisch: `expanding`, `alpha` 0

## 4. Ergebnisse (neue Pipeline)

Folgt nach den Auswahlläufen in #51.

## 5. Belege

- **Code:** `src/energy_price/models/baseline_model/model.py`, `src/energy_price/models/shared/linear.py`
- **Tests:** `tests/test_baseline_model.py`
- **Ergebnisse:** `reports/baseline_model/select_*` (nicht versioniert): `scores.csv` je Kandidat und Monatsblock mit Rang, `selection.json` mit `dropped_days`, `runtime.csv`, `forecasts.parquet`, `coefficients.csv`, `breakdowns.csv`. Grafiken in `docs/figures/baseline_model/select_*.png`
- **Literatur:** [Lago et al. 2021](https://arxiv.org/abs/2008.08004), [Klæboe et al. 2015](https://ideas.repec.org/p/gwc/wpaper/2013-006.html)

```bash
python -m energy_price.evaluation.experiment baseline_model select --days 2026-04-01 2026-04-03   # Probelauf
make select MODEL=baseline_model TRACK=1 JOBS=4   # 30 Kandidaten auf allen Validierungstagen
# CHOSEN in model.py festhalten und committen, dann:
make compare
```

## Historie (alte Pipeline)

> Gewählt auf April und Mai, verglichen auf Juni bis August, wöchentlich neu trainiert, Fenster `expanding` oder 56 Tage. Juni bis August waren nicht mehr blind (#21). Ersetzt durch #36. Stand der Zahlen 20.10.2026, Issue und PR: #6, PR #12, Struktur seit #13 (PR #15).

**Tuning:** April und Mai, Varianten B1 und B2. **Gewählt:** B2. Im Tuning noch 4 % besser als die Referenz, danach nichts mehr angepasst.

| Test Juni bis August, 91 Tage (ohne 24.06.), ct/kWh | Pinball-Loss | MAE P50 | Coverage | Intervallbreite |
|---|---|---|---|---|
| Referenz | **3.20** | 9.92 | **75.2 %** | 26.3 |
| B1 | 3.25 | 9.94 | 73 % | – |
| B2 | 3.23 | **9.81** | 73.6 % | 25.5 |

- **Nicht besser als der Blick in die Vergangenheit:** Pinball-Loss 3.23 statt 3.20, Unterschied unter 1 %. Nur an 53 % der Tage besser als die Referenz, kaum mehr als ein Münzwurf
- **Die Sonne hilft punktuell:** Von 9 bis 16 Uhr und bei negativen Preisen ist B2 besser als die Referenz, abends schlechter. Abends fehlt Information, etwa eine Lastprognose. B2 gegenüber B1 nur leicht besser (3.23 statt 3.25)
- **Die Bänder sind zu schmal:** Coverage 74 % statt 80 %, etwa jede vierte statt jede fünfte Viertelstunde liegt ausserhalb. Der Nutzen liegt im Band P10 bis P90, das aber noch kalibriert werden muss
- **Entscheid:** Bleibt als Massstab, weil das Modell gleich gut wie die Referenz, reproduzierbar, gegen Data Leakage getestet und erklärbar ist. Ein späteres Modell muss einen Pinball-Loss unter 3.20 erreichen

![Pinball-Loss pro Stunde](../../../../docs/figures/baseline_model/test_pinball_by_hour.png)

![Coverage pro Monat](../../../../docs/figures/baseline_model/test_coverage_by_month.png)
