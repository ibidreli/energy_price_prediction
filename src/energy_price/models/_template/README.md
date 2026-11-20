# <Modellname>

**Status:** in Arbeit / abgeschlossen, übernommen / abgeschlossen, nicht übernommen · **Verantwortlich:** <Name> · **Issue:** #<Nr> · **Stand der Zahlen:** <TT.MM.JJJJ>

<!-- Steckbrief des Modells. Nach dem Kopieren alle <...> ersetzen.
     Aufbau und Regeln: docs/models/README.md, Ablauf und Perioden: docs/training_pipeline.md
     Zahlen nur aus den Ergebnisdateien übernehmen (reports/<name>/select_scores.csv, reports/compare/), nie von Hand ändern. -->

## 1. Hypothese

<Was erwarten wir, und warum? Ein bis drei Sätze, geschrieben vor dem Auswahllauf.>

## 2. Herleitung

<Warum genau dieses Modell?>

- **Ausgangspunkt:** <EDA-Erkenntnis (H1 bis H12 in docs/eda.md) oder Überlegung aus der Domäne>
- **Erwarteter Vorteil:** <was das Modell besser kann als Referenz und Baseline B2>
- **Verworfene Alternativen:** <was nicht gewählt wurde und warum>

## 3. Vorgehen

- **Modell:** <Verfahren in einem Satz>
- **Features:** <Feature-Set B1 bis B4 oder neue Blöcke, mit Datenquelle>
- **Auswahl:** `GRID` <Settings ohne Fenster>, jedes mit den Fenstern aus `WINDOWS` (Standard `expanding`, `rolling_56d`, `rolling_90d`) und ihrem Mittel `window_mean`. Beispiel: 4 Settings × 3 Fenster + 4 Mittel = 16 Kandidaten, gewählt wird der kleinste mittlere tägliche Pinball-Loss über alle Validierungstage
- **Neutraining:** täglich, ausser `REFIT_EVERY` ist gesetzt (nur wenn `select` länger als 8 h dauert)
- **Gewählt:** <`CHOSEN` inklusive Fenster, aus `select_selection.json` übernommen, vor `make compare` festgelegt und danach nicht mehr geändert>

## 4. Ergebnisse

Validierung 01.04. bis 31.08.2026, <Anzahl> Tage. Mittlerer täglicher Pinball-Loss in ct/kWh, je Monatsblock aus `reports/compare/validation_*`.

| Modell | 2026-04 | 2026-05 | 2026-06 | 2026-07 | 2026-08 | alle | Coverage |
|---|---|---|---|---|---|---|---|
| Referenz | | | | | | | |
| Baseline B2 | | | | | | | |
| <Modell> | | | | | | | |

<Erkenntnisse: fette Untertitel, die die Aussage nennen, darunter kurze Bullets.>

## 5. Entscheid

**<Übernommen / nicht übernommen / weiterverfolgen>**, weil <Begründung>.

## 6. Belege

- **Issue und PR:** #<Nr>, PR #<Nr>
- **Code:** `src/energy_price/models/<name>/model.py`
- **Tests:** `tests/test_<name>.py`
- **Ergebnisse:** `reports/<name>/select_*` (nicht versioniert), Grafiken in `docs/figures/<name>/select_*.png`

```bash
python -m energy_price.evaluation.experiment <name> select --days 2026-04-01 2026-04-03   # kurzer Probelauf
make select MODEL=<name>    # alle Kandidaten, optional TRACK=1 und JOBS=<n>
make compare                # nachdem CHOSEN in model.py festgehalten und committet ist
```
