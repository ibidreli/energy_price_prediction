# <Modellname>

**Status:** in Arbeit / abgeschlossen, übernommen / abgeschlossen, nicht übernommen · **Verantwortlich:** <Name> · **Issue:** #<Nr> · **Stand der Zahlen:** <TT.MM.JJJJ>

<!-- Steckbrief des Modells. Nach dem Kopieren alle <...> ersetzen.
     Aufbau und Regeln: docs/models/README.md
     Zahlen nur aus den Ergebnisdateien übernehmen (reports/<name>/test_scores.csv), nie von Hand ändern. -->

## 1. Hypothese

<Was erwarten wir, und warum? Ein bis drei Sätze, geschrieben vor dem Tuning.>

## 2. Herleitung

<Warum genau dieses Modell?>

- **Ausgangspunkt:** <EDA-Erkenntnis (H1 bis H12 in docs/eda.md) oder Überlegung aus der Domäne>
- **Erwarteter Vorteil:** <was das Modell besser kann als Referenz und Baseline B2>
- **Verworfene Alternativen:** <was nicht gewählt wurde und warum>

## 3. Vorgehen

- **Modell:** <Verfahren in einem Satz>
- **Features:** <Feature-Set B1 bis B6 oder neue Blöcke, mit Datenquelle>
- **Tuning:** <Grid, Trainingsfenster, Zeitraum>
- **Gewählt:** <Setting `CHOSEN`, vor dem Test festgelegt und danach nicht mehr geändert>

## 4. Erkenntnisse

<Was haben wir gelernt, mit Bezug zur Domäne? Fette Untertitel, die die Aussage nennen, darunter kurze Bullets.>

## 5. Outcome

Test <Zeitraum>, <Anzahl> Tage. Preise, Fehler und Breiten in ct/kWh.

| Modell | Pinball-Loss | MAE P50 | Coverage | Intervallbreite |
|---|---|---|---|---|
| Referenz | | | | |
| Baseline B2 | | | | |
| <Modell> | | | | |

## 6. Entscheid

**<Übernommen / nicht übernommen / weiterverfolgen>**, weil <Begründung>.

## 7. Belege

- **Issue und PR:** #<Nr>, PR #<Nr>
- **Code:** `src/energy_price/models/<name>/model.py`
- **Tests:** `tests/test_<name>.py`
- **Ergebnisse:** `reports/<name>/` (nicht versioniert), Grafiken in `docs/figures/<name>/`

```bash
make backtest MODEL=<name> STEP=tune
make backtest MODEL=<name> STEP=test
```
