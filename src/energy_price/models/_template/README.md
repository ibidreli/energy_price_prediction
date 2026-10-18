# <Modellname>

<!-- Steckbrief des Modells, gleiche Felder wie die EXP-Issue-Vorlage. Nach dem Kopieren alle <...> ersetzen.
     Anleitung: docs/models/README.md -->

**Issue:** #<n> · **Status:** in Arbeit / abgeschlossen / verworfen

## Hypothese

<Was wird getestet, und welche Verbesserung gegenüber Referenz und Baseline B2 wird erwartet? Warum?>

## Modell und Features

<Verfahren in einem Satz, Feature-Set (B1, B2 oder neue Blöcke) und Datenquellen.>

## Ergebnis (Test Juni bis August)

| Modell | MAE P50 | Pinball Loss | Coverage |
|---|---|---|---|
| Referenz | 9.92 | 3.20 | 75 % |
| Baseline B2 | 9.81 | 3.23 | 74 % |
| <Modell> | | | |

**Entscheid:** <übernommen / verworfen / weiterverfolgen mit ...>

## Ausführen

```bash
make backtest MODEL=<name> STEP=tune
make backtest MODEL=<name> STEP=test
```
