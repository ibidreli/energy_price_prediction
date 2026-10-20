# Baseline: lineare Quantilregression

**Status:** abgeschlossen, Massstab für alle weiteren Modelle · **Doku:** [`docs/models/baseline_model.md`](../../../../docs/models/baseline_model.md)

## Hypothese

Kalendermerkmale und die Sonnenprognose bringen mehr als der Blick in die Vergangenheit, also die Referenz aus den Preisen derselben Uhrzeit der letzten 56 Tage.

## Modell und Features

Lineare Quantilregression (`models/shared/linear.py`), eine Gleichung je Quantil P10, P50 und P90.

- **B1:** Uhrzeit, Position der Viertelstunde in der Stunde, Wochenende, Feiertage gewichtet nach PV-Leistung
- **B2:** B1 plus PV-gewichtete Strahlung und ihre Änderung in der Stunde (ECMWF-Lauf D−2 18 UTC)

## Ergebnis (Test Juni bis August)

| Modell | MAE P50 | Pinball Loss | Coverage |
|---|---|---|---|
| Referenz | 9.92 | 3.20 | 75 % |
| B1 | 9.94 | 3.25 | 73 % |
| B2 | 9.81 | 3.23 | 74 % |

**Entscheid:** Gleich gut wie die Referenz, nicht besser. Die Sonne hilft mittags und bei negativen Preisen, die Bänder sind zu schmal. Die Baseline bleibt als erklärbarer Massstab.

## Ausführen

```bash
make baseline-tune    # = make backtest MODEL=baseline_model STEP=tune
make baseline         # = make backtest MODEL=baseline_model STEP=test
```
