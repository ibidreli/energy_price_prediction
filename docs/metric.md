# Metrik

> Gilt für jede Day-Ahead-Prognose (D−1 11:00, alle Viertelstunden von D) des AEP `aep_ct_kwh` in ct/kWh. Code: [`src/energy_price/metrics.py`](../src/energy_price/metrics.py). Issue #7.

## Hauptmetrik: MAE

$$
\mathrm{MAE} = \frac{1}{N}\sum_{t=1}^{N} \lvert y_t - \hat{y}_t \rvert
$$

- $y_t$: AEP der Viertelstunde $t$. $\hat{y}_t$: Punktprognose, das ist das Quantil 0.5.
- $N$: alle Viertelstunden aller Testtage im rollierenden Backtest, jede gleich gewichtet.
- **Baseline:** Median des AEP derselben Viertelstunde an D−2 bis D−8. D−2 ist der letzte Tag, der um 11:00 vollständig bekannt ist.
- **Modelle werden nach MAE in ct/kWh gerankt.** Ein Modell muss den MAE der Baseline unterbieten.

## Unsicherheit

Jedes Modell liefert die Quantile $q_{0.1}$, $q_{0.5}$, $q_{0.9}$.

$$
\mathrm{Pinball} = \frac{1}{3N}\sum_{\tau \in \{0.1,\,0.5,\,0.9\}} \sum_{t=1}^{N} \max\bigl(\tau\,(y_t - q_{\tau,t}),\ (\tau - 1)\,(y_t - q_{\tau,t})\bigr)
$$


| Symbol       | Bedeutung                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| $\tau$       | Quantilniveau: 0.1, 0.5 oder 0.9. Bei $\tau = 0.9$ sollen 90 % der Preise unter der Prognose liegen. |
| $q_{\tau,t}$ | Prognose des Modells für das Quantil $\tau$ in Viertelstunde $t$, in ct/kWh                          |
| $y_t$        | tatsächlicher AEP in Viertelstunde $t$, in ct/kWh                                                    |
| $N$          | Anzahl Viertelstunden im Test                                                                        |
| $3$          | Anzahl Quantilniveaus, über die gemittelt wird                                                       |


**So funktioniert er:** Der Pinball-Loss ist ein absoluter Fehler, der je nach Seite anders gewichtet wird. Liegt der Preis über der Prognose, kostet jede ct/kWh Abstand $\tau$. Liegt er darunter, kostet sie $1 - \tau$. Bei $\tau = 0.9$ ist eine zu tiefe Prognose also neunmal teurer als eine zu hohe. Den kleinsten Loss erreicht ein Modell, wenn es das echte Quantil trifft. Ein zu breites Intervall wird ebenso bestraft wie ein zu schmales. Bei $\tau = 0.5$ ist der Pinball-Loss genau der halbe MAE. Video: [Quantile Loss (Meerkat Statistics)](https://www.youtube.com/watch?v=f_XfuS8NwQ4).


| Kennzahl     | Definition                                    | Ziel                                |
| ------------ | --------------------------------------------- | ----------------------------------- |
| Pinball-Loss | Formel oben                                   | tiefer als Baseline                 |
| Abdeckung    | Anteil der $y_t$ in $[q_{0.1,t},\ q_{0.9,t}]$ | 80 %                                |
| Breite       | Mittel von $q_{0.9,t} - q_{0.1,t}$            | möglichst schmal bei 80 % Abdeckung |


Zusätzlich wird der MAE getrennt für Viertelstunden mit negativem Preis berichtet.