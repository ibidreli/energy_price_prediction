# Regime-Modell short/long

**Status:** abgeschlossen, nicht übernommen · **Verantwortlich:** Rodrigo Urech · **Issue:** #16 · **Stand der Zahlen:** 05.11.2026

## 1. Hypothese

Ein Modell, das zuerst schätzt, ob die Regelzone short oder long ist, und den Preis danach für beide Richtungen getrennt vorhersagt, schlägt die Referenz im Pinball-Loss.

## 2. Herleitung

- **Ausgangspunkt aus der EDA:**
    - Der Preis besteht aus zwei Wolken, bei short meist um 19 ct/kWh (Preis A), bei long um 6 ct/kWh mit negativen Ausreissern (Preis B)
    - Die Baseline landet mit ihrem Median dazwischen, wo kaum echte Preise liegen
    - Die Richtung hängt von Stunde und Position in der Stunde ab (H8), sie sollte sich also teilweise schätzen lassen
- **Aus dem Review:** Der Sprung des Preises bei unterdeckter Regelzone wurde als zentraler Punkt genannt
- **Zusatznutzen:** Die geschätzte Wahrscheinlichkeit für short zeigt, ob eine Bilanzgruppe bei einer Abweichung zahlt oder vergütet wird
- **Verworfene Alternative:** ein einzelnes Modell ohne Trennung, wie die Baseline. Es mittelt die beiden Situationen

## 3. Vorgehen

- **Modell in drei Schritten:**
    1. **Richtung:** logistische Regression für P(short) pro Viertelstunde. Das Label `short` (TSI < 0) ist wie der Preis nur Trainingsziel bis D−2, nie Feature
    2. **Preis je Richtung:** lineare Quantilregression wie in der Baseline, getrennt auf den short- und den long-Viertelstunden, für 13 Quantile von 2 % bis 98 %
    3. **Mischung:** $F(x) = p \, F_{\text{short}}(x) + (1-p) \, F_{\text{long}}(x)$. P10, P50 und P90 sind die Werte, bei denen $F$ 0.1, 0.5 und 0.9 erreicht
- **Features:** B1 (Kalender) oder B2 (Kalender plus Sonne) für beide Teile, wie die Baseline. Zusätzlich liefert das Modell `p_short`
- **Tuning:** April und Mai

| Variante | Pinball-Loss | gegenüber Referenz |
|---|---|---|
| Referenz | 3.449 | – |
| B1, alle Daten ab 01.01. | 3.496 | −1.4 % |
| B1, letzte 56 Tage | 3.732 | −8.2 % |
| **B2, alle Daten ab 01.01.** | **3.323** | **+3.7 %** |
| B2, letzte 56 Tage | 3.664 | −6.3 % |

- **Gewählt:** B2 mit allen Daten ab 01.01., vor dem Test als `CHOSEN` (Variante `regime_mixture`) festgelegt und danach nicht mehr verändert

## 4. Erkenntnisse

**Die Trennung in zwei Richtungen ist richtig**
- Mit bekannter Richtung sinkt der Pinball-Loss um 39 % unter die Referenz, an 99 % der Tage
- Die Preisteile pro Richtung sind brauchbar, der Sprung zwischen Preis A und Preis B ist der grösste Teil des Fehlers

**Die Richtung ist am Vortag nur schwach vorhersagbar**
- Der Brier Score ist nur 4 % besser als die naive Prognose, die AUC liegt bei 0.62
- Der Anteil short pro Viertelstunde erreicht fast dasselbe (Pinball-Loss 3.236, Brier 0.240, AUC 0.61). Das Signal am Vortag ist fast nur Uhrzeit und Position in der Stunde (H8)
- Das bestätigt H2: Die Richtung von D−2 stimmt nur in 54 % mit dem Liefertag überein
- Von der möglichen Verbesserung gegenüber einem konstanten $p$ nutzt das Modell nur $(3.321 - 3.227)/(3.321 - 1.954) \approx 7\,\%$

**Das Muster in der Stunde ist echt, aber zu schwach**
- Um 17 Uhr steigt P(short) von 0.21 (:00) auf 0.85 (:45), beobachtet 0.34 und 0.74 (H8)
- Über den Tag gemittelt liegt P(short) nahe beim Grundanteil von 0.55. Die Mischung entspricht dann fast der unbedingten Verteilung, und die Trennung bringt wenig

**P(short) ist in der Mitte gut kalibriert, an den Rändern zu sicher**
- Bei einer Prognose um 0.17 war die Regelzone in 32 % der Fälle short, bei 0.83 in 75 %
- Im Mittel liegt die Prognose mit 0.55 leicht über dem beobachteten Anteil von 0.53

**Tageszeit und Richtung**
- Von 9 bis 15 Uhr Pinball-Loss 4.61 gegenüber 4.84 der Referenz, übrige Stunden 2.66 gegenüber 2.58. Gleiches Muster wie die Baseline, weil der Preisteil dieselben Features nutzt
- Das Modell verliert vor allem bei short, wo Preis A gilt
- Wie bei der Baseline war das Tuning (+3.7 %) zu optimistisch

## 5. Outcome

Test Juni bis August, 91 Tage. Preise, Fehler und Breiten in ct/kWh.

| Modell | Pinball-Loss | MAE P50 | Coverage | Intervallbreite |
|---|---|---|---|---|
| Referenz | **3.20** | 9.92 | **75.2 %** | 26.3 |
| Baseline B2 | 3.23 | 9.81 | 73.6 % | 25.5 |
| Regime-Modell | 3.23 | **9.80** | 74.9 % | 25.9 |

- Gegenüber der Referenz −0.8 %, an 54 % der Tage besser
- Coverage leicht besser als die Baseline, aber weiter unter 80 %

| Klassifikator P(short) im Test | Wert |
|---|---|
| Brier Score Modell | 0.239 |
| Brier Score naive Prognose (Anteil short im Training, 0.556) | 0.250 |
| AUC | 0.62 |
| Treffer bei Schwelle 0.5 | 59 % |

**Wie viel ist die Richtung wert?** `diagnostics.py` behält die wöchentlichen Fits der Preisteile unverändert und tauscht nur $p$ in der Mischung aus. Es wird kein Modell ausgewählt, deshalb darf die Testperiode verwendet werden.

| $p$ | Bedeutung | Pinball-Loss | MAE P50 | Brier | AUC |
|---|---|---|---|---|---|
| konstant | Anteil short im Training | 3.321 | 10.30 | 0.249 | 0.50 |
| pro Viertelstunde | Anteil short je Uhrzeit und Position im Training | 3.236 | 9.86 | 0.240 | 0.61 |
| Modell | logistische Regression (`CHOSEN`) | 3.227 | 9.80 | 0.239 | 0.62 |
| bekannte Richtung | tatsächliche Richtung des Liefertags, 0 oder 1 | **1.954** | 5.57 | – | – |
| Referenz | | 3.202 | 9.92 | – | – |

Die bekannte Richtung ist am Vortag um 11:00 nicht verfügbar. Sie ist eine Schranke, besser kann dieses Modell mit denselben Preisteilen nicht werden.

**Nach tatsächlicher Richtung** (`reports/regime_mixture/test_breakdowns.csv`):

| Pinball-Loss | long | short |
|---|---|---|
| Referenz | 3.27 | 3.14 |
| Baseline B2 | 3.14 | 3.30 |
| Regime-Modell | 3.19 | 3.26 |

![Pinball-Loss pro Stunde](../../../../docs/figures/regime_mixture/test_pinball_by_hour.png)

![Coverage pro Monat](../../../../docs/figures/regime_mixture/test_coverage_by_month.png)

![Kalibrierung von P(short)](../../../../docs/figures/regime_mixture/test_direction_reliability.png)

## 6. Entscheid

**Nicht übernommen**, weil das Gradient Boosting (`gbm_quantile`, Pinball-Loss 3.16) besser ist. Der Ansatz bleibt als dokumentiertes negatives Ergebnis. Der Sprung zwischen Preis A und Preis B ist real und mit bekannter Richtung 39 % des Pinball-Loss wert. Am Vortag um 11:00 ist aber zu wenig über die Richtung bekannt, um ihn auszunutzen. Eine bessere Richtungsprognose bräuchte Informationen, die erst kurz vor der Lieferung entstehen (Intraday-Handel, aktuelle Systembilanz). Für eine Prognose am Vortag ist das kein Weg, für eine kurzfristige Prognose wäre es der wichtigste Ansatz.

## 7. Belege

- **Issue und PR:** #16, PR #29
- **Code:** `src/energy_price/models/regime_mixture/model.py`, `src/energy_price/models/regime_mixture/diagnostics.py`, Label in `src/energy_price/features/dataset.py`
- **Tests:** `tests/test_regime_mixture.py`, Label in `tests/test_features_dataset.py`
- **Ergebnisse:** `reports/regime_mixture/` (nicht versioniert), Grafiken in `docs/figures/regime_mixture/`

```bash
make backtest MODEL=regime_mixture STEP=tune   # Varianten auf April und Mai
make backtest MODEL=regime_mixture STEP=test   # CHOSEN auf Juni bis August, mit Grafiken
python -m energy_price.models.regime_mixture.diagnostics   # Wert der Richtung, Kalibrierung von P(short)
```

Ein Lauf dauert auf zwei Kernen etwa 20 Minuten (26 Quantilregressionen pro wöchentlichem Fit), die Diagnose ebenso.
