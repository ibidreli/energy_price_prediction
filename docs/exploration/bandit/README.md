# Bandit über die Modelle: explorative Architekturprüfung

> **Status: getestet und verworfen (#21).** Der Bandit erreichte 3.207 gegenüber 3.163 von `gbm_quantile` (Pinball, Juni bis August, alte Pipeline) und wählte fast zufällig.
> Der Code bleibt lauffähig, damit der Befund nachvollziehbar ist; er ist kein Modellkandidat.
> Neue Pipeline: `make bandit-exploration STEP=fit-spans|evaluate-spans`, nur auf Validierungstagen (April und Mai, Juni bis August).

**Status:** abgeschlossen, nicht weiterverfolgt, kein Modellkandidat nach Standardablauf · **Verantwortlich:** Luca Manna · **Issue:** [#21](https://github.com/ibidreli/energy_price_prediction/issues/21) · **Stand der Zahlen:** 05.11.2026

Lesbarer Einstieg: [`notebooks/03_bandit_architektur.ipynb`](../../../notebooks/03_bandit_architektur.ipynb). Die Testperiode (September und Oktober) wurde nie verwendet.

## Ablauf in der neuen Pipeline

Die Prüfung teilt die Validierung (01.04. bis 31.08.2026) selbst auf und verweigert jeden Tag ausserhalb davon (`check_validation`). Konzept der Pipeline: [`training_pipeline.md`](../../training_pipeline.md).

```mermaid
gantt
    title Bandit-Prüfung innerhalb der Validierung 2026
    dateFormat YYYY-MM-DD
    todayMarker off
    axisFormat %b
    tickInterval 1month
    section Validierung
    fit-spans, Einstellungen wählen      :active, 2026-04-01, 2026-06-01
    evaluate-spans, CHOSEN auswerten     :active, 2026-06-01, 2026-09-01
    section Test
    nie verwendet                        :crit, 2026-09-01, 2026-11-01
```

| Schritt | Tage | schreibt nach `results/` |
|---|---|---|
| `make bandit-exploration STEP=fit-spans` | 01.04. bis 31.05. | `fit_selection.csv` (Einstellung je Strategie) |
| `make bandit-exploration STEP=evaluate-spans` | 01.06. bis 31.08. | `evaluate_results.csv`, `evaluate_persistence.csv`, `evaluate_bandit_shares.csv`, `evaluate_strategies.png` |

Die Arme (Referenz, B1, B2, `gbm_quantile`) entstehen im täglichen Walk-forward mit dem `CHOSEN` der Modelle. Mit der neuen Pipeline ist die Prüfung noch nicht gerechnet, die Zahlen unten stammen aus der alten.

| | Standardablauf | Diese Prüfung |
|---|---|---|
| Start | `make select MODEL=<name>` | `make bandit-exploration STEP=fit-spans\|evaluate-spans` |
| Reihenfolge | Hypothese vor dem Test | Hypothese nach dem Vorlauf |
| Testperiode | einmal am Schluss (#53) | keine, nur Validierungstage |

## Entscheid

**Nicht weiterverfolgt**, weil Thompson Sampling als Architektur nicht zu diesem Problem passt und die laufende Gewichtung keinen gesicherten Vorteil bringt, der die zusätzliche Komplexität rechtfertigt.

## Historie (alte Pipeline)

> Stand 05.11.2026. Alte Pipeline: Auswahl auf April und Mai, Test auf Juni bis August, Modelle wöchentlich neu trainiert. Die Dateien `results/tune_*` und `results/test_*` gehören dazu. In der neuen Pipeline heissen die Schritte `fit-spans` und `evaluate-spans`, die festen Gewinner `*_from_fit`.

### 1. Hypothese

Ein Bandit kann pro Viertelstunde laufend das beste bestehende Modell wählen und so `gbm_quantile` im Pinball-Loss schlagen (3.1633, Juni bis August).

**Kriterium:** Pinball-Loss tiefer als `gbm_quantile`, und das 95-%-Intervall der täglichen Differenz liegt ganz unter 0 (Block-Bootstrap über Tage, Blöcke von 7 Tagen, 5000 Wiederholungen).

### 2. Herleitung

- **Herkunft der Idee:** ein Vortrag von CH Media. Dort verteilt ein Bandit mit Thompson Sampling den Traffic live auf mehrere Empfehlungs-Blends, weil sich die Nachfrage mit der Nachrichtenlage ändert
- **Übertragung:** Kein Modell ist zu jeder Tageszeit das beste. Mittags hilft die Sonne, abends ist oft die Referenz besser (`gbm_quantile`, `baseline_model`)
- **Grenzen der Aussage:** Diese Prüfung wich vom damaligen Standardablauf ab: Die Modelle wurden wöchentlich neu trainiert, der Bandit entschied täglich mit Fehlern bis D−2, und die Hypothese entstand erst nach dem Vorlauf. Der Vergleich mit den regulären Modellen war deshalb nicht gleichwertig

### 3. Vorgehen

1. Der [Vorlauf](vorlauf/) mit fünf Modellen, inklusive Regime-Modell, wertete die Testmonate am 05.11.2026 aus
2. Erst danach wurden Frage und Kriterium festgehalten, und danach entstand diese versionierte Fassung mit vier Modellen. Der Test ist deshalb **nicht blind**, die Einstellungen stammen aber nur aus April und Mai

Die Arme sind die Prognosen von Referenz, B1, B2 und Gradient Boosting aus dem gemeinsamen Backtest. Jede Entscheidung für den Liefertag D nutzt nur Fehler bis D−2.

| Strategie | Was sie tut |
|---|---|
| `thompson_bandit` | Der eigentliche Bandit: pro Viertelstunde ein Modell per Thompson Sampling, lernt nur vom gewählten Modell, jedes Modell behält mindestens 5 % |
| `thompson_bandit_no_floor` | Gleich, ohne Mindestanteil |
| `thompson_full_info` | Thompson Sampling, sieht aber die Fehler aller Modelle |
| `follow_leader_slot` | Pro Viertelstunde das Modell mit dem tiefsten bisherigen Fehler, ohne Zufall |
| `hedge_global`, `hedge_hour`, `hedge_slot` | Gewichtetes Mittel aller Modelle, Gewichte pro Quantil und pro Tag, Stunde oder Viertelstunde, alte Fehler verblassen |
| `slot_winner_from_tune`, `hour_winner_from_tune` | Fester Gewinner pro Viertelstunde oder Stunde aus April und Mai |
| `mean_all` | Einfaches Mittel aller Modelle |

Einstellungen: [`tune_selection.csv`](results/tune_selection.csv). Thompson wird über 200 Seeds gemittelt.

### 4. Erkenntnisse

**Der Bandit lernt praktisch nichts**
- Die Anteile bleiben nahe bei je 25 %, er wählt also fast zufällig
- Sein Intervall liegt ganz über 0, er ist klar schlechter als das Gradient Boosting und schlägt auch die Referenz nicht

**Das Problem des Bandits haben wir nicht**
- Ein Bandit ist nötig, wenn nur die Rückmeldung des gewählten Arms sichtbar ist
- Hier prognostizieren alle Modelle, und alle Fehler werden bekannt („Prediction with Expert Advice“). Exploration kostet Genauigkeit und bringt keine zusätzliche Information

**Zu wenig Rückmeldung pro Entscheid**
- CH Media verteilt alle 10 Minuten neu
- Hier gibt es pro Viertelstunde eine Beobachtung am Tag, erst nach rund zwei Tagen bekannt, bei stark streuenden Verlusten

**Das „perfekte Modell pro Slot“ ist überwiegend Rauschen**
- Die −14.55 % setzen voraus, dass der Preis schon bekannt ist
- Gewinner pro Viertelstunde sind kaum stabil, und aus April und Mai übernommen sind sie schlechter als das Einzelmodell

**Die laufende Gewichtung ist kein gesicherter Gewinn**
- Alle Intervalle schliessen 0 ein
- Die beste Variante ist instabil. Im Vorlauf lag die Gewichtung pro Stunde vorn (−0.46 %), hier die pro Viertelstunde (−0.45 %)
- Der Bandit war auch im Vorlauf klar schlechter (+1.5 %)

**Die Coverage bleibt zu tief**
- Auch die Gewichtungen erreichen nur rund 76 % statt 80 %. Dafür sind die kalibrierten Bänder vorgesehen (#19)

**Grenzen:** nur eine Saison und ein Dutzend verglichene Strategien ohne Korrektur für Mehrfachtests. Ein kleiner Nutzen laufender Gewichte in einer anderen Saison oder mit unterschiedlicheren Modellen ist damit nicht ausgeschlossen.

### 5. Outcome

Test Juni bis August, 91 Tage. Differenz zu `gbm_quantile`, das auch auf April und Mai das beste Einzelmodell war. Alle Werte: [`test_results.csv`](results/test_results.csv).

| Strategie | Pinball-Loss | MAE P50 | Differenz | 95-%-Intervall (ct/kWh) | Coverage |
|---|---|---|---|---|---|
| Thompson-Bandit ohne Minimum | 3.207 | 9.851 | +1.37 % | +0.030 bis +0.059 | 74.0 % |
| **Thompson-Bandit, 5 % Minimum** | **3.207** | **9.855** | **+1.37 %** | **+0.030 bis +0.059** | **74.0 %** |
| Referenz | 3.202 | 9.917 | +1.22 % | −0.011 bis +0.095 | 75.2 % |
| Thompson, alle Fehler sichtbar | 3.201 | 9.838 | +1.18 % | +0.024 bis +0.052 | 74.1 % |
| Fester Gewinner pro Stunde aus April/Mai | 3.195 | 9.805 | +0.99 % | +0.019 bis +0.053 | 73.2 % |
| Fester Gewinner pro Viertelstunde aus April/Mai | 3.180 | 9.787 | +0.52 % | +0.005 bis +0.031 | 73.9 % |
| Mittel aller Modelle | 3.167 | 9.787 | +0.11 % | −0.011 bis +0.020 | 75.7 % |
| **Gradient Boosting allein** | **3.163** | **9.792** | | | **74.1 %** |
| Gewichte für den ganzen Tag | 3.159 | 9.782 | −0.13 % | −0.019 bis +0.012 | 76.2 % |
| Gewichte pro Stunde | 3.155 | 9.772 | −0.27 % | −0.025 bis +0.007 | 75.9 % |
| Laufend bester pro Viertelstunde | 3.151 | 9.770 | −0.38 % | −0.050 bis +0.017 | 75.1 % |
| Gewichte pro Viertelstunde | 3.149 | 9.760 | −0.45 % | −0.033 bis +0.004 | 76.0 % |
| *Gewinner pro Viertelstunde im Nachhinein* | 3.109 | 9.708 | −1.72 % | nicht erreichbar | 75.0 % |
| *Bestes Modell für jede einzelne Prognose* | 2.703 | 8.207 | −14.55 % | nicht erreichbar | 80.7 % |

![Differenz zum Gradient Boosting pro Strategie](results/test_strategies.png)

Stabilität der Gewinner ([`test_persistence.csv`](results/test_persistence.csv)) und Anteile im Bandit ([`test_bandit_shares.csv`](results/test_bandit_shares.csv)):

| Prüfung | Wert |
|---|---|
| Anteile im Bandit im Test | Referenz 24.7 %, B1 23.5 %, B2 25.8 %, Boosting 26.1 % (Gleichverteilung wäre 25 %) |
| Gleicher Gewinner pro Viertelstunde, April/Mai gegen Test | 32 % (Zufall wäre etwa 25 %) |
| Gleicher Gewinner, ungerade gegen gerade Testtage | 40 % |
| Gewinner auf den jeweils anderen Testtagen gewählt | nur −0.27 % statt −1.72 % im Nachhinein |
| Tage für 80 % Power beim Vergleich Boosting gegen Referenz in einer Viertelstunde | Median 651, vorhanden sind 91 |

## Belege

- **Issue und PR:** #21, PR #22
- **Code:** [`src/energy_price/exploration/bandit/`](../../../src/energy_price/exploration/bandit/). Ein Schritt auf der Testperiode existiert nicht
- **Tests:** `tests/test_bandit_exploration.py`. Sie prüfen unter anderem, dass keine Entscheidung Preise nach D−2 sieht und der Bandit nur vom gewählten Modell lernt
- **Kontrolle (alte Pipeline):** Die Arme reproduzierten die dokumentierten Test-Scores der Modelle (Boosting 3.1633, Referenz 3.2018, B2 3.2254)
- **Vorlauf:** [`vorlauf/`](vorlauf/) enthält die Skripte und die Rohausgabe unverändert. Die Skripte brauchen die Prognosen aller fünf Modelle, das Regime-Modell ist seit #29 in `main`

```bash
make bandit-exploration STEP=fit-spans        # Einstellungen auf April und Mai wählen
make bandit-exploration STEP=evaluate-spans   # CHOSEN auf Juni bis August, schreibt results/
pytest tests/test_bandit_exploration.py
```
