# Baseline: lineare Quantilregression

> Issue [#6](https://github.com/ibidreli/energy_price_prediction/issues/6) · Stand 13.10.2026 · Code in `src/energy_price/model/`

## Kurzfassung

- **Was:** Am Vortag um 11:00 sagen wir für jede Viertelstunde des nächsten Tages drei Preise voraus: eine untere Grenze (P10), einen mittleren Wert (P50) und eine obere Grenze (P90).
- **Modell:** lineare Quantilregression, ein einfaches und erklärbares Modell.
- **Vergleich:** eine Referenz, die einfach die Preise derselben Uhrzeit der letzten 56 Tage nimmt.
- **Ergebnis:** Das Modell ist etwa gleich gut wie die Referenz, aber nicht besser.

## 1. Ausführen

```bash
make test              # Tests, unter anderem gegen Data Leakage
make baseline-tune     # Einstellungen auf April und Mai wählen
make baseline          # Backtest auf Juni bis August, mit Grafiken
```

Ergebnisse landen in `reports/baseline/` (nicht im Git), Grafiken in `docs/figures/`. Unter Windows zuerst die Umgebung aktivieren ([README](../README.md#windows--powershell)).

## 2. Änderungen im Repo

**Neu** (Code in `src/energy_price/model/`):

| Datei | Inhalt | Warum |
|---|---|---|
| `dataset.py` | Preis und Features pro Viertelstunde | gemeinsame Grundlage; nur Daten, die um 11:00 bekannt sind |
| `models.py` | Referenz und lineare Quantilregression | Massstab und Baseline an einem Ort |
| `scores.py` | Pinball Loss, MAE, Coverage, Skill Score | Bewertung vorab festgelegt, für alle Modelle gleich |
| `backtest.py` | Backtest, Grafiken, W&B-Logging | realistischer Test ohne Data Leakage |
| `tests/test_model_*.py` | 31 Tests | prüfen automatisch, dass keine späteren Daten ins Modell gelangen |
| `docs/baseline.md` | dieses Dokument | Vorgehen und Ergebnisse für das Team |
| `docs/figures/baseline_test_*.png` | 3 Grafiken | Ergebnisse auf einen Blick |

**Geändert:**

| Datei | Änderung | Warum |
|---|---|---|
| `pyproject.toml`, `requirements.lock` | `scikit-learn` ergänzt | liefert die Quantilregression |
| `Makefile` | `make baseline-tune`, `make baseline` | Backtest mit einem Befehl |
| `.gitignore` | `reports/` | Ergebnisse entstehen bei jedem Lauf neu |
| `README.md` | Verweis auf dieses Dokument | auffindbar für alle |

Die bestehenden Datenmodule (`availability.py`, `weather.py` usw.) sind unverändert.

## 3. Begriffe

| Begriff | Bedeutung |
|---|---|
| **D** | Liefertag: der Tag, für den wir den Preis vorhersagen |
| **D−1** | Vortag: Um 11:00 wird die Prognose erstellt |
| **D−2** | Letzter Tag, dessen Preise um 11:00 schon bekannt sind |
| **P10** | Untere Grenze: Nur 10 % der Preise liegen darunter |
| **P50** | Mittlere Prognose (Median) |
| **P90** | Obere Grenze: Nur 10 % der Preise liegen darüber |
| **Data Leakage** | Das Modell nutzt Wissen, das um 11:00 noch nicht da war. Der Test sieht dann zu gut aus |
| **Features** | Eingaben des Modells, z. B. Uhrzeit oder Sonneneinstrahlung |
| **Referenz** | Einfachste Prognose ohne Modell: P10, P50 und P90 der Preise derselben Uhrzeit in den letzten 56 Tagen ([Abschnitt 5](#referenz)) |
| **B1** | Modell nur mit Kalender: Uhrzeit, Wochenende, Feiertage |
| **B2** | B1 plus Sonneneinstrahlung aus der Wetterprognose |
| **Backtest** | Die Vergangenheit Tag für Tag nachspielen, jeweils nur mit damals bekannten Daten |
| **Pinball Loss** | Fehler von P10, P50 und P90 zusammen. Tiefer ist besser. Unsere Hauptkennzahl |
| **MAE** | Mittlere Abweichung von P50 zum echten Preis in ct/kWh |
| **Coverage** | Anteil der echten Preise zwischen P10 und P90. Ziel: 80 % |
| **Skill Score** | Um wie viel das Modell besser ist als die Referenz. +10 % heisst 10 % weniger Fehler |

## 4. Warum dieses Modell

### Der Preis hat zwei typische Bereiche

- **Fehlt Strom im Netz (short):** Preis meist um 19 ct/kWh, nie negativ.
- **Hat es zu viel Strom (long):** Preis meist um 6 ct/kWh, teils stark negativ (bis −688 ct/kWh).
- **Welcher Fall eintritt,** lässt sich am Vortag kaum vorhersagen.
- **Ein normales lineares Modell** sagt einen Mittelwert dazwischen voraus, wo kaum echte Preise liegen.

### Die Quantilregression löst das

- Sie lernt für P10, P50 und P90 je eine eigene Gleichung.
- Die Sonne kann so P10 stark senken und P90 kaum verändern.
- Die Unsicherheit kommt direkt aus dem Modell.

### Sie ist erklärbar

- Jeder Koeffizient ist eine Verschiebung in ct/kWh.
- Beispiel: „Pro 100 W/m² Sonne sinkt P10 um 3 ct/kWh.“

### Was die Literatur erwarten lässt

- Lineare Modelle sind übliche Vergleichsmodelle für Strompreise ([Lago et al. 2021](https://arxiv.org/abs/2008.08004)).
- Ausgleichspreise schwanken viel stärker als Day-Ahead-Preise ([Verstraeten et al. 2026](https://arxiv.org/abs/2605.17054)).
- Am Vortag sind sie kaum vorhersagbar ([Klæboe et al. 2015](https://ideas.repec.org/p/gwc/wpaper/2013-006.html)).
- Ein grosser Vorsprung war darum nicht zu erwarten.

## 5. Aufbau

### Referenz

Die Referenz ist die einfachste Prognose, die ohne Modell möglich ist. Unser Modell muss zeigen, dass es mehr bringt.

**So rechnet sie**, Beispiel für die Viertelstunde 12:00 am Liefertag D:

1. Die Preise um 12:00 der letzten 56 Tage nehmen (D−57 bis D−2): 56 Werte.
2. Sortieren, vom tiefsten zum höchsten.
3. Ablesen:
   - **P10:** etwa der 6. tiefste Wert
   - **P50:** der mittlere Wert (Median)
   - **P90:** etwa der 6. höchste Wert

Das sind **Quantile, kein Mittelwert.** Ein Mittelwert würde von einzelnen Extrempreisen (bis −688 ct/kWh) stark verzogen.

**Warum 56 Tage:**

| Grund | Erklärung |
|---|---|
| In der EDA am besten | Verglichen wurden 7, 14, 28, 56 Tage und alle Daten. 56 Tage hatten den kleinsten Fehler und die beste Coverage (EDA H11, PR #5) |
| Ganze Wochen | 56 Tage = 8 Wochen. Jeder Wochentag kommt gleich oft vor |
| Genug Werte | 56 Werte pro Uhrzeit reichen für stabile P10 und P90; bei 7 Tagen wären es nur 7 |
| Aktuell genug | Ältere Daten, etwa aus dem Winter, fallen heraus. Die Referenz passt sich laufend an |

**Hinweis:** Die 56 Tage wurden in der EDA auf Daten gewählt, die auch den Testzeitraum enthalten. Das macht die Referenz eher stärker. Der Vergleich ist für unser Modell also eher streng als zu günstig.

### Features

Alle Features sind um 11:00 am Vortag bekannt.

| Feature | B1 | B2 |
|---|---|---|
| Uhrzeit | ja | ja |
| Position der Viertelstunde in der Stunde (der Preis kippt oft innerhalb der Stunde) | ja | ja |
| Wochenende | ja | ja |
| Feiertage, gewichtet nach Solarleistung im Kanton | ja | ja |
| Sonneneinstrahlung, Wetterprognose vom Abend D−2 (sicher vor 11:00 verfügbar) | – | ja |
| Zu- oder Abnahme der Sonne in dieser Stunde | – | ja |

### Bewertung mit dem Pinball Loss

Der Pinball Loss misst, wie gut die drei Grenzen liegen. Er bestraft Bänder, die zu schmal sind (Preis oft ausserhalb), und solche, die unnötig breit sind.

**Beispiel:** sonniger Mittag, der echte Preis ist −30 ct/kWh.

| | P10 | P50 | P90 |
|---|---|---|---|
| Prognose | −21 | 7 | 24 |
| Abstand zum echten Preis | 9 | 37 | 54 |
| Gewicht | 0.9 | 0.5 | 0.1 |
| **Pinball Loss** | **8.1** | **18.5** | **5.4** |

- **Gewicht:** Liegt der Preis *unter* der Prognose, zählt der Abstand mit 1 − Quantil (bei P10 also 0.9), liegt er darüber, mit dem Quantil selbst.
- **Deutung:** P10 hätte tiefer liegen müssen, das kostet viel. Dass P90 weit über dem Preis liegt, kostet wenig, denn P90 soll ja hoch liegen.
- **Gesamt:** Mittel der drei Werte = **10.7**.

## 6. Backtest ohne Data Leakage

- **Tag für Tag nachgespielt:** Für jeden Tag wird so prognostiziert, wie es um 11:00 am Vortag möglich gewesen wäre.
- **Training nur mit Preisen bis D−2.**
- **Wöchentlich neu trainiert.**
- **Automatisch geprüft:** Die Tests in `tests/test_model_*.py` stellen sicher, dass keine späteren Daten ins Modell gelangen.

| Zeitraum | Zweck |
|---|---|
| Januar bis März | nur Training |
| April und Mai | Einstellungen wählen |
| Juni bis August | Test |
| September | Schlusstest (Holdout) mit Daten, die vorher niemand angeschaut hat |

**Gewählte Einstellungen:** Auf April und Mai wurden Trainingsfenster (alle Daten ab 01.01. oder nur die letzten 56 Tage) und Regularisierung (Bremse gegen zu grosse Koeffizienten) verglichen. Am besten waren **alle Daten ab 01.01. ohne Regularisierung**, für B1 und B2.

**Restrisiko:** Trainiert wird mit den finalen Preisen. Um 11:00 gab es nur provisorische Werte. Die Ergebnisse sind darum leicht zu gut.

## 7. Aktuell noch offen

- **W&B:** Backtest hochladen mit `make baseline TRACK=1`. Die Team-Einladung fehlt noch.
- **Schlusstest September:** einmal mit unveränderten Einstellungen, sobald die finalen Preise da sind.
- **Grenze D−2 bestätigen:** Aktualisiert Swissgrid die Preise täglich? Bis dahin ist D−2 eine Annahme.
- **Hauptkennzahl im Team festlegen:** Hier ist es der Pinball Loss, PR #9 nennt den MAE.

## 8. Ergebnisse

| Juni bis August (91 Tage) | Pinball Loss | Coverage | gegenüber Referenz |
|---|---|---|---|
| Referenz | **3.20** | **75 %** | – |
| B1: Kalender | 3.25 | 73 % | −1.5 % |
| B2: Kalender + Sonne | 3.23 | 74 % | −0.7 % |

- **Gleich gut, nicht besser:** B2 liegt knapp hinter der Referenz und ist an gut der Hälfte der Tage besser.
- **Die Sonne hilft mittags:** Von 9 bis 16 Uhr und bei negativen Preisen ist B2 besser als die Referenz, abends schlechter.
- **Die Bänder sind zu schmal:** Alle Modelle erreichen weniger als 80 % Coverage.
- **Einstellungen:** In der Abstimmung (April und Mai) war B2 noch 4 % besser. Danach wurde nichts mehr angepasst, damit der Test fair bleibt.
- **Ein Tag fehlt:** Der 24.06. ist wegen einer Lücke in den Wetterdaten für alle Modelle ausgenommen.

![Pinball Loss pro Stunde](figures/baseline_test_pinball_by_hour.png)

![Coverage pro Monat](figures/baseline_test_coverage_by_month.png)

![Koeffizienten von B2](figures/baseline_test_coefficients.png)

**Was das Modell gelernt hat:** Pro 100 W/m² Sonne sinkt P10 um 3 ct/kWh, P90 nur um 0.6. Sonne macht tiefe Preise also wahrscheinlicher, hohe kaum seltener.

## 9. Fazit

### Was verglichen wurde

- **Referenz:** die einfachste Prognose ohne Modell und ohne Wetterdaten: P10, P50 und P90 der Preise derselben Uhrzeit aus den letzten 56 Tagen ([Abschnitt 5](#referenz)).
- **Baseline B2:** unser Modell. Es nutzt zusätzlich Kalender und Sonnenprognose.
- **Frage:** Bringt das Modell mehr als der Blick in die Vergangenheit?

### Was die Zahlen bedeuten

| Kennzahl | B2 gegenüber Referenz | Einordnung |
|---|---|---|
| Pinball Loss | 3.23 statt 3.20 | Unterschied unter 1 %: praktisch gleich gut |
| Tage besser | 53 % | kaum mehr als ein Münzwurf |
| MAE von P50 | 9.8 statt 9.9 ct/kWh | P50 liegt im Schnitt rund 10 ct/kWh daneben. Bei typischen Preisen von 6 bis 19 ct/kWh ist das viel |
| Coverage | 74 % statt 75 % | Ziel 80 %: Etwa jede vierte statt jede fünfte Viertelstunde liegt ausserhalb des Bands |
| B2 gegenüber B1 | 3.23 statt 3.25 | Die Sonne verbessert das Modell, aber nur leicht |

### Gesamtfazit

- **Die Antwort auf die Frage ist nein:** Das Modell ist nicht besser als der Blick in die Vergangenheit, aber auch nicht schlechter.
- **Das ist erwartbar:** Laut Literatur sind Ausgleichspreise am Vortag kaum vorhersagbar.
- **Die Sonne hilft punktuell:** mittags und bei negativen Preisen. Abends fehlt Information, etwa eine Lastprognose.
- **Der Nutzen liegt im Band P10 bis P90,** nicht im einzelnen Wert P50. Das Band ist aber noch zu schmal und muss kalibriert werden.
- **Die Baseline steht:** reproduzierbar (ein erneuter Lauf ergibt exakt dieselben Zahlen), gegen Data Leakage getestet und erklärbar.
- **Der Massstab für spätere Modelle:** ein Pinball Loss unter 3.20, also besser als die Referenz.
