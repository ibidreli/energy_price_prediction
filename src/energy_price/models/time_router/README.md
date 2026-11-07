# Zeitabhängige Modellarchitektur · Issue #25

Stand 07.11.2026: Implementierung und April/Mai-Festlegung. Der gemeinsame
Holdout bleibt bis zum endgültigen Pool-Freeze am 27.11. und zur Bestätigung
finaler Daten gesperrt. Juni–August sind Entwicklungsdaten, kein blinder Test.

## Pool und Struktur

`model.py` folgt dem Modellordner-Schema mit `MODEL`, `GRID`, `CHOSEN` und
`ISSUE`. Der zusätzliche `EXPERIMENT_RUNNER` verarbeitet OOS-Prognosen statt
Rohfeatures. Der normale Befehl funktioniert deshalb auch für dieses Modell:

```bash
make backtest MODEL=time_router STEP=tune JOBS=2
# Design prüfen und committen; die Testphase verlangt den Commit.
make backtest MODEL=time_router STEP=test JOBS=2 TRACK=1
```

Alternativ `make time-router STEP=pool|tune|test|readiness JOBS=2`.
`pool` erzeugt ausschliesslich Januar–Mai; `tune` erzeugt das Design.
`make time-router STEP=test JOBS=2 TRACK=1` reproduziert den gesamten erlaubten
Lauf aus Rohdaten und dem zuvor committeten Design mit einem Befehl.
Ohne GNU Make: `.venv/Scripts/python.exe -m energy_price.evaluation.time_router test --jobs 2 --track`
(vorher die verarbeiteten Datentabellen erzeugen).

Die vier Experten sind Referenz (56 Tage), lineare Baseline B2, Quantile Boosting
und Regime-Mischung. Ihre Einstellungen stammen unverändert aus ihren bisherigen
Experimenten auf April/Mai. Referenz bleibt täglich, die anderen Fits bleiben
wöchentlich mit den bisherigen Ankern: 30.01., 01.04. und 01.06. Historische
Preise und Richtungslabels reichen höchstens bis zum jeweiligen Fit-Tag minus
zwei Tage. Die tatsächliche Richtung wird vom Router ausschliesslich in der
nachfolgenden Auswertung verwendet.

`POOL` ist eine Einstellung, kein Namensschalter im Auswahlcode. Ein neuer
Experte benötigt einen Eintrag mit `module` und seinen eingefrorenen `settings`
(inklusive Trainingsfenster). Seine Modellklasse bietet `fit`/`predict` und
P10/P50/P90 wie die vorhandene Vorlage. Aufnahme erfordert zusätzlich den
Nachweis: gemeinsame Backtest-Folds, D−2-konforme Eingaben und ausschliesslich
auf April/Mai gewählte Einstellungen. Danach Design und Test neu rechnen;
vor dem finalen Freeze keine alten Gewichte oder Ergebnisse weiterverwenden.

Wiederverwendbare Bausteine liegen in `models/shared/forecast_pool.py`,
`models/shared/routing.py` und `models/shared/calibration.py`; die Ausführung
liegt unter `evaluation/time_router.py`. Der inhaltsbasierte Cache enthält
Trainings- und Prognoseeingaben, Einstellungen, Expertencode und Paketversionen.
Vorhandene Reportdateien anderer Experimente werden nicht ungeprüft übernommen.

## Festlegung ausschliesslich auf April/Mai

Diese Festlegung und `design.json` werden vor der ersten neuen
Juni–August-Auswertung committet und in Issue #25 verlinkt.

| Experte | Pinball April/Mai |
|---|---:|
| Referenz | 3.448507 |
| B2 | 3.311188 |
| Boosting | **3.308685** |
| Regime-Mischung | 3.322611 |

**Standardmodell: Boosting.** Die signierten P50-Fehler korrelieren zwischen den
Experten mit 0.987 bis 0.997. Das begrenzt die erwartbare Diversifikation; eine
hohe Korrelation allein beweist allerdings nicht, dass jede Kombination scheitert.

Fachliche Blöcke: 00–08 Uhr (Nacht/Frühmorgen), 08–16 Uhr (PV-Tagesabschnitt),
16–24 Uhr (Abend). Robustheit: 4×6 Stunden, ohne Werktag/Wochenende-Aufteilung.
Die vollständige Literaturentscheidung und die **vorab definierte** Power-Regel
stehen in [research.md](research.md).

Für einen Vorteil von 1 % und 80 % Power benötigen die gepaarten Vergleiche
auf Basis von April/Mai näherungsweise:

| Raster | Minimum benötigte Tage | Median | Maximum |
|---|---:|---:|---:|
| ein Tagesblock | 210 | 215 | 1930 |
| drei fachliche Blöcke | 363 | 862 | 3583 |
| vier 6-Stunden-Blöcke | 619 | 1019.5 | 6226 |

Vorhanden sind 61 Tage. Damit wird **kein Raster für diesen kleinen Effekt
ausreichend abgesichert**. Die Hauptvariante hat als Fallback einen Tagesblock;
sie behauptet keine belastbare Tageszeit-Spezialisierung. Vier 6-Stunden-Blöcke
bleiben eine diagnostische Robustheitsprüfung und dürfen nach Testeinsicht nicht
zum Hauptkandidaten befördert werden. Die Zahlen sind grobe Planungswerte mit
Normalapproximation, keine garantierte Stichprobengrösse bei Extrempreisen.

Die drei Kandidaten sind:

1. Gleiches Gewicht für alle Experten.
2. Auswahl eines Experten für alle drei Quantile gemeinsam. Wöchentlich nur
   wechseln, wenn das 95-%-Intervall der täglichen Pinball-Differenz unter null
   liegt; zusätzlich Bonferroni über `(Poolgrösse−1) × Blockzahl` je Fit.
   Sonst Standardmodell. Alle verfügbaren OOS-Fehler bis zum Wochenbeginn−2.
3. Konvexe Gewichte je zugelassenem Block und Quantil: Pinball-LP, letzte 56
   Kalendertage, 50 % Schrumpfung zum Standardmodell, mindestens 28 Tage.

Bootstrap: **7 Kalendertage**, **5000 Wiederholungen**, **Seed 0**. Fehlende Tage
bleiben Kalenderlücken. Vorhersagen einer Woche verwenden auch an ihren späteren
Tagen nur den Stand vom Wochenbeginn. Das ist konservativer als täglich D−2.

## Datenlecks, Kalibrierung und Nachvollziehbarkeit

Die lange Prognosetabelle enthält je Experte UTC-Zeitstempel, Prognose,
`fit_day` und `train_last_day`. Die Prüfung verweigert fehlende/duplizierte
Viertelstunden, widersprüchliche Zielwerte und eine verletzte D−2-Grenze.
UTC erhält die wiederholte Oktoberstunde. Fehlende Prognosen/Preise schliessen
den ganzen Tag für alle Experten und Kandidaten aus. Juni 24 bleibt so wie bei
den bisherigen Modellen wegen fehlenden Wetters ausgeschlossen.

Der Router sieht beobachtete Fehler ausschliesslich bis D−2. Seine Ausgabe
enthält `chosen_expert`, quantilspezifische `weights`, `block`, `fit_day` und
`error_last_day`. Nach quantilspezifischer Gewichtung werden Kreuzungen sortiert;
`preorder_q10/q50/q90` und `quantiles_reordered` machen diesen Schritt sichtbar.

Kalibrierung folgt **einmal nach der Auswahl**: 56 Tage, 6-Stunden-Gruppen,
separate 10-%-Schwanzkorrekturen und unverändertes P50 wie im GBM-Experiment #19.
Sie verwendet gespeicherte Ausgaben der damaligen Router-Fits, keine rückwirkend
mit heutigen Gewichten berechneten Prognosen. `calibration_last_day` weist die
Grenze aus. Rohe und kalibrierte Kandidaten werden auf identischen Viertelstunden
mit dem rohen beziehungsweise kalibrierten Standardmodell verglichen.

April/Mai legen auch das Standardmodell und die Expertenparameter fest. Die
Warmhistorie ist deshalb OOS hinsichtlich der einzelnen Modellfits, aber keine
vollständig vorab festgelegte Hyperparameter-Evaluation von Januar bis Mai.
Sie dient der Entwicklung und Kalibrierung; erst ab Juni gilt das eingefrorene
Design. Die historischen Swissgrid-Dateien enthalten überarbeitete Werte und
keine vollständigen damaligen Datenstände. D−2 verhindert zeitliche Lecks im
Backtest, beseitigt aber diese Einschränkung historischer Revisionen nicht.

## Juni–August und Entscheid

Noch nicht ausgeführt beim Commit der Festlegung. Nach dem Lauf werden hier
Kennzahlen, gepaarte Intervalle und der nominierte Holdout-Kandidat ergänzt.
Die drei Kandidaten werden nach kalibriertem Pinball verglichen. Der tiefste
Wert nominiert einen Kandidaten; daraus folgt keine Übernahme. Der einzige
blinde Entscheid erfolgt später im Holdout: Pinball tiefer, 95-%-Intervall ganz
negativ und Coverage mindestens so hoch wie beim kalibrierten Standardmodell.

## Holdout: erst nach finalen Daten und Pool-Freeze

Der Nutzer hat am 07.11. bestätigt, den Zeitplan beizubehalten. Deshalb gibt es
hier noch **keinen Holdout-Lauf und keinen endgültigen Architekturentscheid**.
26 fehlende ECMWF-Läufe wurden archiviert; die Wetterfeatures sind für alle
5860 erwarteten Viertelstunden September/Oktober vollständig. Preise fehlen
lokal noch für alle 5860 Viertelstunden, Richtungslabels für 2692.

```bash
make holdout-data-check
# Später: finale Swissgrid-XML September/Oktober einlesen, CAB-Snapshot holen.
make fetch-cab
make data
```

Finalität muss anhand der Swissgrid-Veröffentlichung bestätigt werden. Ein
Datum allein gilt nicht als Nachweis. Erwartet werden finale Oktoberpreise
etwa ab 20.11.; der geplante Pool-Freeze bleibt 27.11. Erweiterungen des Pools
vorher verlangen erneute Entwicklungsschritte. Die abschliessende Erklärung
`final_data.json` liegt lokal und enthält:

```json
{
  "final_prices_confirmed": true,
  "months": ["2026-09", "2026-10"],
  "source_url": "URL zur geprüften offiziellen Veröffentlichung",
  "verified_by": "Name des prüfenden Teammitglieds",
  "verified_on": "2026-11-27",
  "explanation": "Nachweis, dass beide XML-Dateien finale Preise enthalten",
  "data_fingerprint": "Wert aus holdout_readiness.json nach make data"
}
```

Erst dann:

```bash
make time-router STEP=freeze FINAL_DATA_EVIDENCE=final_data.json
git add src/energy_price/models/time_router/pool_freeze.json
git commit -m "exp: freeze final model pool and holdout inputs (#25)"
make time-router STEP=holdout JOBS=2 TRACK=1 FINAL_DATA_EVIDENCE=final_data.json
```

Der Freeze bindet Pool, Design, Nominierung und Daten-Hash. Ein atomar angelegtes
`holdout_protocol.json` verbraucht die Freigabe vor dem Rechnen. Auch nach einem
Abbruch erlaubt kein CLI-Schalter eine zweite Auswertung. Ein technischer
Fehler muss mit sichtbarem Protokoll untersucht werden. Die endgültigen
Einzelexperten und der nominierte Kandidat werden auf denselben Viertelstunden
bewertet; Monats-/Block-/Richtungstabellen entstehen erst nach der Prognose.

## Prüfung und Review

`tests/test_time_router.py` prüft manipulierte Zukunftspreise, tatsächliche
Richtung, wöchentliche Grenzen, historische Router-Ausgaben in der Kalibrierung,
einmalige Kalibrierung, neue Experten ohne Codeänderung, Quantilkreuzungen,
Blockpartition, UTC/DST, vollständige gemeinsame Quartale, Bootstrap,
eingeschränkte LP-Gewichte und die atomare Holdout-Sperre.

Die Branches aus #19 und #21 wurden in diesen Arbeitsbranch integriert. Das ist
keine Freigabe durch ein Teammitglied und kein Merge nach `main`. Teamreview
und der spätere gemeinsame Holdout bleiben offene Abschlussbedingungen.
