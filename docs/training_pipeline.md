# Trainingspipeline

```mermaid
flowchart LR
    TRAIN["Training<br/>Januar bis März"]
    VAL["Validation (tune)<br/>April und Mai<br/>Einstellungen wählen"]
    TEST["Test<br/>Juni bis August<br/>Modelle vergleichen"]
    HOLD["Holdout<br/>September und Oktober<br/>gesperrt bis zur Freigabe"]

    TRAIN --> VAL --> TEST --> HOLD

    classDef train fill:#e8e8e3,stroke:#8a8983,color:#0b0b0b
    classDef val fill:#d6e6f8,stroke:#2a78d6,color:#0b0b0b
    classDef test fill:#86b6ef,stroke:#104281,color:#0b0b0b
    classDef hold fill:#fbdccf,stroke:#eb6834,color:#0b0b0b
    class TRAIN train
    class VAL val
    class TEST test
    class HOLD hold
```

```mermaid
gantt
    title Training und bewertete Perioden 2026
    dateFormat YYYY-MM-DD
    axisFormat %b
    tickInterval 1month

    section Validation
    Training beim Start      :done, 2026-01-01, 2026-04-01
    April und Mai bewertet   :active, 2026-04-01, 2026-06-01

    section Test
    Training beim Start      :done, 2026-01-01, 2026-06-01
    Juni bis August bewertet :active, 2026-06-01, 2026-09-01

    section Holdout
    Training beim Start      :done, 2026-01-01, 2026-09-01
    September Holdout        :crit, 2026-09-01, 2026-10-01
    Oktober Holdout          :crit, 2026-10-01, 2026-11-01
```

Der Holdout umfasst September und Oktober 2026 und bleibt gemäss Modell-Dokumentation
gesperrt, bis er ausdrücklich einmal freigegeben wird. Für jedes Liefertag D nutzt
das Training nur Preise bis D−2. Die Verfügbarkeitsgrenze ist D−2, 24:00, basierend
auf der Projektbestätigung vom 22.10.2026. Der Laptop-Task wurde getestet, seine Logs
gingen wegen eines Scriptfehlers verloren; die Projektentscheidung ist, die Daten
aufgrund der verfügbaren Indizien weiter zu verwenden. Details:
[Datengrundlage](data.md#prüfung-des-laptop-tasks).
