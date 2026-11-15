# Metrik

Dieses Dokument ist in [`docs/evaluation.md`](evaluation.md) aufgegangen. Dort stehen Ausgabe, Metriken, Referenz und Bewertungsverfahren mit Begründung.

Die frühere Festlegung hier (MAE als Hauptmetrik, Median der gleichen Viertelstunde an D−2 bis D−8 als Baseline) gilt nicht mehr. Hauptmetrik ist der mittlere Pinball-Loss, Referenz sind die Quantile der letzten 56 Tage.
