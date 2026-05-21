# MPS Performance Guide

Dieses Dokument ist bewusst auf **noch offene** MPS-Performance-Themen im
`tensor`-Runtime reduziert. Bereits erledigte Arbeiten und historische
Messreihen sind aus dem laufenden Arbeitsdokument entfernt.

Leitplanken bleiben unverändert:

- GPU-Daten möglichst GPU-resident halten.
- `readBytes(...)` nur an klaren API-Grenzen.
- `waitUntilCompleted` nicht implizit pro Operator.
- Reuse von Graphen, Buffern und Command Buffers.
- Jede Änderung muss benchmark-seitig nachvollziehbar sein.

## Fortschritt (kurz)

### 2026-05-21 — Schritt 1 (P0) abgeschlossen: CSV-Basis erweitert

Umgesetzt:

- `BENCH_CSV`-Export für `//app/tensor:bench_st_pipeline` ergänzt.
- `BENCH_CSV`-Export für `//app/tensor:bench_st_transfer` ergänzt.
- Default-Textausgabe beider Benchmarks unverändert beibehalten.

Verifiziert:

- `bazel build //app/tensor:bench_st_pipeline //app/tensor:bench_st_transfer` ✅
- `bazel run //app/tensor:bench_st_pipeline` ✅
- `BENCH_CSV=1 bazel run //app/tensor:bench_st_pipeline` ✅
- `bazel run //app/tensor:bench_st_transfer` ✅
- `BENCH_CSV=1 bazel run //app/tensor:bench_st_transfer` ✅

Kurzbefund Performance:

- Kein reproduzierbarer Regressionshinweis durch CSV-Pfad.
- Transfer-Benchmark bestätigt weiter klaren Vorteil für metal-residenten Pfad
  (Session-Lauf: `transfer-medium` ~`2.05ms` vs `3.12ms`,
  `transfer-large` ~`16.43ms` vs `35.77ms`).

### 2026-05-21 — Schritt 2 (P0) gestartet: Profil-Kalibrierung angebahnt

Umgesetzt:

- CSV-Ausgabe enthält jetzt zusätzlich `async_profile` zur sauberen
  A/B-Zuordnung in Profil-Läufen.
- Skript `share/benchmarks/scripts/run_st_profile_matrix.sh` ergänzt,
  erzeugt Profil-Matrix-Artefakte für `pipeline` und `transfer`.
- Erste Stichprobe für `conv+bn+pool(fused), boundary_sync_only, pipe-large`
  aufgenommen:
  - `throughput`: ~`25.64ms/iter`
  - `balanced`: ~`25.84ms/iter`
  - `stable`: ~`28.66ms/iter`

Neue Artefakte (aktueller Lauf):

- `share/benchmarks/results/st_pipeline_profile_matrix_20260521_213338.csv`
- `share/benchmarks/results/st_transfer_profile_matrix_20260521_213338.csv`

Kurzbefund:

- In dieser Stichprobe liegt `stable` hinter `throughput/balanced`.
- Einzelne Transfer-Mittelwerte schwanken stark zwischen Läufen; daher sind
  Einzelmessungen kein belastbarer Regressionsindikator.
- Für eine belastbare Default-Entscheidung folgt als Nächstes die
  vollständige Matrix über weitere Cases/Größen.

### 2026-05-21 — Schritt 3/4 (P0) umgesetzt: Schema + Matrix-Auswertung

Umgesetzt:

- Einheitliches CSV-Kernschema dokumentiert:
  `share/benchmarks/csv-schema.md`.
- `single_op`-CSV auf Kernspalten `suite` und `async_profile` erweitert.
- Generischer Parser/Validator ergänzt:
  `share/benchmarks/scripts/parse_st_bench_csv.py`.
- Profil-Matrix-Resumen ergänzt:
  `share/benchmarks/scripts/summarize_st_profile_matrix.py`.

Verifiziert:

- Parser liest `single_op`, `pipeline`, `transfer` mit derselben Kernlogik:
  - `share/benchmarks/results/st_single_op_latest.csv`
  - `share/benchmarks/results/st_pipeline_latest.csv`
  - `share/benchmarks/results/st_transfer_latest.csv`
- Matrix-Artefakte und Auswertung aus aktuellem Lauf:
  - `share/benchmarks/results/st_pipeline_profile_matrix_20260521_213722.csv`
  - `share/benchmarks/results/st_transfer_profile_matrix_20260521_213722.csv`
  - `share/benchmarks/results/st_profile_matrix_summary_20260521_213722.md`

Kurzbefund:

- Aktuelle Matrix-Empfehlung: `default` (aggregiert beste Mean-Metrik über
  `conv_bn_boundary`, `conv_bn_pool_boundary`, `transfer_metal`).
- `throughput` ist zwar bei `conv_bn_boundary` am schnellsten, verliert aber in
  `conv_bn_pool_boundary` und `transfer_metal`.

### 2026-05-21 — Schritt 5 (P0) umgesetzt: Empfehlung + Guardrails

Umgesetzt:

- Klassen-spezifische Profil-Policy ergänzt:
  `share/benchmarks/st_async_profile_policy.json`
  - `conv_bn_boundary`: `throughput`
  - `conv_bn_pool_boundary`: `default`
  - `transfer_metal`: `default`
- Regression-Guard ergänzt:
  `share/benchmarks/scripts/check_st_profile_policy.py`
- Guard-Report aus aktuellem Matrix-Lauf erzeugt:
  `share/benchmarks/results/st_profile_policy_guard_20260521_213722.md`

Guardrail:

- Maximal erlaubte Regression gegenüber Klassen-Bestwert: `10%`.
- Aktueller Datensatz: alle Policy-Klassen bestehen den Guard (`ok`).

### 2026-05-21 — Schritt 6 (P1) umgesetzt: Fastpath-Helper konsolidiert

Umgesetzt:

- Neuer gemeinsamer Helper im MPS-Backend:
  `st_mps_encode_preallocated_async(...)` in
  `app/tensor/src/st_backend_mps.m`.
- Verwendet in drei preallocated Fastpaths:
  - `mps_maxpool2d_forward_preallocated`
  - `mps_avgpool2d_forward_preallocated`
  - `mps_batchnorm2d_forward_preallocated`

Wirkung:

- Gemeinsame Encode/Finalize-Logik statt dreifacher Duplikation.
- Keine Verhaltensänderung an Call-Sites (gleiche async/commit-Semantik).

Verifiziert:

- `bazel build //app/tensor:bench_st_single_op //app/tensor:bench_st_pipeline //app/tensor:bench_st_transfer` ✅
- Fokus-Benchmarks nach Konsolidierung ohne Regressionshinweis:
  - `single_op`: `maxpool2d`, `avgpool2d`, `batchnorm2d`
  - `pipeline`: `conv+bn` und `conv+bn+pool` (boundary_sync_only, medium/large)

### 2026-05-21 — Schritt 7 (P1) umgesetzt: Layout-Policy Draft

Umgesetzt:

- Layout-Policy mit If/Then-Regeln ergänzt:
  `share/benchmarks/st_layout_policy.md`
- Messartefakte für den Draft erzeugt:
  - `share/benchmarks/results/st_layout_policy_input_20260521.csv`
  - `share/benchmarks/results/st_nhwc_vs_nchw_policy_input_20260521.csv`
- Kurz-Auswertung abgelegt:
  `share/benchmarks/results/st_layout_policy_summary_20260521.md`

Kurzbefund:

- NCHW↔NHWC-Roundtrip ist für mehrere Cases deutlich teurer als Conv selbst
  (`+544%` bis `+2106%`).
- NHWC zeigt im aktuellen Satz nur für `resnet_s1` einen Vorteil (`-11.06%`),
  in den übrigen Cases ist NHWC langsamer.
- Empfehlung daher: Default bleibt NCHW, NHWC nur case-spezifisch freigeben.

### 2026-05-21 — Schritt 8 (P2) umgesetzt: Custom-Kernel Kandidatenliste

Umgesetzt:

- Kandidatenliste + Go/No-Go-Hypothesen + A/B-Messplan ergänzt:
  `share/benchmarks/st_custom_kernel_candidates.md`

Top-Kandidaten (priorisiert):

1. fused `BN(+ReLU)+Pool` epilogue kernel
2. channelwise BatchNorm inference kernel
3. Pool2D kernel family (`k=2..3`, `stride=1..2`)

Explizit aktuell **kein** Kandidat:

- Transferpfad als „Kernelproblem“ (primär Residency/Boundary-Thema).

## Offene Themen (priorisiert)

### P0 — Messbasis vereinheitlichen (CSV überall)

**Ziel:** Maschinenlesbare Ausgabe konsistent für die zentralen MPS-Benchmarks.

Offen:

- CSV-Export in `bench_st_pipeline` ergänzen
- CSV-Export in `bench_st_transfer` ergänzen
- gemeinsames Spalten-Set für Vergleichbarkeit definieren

Definition of Done:

- `bench_st_single_op`, `bench_st_pipeline`, `bench_st_transfer` liefern per
  Env-Flag konsistente CSV-Ausgabe.
- ein Parser/Notebook kann alle drei Formate ohne Sonderfälle einlesen.

---

### P0 — Commit-/Sync-Strategie kalibrieren

**Ausgangslage:** Profile (`throughput`, `balanced`, `stable`) sind vorhanden,
aber noch nicht belastbar kalibriert.

Offen:

- systematische A/B-Matrix pro Pipeline-Typ (`conv+bn`, `conv+bn+pool`,
  transfer-lastig)
- klare Empfehlung, wann welches Profil Standard sein soll
- Regression-Guards für ungünstige Profile

Definition of Done:

- dokumentierte Profil-Empfehlung pro Pipeline-Klasse
- reproduzierbare Messung mit mindestens zwei Problemgrößen je Klasse
- kein Profil mit signifikantem Regressionsrisiko als Default

---

### P1 — Fastpath-Helfer weiter vereinheitlichen

**Ziel:** weniger Duplikation, schnelleres Hinzufügen neuer GPU-residenter Ops.

Offen:

- einheitliche Helper für Feed-Mapping und preallocated Outputs
- klarer Reuse-Pfad für Executable-/Graph-Caching
- konsistentes async finalize/commit handling für neue Ops

Definition of Done:

- neue oder angepasste Op-Fastpaths verwenden dieselben Kern-Helper
- weniger op-spezifische Sonderlogik im Encode-/Finalize-Pfad

---

### P1 — Layout-Strategie festziehen (NCHW/NHWC)

**Ausgangslage:** NHWC-Vorarbeit ist vorhanden, aber keine verbindliche
Strategie über den gesamten Pipeline-Fluss.

Offen:

- bevorzugtes internes Layout für Conv-dominierte GPU-Pipelines festlegen
- Transpose-Budget/Regeln definieren (wann erlaubt, wann zu teuer)
- robuster Umgang mit nicht-kontiguierlichen Tensoren im Layout-Wechsel

Definition of Done:

- dokumentierte Layout-Policy mit klaren Entscheidungsregeln
- Benchmarks zeigen reduzierte oder stabilere Transpose-Overheads

---

### P2 — Custom-Metal-Kernels gezielt evaluieren

**Ziel:** nur dort eigene Kernel bauen, wo MPSGraph klar limitiert.

Offen:

- Hot-Path-Kandidaten anhand von Profiling priorisieren
- für jeden Kandidaten: MPSGraph vs. Custom-Kernel A/B messen
- nur Kandidaten mit robustem Vorteil übernehmen

Definition of Done:

- kurze Kandidatenliste mit Messdaten und Go/No-Go-Entscheidungen
- keine „vorsorgliche“ Kernel-Implementierung ohne Benchmark-Nachweis

## Mindestprüfung für jede neue MPS-Änderung

Vor Merge müssen mindestens diese Fragen positiv beantwortet sein:

1. Bleiben Daten länger GPU-resident?
2. Wurden Readbacks oder unnötige Waits reduziert?
3. Ist der Effekt in Benchmarks/CSV sichtbar?
4. Nutzt die Änderung bestehende Stream-/Buffer-/Fastpath-Infrastruktur?
5. Gibt es einen klaren Regression-Check für den betroffenen Pfad?

## Kurz-Roadmap (empfohlen)

1. **P0 umsetzen:** CSV für Pipeline/Transfer + Profil-Kalibrierung.
2. **P1 stabilisieren:** Fastpath-Helper und Layout-Policy konsolidieren.
3. **P2 selektiv:** Custom-Kernel nur datengetrieben nachziehen.

## Sprint-Backlog (2 Wochen, direkt umsetzbar)

Die folgende Liste ist so formuliert, dass sie 1:1 als GitHub-Issues übernommen
werden kann.

### Sprint-Ziel

- Reproduzierbare MPS-Messbasis für Pipeline/Transfer herstellen.
- Commit-/Sync-Profile belastbar kalibrieren.
- P1-Themen vorbereiten, ohne unnötige Refactors zu starten.

### Ticket 1 — CSV-Export für `bench_st_pipeline`

**Priorität:** P0  
**Aufwand:** S

- [x] Env-Flag analog zu `BENCH_CSV` in `bench_st_single_op` ergänzen.
- [x] CSV-Header und Rows für alle Pipeline-Varianten ausgeben.
- [x] Menschlich lesbare Ausgabe als Default beibehalten.

**Akzeptanzkriterien**

- [x] `bench_st_pipeline` liefert mit Env-Flag parsebare CSV-Zeilen.
- [x] Ohne Env-Flag bleibt bestehende Console-Ausgabe unverändert.

### Ticket 2 — CSV-Export für `bench_st_transfer`

**Priorität:** P0  
**Aufwand:** S

- [x] Env-Flag für CSV ergänzen.
- [x] Transfer-spezifische Kennzahlen als Spalten aufnehmen.
- [x] Werte für Metal vs. CPU/Blit pro Case eindeutig markieren.

**Akzeptanzkriterien**

- [x] `bench_st_transfer` liefert reproduzierbare CSV-Zeilen pro Case.
- [x] Spaltennamen sind stabil und dokumentiert.

### Ticket 3 — Einheitliches CSV-Schema definieren

**Priorität:** P0  
**Aufwand:** S

- [x] Kernspalten festlegen (suite, case, variant, iters, ms, counter deltas).
- [x] Pflicht-/optional-Spalten dokumentieren.
- [x] Kurz-Doku im Repo ergänzen (z. B. in `share/` oder `doc/`).

**Akzeptanzkriterien**

- [x] Ein Parser kann `single_op`, `pipeline`, `transfer` ohne Sonderfälle lesen.
- [x] Keine mehrdeutigen Spaltennamen zwischen Benchmarks.

### Ticket 4 — Profil-Matrix für Commit-/Sync-Strategie aufsetzen

**Priorität:** P0  
**Aufwand:** M

- [x] Testmatrix definieren für `throughput|balanced|stable`.
- [x] Mindestens 3 Pipeline-Klassen abdecken (`conv+bn`, `conv+bn+pool`, transfer).
- [x] Je Klasse mindestens 2 Problemgrößen messen.

**Akzeptanzkriterien**

- [x] Vollständige Ergebnistabelle mit identischer Messmethodik vorhanden.
- [x] Messergebnisse sind per CSV/Artefakt nachvollziehbar.

### Ticket 5 — Default-Profil-Empfehlung mit Guardrails

**Priorität:** P0  
**Aufwand:** M

- [x] Profil-Empfehlung pro Pipeline-Klasse aus Messmatrix ableiten.
- [x] Regressionsgrenzen festlegen (z. B. max. tolerierbare Verschlechterung).
- [x] Guard-Check in Benchmark-/CI-Workflow aufnehmen (wo sinnvoll).

**Akzeptanzkriterien**

- [x] Dokumentierte Default-Empfehlung im Repo vorhanden.
- [x] Kein Default mit bekannter signifikanter Regression.

### Ticket 6 — Fastpath-Helfer: minimaler Konsolidierungsschnitt

**Priorität:** P1  
**Aufwand:** M

- [x] Duplizierte Feed-/Finalize-Muster in bestehenden MPS-Ops inventarisieren.
- [x] 1–2 kleine gemeinsame Helper extrahieren (ohne Großrefactor).
- [x] Bestehendes Verhalten mit Bench/Test absichern.

**Akzeptanzkriterien**

- [x] Reduzierte Duplikation in mindestens zwei Fastpaths.
- [x] Keine funktionale Regression in betroffenen Benchmarks.

### Ticket 7 — Layout-Policy Entwurf (NCHW/NHWC)

**Priorität:** P1  
**Aufwand:** M

- [x] Entscheidungsregeln für internes Layout als Draft dokumentieren.
- [x] Transpose-Budget definieren (wann Wechsel sinnvoll/zu teuer).
- [x] Bezug auf vorhandene Layout-Benchmarks herstellen.

**Akzeptanzkriterien**

- [x] Policy-Dokument mit klaren If/Then-Regeln vorhanden.
- [x] Mindestens ein Benchmark-Befund stützt jede Kernregel.

### Ticket 8 — Custom-Kernel Kandidatenliste (No-Implementation)

**Priorität:** P2  
**Aufwand:** S

- [x] Top-2/Top-3 Kandidaten aus Profiling ableiten.
- [x] Für jeden Kandidaten Go/No-Go-Hypothese formulieren.
- [x] Messplan für spätere A/B-Validierung definieren.

**Akzeptanzkriterien**

- [x] Kandidatenliste ist kurz, begründet und priorisiert.
- [x] Keine Kernel-Implementierung ohne vorherige Messgrundlage.

## Sprint-Exit-Kriterien

Der Sprint gilt als erfolgreich, wenn alle Punkte erfüllt sind:

- [x] CSV in `single_op`, `pipeline`, `transfer` ist einheitlich nutzbar.
- [x] Profil-Kalibrierung liefert belastbare Default-Empfehlungen.
- [x] P1 ist mit klaren, risikoarmen Folgeschritten vorbereitet.
- [x] Für P2 liegt eine datengetriebene Kandidatenliste vor.
