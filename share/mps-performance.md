# MPS Performance Guide

Dieses Dokument fasst den aktuellen Stand der MPS-Optimierungen im
`tensor`-Runtime zusammen. Der Fokus liegt auf drei Fragen:

- Was wurde bereits umgesetzt?
- Was wurde dadurch messbar erreicht?
- Welche Themen sind noch offen?

Der übergeordnete Performance-Anspruch bleibt unverändert:

- GPU-Daten sollen möglichst auf der GPU bleiben.
- `readBytes(...)` soll nur an klaren API-Grenzen oder für explizite
  Host-Ausgaben auftreten.
- `waitUntilCompleted` soll nicht implizit pro Operator passieren.
- Graphen, Buffer und Command Buffers sollen wiederverwendet werden.
- Änderungen müssen über Benchmarks nachvollziehbar sein.

## Bereits umgesetzt

### GPU-residente Outputs für Pool und BatchNorm

Die größte konkrete Lücke lag in Pool- und BatchNorm-Pfaden, die GPU-Arbeit zu
früh wieder auf die CPU zurückgeführt haben. Diese Lücke ist für die relevanten
Inference-Pfade geschlossen.

Umgesetzt wurden:

- `mps_maxpool2d_forward_preallocated()` und
  `mps_avgpool2d_forward_preallocated()` in `src/st_backend_mps.m`
- GPU-residenter BatchNorm-Fastpath
  `mps_batchnorm2d_forward_preallocated()` in `src/st_backend_mps.m`
- Preallocated Metal-Outputs statt `readBytes(...)` im Fastpath
- Asynchrone Ausführung ohne implizites `waitUntilCompleted`
- CPU-/Readback-Fallback für Fälle ohne Metal-Handle bzw. für Training mit
  Mean/Variance-Outputs

Ergebnis auf Architektur-Ebene:

- GPU-residente Pipelines brechen deutlich seltener auf
- Conv/Pool/BatchNorm folgen nun demselben Fastpath-Muster
- Readback ist von der Hot-Path-Ausführung in die Fallbacks verschoben

### Gemeinsame MPS-Stream-Infrastruktur

Die Runtime besitzt inzwischen eine zentrale Stream-Schicht, statt die
Command-Buffer-Steuerung vollständig op-lokal zu halten.

Umgesetzt wurden:

- `src/st_stream_mps.h/.m` als gemeinsame Stream-Abstraktion
- Zentraler Zugriff auf Queue, Command Buffer und Commit-Pfad
- `st_mps_stream_flush()` vor Boundary-Sync in `src/st_buffer.c`
- Konfigurierbare Commit-Frequenz über
  `MMATRIX_ST_STREAM_COMMIT_EVERY`
- Selektiver Defer-Commit für asynchrone Fastpaths über
  `MMATRIX_ST_STREAM_DEFER_ASYNC=1`
- Konservative Heuristik: Defer aktiv für `conv+bn`, deaktiviert für
  `conv+bn+pool`, weil dort Regressionen beobachtet wurden

Ergebnis auf Architektur-Ebene:

- Stream-Steuerung ist zentralisiert
- Uncommitted GPU-Arbeit wird vor Boundary-Sync deterministisch committed
- Commit-Verhalten kann jetzt gezielt kalibriert werden statt implizit zu
  variieren

### Wiederverwendbarer Metal-Buffer-Pool

Ein globaler Metal-Buffer-Pool wurde eingeführt, um wiederholte
`newBufferWithLength`-Allokationen zu vermeiden.

Umgesetzt wurden:

- Thread-sicherer Reuse-Pool in `src/st_buffer_metal.m`
- Wiederverwendung exakter Größen vor Neuallokation
- Rückgabe in den Pool auch bei async deferred release
- Schutzkorrekturen aus dem Review:
  - `NSCache`-Limits für Pooling
  - Leak-Fix bei Fehlerpfaden nach Pool-Take

Zusätzlich erfasst die Runtime jetzt Allocator-Metriken:

- `alloc_requests`
- `pool_hit`
- `new`
- `stores`
- `drops`

Die API dafür liegt in `src/st_buffer.h` / `src/st_buffer.c`.

### Benchmarks und Profiling-Basis

Die vorhandenen Benchmarks decken die wichtigsten MPS-Fragen inzwischen
brauchbar ab:

- `tests/bench_st_single_op.c` für isolierte Operatoren
- `tests/bench_st_pipeline.c` für Pipeline- und Boundary-Sync-Verhalten
- `tests/bench_st_transfer.c` für Host↔Device-Transferkosten
- `share/simple_benchmark/bench_st_pool.c` für Pool-spezifische Diagnostik

Außerdem ist der CSV-Export im Single-Op-Benchmark vorhanden:

- `BENCH_CSV=1 bazel run //app/tensor:bench_st_single_op`

Damit gibt es bereits eine reproduzierbare Maschinen-Ausgabe für
Single-Op-Messungen; der gleiche Export für weitere Benchmarks ist noch nicht
vollständig ausgerollt.

## Was erreicht wurde

### BatchNorm profitiert deutlich von GPU-Resident-Outputs

Der BatchNorm-Fastpath zeigt gegenüber dem CPU-/Readback-Pfad einen klaren und
messbaren Gewinn:

- `bn-small`: `1.205 ms/iter` statt `1.646 ms/iter`
- `bn-medium`: `6.021 ms/iter` statt `8.935 ms/iter`
- `bn-large`: `19.170 ms/iter` statt `33.447 ms/iter`

Das entspricht einem Vorteil von etwa **36 % bis 74 %** für den GPU-residenten
Pfad. Der Gewinn steigt mit der Tensorgröße, was zum erwarteten Readback-
Overhead passt.

### Pool-Pfade vermeiden den früheren strukturellen Readback-Verlust

Die Pool-Fastpaths laufen im Diagnose-Benchmark mit erfolgreichem MPS-Treffer
und ohne systematischen Fallback:

- `mps_hit = iters`
- `fallback = 0`

Wichtig ist dabei die Einordnung: Die früher beobachtete große xlarge-Abweichung
war in späteren Läufen **nicht stabil reproduzierbar**. Der aktuelle Stand
spricht eher für Laufzeit-/Queue-Effekte als für einen festen MaxPool-Defekt.

### Transfer- und Allocator-Optimierungen liefern sichtbare End-to-End-Gewinne

Die Transfer-Benchmarks zeigen einen klaren Vorteil für device-residente
Ausführung:

- `transfer-medium`: `2.305 ms/iter` statt `3.477 ms/iter`
- `transfer-large`: `17.160 ms/iter` statt `38.048 ms/iter`

Das bestätigt zwei Dinge:

- Zero-copy/Metal-residente Datenpfade lohnen sich messbar
- Der Buffer-Pool und die Vermeidung unnötiger Host↔Device-Übergänge greifen
  in der Praxis zusammen

### Matrix-GEMM profitiert erst mit GPU-residenter Ausführung

`sm` bleibt bewusst CPU-/BLAS-orientiert. Der separate Apple-Silicon-Pfad
`sm_mps` zeigt aber dieselbe Grundregel wie die Tensor-Runtime: MPS lohnt sich
erst, wenn Daten resident bleiben und Synchronisation an echte Grenzen
verschoben wird.

Der Benchmark `//share/simple_benchmark:bench_sm_mps` vergleicht aktuell:

- `oneshot`: Host-Input, pro Call MTLBuffer anlegen, GEMM, wait, Host-Output
- `resident_sync`: A/B/C als wiederverwendete `SmMpsMatrix`, aber ein Wait pro GEMM
- `resident_async_batch`: mehrere GEMMs in einen Stream encoden, ein Boundary-Wait
- `resident_async_plan`: wie async batch, aber mit wiederverwendetem GEMM-Plan

Messstand vom 2026-05-20 auf Apple Silicon:

- Kleine Matrizen (`128^3`, `256^3`) bleiben klar bei Accelerate.
- `512^3` profitiert stark von async/plan, erreicht Accelerate aber noch nicht stabil.
- `1024^3` ist die aktuelle Crossover-Zone; je nach Lauf ist async/plan gleichauf
  bis schneller.
- `2048^3` ist mit GPU-residentem async/plan klar schneller als Accelerate.

Die Counter bestätigen die Ursache: `resident_async_plan` reduziert die timed
CommandBuffer/Waits auf wenige Boundary-Operationen und die Plan-Allokationen
auf 1, während `oneshot` pro Iteration Upload, CommandBuffer, Wait und Download
bezahlt.

### Die Infrastruktur ist messbar robuster geworden

Neben den reinen Laufzeiten wurde auch die technische Basis verbessert:

- Pool, BatchNorm und Conv nutzen nun ein konsistenteres Fastpath-Modell
- Asynchrone GPU-Arbeit wird sauberer getrackt
- Allocator-Reuse ist über Counter sichtbar
- Build- und Test-Basis für die betroffenen Benchmarks ist intakt

Zuletzt verifiziert:

- `bazel test //app/tensor:test_st_batchnorm` ✅
- `bazel build //app/tensor:bench_st_single_op` ✅
- `bazel build //app/tensor:bench_st_pipeline` ✅
- `bazel build //app/tensor:bench_st_transfer` ✅

## Noch ausstehende Themen

### Commit-Strategie weiter kalibrieren

Die gemeinsame Stream-Schicht ist vorhanden, aber die optimale Commit-Politik
ist noch nicht abschließend gefunden.

Offen sind insbesondere:

- belastbare Profile wie `stable`, `balanced`, `throughput`
- systematische A/B-Auswertung pro Pipeline-Typ
- optionale Integration der Stream-Mechanik auch in weitere Fallback-Pfade

### Fastpath-Helfer weiter verallgemeinern

Der Conv-Fastpath ist leistungsstark, aber noch zu speziell. Die Extraktion
gemeinsamer Helper für

- Feed-Mapping
- preallocated outputs
- Executable-Reuse
- async commit handling

steht noch aus.

Damit würde die Einführung weiterer GPU-residenter Operatoren einfacher und
weniger dupliziert.

### Profiling weiter ausbauen

Die Allocator-Counter sind vorhanden, aber das Profiling ist noch nicht am
Ziel.

Offen sind:

- CSV-Export auch für `bench_st_pipeline` und `bench_st_transfer`
- Profiling der Command-Buffer-Commit-Zyklen
- eine noch einheitlichere Maschinen-Ausgabe über alle MPS-Benchmarks hinweg

### Custom-Metal-Kernels für Hot Paths prüfen

Der aktuelle Ansatz ist stark MPSGraph-zentriert. Das ist für viele Fälle gut,
aber nicht zwingend optimal für kleine, sehr häufige oder bandbreitenlimitierte
Operatoren.

Noch offen ist daher die Frage, welche Hot Paths von eigenen Metal-Kernels
profitieren würden und welche weiterhin besser bei MPSGraph bleiben.

### Layout-Strategie für GPU-Pipelines festziehen

Einzelne Vorarbeiten wie das NHWC-Experiment im Conv-Pfad existieren, aber eine
klare interne Layout-Strategie fehlt noch.

Offen sind:

- bevorzugtes GPU-Layout für Conv-dominierte Pipelines
- Reduktion unnötiger Transposes
- robustere Behandlung nicht-kontiguierlicher Tensoren

## Praktische Mindestprüfung für weitere MPS-Änderungen

Für jede weitere Backend-Optimierung sollten mindestens diese Fragen erneut
beantwortet werden:

1. Bleiben Daten länger GPU-resident?
2. Wurden Readbacks oder unnötige Waits reduziert?
3. Ist der Effekt in Benchmarks sichtbar?
4. Wurde bestehende Infrastruktur wiederverwendet statt neue Sonderlogik
   einzuführen?
