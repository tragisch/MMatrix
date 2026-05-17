# MPS Performance Guide

## Ziel

Dieses Dokument uebersetzt die aktuelle Analyse des `tensor`-Runtimes in eine
klare technische Richtung fuer Apple-Silicon-Performance.

Leitidee:

> `tensor` soll sich von einer Sammlung schneller Einzeloperatoren zu einer
> kleinen GPU-Execution-Engine entwickeln.

## Performance Contract

Fuer neue oder geaenderte MPS-Pfade gelten diese Regeln:

- GPU-Daten sollen moeglichst auf der GPU bleiben.
- `readBytes` nur an API-Grenzen oder fuer explizite Host-Ausgaben.
- `waitUntilCompleted` nicht implizit pro Operator, sondern nur an klaren
  Synchronisationspunkten.
- Graphen, Buffer und Command Buffers sollen wiederverwendet werden.
- Backend-Aenderungen muessen ueber bestehende Benchmarks messbar belegt werden.

## Aktueller Stand im Repo

Die Runtime hat bereits mehrere wichtige Bausteine:

- Zero-copy Metal-Buffer mit `MTLResourceStorageModeShared`
  (`src/st_buffer_metal.m`)
- MPSGraph-Caches fuer Conv, Pool und BatchNorm
  (`src/st_backend_mps.m`, `src/st_mps.m`)
- Async-Tracking ausstehender GPU-Arbeit pro Buffer
  (`src/st_buffer.c`)
- Conv-Fastpath mit vorallokiertem GPU-Output und kompiliertem
  `MPSGraphExecutable` (`src/st_backend_mps.m`)
- Explizite Fusionspfade fuer
  - Conv + BatchNorm
  - Conv + BatchNorm + Pool

Diese Basis ist gut. Der Unterschied zu PyTorch liegt nicht in fehlender
Grundidee, sondern in fehlender Verallgemeinerung der Infrastruktur.

## Unterschied zum PyTorch-Ansatz

PyTorch optimiert MPS nicht primär durch "eine magische Kernel-Implementierung",
sondern durch systematische Overhead-Reduktion:

- globaler Caching-Allocator statt direkter Buffer-Allokation
- gemeinsames Stream-/Command-Buffer-Modell statt op-lokaler Steuerung
- MPSGraph **und** Custom-Metal-Kernels je nach Workload
- GPU-residente Tensorpfade ueber viele Operatoren hinweg
- Spezialisierung fuer Layout, DType, Dense/Strided-Faelle

`tensor` ist aktuell bewusst schlanker:

- wenige, gezielt optimierte Ops
- explizite Fusion
- threshold-basierte Backendwahl
- MPSGraph-zentrierte GPU-Ausfuehrung

Das ist kein Fehler. Die naechste Ausbaustufe sollte aber die gemeinsame
Execution-Infrastruktur staerken.

## Hauptluecken mit hohem Performance-Impact

### 1. Pool und BatchNorm lesen oft noch auf die CPU zurueck

Der groesste konkrete Verlust ist derzeit nicht Conv, sondern das Ende von
Pool-/BatchNorm-Pfaden, die haeufig ueber `readBytes(...)` abschliessen
(`src/st_mps.m`).

Folge:

- GPU-Resident-Pipelines werden unterbrochen
- CPU/GPU-Synchronisation steigt
- Fusionsvorteile schrumpfen

Ziel:

- dieselbe preallocated-output-Fastpath-Idee wie bei Conv auch fuer Pool und
  BatchNorm anwenden

### 2. Kein globaler Metal-Buffer-Pool

Aktuell werden Metal-Buffer direkt erzeugt und ueber async release verwaltet
(`src/st_buffer_metal.m`).

Das ist funktional, aber noch kein echter Caching-Allocator.

Ziel:

- Buffer-Reuse statt wiederholter `newBufferWithLength`
- kleine und grosse Freilisten
- Rueckgabe erst nach Abschluss ausstehender Command Buffers

### 3. Kein gemeinsames MPS-Stream-Modell

Aktuell wird GPU-Arbeit sinnvoll getrackt, aber eher aus Sicht einzelner Buffer
und Operatoren.

Ziel:

- gemeinsamer Laufzeitkontext fuer
  - Command Queue
  - aktuelles Command Buffer
  - Commit-Strategie
  - Boundary-Sync

### 4. Zu starke Abhaengigkeit von MPSGraph fuer alles

MPSGraph ist praktisch, aber nicht fuer jeden Hot Path optimal.

Ziel:

- MPSGraph fuer strukturierte groessere Ops
- Custom-Metal-Kernels fuer einfache, haeufige, bandbreitenlimitierte oder
  kleine Operatoren

### 5. Layout- und Stride-Spezialisierung ist noch punktuell

Es gibt bereits ein NHWC-Experiment im Conv-Pfad, aber noch keine allgemeine
Strategie fuer interne Execution-Layouts.

Ziel:

- klares bevorzugtes GPU-Layout fuer Conv-intensive Pfade
- moeglichst wenige wiederholte Transposes
- mittelfristig robustere Behandlung nicht-kontiguierlicher Tensors

## Priorisierte Roadmap

### Phase 1: groesste Flaschenhaelse entfernen

1. ✅ Pool ohne `readBytes`-Fallback fuer GPU-Outputs
   - `mps_maxpool2d_forward_preallocated()` implementiert (async, kein readBytes)
   - `mps_avgpool2d_forward_preallocated()` implementiert (async, kein readBytes)
   - Fallback auf `readBytes` wenn Metal-Handle nicht verfuegbar
   - Tests: alle 14 test_st_pool Tests bestehen
   - Benchmark messbar: xlarge config 168ms MaxPool / 45ms AvgPool
   - Hinweis: Die xlarge-MaxPool/AvgPool-Divergenz ist ein diagnostisches
     Signal fuer verbleibenden Overhead (moeglich: Sync- oder
     Executable-Cache-Kosten) und priorisiert Phase 1.1 (Allocator/Stream).

2. ⚠️ BatchNorm ohne `readBytes`-Fallback fuer GPU-Outputs
   - Fallback-only implementiert (vtable wrapper nutzt st_batchnorm2d_mps)
   - Preallocated Pfad deferred (MPSGraph Selector-Komplikationen)
   - Funktional aber noch nicht GPU-resident optimiert

3. ✅ Benchmarks erweitern, damit `readbytes` und boundary sync sichtbar werden
   - `share/simple_benchmark/bench_st_pool.c` erstellt
   - BUILD target hinzugefuegt
   - Baseline-Messungen fuer Pool-Operationen verfuegbar

### Phase 2: Infrastruktur ausbauen

4. Einfachen Metal-Buffer-Pool einfuehren
5. Reuse-Statistiken fuer Buffer-Allokation erfassen
6. Gemeinsame MPS-Stream-Abstraktion einfuehren

### Phase 3: breitere Optimierung

7. Conv-Fastpath-Muster fuer weitere Ops verallgemeinern
8. einfache Custom-Metal-Kernels fuer Hot Ops einfuehren
9. Layout-Strategie fuer GPU-Pipelines festlegen

## Definition of Done fuer Backend-Optimierungen

Eine MPS-Optimierung ist erst dann "fertig", wenn sie:

- funktional korrekt ist
- keine implizite pro-Op-Synchronisation einfuehrt
- GPU-Resident-Pfade verbessert oder mindestens nicht verschlechtert
- in Benchmarks sichtbar nachvollziehbar ist

Mindestens relevante Messsignale:

- `fastpath_hit`
- `conv_readbytes`
- boundary-sync-only Zeit aus `tests/bench_st_pipeline.c`
- Single-op Zeit aus `tests/bench_st_single_op.c`
- Allocator-Reuse-Metriken:
  - `alloc_requests`
  - `pool_hit`
  - `new`
  - `stores`
  - `drops`

## Konkrete Issue-Liste

### Issue 1: MPS Pool output without readback ✅ ERLEDIGT

**Titel**
`MPS: add GPU-resident output fastpath for maxpool/avgpool`

**Status**
✅ Implementiert und getestet (17. Mai 2026)

**Lösung**
- `mps_maxpool2d_forward_preallocated()` in st_backend_mps.m
- `mps_avgpool2d_forward_preallocated()` in st_backend_mps.m
- Async execution (kein waitUntilCompleted)
- Fallback zu readBytes wenn Metal-Handle nicht vorhanden
- Tests bestehen: 14/14 in test_st_pool
- Benchmark verfuegbar: bench_st_pool

**Problem**
Pool-Operatoren in `src/st_mps.m` enden aktuell ueber `readBytes(...)` und
brechen GPU-residente Pipelines auf.

**Ziel**
Preallocated GPU-Output fuer Pool-Pfade unterstuetzen, analog zum Conv-Fastpath.

**Betroffene Dateien**

- ✅ `src/st_backend_mps.m` (mps_maxpool2d_forward_preallocated, mps_avgpool2d_forward_preallocated)
- ✅ `src/st_mps.m` (fallback path)
- ✅ `src/st_buffer.c` (st_buffer_track_pending_cmd)
- ✅ `src/st_buffer_metal.m` (st_buffer_metal_handle)

**Messkriterium**

- ✅ weniger oder keine Pool-bedingten Readbacks in Pipeline-Benchmarks
- ✅ bessere Zeit bei boundary-sync-only
- ✅ bench_st_pool zeigt konsistente Performance
- 📌 xlarge-Anomalie dokumentiert: 168ms MaxPool vs 45ms AvgPool als
  Diagnose-Hinweis fuer verbleibenden Runtime-Overhead

### Issue 2: MPS BatchNorm output without readback ⚠️ TEILWEISE ERLEDIGT

**Titel**
`MPS: keep batchnorm outputs GPU-resident`

**Status**
⚠️ Fallback-only implementiert (17. Mai 2026). Preallocated Pfad deferred wegen
MPSGraph Selector-Komplikationen (rsqrtTensor, averagePooling2D nicht verfuegbar
oder unterschiedlich benannt).

**Ziel**
GPU-resident Output, optional getrennt von mean/variance-Readback im
Inference-Pfad.

**Betroffene Dateien**

- ✅ `src/st_backend_mps.m` (fallback-only wrapper)
- ⚠️ `src/st_mps.m` (Fallback-Pfad bleibt, aber kein preallocated)

**Nächste Schritte**
- Untersuche korrekte MPSGraph Selectors fuer Custom-Graph
- Oder: Nutze existierende MPSGraph Ops wenn verfuegbar
- Oder: Behalte Fallback-only bis gewinn hoch genug ist

**Messkriterium**

- weniger Readbacks (noch nicht gemessen)
- bessere fused-pipeline Zeit fuer Conv+BN (noch nicht gemessen)

### Issue 3: Introduce shared MPS stream layer 📋 NICHT GESTARTET

**Titel**
`MPS: introduce shared stream and command-buffer abstraction`

**Problem**
Command-Buffer-Lebenszeit und Synchronisation sind noch zu stark op-lokal.

**Ziel**
Eine gemeinsame Schicht fuer Queue, aktuelles Command Buffer, Commit und
Boundary-Sync schaffen.

**Betroffene Dateien**

- neues `src/st_stream_mps.*` oder aehnlich
- `src/st_backend_mps.m`
- `src/st_mps.m`
- `src/st_buffer.c`

**Messkriterium**

- weniger explizite Waits
- stabilere Pipeline-Laufzeiten

### Issue 4: Add pooled Metal allocator 🚧 GESTARTET

**Titel**
`MPS: add reusable Metal buffer pool`

**Status**
🚧 Angefangen (17. Mai 2026): einfacher globaler MTLBuffer-Reuse-Pool mit
exakter Groessenwiederverwendung in `st_buffer_metal.m`.

- Reuse beim Allocate-Pfad vor `newBufferWithLength`
- Rueckgabe in den Pool bei normalem Release und async deferred release
- Pool begrenzt (max. Slots und max. Buffer-Groesse), um Speicherwachstum zu
  deckeln

Noch offen:
- Feineres Klassieren (z. B. size classes statt exakter Groesse)
- Reuse-Hit-Rate in gezieltem Allocator-Microbenchmark validieren

**Problem**
Direkte Buffer-Allokation ist teurer als noetig und amortisiert sich schlecht
ueber viele Operatoren.

**Ziel**
Ein einfacher Caching-Allocator mit Reuse einbauen.

**Betroffene Dateien**

- `src/st_buffer.c`
- `src/st_buffer.h`
- `src/st_buffer_metal.m`

**Messkriterium**

- sinkende Zahl direkter Neuallokationen
- bessere Laufzeit in wiederholten Benchmark-Loops

### Issue 5: Generalize executable fastpath helpers 📋 NICHT GESTARTET

**Titel**
`MPS: extract reusable executable + preallocated-output helper`

**Problem**
Der Conv-Fastpath ist leistungsstark, aber noch zu speziell implementiert.

**Ziel**
Hilfslogik fuer Feed-Mapping, preallocated outputs und async commit extrahieren,
um sie fuer weitere Ops wiederzuverwenden.

**Betroffene Dateien**

- `src/st_backend_mps.m`
- `src/st_mps.m`

**Messkriterium**

- weniger duplizierte Fastpath-Logik
- schnellere Einfuehrung weiterer GPU-resident Ops

### Issue 6: Add allocator and execution profiling 📋 NICHT GESTARTET

**Titel**
`MPS: add allocator reuse and command-buffer profiling counters`

**Status**
🚧 Gestartet (17. Mai 2026): erste Allocator-Reuse-Counter implementiert und
in `bench_st_transfer` ausgegeben.

- API: `st_buffer_metal_allocator_stats_get/reset`
- Signale: requests, pool_hit, new, stores, drops
- Naechster Schritt: CSV-orientierte Ausgabe in weiteren Benchmarks

## Laufendes Fortschritts- und Benchmark-Log

Ab jetzt wird jede relevante MPS-Infrastruktur-Aenderung hier mit Datum,
Kurzbeschreibung und Benchmark-Deltas nachgetragen.

### Vorlage fuer neue Eintraege (copy/paste)

```md
### YYYY-MM-DD — Schritt N: <Kurzname>

**Aenderung**
- <Codepfad/Funktionsname + was geaendert wurde>
- <weitere relevante Aenderung>

**Benchmark (`//app/tensor:<target>` )**
- <case-1>: metal `<x.xxx ms/iter>`, cpu `<y.yyy ms/iter>`
- <case-2>: metal `<x.xxx ms/iter>`, cpu `<y.yyy ms/iter>`

**Zusammenfassung der Benchmark-Ergebnisse**
- <kurzes Fazit in 1-2 Bulletpoints>
- <z. B. "metal schneller als cpu in beiden Cases" oder "gemischtes Bild">

**Delta vs. vorherigem Schritt**
- <case-1>: metal `<+/-x.xx%>`, cpu `<+/-y.yy%>`
- <case-2>: metal `<+/-x.xx%>`, cpu `<+/-y.yy%>`

**Counter-Ausgabe (falls vorhanden)**
- <case-1>: `<counter_a=... counter_b=...>`
- <case-2>: `<counter_a=... counter_b=...>`

**Einordnung**
- <1-2 Saetze zu Aussagekraft / Messstreuung / naechstem Schritt>
```

### 2026-05-17 — Schritt 1: Globaler Metal-Buffer-Pool

**Aenderung**
- einfacher globaler, thread-sicherer MTLBuffer-Reuse-Pool in
  `src/st_buffer_metal.m`
- Reuse vor `newBufferWithLength`
- Rueckgabe bei normalem und async deferred release

**Benchmark (`//app/tensor:bench_st_transfer`)**
- transfer-medium: metal `2.303 ms/iter`, cpu `3.462 ms/iter`
- transfer-large: metal `16.831 ms/iter`, cpu `38.799 ms/iter`

**Zusammenfassung der Benchmark-Ergebnisse**
- Metal ist in beiden Cases schneller als CPU (zero-copy Vorteil sichtbar).
- Der Abstand ist bei `transfer-large` deutlich groesser als bei
  `transfer-medium`.

### 2026-05-17 — Schritt 2: Allocator-Counter + Sichtbarkeit im Benchmark

**Aenderung**
- API in `src/st_buffer.h` + Wrapper in `src/st_buffer.c`:
  `st_buffer_metal_allocator_stats_get/reset`
- Counter in `src/st_buffer_metal.m`:
  `alloc_requests`, `pool_hit`, `new`, `stores`, `drops`
- Ausgabe in `tests/bench_st_transfer.c`

**Benchmark (`//app/tensor:bench_st_transfer`)**
- transfer-medium: metal `2.235 ms/iter`, cpu `3.177 ms/iter`
- transfer-large: metal `17.441 ms/iter`, cpu `36.598 ms/iter`

**Zusammenfassung der Benchmark-Ergebnisse**
- Metal bleibt in beiden Cases schneller als CPU.
- Gegenueber Schritt 1 ergibt sich ein gemischtes Bild:
  `transfer-medium` verbessert, `transfer-large` leicht schlechter.

**Delta vs. Schritt 1**
- transfer-medium: metal `-2.95%`, cpu `-8.23%`
- transfer-large: metal `+3.62%`, cpu `-5.67%`

**Neue Counter-Ausgabe (aktueller Lauf)**
- medium: `requests=8, pool_hit=0, new=8, stores=0, drops=0`
- large: `requests=16, pool_hit=0, new=16, stores=8, drops=0`

Hinweis: gemischte Laufzeitdeltas zwischen zwei Einzelruns sind erwartbar.
Fuer belastbare Aussagen Mehrfachlaeufe mit identischem Setup verwenden.

**Problem**
Es gibt bereits gute Dispatch-Counter, aber noch keine vollstaendige Sicht auf
Allocation-Reuse und Command-Buffer-Verhalten.

**Ziel**
Metriken fuer:

- Buffer-Reuse
- Neuallokation
- Pending depth
- Commit/Wait-Verhalten

**Betroffene Dateien**

- `src/st_buffer.c`
- `src/st_backend.c`
- Benchmarks unter `tests/`

**Messkriterium**

- reproduzierbare Vorher/Nachher-Messung fuer Infrastruktur-Aenderungen

## Benchmark-Empfehlung

Fuer jede MPS-Aenderung mindestens ausfuehren:

- `tests/bench_st_single_op.c`
- `tests/bench_st_pipeline.c`

Und fuer GPU-Pfade zusaetzlich protokollieren:

- `mps_hit`
- `mps_miss`
- `conv_readbytes`
- `fastpath_hit`
- neue Allocator-/Stream-Counter, sobald vorhanden

## Kurzfassung fuer Pull Requests

Wenn eine Aenderung die MPS-Performance betrifft, sollte die PR-Beschreibung
mindestens beantworten:

1. Bleiben Daten laenger GPU-resident?
2. Wurden Readbacks oder Waits reduziert?
3. Wird vorhandene Infrastruktur wiederverwendet statt neuer Speziallogik?
4. Welche Benchmarks zeigen den Effekt?
