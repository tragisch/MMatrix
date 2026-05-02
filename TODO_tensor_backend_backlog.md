# app/tensor – Performance & Backend Reliability Backlog

## Milestone: MPS Dispatch Robustness v1

## 1) [P0] ~~Fallback-Gründe standardisieren und sichtbar machen~~ ✅ DONE
**Labels:** `tensor`, `backend`, `debug`, `observability`  
**Aufwand:** M  

Einheitliche Reason-Codes für Backend-Entscheidungen/Fallbacks einführen (z. B. `THRESHOLD`, `DTYPE_UNSUPPORTED`, `NON_CONTIGUOUS`, `NO_MPS_DEVICE`, `BACKEND_FORCED_CPU`).

### Akzeptanzkriterien
- Zentraler Enum/Code-Satz für Fallback-Gründe vorhanden.
- Für Conv/Pool/BatchNorm wird bei Fallback ein Reason-Code gesetzt.
- Optionaler Debug-Output (abschaltbar) zeigt „selected backend + reason“.

### Done when
Reason-Codes sind in Runtime-Pfaden integriert und in mindestens einem Test/Debug-Run sichtbar.

---

## 2) [P0] ~~Backend-Hit/Miss-Counter ergänzen~~ ✅ DONE
**Labels:** `tensor`, `backend`, `metrics`  
**Aufwand:** S  

Leichtgewichtige Counter einführen: `mps_hit`, `mps_miss`, `fallback_gemm`, `fallback_reference`.

### Akzeptanzkriterien
- Counter sind pro Prozesslauf abrufbar (API oder Debug-Print).
- Kein relevanter Overhead im normalen Release-Betrieb.
- Mindestens ein Test validiert, dass Counter in einem bekannten Pfad steigen.

### Done when
Counter sind im Bench/Test sichtbar und reproduzierbar.

---

## 3) [P1] ~~Benchmark-Split: Single-Op vs Pipeline~~ ✅ DONE
**Labels:** `tensor`, `benchmark`, `performance`  
**Aufwand:** M  

Benchmarks in zwei Klassen strukturieren:
1. Single-Op (isoliert)
2. Pipeline (z. B. Conv→BN→ReLU)

### Akzeptanzkriterien
- Mindestens ein Pipeline-Benchmark-Target existiert.
- Reports trennen klar zwischen Single-Op und Pipeline-Ergebnissen.
- Ergebnisse sind über mehrere Runs stabil (Varianz dokumentiert).

### Done when
Benchmark-Output enthält beide Kategorien mit vergleichbarer Darstellung.

---

## 4) [P1] ~~Transferkosten explizit messen (E2E vs device-resident)~~ ✅ DONE
**Labels:** `tensor`, `benchmark`, `mps`  
**Aufwand:** M  

Für relevante Pfade zwei Messmodi ausweisen:
- End-to-End inkl. Host↔Device
- Device-resident (ohne zusätzliche Transfers)

### Akzeptanzkriterien
- Für mindestens Conv+BN existieren beide Messmodi.
- Report enthält beide Werte nebeneinander.
- Kurze Interpretation in Doku, wann welcher Wert entscheidend ist.

### Done when
Transfereffekte sind explizit quantifiziert und dokumentiert.

---

## 5) [P1] ~~Numerik-Metriken in Performance-Reports integrieren~~ ✅ DONE
**Labels:** `tensor`, `quality`, `benchmark`  
**Aufwand:** M  

Neben Laufzeit immer Genauigkeit reporten (`max_abs`, `max_rel`, optional ULP).

### Akzeptanzkriterien
- Jede relevante Benchmarkgruppe schreibt Genauigkeitsmetriken aus.
- Toleranzen sind pro Dtype dokumentiert (`F32`, `BF16`).
- Mindestens ein Fail-Case bei zu hoher Abweichung ist testbar.

### Done when
Kein Performance-Resultat mehr ohne Accuracy-Kontext.

---

## 6) [P1] ~~Dispatch-Matrix dokumentieren (When GPU/CPU)~~ ✅ DONE
**Labels:** `tensor`, `docs`, `backend`  
**Aufwand:** S  

Pro Op eine kompakte Entscheidungs-Matrix dokumentieren (Shape, MACs, Dtype, Layout, Buffer).

### Akzeptanzkriterien
- Matrix für Conv2D, MaxPool2D, AvgPool2D, BatchNorm2D vorhanden.
- Pro Regel kurze Begründung (warum).
- Matrix referenziert reale Bench-Ergebnisse.

### Done when
Dokumentation ermöglicht reproduzierbare Backend-Entscheidungen.

---

## 7) [P2] ~~Anti-Patterns pro Op dokumentieren~~ ✅ DONE
**Labels:** `tensor`, `docs`, `performance`  
**Aufwand:** S  

„Nicht auf MPS gehen wenn …“-Regeln ergänzen (kleine Shapes, ungünstige Layouts etc.).

### Akzeptanzkriterien
- Mindestens 2–3 Anti-Patterns pro Kern-Op.
- Jeder Anti-Pattern hat eine Bench-/Mess-Referenz.
- Klar getrennt von allgemeinen Empfehlungen.

### Done when
Fehlentscheidungen im Alltag werden durch Doku sichtbar reduziert.

---

## 8) [P2] ~~BF16-Policy festziehen (native vs Promotion)~~ ✅ DONE
**Labels:** `tensor`, `dtype`, `quality`  
**Aufwand:** M  

Pro Op verbindlich definieren: native BF16 oder interne F32-Promotion inkl. Rückkonvertierung.

### Akzeptanzkriterien
- Schriftliche Policy vorhanden.
- Tests decken beide Pfade ab.
- Genauigkeits- und Performance-Tradeoffs dokumentiert.

### Done when
BF16-Verhalten ist konsistent, testbar und transparent.

---

## 9) [P2] ~~Shape-aware MPSGraph Warmup (optional)~~ ✅ DONE
**Labels:** `tensor`, `mps`, `latency`  
**Aufwand:** M  

Optionales Warmup für häufige Shape-Signaturen zur Reduktion der First-Run-Latenz.

### Akzeptanzkriterien
- Warmup ist per Flag/Config aktivierbar.
- Erste Ausführung bei häufigen Shapes wird messbar schneller.
- Kein negativer Effekt auf Standardpfad ohne Warmup.

### Done when
Warmup liefert in mindestens einem praxisnahen Szenario klaren Vorteil.

---

## 10) [P2] ~~Reproduzierbares Threshold-Tuning etablieren~~ ✅ DONE
**Labels:** `tensor`, `tuning`, `performance`  
**Aufwand:** S  

Kurze Prozedur für `MMATRIX_ST_CONV_MPS_MACS_THRESHOLD` und `MMATRIX_ST_CONV_MPS_OUT_ELEMS_THRESHOLD` erstellen.

### Akzeptanzkriterien
- Schritt-für-Schritt-Guide vorhanden.
- Mindestens ein validiertes Preset pro Hardwareklasse.
- Regression-Check gegen Referenz-Bench integriert.

### Done when
Threshold-Anpassungen sind reproduzierbar statt Trial & Error.

---

## Empfohlene Umsetzungsreihenfolge
1. Fallback-Gründe
2. Counter
3. Benchmark-Split
4. Transferkosten
5. Numerik-Metriken
6. Dispatch-Matrix
7. BF16-Policy
8. Threshold-Tuning
9. Anti-Patterns
10. Warmup

---

## Definition of Done (Milestone gesamt)
- Backend-Entscheidungen sind transparent (Reason-Codes + Counter).
- Performance-Messungen sind aussagekräftig (Single-Op + Pipeline + Transferkontext).
- Numerik ist systematisch abgesichert (Accuracy-Metriken + Dtype-Policy).
- Doku erlaubt reproduzierbare Entscheidungen und Tuning.
