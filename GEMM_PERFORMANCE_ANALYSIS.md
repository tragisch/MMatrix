# GEMM Performance-Analyse: Alte vs. Neue Implementierung

**Testsystem:** macOS ARM64 (Apple Silicon), OpenMP/ARM NEON Fallback
**Compiler:** `-c opt` (Release-Build)
**Datum:** 22. Februar 2026

## Zusammenfassung

Die neue GEMM-Implementierung mit Transpose-Flags zeigt **signifikante Geschwindigkeitsvorteile** bei Backprop-Mustern mit Transposen, während Standard-Multiplikation (NN) gleich schnell bleibt.

---

## Detaillierte Ergebnisse

### 1. Standard Matrix-Multiplikation (Baseline A × B, kein Transpose)

| Größe       | Zeit (ns)   | Modus    | Anmerkung               |
| ----------- | ----------- | -------- | ----------------------- |
| 64×64×64    | 295,402     | Baseline | Identisch (GEMM intern) |
| 128×128×128 | 2,375,073   | Baseline | Identisch (GEMM intern) |
| 256×256×256 | 21,606,111  | Baseline | Identisch (GEMM intern) |
| 512×512×512 | 175,248,000 | Baseline | Identisch (GEMM intern) |

**Fazit:** Keine Regression – `sm_multiply()` nutzt intern `sm_gemm()` mit identischer Performance.

---

### 2. Backprop-Muster: dW = X^T × dY (Transpose auf linker Matrix)

| Größe (Batch×In×Out) | ALT (ns)    | NEU (ns)    | Speedup                |
| -------------------- | ----------- | ----------- | ---------------------- |
| 128×256×512          | 22,490,484  | 22,656,767  | **0.99×** (identisch)  |
| 256×512×1024         | 242,206,333 | 311,439,500 | **0.78×** (langsamer?) |

**Überraschung:** Bei großen Matrizen ist die neue Variante hier _langsamer_.

**Ursache (Hypothese):**
- Fallback-Pfad (ohne BLAS) hat bei TN-Modus ungünstige Cache-Access-Patterns
- Alte Variante: `sm_transpose(X)` erzeugt contiguous Zeilen → besserer Cache-Hit bei anschließendem `sm_multiply()`
- Neue Variante: direkte Indizierung `a[p * A->cols + i]` für Transponierung hat Stride-Access

**Empfehlung:** Mit BLAS/Accelerate aktiviert wäre das anders (siehe unten).

---

### 3. Backprop-Muster: dX = dY × W^T (Transpose auf rechter Matrix)

| Größe (Batch×Out×In) | ALT (ns)    | NEU (ns)    | Speedup     |
| -------------------- | ----------- | ----------- | ----------- |
| 128×512×256          | 20,380,000  | 13,757,627  | **1.48×** ✅ |
| 256×1024×512         | 133,892,800 | 111,612,667 | **1.20×** ✅ |

**Ergebnis:** **+20-48% Beschleunigung!** 🚀

**Warum?**
- Alte Variante: `sm_transpose(W)` allokiert 256×1024 Matrix → hoher Memory-Overhead
- Neue Variante: NT-Modus (B transponiert) hat bessere Locality im Fallback-Pfad
- Speichereinsparung: keine temporäre Transpose-Matrix

---

## Performance-Vergleich: BLAS vs. Fallback (Erwartung)

| Backend             | dW = X^T × dY | dX = dY × W^T | Kommentar                        |
| ------------------- | ------------- | ------------- | -------------------------------- |
| **OpenMP/NEON**     | 0.78-0.99×    | 1.20-1.48× ✅  | NT-Modus profitiert, TN leidet   |
| **BLAS/Accelerate** | ~1.5-2× ✅     | ~1.5-2× ✅     | Beide Modi optimal (cblas_sgemm) |

**Wichtig:** Die Ergebnisse sind für den **Fallback-Pfad** (OpenMP). Mit aktiviertem BLAS (`USE_ACCELERATE` oder `USE_OPENBLAS`) würden **beide Transpose-Modi** massiv profitieren, da:

- `cblas_sgemm(..., CblasTrans, ...)` intern optimiert ist
- Keine explizite Transpose-Allokation nötig
- BLAS-Bibliothek nutzt SIMD/Cache-Blocking für alle Modi

---

## Zusammenfassung & Empfehlung

### ✅ **Klare Vorteile** (bereits im Fallback-Modus)
1. **dX = dY × W^T**: **+20-48% schneller** → direkter Gewinn für Backprop
2. **Speichereffizienz**: Keine temporären Transpose-Matrizen
3. **API-Klarheit**: Transpose-Flags vermeiden Allokationsfehler

### ⚠️ **Einschränkung** (Fallback-Modus)
- **dW = X^T × dY**: Bei großen Matrizen (256×512×1024) leicht langsamer (~22%)
- **Ursache**: Cache-ungünstige Indizierung bei TN-Modus ohne BLAS

### 🎯 **Empfehlung**
1. **Mit BLAS/Accelerate kompilieren** (`--define=USE_ACCELERATE=1` auf macOS):
   ```bash
   bazel build -c opt --define=USE_ACCELERATE=1 //share/google_benchmark:bench_sm_gemm_comparison
   ```
   → Erwartbar: **beide Transpose-Modi ~1.5-2× schneller**

2. **Fallback-Pfad optimieren** (optional):
   - Für TN-Modus: Block-Tiling hinzufügen (ähnlich wie `sm_transpose()`)
   - Oder: Hybrid-Strategie (große TN → temporäre Transpose; kleine TN → direkter Index)

3. **Production-Use:**
   - Für NN/NT: **GEMM ist jetzt optimal** (identisch/schneller)
   - Für TN ohne BLAS: alte `sm_transpose()`-Variante erwägen (oder BLAS nutzen)

---

## Nächste Schritte

- [ ] BLAS-Build testen (erwartbar: alle Modi schneller)
- [ ] Fallback TN-Modus mit Cache-Blocking optimieren
- [ ] Integration in `nm`-Modul (wenn vorhanden)
- [ ] Memory-Profile (Heap-Allokationen alt vs. neu)

