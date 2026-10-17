#ifndef SIMPLE_BENCH_H
#define SIMPLE_BENCH_H

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

/*****************************
 *  Compiler barrier
 *
 *  Prevents the compiler from optimizing away the benchmarked code.
 *  Usage:  BENCH_DO_NOT_OPTIMIZE(result);
 *******************************/

#define BENCH_DO_NOT_OPTIMIZE(val)                     \
  __asm__ volatile("" : : "r,m"(val) : "memory")

#define BENCH_CLOBBER_MEMORY() __asm__ volatile("" ::: "memory")

/*****************************
 *  Color codes
 *******************************/

#define BENCH_BLUE "\x1B[34m"
#define BENCH_RED "\x1B[31m"
#define BENCH_GREEN "\x1B[32m"
#define BENCH_WHITE "\x1B[37m"
#define BENCH_RESET "\x1B[0m"

/* backward compat */
#define BLUE BENCH_BLUE
#define RED BENCH_RED
#define WHITE BENCH_WHITE
#define RESET BENCH_RESET

/*****************************
 *  Statistics helpers
 *******************************/

typedef struct {
  double min_ns;
  double max_ns;
  double mean_ns;
  double median_ns;
  double stddev_ns;
  int n;
} bench_stats_t;

static inline int bench_cmp_ll_(const void* a, const void* b) {
  long long va = *(const long long*)a;
  long long vb = *(const long long*)b;
  return (va > vb) - (va < vb);
}

/* Compute statistics from an array of nanosecond timings. */
static inline bench_stats_t bench_compute_stats(long long* samples, int n) {
  bench_stats_t s = {0};
  if (n <= 0) return s;

  s.n = n;

  /* sort for median */
  qsort(samples, (size_t)n, sizeof(long long), bench_cmp_ll_);

  s.min_ns = (double)samples[0];
  s.max_ns = (double)samples[n - 1];

  if (n % 2 == 1) {
    s.median_ns = (double)samples[n / 2];
  } else {
    s.median_ns = (double)(samples[n / 2 - 1] + samples[n / 2]) / 2.0;
  }

  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += (double)samples[i];
  s.mean_ns = sum / n;

  if (n > 1) {
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
      double d = (double)samples[i] - s.mean_ns;
      var += d * d;
    }
    s.stddev_ns = sqrt(var / (n - 1));
  }

  return s;
}

/*****************************
 *  Formatting helpers
 *******************************/

/* Human-readable time format with fractional precision. */
static inline void format_duration(long long nanoseconds, char* buffer,
                                   size_t buffer_size) {
  const double ns = (double)nanoseconds;
  const double us = ns / 1e3;
  const double ms = ns / 1e6;
  const double sec = ns / 1e9;

  if (ns < 1e3) {
    snprintf(buffer, buffer_size, "%lld ns", nanoseconds);
  } else if (ns < 1e6) {
    snprintf(buffer, buffer_size, "%.2f us", us);
  } else if (ns < 1e9) {
    snprintf(buffer, buffer_size, "%.2f ms", ms);
  } else {
    long long minutes = (long long)(sec / 60.0);
    double remaining_sec = sec - minutes * 60.0;
    if (minutes > 0) {
      snprintf(buffer, buffer_size, "%lldm %.1fs", minutes, remaining_sec);
    } else {
      snprintf(buffer, buffer_size, "%.2f s", sec);
    }
  }
}

/* Human-readable memory format. */
static inline void format_memory_usage(long memory_usage, char* buffer,
                                       size_t buffer_size) {
  const long KB = 1024;
  const long MB = 1024 * KB;

#ifdef __linux__
  /* On Linux, ru_maxrss is in kilobytes */
  long memory_kilobytes = memory_usage;
#else
  /* On BSD and macOS, ru_maxrss is in bytes */
  long memory_kilobytes = memory_usage / KB;
#endif

  if (memory_kilobytes < KB) {
    snprintf(buffer, buffer_size, "%ld KB", memory_kilobytes);
  } else if (memory_kilobytes < MB) {
    snprintf(buffer, buffer_size, "%.2f MB", memory_kilobytes / (double)KB);
  } else {
    snprintf(buffer, buffer_size, "%.2f GB", memory_kilobytes / (double)MB);
  }
}

/*****************************
 *  Output helpers
 *******************************/

static inline void print_benchmark_result(const char* formatted_time,
                                          const long long memory,
                                          const char* fcall, const char* file,
                                          int line) {
  char formatted_memory[64];
  format_memory_usage(memory, formatted_memory, sizeof(formatted_memory));
  printf(BENCH_BLUE "ℬ|" BENCH_RESET " Wall-Time: " BENCH_RED
                    "%s\t" BENCH_RESET "Peak-RSS:" BENCH_RED
                    " %s\t" BENCH_RESET "| %s in " BENCH_WHITE " %s:%i\n",
         formatted_time, formatted_memory, fcall, file, line);
}

static inline void print_benchmark_stats(const char* label,
                                         const bench_stats_t* s,
                                         const long peak_rss) {
  char fmt_min[64], fmt_med[64], fmt_mean[64], fmt_std[64], fmt_mem[64];
  format_duration((long long)s->min_ns, fmt_min, sizeof(fmt_min));
  format_duration((long long)s->median_ns, fmt_med, sizeof(fmt_med));
  format_duration((long long)s->mean_ns, fmt_mean, sizeof(fmt_mean));
  format_duration((long long)s->stddev_ns, fmt_std, sizeof(fmt_std));
  format_memory_usage(peak_rss, fmt_mem, sizeof(fmt_mem));

  printf(BENCH_BLUE "ℬ|" BENCH_RESET " %s  " BENCH_GREEN "[%d runs]"
         BENCH_RESET "\n", label, s->n);
  printf("    min: " BENCH_RED "%s" BENCH_RESET
         "  median: " BENCH_RED "%s" BENCH_RESET
         "  mean: " BENCH_RED "%s" BENCH_RESET
         "  stddev: %s  Peak-RSS: %s\n",
         fmt_min, fmt_med, fmt_mean, fmt_std, fmt_mem);
}

static inline void update_countdown(int remaining) {
  printf("\r" BENCH_BLUE "ℬ|" BENCH_RESET " %i ", remaining);
  fflush(stdout);
}

static inline void clear_line(void) { printf("\r\033[K"); }

/*****************************
 *  Memory measurement
 *******************************/

static inline long get_memory_usage(void) {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_maxrss; /* peak RSS (bytes on macOS, KB on Linux) */
}

/*****************************
 *  File output
 *******************************/

/* Write a benchmark result line to a TSV file.
 * Renamed from write_double_to_file for clarity. */
static inline void bench_write_result(const char* filename,
                                      const char* function_name,
                                      const char* time_usage,
                                      const char* memory_usage,
                                      const char* comment) {
  FILE* file = fopen(filename, "a");
  if (file == NULL) {
    perror("fopen");
    exit(1);
  }

  struct utsname sysinfo;
  if (uname(&sysinfo) != 0) {
    strcpy(sysinfo.sysname, "UnknownOS");
  }

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    strcpy(hostname, "UnknownHost");
  }

  time_t tm = 0;
  time(&tm);
  char* timestr = ctime(&tm);
  /* ctime() appends a newline — strip it */
  if (timestr) {
    size_t len = strlen(timestr);
    if (len > 0 && timestr[len - 1] == '\n') {
      timestr[len - 1] = '\0';
    }
  }

  /* Columns: Function | Time | Memory | Comment | OS | Host | Timestamp */
  fprintf(file, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", function_name, time_usage,
          memory_usage, comment, sysinfo.sysname, hostname,
          timestr ? timestr : "unknown");
  fclose(file);
}

/* Backward-compatible alias */
static inline void write_double_to_file(char* filename,
                                        const char* function_name,
                                        const char* time_usage,
                                        const char* memory_usage,
                                        const char* comment) {
  bench_write_result(filename, function_name, time_usage, memory_usage,
                     comment);
}

/* ---- JSON string escaping (minimal, no alloc) ---- */
static inline void bench_json_escape_(const char* src, char* dst,
                                      size_t dst_size) {
  size_t j = 0;
  for (size_t i = 0; src[i] && j + 6 < dst_size; ++i) {
    switch (src[i]) {
      case '"':  dst[j++] = '\\'; dst[j++] = '"';  break;
      case '\\': dst[j++] = '\\'; dst[j++] = '\\'; break;
      case '\n': dst[j++] = '\\'; dst[j++] = 'n';  break;
      case '\r': dst[j++] = '\\'; dst[j++] = 'r';  break;
      case '\t': dst[j++] = '\\'; dst[j++] = 't';  break;
      default:   dst[j++] = src[i]; break;
    }
  }
  dst[j] = '\0';
}

/* Write a single benchmark result as one JSON line (.jsonl).
 *
 * Stores both the raw nanosecond value (for machine processing) and a
 * human-readable formatted string.  Appends one line per call.
 *
 * Usage:
 *   bench_write_jsonl("results.jsonl",
 *                     "dm_multiply (1000x1000)", elapsed_ns,
 *                     peak_rss, "OpenBLAS");
 */
static inline void bench_write_jsonl(const char* filename,
                                     const char* function_name,
                                     long long wall_ns,
                                     long peak_rss,
                                     const char* comment) {
  FILE* file = fopen(filename, "a");
  if (file == NULL) {
    perror("fopen");
    exit(1);
  }

  struct utsname sysinfo;
  if (uname(&sysinfo) != 0) {
    strcpy(sysinfo.sysname, "UnknownOS");
  }

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    strcpy(hostname, "UnknownHost");
  }

  /* ISO 8601 timestamp */
  time_t tm = 0;
  time(&tm);
  char iso_ts[64];
  strftime(iso_ts, sizeof(iso_ts), "%Y-%m-%dT%H:%M:%S%z", localtime(&tm));

  /* Human-readable strings */
  char fmt_time[64], fmt_mem[64];
  format_duration(wall_ns, fmt_time, sizeof(fmt_time));
  format_memory_usage(peak_rss, fmt_mem, sizeof(fmt_mem));

  /* Escape strings that may contain special characters */
  char esc_func[256], esc_comment[256];
  bench_json_escape_(function_name, esc_func, sizeof(esc_func));
  bench_json_escape_(comment ? comment : "", esc_comment, sizeof(esc_comment));

  fprintf(file,
          "{\"function\":\"%s\""
          ",\"wall_ns\":%lld"
          ",\"wall_fmt\":\"%s\""
          ",\"peak_rss\":%ld"
          ",\"peak_rss_fmt\":\"%s\""
          ",\"comment\":\"%s\""
          ",\"os\":\"%s\""
          ",\"host\":\"%s\""
          ",\"timestamp\":\"%s\"}\n",
          esc_func, wall_ns, fmt_time, peak_rss, fmt_mem,
          esc_comment, sysinfo.sysname, hostname, iso_ts);
  fclose(file);
}

/* Write statistics from a BENCH_N run as one JSON line (.jsonl).
 *
 * Usage:
 *   bench_write_stats_jsonl("results.jsonl",
 *                           "dm_multiply (1000x1000)",
 *                           &stats, peak_rss, "OpenBLAS");
 */
static inline void bench_write_stats_jsonl(const char* filename,
                                           const char* function_name,
                                           const bench_stats_t* s,
                                           long peak_rss,
                                           const char* comment) {
  FILE* file = fopen(filename, "a");
  if (file == NULL) {
    perror("fopen");
    exit(1);
  }

  struct utsname sysinfo;
  if (uname(&sysinfo) != 0) {
    strcpy(sysinfo.sysname, "UnknownOS");
  }

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    strcpy(hostname, "UnknownHost");
  }

  char iso_ts[64];
  time_t tm = 0;
  time(&tm);
  strftime(iso_ts, sizeof(iso_ts), "%Y-%m-%dT%H:%M:%S%z", localtime(&tm));

  char fmt_mem[64];
  format_memory_usage(peak_rss, fmt_mem, sizeof(fmt_mem));

  char esc_func[256], esc_comment[256];
  bench_json_escape_(function_name, esc_func, sizeof(esc_func));
  bench_json_escape_(comment ? comment : "", esc_comment, sizeof(esc_comment));

  fprintf(file,
          "{\"function\":\"%s\""
          ",\"iterations\":%d"
          ",\"min_ns\":%.0f"
          ",\"max_ns\":%.0f"
          ",\"mean_ns\":%.0f"
          ",\"median_ns\":%.0f"
          ",\"stddev_ns\":%.0f"
          ",\"peak_rss\":%ld"
          ",\"peak_rss_fmt\":\"%s\""
          ",\"comment\":\"%s\""
          ",\"os\":\"%s\""
          ",\"host\":\"%s\""
          ",\"timestamp\":\"%s\"}\n",
          esc_func, s->n, s->min_ns, s->max_ns, s->mean_ns,
          s->median_ns, s->stddev_ns, peak_rss, fmt_mem,
          esc_comment, sysinfo.sysname, hostname, iso_ts);
  fclose(file);
}

/*****************************
 *  Environment info
 *******************************/

static inline void print_environment_info(void) {
  struct utsname sysinfo;
  if (uname(&sysinfo) == 0) {
    printf("\r" BENCH_BLUE "ℬ|" BENCH_RESET " OS: %s %s (%s)",
           sysinfo.sysname, sysinfo.release, sysinfo.machine);
  }

  long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  if (nprocs > 0) {
    printf(" | Cores: %ld", nprocs);
  }

  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    double mem_gb = (pages * page_size) / (1024.0 * 1024.0 * 1024.0);
    printf(" | RAM: %.2f GB", mem_gb);
  }

  double loadavg[3];
  if (getloadavg(loadavg, 3) == 3) {
    printf(" | Load: %.2f / %.2f / %.2f", loadavg[0], loadavg[1], loadavg[2]);
  }

  printf("\n");
  fflush(stdout);
}

/*****************************
 *  BENCH MACROs
 *******************************/

/* ---- Single-shot benchmark (wall-time + peak RSS) ---- */
#define BENCH(FCALL)                                                      \
  do {                                                                    \
    long long bench_elapsed_, bench_rss_;                                 \
    WALLTIME_RSS(FCALL, &bench_elapsed_, &bench_rss_);                    \
    char bench_fmt_[64];                                                  \
    format_duration(bench_elapsed_, bench_fmt_, sizeof(bench_fmt_));      \
    print_benchmark_result(bench_fmt_, bench_rss_, #FCALL, __FILE__,     \
                           __LINE__);                                     \
  } while (0)

/* ---- Multi-iteration benchmark with warmup & statistics ---- *
 *
 * Usage:
 *   BENCH_N(5, 2, {
 *     DoubleMatrix *r = dm_multiply(a, b);
 *     dm_destroy(r);
 *   });
 *
 * Parameters:
 *   ITERS   - number of timed iterations
 *   WARMUP  - number of warmup iterations (not measured)
 *   FCALL   - code block to benchmark
 */
#define BENCH_N(ITERS, WARMUP, FCALL)                                      \
  do {                                                                     \
    /* Warmup */                                                           \
    for (int bench_w_ = 0; bench_w_ < (WARMUP); ++bench_w_) {             \
      FCALL;                                                               \
    }                                                                      \
    /* Timed iterations */                                                 \
    long long bench_samples_[(ITERS)];                                     \
    long bench_peak_rss_ = get_memory_usage();                             \
    for (int bench_i_ = 0; bench_i_ < (ITERS); ++bench_i_) {              \
      update_countdown((ITERS) - bench_i_);                                \
      struct timespec bench_ts0_, bench_ts1_;                              \
      clock_gettime(CLOCK_MONOTONIC, &bench_ts0_);                         \
      FCALL;                                                               \
      clock_gettime(CLOCK_MONOTONIC, &bench_ts1_);                         \
      bench_samples_[bench_i_] =                                           \
          (bench_ts1_.tv_sec - bench_ts0_.tv_sec) * 1000000000LL +         \
          (bench_ts1_.tv_nsec - bench_ts0_.tv_nsec);                       \
    }                                                                      \
    long bench_peak_rss_after_ = get_memory_usage();                       \
    if (bench_peak_rss_after_ > bench_peak_rss_)                           \
      bench_peak_rss_ = bench_peak_rss_after_;                             \
    clear_line();                                                          \
    bench_stats_t bench_st_ = bench_compute_stats(bench_samples_, (ITERS));\
    print_benchmark_stats(#FCALL, &bench_st_, bench_peak_rss_);            \
  } while (0)

/* ---- Wall-time expression macro (GCC/Clang extension) ---- *
 *
 * Returns elapsed nanoseconds as long long.
 * Note: uses GCC statement-expression ({...}).
 *
 * Usage:  long long ns = WALLTIME({ do_work(); });
 */
#define WALLTIME(FCALL)                                         \
  ({                                                            \
    struct timespec bench_s_, bench_e_;                          \
    long long bench_elapsed_;                                   \
    if (clock_gettime(CLOCK_MONOTONIC, &bench_s_) != 0) {       \
      perror("clock_gettime");                                  \
      bench_elapsed_ = -1;                                      \
    } else {                                                    \
      FCALL;                                                    \
      BENCH_CLOBBER_MEMORY();                                   \
      if (clock_gettime(CLOCK_MONOTONIC, &bench_e_) != 0) {     \
        perror("clock_gettime");                                \
        bench_elapsed_ = -1;                                    \
      } else {                                                  \
        bench_elapsed_ =                                        \
            (bench_e_.tv_sec - bench_s_.tv_sec) * 1000000000LL +\
            (bench_e_.tv_nsec - bench_s_.tv_nsec);              \
      }                                                         \
    }                                                           \
    bench_elapsed_;                                             \
  })

/* backward-compatible alias */
#define CPUTIME(FCALL) WALLTIME(FCALL)

/* ---- Wall-time + RSS measurement ---- */
#define WALLTIME_RSS(FCALL, TIME_PTR, RSS_PTR)                    \
  do {                                                            \
    struct timespec bench_s_, bench_e_;                            \
    long bench_rss_before_ = get_memory_usage();                  \
                                                                  \
    if (clock_gettime(CLOCK_MONOTONIC, &bench_s_) != 0) {         \
      perror("clock_gettime");                                    \
      *(TIME_PTR) = -1;                                           \
    } else {                                                      \
      FCALL;                                                      \
      BENCH_CLOBBER_MEMORY();                                     \
      if (clock_gettime(CLOCK_MONOTONIC, &bench_e_) != 0) {       \
        perror("clock_gettime");                                  \
        *(TIME_PTR) = -1;                                         \
      } else {                                                    \
        *(TIME_PTR) =                                             \
            (bench_e_.tv_sec - bench_s_.tv_sec) * 1000000000LL +  \
            (bench_e_.tv_nsec - bench_s_.tv_nsec);                \
      }                                                           \
    }                                                             \
    long bench_rss_after_ = get_memory_usage();                   \
    *(RSS_PTR) = bench_rss_after_ - bench_rss_before_;            \
  } while (0)

/* backward-compatible alias */
#define CPUTIME_RSS(FCALL, TIME_PTR, RSS_PTR) \
  WALLTIME_RSS(FCALL, TIME_PTR, RSS_PTR)

/* ---- Peak RSS expression macro ---- */
#define RSS(FCALL)                                   \
  ({                                                 \
    long bench_mem_b_ = get_memory_usage();          \
    FCALL;                                           \
    long bench_mem_a_ = get_memory_usage();          \
    (bench_mem_a_ - bench_mem_b_);                   \
  })

#endif /* SIMPLE_BENCH_H */
