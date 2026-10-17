// Investigation-only (OPEN-ITEMS-NODE6 item E). LD_PRELOAD shim on pthread_create.
// For every thread created in the process it logs, from inside the new thread: its tid, the
// creating thread's tid, the CUDA device of the creator's current driver context, the device of
// the new thread's current driver context at start (-1 = none), and the creation call stack's
// libraries. With AIM345_TDEV_FIX=1, a new thread whose creation stack contains
// AIM345_TDEV_FIX_MATCH (default libplugin_LIBFABRIC), whose creator had a current context and
// which itself has none, makes the creator's device's (or AIM345_TDEV_FORCE_DEV's) primary context
// current before running. Every matched thread is renamed a345-lf-thread so host-side tracing
// can name it.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/prctl.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef int (*ctxget_t)(void **);
typedef int (*ctxdev_t)(int *);
typedef int (*retain_t)(void **, int);
typedef int (*ctxset_t)(void *);
typedef int (*alloc_t)(unsigned long long *, size_t);
typedef int (*dtod_t)(unsigned long long, unsigned long long, size_t);
typedef int (*memcpy_t)(unsigned long long, unsigned long long, size_t);
static ctxget_t p_get; static ctxdev_t p_dev; static retain_t p_retain; static ctxset_t p_set;
static alloc_t p_alloc; static dtod_t p_dtod; static memcpy_t p_memcpy;
static int (*real_create)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
static int fix_mode = -1;

static void load(void) {
  if (!real_create) real_create = dlvsym(RTLD_NEXT, "pthread_create", "GLIBC_2.34");
  if (!real_create) real_create = dlvsym(RTLD_NEXT, "pthread_create", "GLIBC_2.2.5");
  if (fix_mode < 0) { const char *f = getenv("AIM345_TDEV_FIX"); fix_mode = f && f[0] == '1'; }
  if (!p_get) {
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (h) { p_get = dlsym(h, "cuCtxGetCurrent"); p_dev = dlsym(h, "cuCtxGetDevice");
             p_retain = dlsym(h, "cuDevicePrimaryCtxRetain"); p_set = dlsym(h, "cuCtxSetCurrent");
             p_alloc = dlsym(h, "cuMemAlloc_v2"); p_dtod = dlsym(h, "cuMemcpyDtoD_v2");
             p_memcpy = dlsym(h, "cuMemcpy"); }
  }
}
static int cur_dev(void) {
  void *ctx = NULL; int d = -1;
  if (!p_get || p_get(&ctx) != 0 || !ctx) return -1;
  if (p_dev(&d) != 0) return -2;
  return d;
}

// Optional load (AIM345_TDEV_FOREIGN_LOAD=1): from a helper thread of this process, run 1 MiB
// device-to-device copies every AIM345_TDEV_FOREIGN_PERIOD_US on device AIM345_TDEV_FOREIGN_DEV,
// i.e. this process's work on another rank's GPU, without touching the matched thread.
// With AIM345_TDEV_FOREIGN_MODE=peer, the copy instead goes from a buffer on that device into a
// buffer on this process's own device (AIM345_TDEV_OWN_DEV), issued from the foreign device's
// context through cuMemcpy (unified addressing), the shape of an unbound thread's receive copy.
static void *foreign_load(void *v) {
  int dev = (int)(long)v; void *ctx = NULL, *own = NULL; unsigned long long a = 0, b = 0;
  const char *pe = getenv("AIM345_TDEV_FOREIGN_PERIOD_US"); long us = pe ? atol(pe) : 10000;
  const char *mode = getenv("AIM345_TDEV_FOREIGN_MODE"); int peer = mode && !strcmp(mode, "peer");
  const char *od = getenv("AIM345_TDEV_OWN_DEV"); int owndev = od ? atoi(od) : -1;
  int ok = p_retain && p_alloc && p_retain(&ctx, dev) == 0;
  if (ok && peer) ok = owndev >= 0 && p_retain(&own, owndev) == 0 && p_set(own) == 0 && p_alloc(&b, 1 << 20) == 0 && p_memcpy;
  if (ok) ok = p_set(ctx) == 0 && p_alloc(&a, 1 << 20) == 0 && (peer || p_alloc(&b, 1 << 20) == 0);
  if (!ok) { fprintf(stderr, "a345_tdev foreign_load pid=%d dev=%d setup failed\n", getpid(), dev); return NULL; }
  prctl(PR_SET_NAME, "a345-load");
  fprintf(stderr, "a345_tdev foreign_load pid=%d tid=%ld dev=%d mode=%s own_dev=%d period_us=%ld started\n", getpid(), syscall(SYS_gettid), dev, peer ? "peer" : "dtod", owndev, us);
  for (unsigned long i = 1, err = 0;; i++) {
    if ((peer ? p_memcpy(b, a, 1 << 20) : p_dtod(b, a, 1 << 20)) != 0) err++;
    if (i % 1000 == 0) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
      fprintf(stderr, "a345_tdev foreign_load pid=%d wall=%ld copies=%lu errors=%lu\n", getpid(), (long)ts.tv_sec, i, err); }
    usleep(us);
  }
  return NULL;
}
struct tramp { void *(*fn)(void *); void *arg; int pdev; long ptid; char stack[768]; };

static void *start(void *v) {
  struct tramp t = *(struct tramp *)v; free(v);
  load();
  long tid = syscall(SYS_gettid);
  int before = cur_dev(), after = before;
  const char *match = getenv("AIM345_TDEV_FIX_MATCH");
  if (!match || !match[0]) match = "libplugin_LIBFABRIC";
  int matched = strstr(t.stack, match) != NULL;
  // AIM345_TDEV_ROLE_PORT: act (fix, load) only in processes whose parent's command line has
  // "--port <value>", i.e. one engine role of the packed pod.
  const char *rp = getenv("AIM345_TDEV_ROLE_PORT");
  int role_ok = 1;
  if (rp && rp[0]) {
    char path[64], buf[8192] = {0}, want[32]; snprintf(path, sizeof path, "/proc/%d/cmdline", getppid());
    FILE *f = fopen(path, "r"); size_t n = f ? fread(buf, 1, sizeof buf - 1, f) : 0; if (f) fclose(f);
    for (size_t i = 0; i < n; i++) if (!buf[i]) buf[i] = ' ';
    snprintf(want, sizeof want, "--port %s ", rp); role_ok = strstr(buf, want) != NULL;
  }
  const char *force = getenv("AIM345_TDEV_FORCE_DEV");
  int target = (force && force[0]) ? atoi(force) : t.pdev;
  if (fix_mode && matched && role_ok && target >= 0 && before < 0 && p_retain && p_set) {
    void *ctx = NULL;
    if (p_retain(&ctx, target) == 0 && p_set(ctx) == 0) after = cur_dev();
  }
  if (matched) prctl(PR_SET_NAME, "a345-lf-thread");
  const char *fl = getenv("AIM345_TDEV_FOREIGN_LOAD"), *fd = getenv("AIM345_TDEV_FOREIGN_DEV");
  if (matched && role_ok && fl && fl[0] == '1' && fd && fd[0] && t.pdev >= 0 && t.pdev != atoi(fd)) {
    pthread_t h; real_create(&h, NULL, foreign_load, (void *)(long)atoi(fd)); pthread_detach(h);
  }
  struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
  fprintf(stderr, "a345_tdev pid=%d tid=%ld ptid=%ld wall=%ld.%06ld parent_dev=%d start_dev=%d after_fix=%d fix=%d matched=%d role_ok=%d stack=%s\n",
          getpid(), tid, t.ptid, (long)ts.tv_sec, ts.tv_nsec / 1000, t.pdev, before, after, fix_mode, matched, role_ok, t.stack);
  return t.fn(t.arg);
}


// AIM345_TDEV_COPYLOG=1: interpose dlsym so that a library resolving cudaMemcpy by name (libfabric
// does, hmem_cuda.c) gets a wrapper that records, per calling thread, the device of the current
// driver context and the device that owns the destination pointer, then calls the real function.
typedef int (*cumemcpy_rt_t)(void *, const void *, size_t, int);
typedef int (*ptrattr_t)(void *, int, unsigned long long);
static cumemcpy_rt_t real_cudaMemcpy; static ptrattr_t p_ptrattr;
static void *(*real_dlsym)(void *, const char *);
static __thread unsigned long tl_n, tl_mismatch, tl_nodst;
static __thread int tl_last_ctx = -9, tl_last_dst = -9;
static int wrap_cudaMemcpy(void *dst, const void *src, size_t n, int kind) {
  load();
  if (!p_ptrattr) { void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD); if (h) p_ptrattr = (ptrattr_t)real_dlsym(h, "cuPointerGetAttribute"); }
  int ctxd = cur_dev(), dstd = -1;
  if (!p_ptrattr || p_ptrattr(&dstd, 9 /* CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL */, (unsigned long long)dst) != 0) { dstd = -1; tl_nodst++; }
  tl_n++; if (dstd >= 0 && ctxd != dstd) tl_mismatch++;
  if (tl_n <= 2 || ctxd != tl_last_ctx || dstd != tl_last_dst || (tl_n & (tl_n - 1)) == 0 || tl_n % 500 == 0)
    fprintf(stderr, "a345_copy pid=%d tid=%ld n=%lu bytes=%zu ctx_dev=%d dst_dev=%d mismatched=%lu dst_unknown=%lu\n",
            getpid(), syscall(SYS_gettid), tl_n, n, ctxd, dstd, tl_mismatch, tl_nodst);
  tl_last_ctx = ctxd; tl_last_dst = dstd;
  return real_cudaMemcpy(dst, src, n, kind);
}
void *dlsym(void *h, const char *name) {
  if (!real_dlsym) { real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
                     if (!real_dlsym) real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"); }
  void *r = real_dlsym(h, name);
  const char *on = getenv("AIM345_TDEV_COPYLOG");
  if (r && name && on && on[0] == '1' && !strcmp(name, "cudaMemcpy")) { real_cudaMemcpy = (cumemcpy_rt_t)r; return (void *)wrap_cudaMemcpy; }
  return r;
}

int pthread_create(pthread_t *th, const pthread_attr_t *attr, void *(*fn)(void *), void *arg) {
  load();
  struct tramp *t = malloc(sizeof *t);
  if (!t) return real_create(th, attr, fn, arg);
  t->fn = fn; t->arg = arg; t->pdev = cur_dev(); t->ptid = syscall(SYS_gettid); t->stack[0] = 0;
  void *frames[8]; int n = backtrace(frames, 8); size_t off = 0;
  for (int i = 1; i < n && off < sizeof t->stack - 96; i++) {
    Dl_info di; const char *lib = "?";
    if (dladdr(frames[i], &di) && di.dli_fname) { lib = strrchr(di.dli_fname, '/'); lib = lib ? lib + 1 : di.dli_fname; }
    off += snprintf(t->stack + off, sizeof t->stack - off, "%s%s", i > 1 ? "|" : "", lib);
  }
  return real_create(th, attr, start, t);
}
