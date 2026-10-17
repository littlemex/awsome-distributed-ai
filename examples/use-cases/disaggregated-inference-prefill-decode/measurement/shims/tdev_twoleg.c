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
typedef int (*d2h_t)(void *, unsigned long long, size_t);
static d2h_t p_d2h; typedef int (*peer_t)(void *, unsigned int); static peer_t p_peer;
typedef int (*h2d_t)(unsigned long long, const void *, size_t); static h2d_t p_h2d;
typedef int (*hostalloc_t)(void **, size_t); static hostalloc_t p_hostalloc;
typedef int (*streamcreate_t)(void **, unsigned int); static streamcreate_t p_screate;
typedef int (*hostalloc2_t)(void **, size_t, unsigned int); static hostalloc2_t p_hostalloc2;
typedef int (*hostdevptr_t)(unsigned long long *, void *, unsigned int); static hostdevptr_t p_hostdevptr;
typedef int (*waitval_t)(void *, unsigned long long, unsigned int, unsigned int); static waitval_t p_waitval;
typedef int (*ssync_t)(void *); static ssync_t p_ssync;
typedef int (*memset32_t)(unsigned long long, unsigned int, size_t); static memset32_t p_memset32;
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
             p_memcpy = dlsym(h, "cuMemcpy"); p_d2h = dlsym(h, "cuMemcpyDtoH_v2"); p_peer = dlsym(h, "cuCtxEnablePeerAccess"); p_h2d = dlsym(h, "cuMemcpyHtoD_v2"); p_hostalloc = dlsym(h, "cuMemAllocHost_v2");
             p_screate = dlsym(h, "cuStreamCreate"); p_hostalloc2 = dlsym(h, "cuMemHostAlloc");
             p_hostdevptr = dlsym(h, "cuMemHostGetDevicePointer_v2"); p_waitval = dlsym(h, "cuStreamWaitValue32_v2");
             if (!p_waitval) p_waitval = dlsym(h, "cuStreamWaitValue32"); p_ssync = dlsym(h, "cuStreamSynchronize"); p_memset32 = dlsym(h, "cuMemsetD32_v2"); }
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
  int d2h = mode && !strcmp(mode, "d2h"); void *hostbuf = NULL;
  // NODE8 h2d: copy a pinned host buffer into a buffer on this device (the device's DMA engine reads host memory).
  int h2d = mode && !strcmp(mode, "h2d");
  // NODE8 p2pread_h / p2pread_d: from this (foreign) device's context, with peer access to the own device's
  // context enabled, copy a buffer on the own device into host memory (p2pread_h) or into this device (p2pread_d).
  int p2h = mode && !strcmp(mode, "p2pread_h"), p2d = mode && !strcmp(mode, "p2pread_d"), p3 = mode && !strcmp(mode, "thirdparty"); unsigned long long b2 = 0;
  // thirdparty: copy between two buffers on the own device, issued from this device's context with peer access.
  if (p2h || p2d || p3) peer = 1;
  const char *od = getenv("AIM345_TDEV_OWN_DEV"); int owndev = od ? atoi(od) : -1;
  // NODE8: copy size, copies per period (a burst), and which context issues a peer copy
  // (foreign: the other rank's device, own: this process's device).
  const char *sz = getenv("AIM345_TDEV_FOREIGN_SIZE"); size_t n = sz ? (size_t)atol(sz) : (1 << 20);
  const char *bu = getenv("AIM345_TDEV_FOREIGN_BURST"); long burst = bu ? atol(bu) : 1;
  const char *is = getenv("AIM345_TDEV_FOREIGN_ISSUE"); int issue_own = is && !strcmp(is, "own");
  int ok = p_retain && p_alloc && p_retain(&ctx, dev) == 0;
  if (ok && peer) ok = owndev >= 0 && p_retain(&own, owndev) == 0 && p_set(own) == 0 && p_alloc(&b, n) == 0 && p_memcpy;
  if (ok) ok = p_set(ctx) == 0 && p_alloc(&a, n) == 0 && (peer || p_alloc(&b, n) == 0);
  if (ok && peer && issue_own) ok = p_set(own) == 0;
  // NODE8: AIM345_TDEV_PINNED=1 makes the d2h destination pinned host memory instead of pageable.
  const char *pn = getenv("AIM345_TDEV_PINNED"); int pinned = pn && pn[0] == '1';
  if (ok && d2h) ok = p_d2h && (pinned ? (p_hostalloc && p_hostalloc(&hostbuf, n) == 0) : (hostbuf = malloc(n)) != NULL);
  if (ok && h2d) ok = p_h2d && p_hostalloc && p_hostalloc(&hostbuf, n) == 0;
  int peer_rc = -99;
  if (ok && p3) ok = p_set(own) == 0 && p_alloc(&b2, n) == 0 && p_set(ctx) == 0;
  if (ok && (p2h || p2d || p3)) { peer_rc = p_peer ? p_peer(own, 0) : -98; ok = (peer_rc == 0 || peer_rc == 704) && p_set(ctx) == 0; }
  if (ok && p2h) ok = p_d2h && (hostbuf = malloc(n)) != NULL;
  if (!ok) { fprintf(stderr, "a345_tdev foreign_load pid=%d dev=%d setup failed\n", getpid(), dev); return NULL; }
  // NODE8 semwait: no data. Every period, block a stream of this device's context on a host-memory flag for
  // AIM345_TDEV_WAIT_US (cuStreamWaitValue32), then release it from the host. Shows whether a waiting channel
  // of this context, alone, delays other contexts' work on the same GPU.
  // NODE8 semwait_peer: as semwait, but the flag lives in a buffer on the own device (AIM345_TDEV_OWN_DEV), read
  // across PCIe by this device's stream through peer access; the own context releases it with cuMemsetD32.
  if (mode && !strcmp(mode, "semwait_peer")) {
    void *st = NULL, *ownc = NULL; unsigned long long f = 0; int rc = 0;
    const char *ww = getenv("AIM345_TDEV_WAIT_US"); long wait_us = ww ? atol(ww) : 30000;
    if (owndev < 0 || p_retain(&ownc, owndev) || p_set(ownc) || p_alloc(&f, 4096) || p_memset32(f, 0, 1)) rc = 1;
    if (!rc && (p_set(ctx) || !p_peer)) rc = 2;
    int prc = rc ? -1 : p_peer(ownc, 0); if (!rc && prc != 0 && prc != 704) rc = 3;
    if (!rc && (!p_screate || p_screate(&st, 1))) rc = 4;
    prctl(PR_SET_NAME, "a345-load");
    fprintf(stderr, "a345_tdev semwait_peer pid=%d tid=%ld dev=%d own_dev=%d wait_us=%ld period_us=%ld rc=%d peer_rc=%d started\n", getpid(), syscall(SYS_gettid), dev, owndev, wait_us, us, rc, prc);
    if (rc || !p_waitval || !p_ssync || !p_memset32) return NULL;
    for (unsigned long i = 1, err = 0;; i++) {
      if (p_set(ownc) || p_memset32(f, 0, 1)) err++;
      if (p_set(ctx) || p_waitval(st, f, 1, 0x0) != 0) err++;
      usleep(wait_us);
      if (p_set(ownc) || p_memset32(f, 1, 1)) err++;
      if (p_set(ctx) || p_ssync(st) != 0) err++;
      if (i % 20 == 0) fprintf(stderr, "a345_tdev semwait_peer pid=%d waits=%lu errors=%lu\n", getpid(), i, err);
      usleep(us);
    }
  }
  if (mode && !strcmp(mode, "semwait")) {
    void *st = NULL; volatile unsigned int *flag = NULL; unsigned long long dflag = 0;
    const char *ww = getenv("AIM345_TDEV_WAIT_US"); long wait_us = ww ? atol(ww) : 30000;
    int rc1 = p_set(ctx), rc2 = p_screate ? p_screate(&st, 1) : -1, rc3 = p_hostalloc2 ? p_hostalloc2((void **)&flag, 4096, 0x02) : -1;
    int rc4 = (rc3 == 0 && p_hostdevptr) ? p_hostdevptr(&dflag, (void *)flag, 0) : -1;
    prctl(PR_SET_NAME, "a345-load");
    fprintf(stderr, "a345_tdev semwait pid=%d tid=%ld dev=%d wait_us=%ld period_us=%ld rc=%d,%d,%d,%d waitval=%p started\n", getpid(), syscall(SYS_gettid), dev, wait_us, us, rc1, rc2, rc3, rc4, (void *)p_waitval);
    if (rc1 || rc2 || rc3 || rc4 || !p_waitval || !p_ssync) return NULL;
    for (unsigned long i = 1, err = 0;; i++) {
      // NODE8: AIM345_TDEV_FOREIGN_BURST wait/release cycles per period (default 1).
      for (long k = 0; k < burst; k++) {
        *flag = 0; __sync_synchronize();
        if (p_waitval(st, dflag, 1, 0x0) != 0) err++;
        usleep(wait_us);
        *flag = 1; __sync_synchronize();
        if (p_ssync(st) != 0) err++;
      }
      if (i % 20 == 0) fprintf(stderr, "a345_tdev semwait pid=%d periods=%lu errors=%lu\n", getpid(), i, err);
      usleep(us);
    }
  }
  prctl(PR_SET_NAME, "a345-load");
  fprintf(stderr, "a345_tdev foreign_load pid=%d tid=%ld dev=%d mode=%s own_dev=%d issue=%s size=%zu burst=%ld period_us=%ld peer_rc=%d started\n", getpid(), syscall(SYS_gettid), dev, h2d ? "h2d" : p3 ? "thirdparty" : p2h ? "p2pread_h" : p2d ? "p2pread_d" : d2h ? "d2h" : peer ? "peer" : "dtod", owndev, issue_own ? "own" : "foreign", n, burst, us, peer_rc);
  for (unsigned long i = 1, err = 0;; i++) {
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long k = 0; k < burst; k++)
      if ((h2d ? p_h2d(a, hostbuf, n) : p3 ? p_dtod(b2, b, n) : p2h ? p_d2h(hostbuf, b, n) : p2d ? p_dtod(a, b, n) : d2h ? p_d2h(hostbuf, a, n) : peer ? p_memcpy(b, a, n) : p_dtod(b, a, n)) != 0) err++;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (i % 20 == 0) { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
      fprintf(stderr, "a345_tdev foreign_load pid=%d wall=%ld bursts=%lu errors=%lu last_burst_ms=%.2f\n", getpid(), (long)ts.tv_sec, i, err, (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6); }
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
  // NODE8: AIM345_TDEV_ENABLE_PEER=<d>: in a matched thread of a process whose creator is NOT on device d, enable
  // peer access from device d's primary context to the creator's device's primary context, then leave the
  // thread without a current context again, so the mismatch itself is unchanged and only the copy route can change.
  const char *ep = getenv("AIM345_TDEV_ENABLE_PEER");
  if (matched && role_ok && ep && ep[0] && t.pdev >= 0 && t.pdev != atoi(ep) && p_retain && p_set && p_peer) {
    void *other = NULL, *mine = NULL; int rc = -1;
    if (p_retain(&other, atoi(ep)) == 0 && p_retain(&mine, t.pdev) == 0 && p_set(other) == 0) rc = p_peer(mine, 0);
    p_set(NULL);
    fprintf(stderr, "a345_tdev enable_peer pid=%d tid=%ld from_dev=%d to_dev=%d rc=%d now_dev=%d\n", getpid(), tid, atoi(ep), t.pdev, rc, cur_dev());
  }
  if (matched) prctl(PR_SET_NAME, "a345-lf-thread");
  const char *fl = getenv("AIM345_TDEV_FOREIGN_LOAD"), *fd = getenv("AIM345_TDEV_FOREIGN_DEV");
  // NODE8: AIM345_TDEV_FOREIGN_SAMEOK=1 also allows a helper on the creator's own device, and
  // AIM345_TDEV_FOREIGN_ONLY_PDEV=<d> restricts the helper to processes whose creator is on device d.
  const char *so = getenv("AIM345_TDEV_FOREIGN_SAMEOK"), *op = getenv("AIM345_TDEV_FOREIGN_ONLY_PDEV");
  int same_ok = so && so[0] == '1', pdev_ok = !(op && op[0]) || t.pdev == atoi(op);
  if (matched && role_ok && pdev_ok && fl && fl[0] == '1' && fd && fd[0] && t.pdev >= 0 && (same_ok || t.pdev != atoi(fd))) {
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
