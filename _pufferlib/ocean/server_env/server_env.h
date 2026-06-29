/* server_env.h — PufferLib Ocean env that is a PASSTHROUGH to the throwaway
 * betterSkillys C# server (server-as-sim). The real Snake Pit dynamics, obs
 * (SimObsBuilder, bit-identical 9807-float), action apply and reward all live in
 * the C# server; this env only shuttles the PufferLib vec-buffers across two
 * boundaries each step:
 *
 *   - SHARED MEMORY (/dev/shm/rotmg_sim_shm): N*OBS obs, N*4 actions, N reward,
 *     N done, fixed float32 layout written by SimShmBridge.cs. We write actions,
 *     read obs/reward/done.
 *   - REDIS LOCKSTEP GATE (sim:step:cmd / sim:step:ack on the sim redis): one
 *     LPUSH advances all N C# worlds exactly one gated tick; the ack means the
 *     new obs/reward/done are in shm. Spoken as raw RESP over a TCP socket (no
 *     hiredis dependency) — LPUSH cmd, BLPOP ack.
 *
 * One Env holds ALL N agents (num_agents = N), so vecenv.h lays the N agents'
 * buffers out contiguously and env->observations/actions/rewards/terminals are
 * the slot-0 bases with per-agent stride — a 1:1 map onto the shm slots. The obs
 * layout (NUM_CH/GRID/MM/scalars) is IDENTICAL to the dungeon env, so the same
 * DungeonEncoder policy consumes it unchanged.
 */
#ifndef ROTMG_SERVER_ENV_H
#define ROTMG_SERVER_ENV_H

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

/* Obs layout — MUST match SimObsBuilder.cs (and dungeon.h) exactly. */
#define VIS_RADIUS 15
#define GRID 31
#define NUM_CH 7
#define NUM_SCALARS 8
#define MM 32
#define NUM_MM_CH 3
#define GRID_SIZE (NUM_CH * GRID * GRID)
#define MM_SIZE (NUM_MM_CH * MM * MM)
#define OBS_SIZE (GRID_SIZE + MM_SIZE + NUM_SCALARS) /* 9807 */
#define SRV_NUM_ATNS 4

/* shm header (matches SimShmBridge.cs): magic, n_agents, obs_len, n_atns. */
#define SHM_HEADER_INTS 4
#define SHM_MAGIC 0x52544D47 /* 'RTMG' */

/* Pure-shm barrier control words (SIM_SHM_BARRIER=1): two atomic uint32 generation
 * counters at the TAIL of the region (after [obs][act][rew][done]). req: this side bumps
 * == 'actions ready, tick'; done: the C# controller bumps == 'obs/reward/done are in shm'.
 * A monotonic generation (never reused) means we never read a stale obs frame. */
#define SHM_CTRL_INTS 2
#define SRV_SPIN_BUDGET 2000

/* Per-episode legible metrics, divided by n in vecenv. Kept tiny: the C# server
 * owns the real game telemetry; here we log only what crosses the boundary. */
typedef struct {
    float reward;        /* per-step reward, summed */
    float episodes;      /* episodes ended (terminal=1), summed */
    float done_count;    /* same as episodes; kept for a clean rate field */
    float n;             /* step count (MUST be the last field) */
} Log;

/* Minimal config: only the kwargs binding.c reads. N (num_agents) + the connection
 * coordinates for the shm region and the redis gate. */
typedef struct {
    int n_agents;
    int max_steps;
    int redis_port;
    int redis_db;
} Config;

typedef struct {
    Log log;
    float *observations; /* N*OBS_SIZE float32 (slot-0 base) */
    float *actions;      /* N*4 float32 */
    float *rewards;      /* N float32 */
    float *terminals;    /* N float32 */
    int num_agents;      /* N: this env owns all N agents */
    unsigned int rng;    /* env index (always 0: single env) */

    Config cfg;
    uint64_t rng_state;
    int steps;

    /* boundary handles (one shared connection for the single env) */
    int shm_fd;
    float *shm;     /* mmapped region base */
    size_t shm_bytes;
    float *shm_obs; /* into shm, past the header */
    float *shm_act;
    float *shm_rew;
    float *shm_done;
    int redis_fd;   /* TCP socket to the sim redis (RESP) */
    long tick;      /* monotonic tick token sent to the gate */
    int use_barrier;            /* 1 == pure-shm futex barrier, 0 == redis gate */
    volatile int32_t *ctrl_req;  /* &ctrl[0]: we bump to request a tick */
    volatile int32_t *ctrl_done; /* &ctrl[1]: controller bumps when obs are ready */
    int32_t generation;          /* last tick generation we requested */
    int frame_skip;              /* K: env advances K ticks per policy step (action-repeat) */
} ServerEnv;

#define Env ServerEnv

/* ---- redis RESP over a raw socket (no hiredis) ------------------------------ */

static int srv_redis_connect(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int srv_write_all(int fd, const char *buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, buf + off, n - off);
        if (w <= 0) return -1;
        off += (size_t)w;
    }
    return 0;
}

/* Read one line (terminated by \r\n) into buf. Returns length sans CRLF, -1 on err. */
static int srv_read_line(int fd, char *buf, int cap) {
    int i = 0;
    while (i < cap - 1) {
        char c;
        ssize_t r = read(fd, &c, 1);
        if (r <= 0) return -1;
        if (c == '\r') {
            char lf;
            if (read(fd, &lf, 1) <= 0) return -1;
            buf[i] = '\0';
            return i;
        }
        buf[i++] = c;
    }
    buf[i] = '\0';
    return i;
}

/* SELECT the gate's logical db so cmd/ack land where the C# controller reads. */
static int srv_redis_select(int fd, int db) {
    char cmd[64];
    int len = snprintf(cmd, sizeof(cmd), "*2\r\n$6\r\nSELECT\r\n$%d\r\n%d\r\n", db < 10 ? 1 : 2, db);
    if (srv_write_all(fd, cmd, (size_t)len) < 0) return -1;
    char line[64];
    return srv_read_line(fd, line, sizeof(line)); /* +OK */
}

/* LPUSH sim:step:cmd <token>: release one tick across all N C# worlds. */
static int srv_gate_push(Env *env, const char *key, long token) {
    char tok[32];
    int tlen = snprintf(tok, sizeof(tok), "%ld", token);
    char cmd[128];
    int len = snprintf(cmd, sizeof(cmd),
        "*3\r\n$5\r\nLPUSH\r\n$%zu\r\n%s\r\n$%d\r\n%s\r\n",
        strlen(key), key, tlen, tok);
    if (srv_write_all(env->redis_fd, cmd, (size_t)len) < 0) return -1;
    char line[64];
    return srv_read_line(env->redis_fd, line, sizeof(line)); /* :<list-len> */
}

/* BLPOP sim:step:ack 0: block until the C# gate acks the tick completed (obs are
 * in shm). The reply is an array: *2 \r\n $<klen>\r\n<key>\r\n $<vlen>\r\n<val>. */
static int srv_gate_blpop(Env *env, const char *key) {
    char cmd[128];
    int len = snprintf(cmd, sizeof(cmd),
        "*3\r\n$5\r\nBLPOP\r\n$%zu\r\n%s\r\n$1\r\n0\r\n", strlen(key), key);
    if (srv_write_all(env->redis_fd, cmd, (size_t)len) < 0) return -1;
    char line[64];
    if (srv_read_line(env->redis_fd, line, sizeof(line)) < 0) return -1; /* *2 */
    if (line[0] != '*') return -1;
    /* key bulk: $len\r\n<key>\r\n */
    if (srv_read_line(env->redis_fd, line, sizeof(line)) < 0) return -1; /* $klen */
    char tmp[128];
    if (srv_read_line(env->redis_fd, tmp, sizeof(tmp)) < 0) return -1;   /* key */
    /* val bulk: $len\r\n<val>\r\n */
    if (srv_read_line(env->redis_fd, line, sizeof(line)) < 0) return -1; /* $vlen */
    if (srv_read_line(env->redis_fd, tmp, sizeof(tmp)) < 0) return -1;   /* val */
    return 0;
}

/* ---- shared memory --------------------------------------------------------- */

static void srv_open_shm(Env *env) {
    const char *path = getenv("SIM_SHM_PATH");
    if (path == NULL) path = "/dev/shm/rotmg_sim_shm";

    int n = env->cfg.n_agents;
    size_t bytes = (size_t)SHM_HEADER_INTS * sizeof(int)
        + (size_t)n * OBS_SIZE * sizeof(float)
        + (size_t)n * SRV_NUM_ATNS * sizeof(float)
        + (size_t)n * sizeof(float)
        + (size_t)n * sizeof(float);

    /* The C# server creates + sizes the region first. Spin until it exists and the
     * magic is stamped so a too-early open never maps a short/zeroed file. */
    int fd = -1;
    for (int tries = 0; tries < 600; tries++) {
        fd = open(path, O_RDWR);
        if (fd >= 0) {
            struct stat st;
            if (fstat(fd, &st) == 0 && (size_t)st.st_size >= bytes) break;
            close(fd);
            fd = -1;
        }
        usleep(100000); /* 100ms; up to 60s for the server to boot */
    }
    if (fd < 0) {
        fprintf(stderr, "server_env: could not open shm '%s' (is the C# server up with SIM_SHM=1?)\n", path);
        exit(1);
    }

    void *base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "server_env: mmap failed\n");
        exit(1);
    }

    int *hdr = (int *)base;
    for (int tries = 0; tries < 600 && hdr[0] != SHM_MAGIC; tries++)
        usleep(100000);
    if (hdr[0] != SHM_MAGIC || hdr[1] != n || hdr[2] != OBS_SIZE || hdr[3] != SRV_NUM_ATNS) {
        fprintf(stderr, "server_env: shm header mismatch magic=%x n=%d obs=%d atns=%d (want n=%d obs=%d atns=%d)\n",
            hdr[0], hdr[1], hdr[2], hdr[3], n, OBS_SIZE, SRV_NUM_ATNS);
        exit(1);
    }

    env->shm_fd = fd;
    env->shm = (float *)base;
    env->shm_bytes = bytes;
    float *after_hdr = (float *)((char *)base + (size_t)SHM_HEADER_INTS * sizeof(int));
    env->shm_obs = after_hdr;
    env->shm_act = env->shm_obs + (size_t)n * OBS_SIZE;
    env->shm_rew = env->shm_act + (size_t)n * SRV_NUM_ATNS;
    env->shm_done = env->shm_rew + n;
    /* barrier control words sit right after the done array (region tail). */
    int32_t *ctrl = (int32_t *)(env->shm_done + n);
    env->ctrl_req = (volatile int32_t *)&ctrl[0];
    env->ctrl_done = (volatile int32_t *)&ctrl[1];
    /* Continue the shared generation sequence: req/done live in shm and persist across
     * client processes, so seed from the live req (not 0) or a reconnecting C-shim would
     * drive the counter backwards and desync the controller. */
    env->generation = __atomic_load_n(env->ctrl_req, __ATOMIC_ACQUIRE);
}

static void srv_connect(Env *env) {
    if (env->shm == NULL)
        srv_open_shm(env);
    /* SIM_SHM_BARRIER=1 -> pure-shm futex barrier, no redis on the hot path. */
    const char *bar = getenv("SIM_SHM_BARRIER");
    env->use_barrier = (bar != NULL && bar[0] == '1');
    /* SIM_FRAME_SKIP=K (default 1): the env advances K ticks per policy step, repeating
     * the same action, summing reward across the K ticks. K halves the policy/GPU step
     * rate (the dominant lockstep overhead) at the cost of K*100ms reaction latency.
     * Action-repeat lives ENTIRELY in the C-shim: the action stays in shm across the K
     * gate ticks (the C# server re-reads + re-applies it each tick), so the server is
     * unchanged. An episode end mid-skip breaks early so the policy gets the post-reset
     * obs[0] (== the single-step auto-reset convention). */
    const char *fs = getenv("SIM_FRAME_SKIP");
    env->frame_skip = (fs != NULL) ? atoi(fs) : 1;
    if (env->frame_skip < 1) env->frame_skip = 1;
    if (env->frame_skip > 1)
        fprintf(stderr, "server_env: frame_skip K=%d (policy acts every %d ticks)\n",
                env->frame_skip, env->frame_skip);
    if (env->use_barrier) {
        fprintf(stderr, "server_env: pure-shm futex barrier ON (redis bypassed)\n");
        return;
    }
    if (env->redis_fd <= 0) {
        int fd = srv_redis_connect(env->cfg.redis_port);
        if (fd < 0) {
            fprintf(stderr, "server_env: redis connect failed on 127.0.0.1:%d\n", env->cfg.redis_port);
            exit(1);
        }
        srv_redis_select(fd, env->cfg.redis_db);
        env->redis_fd = fd;
    }
}

/* ---- pure-shm futex barrier (no redis on the hot path) --------------------- */

static long srv_futex(volatile int32_t *uaddr, int op, int val) {
    return syscall(SYS_futex, (void *)uaddr, op, val, NULL, NULL, 0);
}

/* Advance all N C# worlds one tick over the shm barrier: bump req to the next
 * generation, wake the C# controller parked on req, then wait until done catches
 * up to that generation (obs/reward/done are now in shm). Adaptive spin first
 * (the controller usually flips done within microseconds), then futex-park. */
static void srv_barrier_tick(Env *env) {
    int32_t gen = ++env->generation;
    __atomic_store_n(env->ctrl_req, gen, __ATOMIC_RELEASE);
    srv_futex(env->ctrl_req, FUTEX_WAKE, 1);

    for (int spin = 0; spin < SRV_SPIN_BUDGET; spin++) {
        if (__atomic_load_n(env->ctrl_done, __ATOMIC_ACQUIRE) >= gen) return;
        __builtin_ia32_pause(); /* PAUSE: busy-spin without yielding to the GPU sweep */
    }
    for (;;) {
        int32_t cur = __atomic_load_n(env->ctrl_done, __ATOMIC_ACQUIRE);
        if (cur >= gen) return;
        /* FUTEX_WAIT(&done, cur): block until done != cur (a wake or value change). */
        srv_futex(env->ctrl_done, FUTEX_WAIT, cur);
    }
}

/* ---- the lockstep barrier: one gated tick --------------------------------- */

/* Advance all N C# worlds one tick: LPUSH cmd -> BLPOP ack (obs now in shm). */
static void srv_gate_tick(Env *env) {
    if (env->use_barrier) {
        srv_barrier_tick(env);
        return;
    }
    env->tick++;
    if (srv_gate_push(env, "sim:step:cmd", env->tick) < 0) {
        fprintf(stderr, "server_env: gate LPUSH failed\n");
        exit(1);
    }
    if (srv_gate_blpop(env, "sim:step:ack") < 0) {
        fprintf(stderr, "server_env: gate BLPOP failed\n");
        exit(1);
    }
}

static void srv_pull_obs(Env *env) {
    int n = env->cfg.n_agents;
    memcpy(env->observations, env->shm_obs, (size_t)n * OBS_SIZE * sizeof(float));
    memcpy(env->rewards, env->shm_rew, (size_t)n * sizeof(float));
    memcpy(env->terminals, env->shm_done, (size_t)n * sizeof(float));
}

/* ---- the vecenv contract: c_reset / c_step / c_close / c_render ------------ */

static void init_globals(void) { /* nothing global to build: the C# server owns state */ }

static void c_reset(Env *env) {
    srv_connect(env);
    env->steps = 0;
    /* Write a no-op action for every agent, tick the gate once, then read obs[0].
     * The C# side spawns agents lazily over the first ticks; the warm-up zeros are
     * harmless and the policy's first real action lands once agents are in. */
    int n = env->cfg.n_agents;
    memset(env->shm_act, 0, (size_t)n * SRV_NUM_ATNS * sizeof(float));
    srv_gate_tick(env);
    srv_pull_obs(env);
    /* a reset is not a terminal step */
    memset(env->terminals, 0, (size_t)n * sizeof(float));
    memset(&env->log, 0, sizeof(Log));
}

static void c_step(Env *env) {
    int n = env->cfg.n_agents;
    /* push this step's actions (vec-buffer float -> shm float, identical layout) */
    memcpy(env->shm_act, env->actions, (size_t)n * SRV_NUM_ATNS * sizeof(float));

    int k = env->frame_skip;
    if (k <= 1) {
        srv_gate_tick(env);
        srv_pull_obs(env);
    } else {
        /* Action-repeat over K gate ticks. The action stays in shm (the server re-reads
         * it each tick), so K ticks all use the same action. Reward is SUMMED across the
         * K ticks; obs/terminals come from the last tick we ran. If any tick ends an
         * episode, stop early -- the server auto-reset, so its obs is the fresh frame the
         * policy must see (the single-step auto-reset convention, just K ticks coarser). */
        float acc[256 * 4]; /* reward accumulator; n*1 <= total_agents, comfortably < 1024 */
        for (int i = 0; i < n; i++) acc[i] = 0.0f;
        for (int t = 0; t < k; t++) {
            srv_gate_tick(env);
            srv_pull_obs(env);
            int any_done = 0;
            for (int i = 0; i < n; i++) {
                acc[i] += env->rewards[i];
                if (env->terminals[i] != 0.0f) any_done = 1;
            }
            if (any_done) break; /* hand the policy the post-reset obs[0] */
        }
        /* overwrite the per-step reward buffer with the summed reward */
        for (int i = 0; i < n; i++) env->rewards[i] = acc[i];
    }
    env->steps++;
    /* legible metrics for the boundary */
    float r = 0.0f, d = 0.0f;
    for (int i = 0; i < n; i++) {
        r += env->rewards[i];
        d += env->terminals[i];
    }
    env->log.reward += r / (float)n;
    env->log.done_count += d;
    env->log.episodes += d;
    env->log.n += 1.0f;
}

static void c_close(Env *env) {
    if (env->shm != NULL && env->shm != MAP_FAILED) {
        munmap(env->shm, env->shm_bytes);
        env->shm = NULL;
    }
    if (env->shm_fd > 0) {
        close(env->shm_fd);
        env->shm_fd = 0;
    }
    if (env->redis_fd > 0) {
        close(env->redis_fd);
        env->redis_fd = 0;
    }
}

static void c_render(Env *env) { (void)env; /* rendering lives in the C# game, not here */ }

#endif /* ROTMG_SERVER_ENV_H */
