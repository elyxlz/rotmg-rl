/* Faithful C port of rotmg_rl.sim.dungeon.DungeonEnv (the Snake Pit dungeon sim).
 *
 * The Python sim is the oracle; this matches its obs layout, action space, and dynamics so a
 * policy trained here transfers. Determinism notes for the parity test: with n_snakes=0,
 * enable_minions=0, boss_wander_speed=0 (stationary boss) and point-mass damage ranges (lo==hi),
 * the whole sim is RNG-free and matches the Python oracle bit-faithfully (positions float32,
 * distances float64, as in numpy<2.0). Snake spawn/drift, boss Wander and minion placement use
 * rand() and are not bit-matched (stochastic by design).
 */
#ifndef ROTMG_DUNGEON_H
#define ROTMG_DUNGEON_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "snakepit_map.h"

#define VIS_RADIUS 15
#define GRID 31
#define HALF 15
#define N_MOVE 8
#define N_AIM 32
#define NUM_CH 7
#define NUM_SCALARS 8
#define GRID_SIZE (NUM_CH * GRID * GRID)
/* Fog-of-war global minimap: MM x MM cells, 3 channels (terrain, player, boss). Obs layout is
 * [grid, minimap, scalars] to match the numpy oracle's flattened Dict order. */
#define MM 32
#define NUM_MM_CH 3
#define MM_SIZE (NUM_MM_CH * MM * MM)
#define SCALAR_OFF (GRID_SIZE + MM_SIZE)
#define OBS_SIZE (GRID_SIZE + MM_SIZE + NUM_SCALARS)

#define CH_WALL 0
#define CH_ENEMY 1
#define CH_EBULLET 2
#define CH_EBVX 3
#define CH_EBVY 4
#define CH_PBULLET 5
#define CH_GRENADE 6

#define BOSS_RETURN_RADIUS 1.0f /* ReturnToSpawn(0.7, 1): the boss anchors within 1 tile of spawn */

#define MAX_PBULLETS 4096
#define MAX_EBULLETS 4096
#define MAX_SNAKES 512
#define MAX_GRENADES 64

/* Per-STEP accumulators (vec_log divides every field by n). This matches the numpy oracle's
 * per-step info semantics: the gym vector backend feeds PuffeRL one info dict per env per step,
 * so environment/boss_hp_frac is a per-step mean (~1.0 while the boss is healthy, dropping as it
 * is damaged) and environment/cleared is the per-step clear rate (~0, spiking only on the death
 * step), NOT per-episode values. n increments every step. */
typedef struct {
    float boss_hp_frac;   /* max(boss_hp,0)/boss_hp_max, summed per step */
    float in_room;        /* fight_active, summed per step */
    float cleared;        /* cleared-this-step flag, summed per step */
    float snakes;         /* alive snake count, summed per step */
    float player_hp_frac; /* player_hp/player_hp_max, summed per step */
    float reward;         /* step reward, summed per step */
    float perf;           /* per-step clear rate */
    /* Per-EPISODE accumulators (added once at episode end, NOT per step). The sweep maximizes
     * `score`: a dense end-of-episode result independent of the tunable reward scale (the guide:
     * "log a score, not raw reward"). score = 1.0 if cleared else (1 - boss_hp_frac_at_end). Both
     * fields are summed then divided by n (step count) in vec_log, so my_log recovers the
     * per-episode mean as score/episodes (the n divisor cancels in the ratio). */
    float score;              /* sum of per-episode end-state scores */
    float clear_count;        /* sum of per-episode cleared flags -> clear_rate = clear_count/episodes */
    float player_hp_end_sum;  /* sum of player_hp_frac at episode end -> player_hp_remaining */
    float death_count;        /* sum of player-death episodes -> death_rate */
    float episodes;           /* count of episodes ended (terminated or truncated) */
    float n;                  /* step count (required as the last field) */
} Log;

typedef struct {
    float x, y, vx, vy, life, dmg;
} Bullet;

typedef struct {
    float x, y, hp, timer, type;
} Snake;

/* Real Snake Pit enemies (EmbeddedData_SnakePitCXML.xml + BehaviorDb.SnakePit.cs), mirroring the
 * numpy oracle's SNAKE_TYPES. Columns: hp, defense, dmg, bvspeed, blife(ticks), count, arc(rad),
 * cooldown(ticks), follow, follow_speed, acquire_range, shoot_range. */
#define N_SNAKE_TYPES 5
#define ST_HP 0
#define ST_DEF 1
#define ST_DMG 2
#define ST_BVS 3
#define ST_BLIFE 4
#define ST_CNT 5
#define ST_ARC 6
#define ST_CD 7
#define ST_FOLLOW 8
#define ST_FSPD 9
#define ST_ACQ 10
#define ST_SRANGE 11
static const float SNAKE_TYPES[N_SNAKE_TYPES][12] = {
    {5.0f, 0.0f, 20.0f, 0.6f, 20.0f, 1.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 20.0f},                      /* Pit Viper */
    {200.0f, 5.0f, 25.0f, 0.8f, 20.0f, 3.0f, (float)(5.0 * M_PI / 180.0), 10.0f, 1.0f, 0.5f, 10.0f, 15.0f},  /* Fire Python */
    {200.0f, 5.0f, 25.0f, 0.8f, 30.0f, 1.0f, 0.0f, 10.0f, 1.0f, 0.5f, 10.0f, 20.0f},                   /* Yellow Python */
    {500.0f, 10.0f, 50.0f, 0.8f, 30.0f, 3.0f, (float)(5.0 * M_PI / 180.0), 10.0f, 1.0f, 0.5f, 10.0f, 15.0f}, /* Greater Pit Snake */
    {500.0f, 10.0f, 50.0f, 0.6f, 30.0f, 1.0f, 0.0f, 3.0f, 1.0f, 0.5f, 10.0f, 15.0f},                   /* Greater Pit Viper */
};
static const float SNAKE_WEIGHTS[N_SNAKE_TYPES] = {0.40f, 0.22f, 0.15f, 0.15f, 0.08f};
#define SNAKE_TIMER_JITTER 10

typedef struct {
    float x, y, fuse, rad, dmg, status;
} Grenade;

typedef struct {
    /* --- DungeonConfig mirror --- */
    float player_speed, player_radius;
    int max_steps;
    float activation_range, spawn_in_room_prob, random_spawn_prob, spawn_in_room_radius;
    float player_hp_max, player_mp_max, player_defense, damage_floor, mp_regen, hp_regen;
    int staff_num;
    float staff_cooldown; /* ticks/shot, fractional (DEX 75 -> 1.25); the accumulator carries the remainder */
    float staff_dmg_lo, staff_dmg_hi, staff_speed, staff_life;
    float staff_radius, staff_offset, spell_cost;
    int spell_cooldown, spell_num;
    float spell_dmg_lo, spell_dmg_hi, spell_speed;
    int spell_life;
    int n_snakes, n_snakes_jitter;
    float snake_speed, snake_radius;
    float boss_hp_max, boss_radius, boss_defense, boss_wander_speed, boss_return_speed;
    int boss_shoots, opening_invuln_ticks, invuln_ticks;
    int blade_cd;
    float blade_radius_p1, blade_radius_p3;
    float ebullet_speed;
    int ebullet_life;
    float ebullet_dmg, ebullet_radius;
    int max_bullets;
    int grenade_fuse, grenade_cd_p1, grenade_cd_p2, grenade_cd_p3_diag;
    float grenade_range_confuse, grenade_petrify_dist;
    float grenade_radius_confuse, grenade_dmg_confuse, grenade_radius_petrify, grenade_dmg_petrify;
    int confused_ticks, petrify_ticks;
    int minion_max, minion_cd;
    float minion_hp;
    int enable_grenades, enable_minions;
    float rew_explore, rew_kill, rew_boss_dmg, rew_reach, rew_survive;
    float rew_damage_taken, rew_clear, rew_death, rew_step;
    float rew_approach; /* potential-based distance-to-boss shaping (0 = off) */
} Config;

typedef struct {
    Log log;
    float* observations;  /* OBS_SIZE float32 */
#ifdef PUFFER4
    /* PufferLib 4.0 vecenv.h wires float* action/reward/terminal buffers and reads num_agents + rng
     * (the env index, used to seed rng_state in my_init). Same env dynamics as the 3.0 build; only
     * the buffer dtypes differ (actions are cast to int per-dim in c_step). */
    float* actions;       /* 4 dims: move, aim, shoot, cast (delivered as float, cast to int) */
    float* rewards;       /* 1 float */
    float* terminals;     /* 1 float */
    int num_agents;       /* 1 game per env */
    unsigned int rng;     /* env index, set by vecenv.h before my_init */
#else
    int* actions;         /* 4 ints: move, aim, shoot, cast */
    float* rewards;       /* 1 float */
    unsigned char* terminals;
#endif

    Config cfg;

    float px, py;          /* player pos (float32 in oracle) */
    float player_hp, player_mp;
    double staff_timer; /* fractional staff cooldown accumulator (double, parity-matched to the oracle) */
    int spell_timer;
    double boss_x, boss_y; /* boss_pos becomes float64 in oracle after first move */
    double boss_spawn_x, boss_spawn_y; /* ReturnToSpawn anchor: the boss is pulled back toward here in P1 */
    double boss_hp; /* float64 like the numpy oracle (Python float), for phase/collision fidelity */
    int phase, fight_active, invuln_timer;
    int confused_timer, petrify_timer, minion_timer;
    int t_p1, t_p3a, t_g1, t_g2, t_g3card, t_g3diag;

    Bullet pbul[MAX_PBULLETS];
    int n_pbul;
    Bullet ebul[MAX_EBULLETS];
    int n_ebul;
    Snake snakes[MAX_SNAKES];
    int n_snake;
    Grenade grenades[MAX_GRENADES];
    int n_gren;

    unsigned char visited[MAP_H * MAP_W];
    unsigned char discovered[MAP_H * MAP_W]; /* fog-of-war: tiles ever within VIS_RADIUS of the player */
    float mm_terr[MM * MM]; /* minimap terrain pool, accumulated as tiles are discovered (+1/-1/0) */
    int boss_seen;          /* boss has been within vision at least once */
    double prev_boss_dist;  /* distance-shaping baseline: player->boss distance last step */
    int steps;
    uint64_t rng_state; /* per-env RNG: thread-safe under OpenMP, independent per env */
    int last_ipx, last_ipy; /* wall-channel cache key (avoid refilling 31x31 walls every step) */
} Dungeon;

/* Shared, read-only after init: the map-derived tables are identical for every env, so we build
 * them once globally instead of per-env (less memory bandwidth, better cache, no per-env malloc). */
static int g_init = 0;
static int g_n_walk = 0;
static int g_walk_x[MAP_H * MAP_W];
static int g_walk_y[MAP_H * MAP_W];
static float g_move_dx[N_MOVE], g_move_dy[N_MOVE];
static float g_aim_dx[N_AIM], g_aim_dy[N_AIM];

static void init_globals(void) {
    if (g_init) return;
    for (int i = 0; i < N_MOVE; i++) {
        g_move_dx[i] = (float)cos(i * M_PI / 4.0);
        g_move_dy[i] = (float)sin(i * M_PI / 4.0);
    }
    for (int i = 0; i < N_AIM; i++) {
        g_aim_dx[i] = (float)cos(i * 2.0 * M_PI / N_AIM);
        g_aim_dy[i] = (float)sin(i * 2.0 * M_PI / N_AIM);
    }
    int k = 0;
    for (int y = 0; y < MAP_H; y++)
        for (int x = 0; x < MAP_W; x++)
            if (MAP_WALKABLE[y * MAP_W + x]) {
                g_walk_x[k] = x;
                g_walk_y[k] = y;
                k++;
            }
    g_n_walk = k;
    g_init = 1;
}

/* --- helpers --- */

static inline uint32_t rng_next(Dungeon* env) {
    env->rng_state = env->rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(env->rng_state >> 32);
}
static inline float frand(Dungeon* env) { return (float)(rng_next(env) >> 8) / (float)(1 << 24); }
static inline float uniform_f(Dungeon* env, float lo, float hi) { return lo + (hi - lo) * frand(env); }

static float randn(Dungeon* env) {
    /* Box-Muller; training-only randomness (boss wander + snake drift), not parity-matched */
    float u1 = frand(env), u2 = frand(env);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static inline int walkable_at(float fx, float fy) {
    int x = (int)fx, y = (int)fy;
    if (x < 0 || x >= MAP_W || y < 0 || y >= MAP_H) return 0;
    return MAP_WALKABLE[y * MAP_W + x] != 0;
}

static double dist_ff(float ax, float ay, float bx, float by) {
    float dx = ax - bx, dy = ay - by;  /* float32 subtraction, as in numpy */
    return sqrt((double)dx * dx + (double)dy * dy);
}

static double dist_df(double ax, double ay, float bx, float by) {
    double dx = ax - bx, dy = ay - by;  /* boss_pos is float64 (double) */
    return sqrt(dx * dx + dy * dy);
}

/* Real DamageWithDefense clamp for a single hit: max(raw*floor, raw - defense). */
static inline double defended(float raw, float defense, float floor) {
    double a = (double)raw * floor, b = (double)raw - defense;
    return a > b ? a : b;
}

/* nearest walkable tile to (x,y), matching _nearest_walkable */
static void nearest_walkable(int x, int y, int* ox, int* oy) {
    if (x >= 0 && x < MAP_W && y >= 0 && y < MAP_H && MAP_WALKABLE[y * MAP_W + x]) {
        *ox = x;
        *oy = y;
        return;
    }
    long best = -1;
    int bi = 0;
    for (int i = 0; i < g_n_walk; i++) {
        long dx = g_walk_x[i] - x, dy = g_walk_y[i] - y;
        long d = dx * dx + dy * dy;
        if (best < 0 || d < best) {
            best = d;
            bi = i;
        }
    }
    *ox = g_walk_x[bi];
    *oy = g_walk_y[bi];
}

static void append_bullet(Bullet* arr, int* n, int cap, int max_keep, Bullet b) {
    if (*n < cap) {
        arr[*n] = b;
        (*n)++;
    } else {
        /* keep newest (drop oldest), mirroring out[-max_bullets:] */
        memmove(arr, arr + 1, sizeof(Bullet) * (cap - 1));
        arr[cap - 1] = b;
    }
    if (max_keep > 0 && *n > max_keep) {
        int drop = *n - max_keep;
        memmove(arr, arr + drop, sizeof(Bullet) * max_keep);
        *n = max_keep;
    }
}

/* --- bullet physics --- */

static void advance_bullets(Bullet* arr, int* n) {
    int w = 0;
    for (int i = 0; i < *n; i++) {
        Bullet b = arr[i];
        b.x += b.vx;
        b.y += b.vy;
        b.life -= 1.0f;
        int ix = (int)b.x;
        if (ix < 0) ix = 0;
        if (ix > MAP_W - 1) ix = MAP_W - 1;
        int iy = (int)b.y;
        if (iy < 0) iy = 0;
        if (iy > MAP_H - 1) iy = MAP_H - 1;
        if (b.life > 0.0f && MAP_WALKABLE[iy * MAP_W + ix]) arr[w++] = b;
    }
    *n = w;
}

/* --- combat --- */

static double resolve_collisions(Dungeon* env) {
    Config* c = &env->cfg;
    double reward = 0.0;

    /* player bullets vs snakes (per-bullet defense clamp by snake type) */
    for (int s = 0; s < env->n_snake; s++) {
        if (env->snakes[s].hp <= 0.0f) continue;
        if (env->n_pbul == 0) break;
        float thr = c->snake_radius + c->staff_radius;
        float sdef = SNAKE_TYPES[(int)env->snakes[s].type][ST_DEF];
        double dmg = 0.0;
        int any = 0, w = 0;
        for (int i = 0; i < env->n_pbul; i++) {
            if (dist_ff(env->pbul[i].x, env->pbul[i].y, env->snakes[s].x, env->snakes[s].y) < thr) {
                dmg += defended(env->pbul[i].dmg, sdef, c->damage_floor);
                any = 1;
            } else {
                env->pbul[w++] = env->pbul[i];
            }
        }
        if (any) {
            env->n_pbul = w;
            env->snakes[s].hp -= (float)dmg;
            if (env->snakes[s].hp <= 0.0f) reward += c->rew_kill;
        }
    }

    /* player bullets vs boss (DEF 19) */
    if (env->n_pbul > 0 && env->invuln_timer == 0 && env->phase > 0) {
        float thr = c->boss_radius + c->staff_radius;
        double dmg = 0.0;
        int any = 0, w = 0;
        for (int i = 0; i < env->n_pbul; i++) {
            if (dist_df(env->boss_x, env->boss_y, env->pbul[i].x, env->pbul[i].y) < thr) {
                dmg += defended(env->pbul[i].dmg, c->boss_defense, c->damage_floor);
                any = 1;
            } else {
                env->pbul[w++] = env->pbul[i];
            }
        }
        if (any) {
            env->n_pbul = w;
            env->boss_hp -= dmg;
            reward += (dmg / c->boss_hp_max) * c->rew_boss_dmg;
        }
    }

    /* enemy bullets vs player (robe DEF 17) */
    if (env->n_ebul > 0) {
        float thr = c->player_radius + c->ebullet_radius;
        double dmg = 0.0;
        int any = 0, w = 0;
        for (int i = 0; i < env->n_ebul; i++) {
            if (dist_ff(env->ebul[i].x, env->ebul[i].y, env->px, env->py) < thr) {
                dmg += defended(env->ebul[i].dmg, c->player_defense, c->damage_floor);
                any = 1;
            } else {
                env->ebul[w++] = env->ebul[i];
            }
        }
        if (any) {
            env->n_ebul = w;
            env->player_hp -= (float)dmg;
            reward -= (dmg / c->player_hp_max) * c->rew_damage_taken;
        }
    }
    return reward;
}

/* --- wizard --- */

static void fire_staff(Dungeon* env, float dx, float dy) {
    Config* c = &env->cfg;
    float perpx = -dy * c->staff_offset, perpy = dx * c->staff_offset;
    float vx = dx * c->staff_speed, vy = dy * c->staff_speed;
    for (int i = 0; i < c->staff_num; i++) {
        float k = (float)i - (c->staff_num - 1) / 2.0f;
        Bullet b = {env->px + perpx * k, env->py + perpy * k, vx, vy, (float)c->staff_life,
                    uniform_f(env, c->staff_dmg_lo, c->staff_dmg_hi)};
        append_bullet(env->pbul, &env->n_pbul, MAX_PBULLETS, c->max_bullets, b);
    }
}

/* Spell of Galactic Creation: a 360-degree BulletNova of spell_num bullets evenly over a full
 * circle, emitted from the player position (point-blank). Aim is irrelevant for a full circle. */
static void cast_spell(Dungeon* env) {
    Config* c = &env->cfg;
    int n = c->spell_num;
    for (int i = 0; i < n; i++) {
        double a = i * (2.0 * M_PI / n);
        Bullet b = {env->px, env->py, (float)cos(a) * c->spell_speed, (float)sin(a) * c->spell_speed,
                    (float)c->spell_life, uniform_f(env, c->spell_dmg_lo, c->spell_dmg_hi)};
        append_bullet(env->pbul, &env->n_pbul, MAX_PBULLETS, c->max_bullets, b);
    }
}

/* --- boss --- */

static void spawn_burst(Dungeon* env, double base_angle, int count, double gap) {
    Config* c = &env->cfg;
    for (int i = 0; i < count; i++) {
        double a = base_angle + (i - (count - 1) / 2.0) * gap;
        Bullet b = {(float)env->boss_x, (float)env->boss_y, (float)cos(a) * c->ebullet_speed,
                    (float)sin(a) * c->ebullet_speed, (float)c->ebullet_life, c->ebullet_dmg};
        append_bullet(env->ebul, &env->n_ebul, MAX_EBULLETS, c->max_bullets, b);
    }
}

/* Blade shot: once the cooldown elapses, fire only if the player is within acquire_radius (P1 is
 * point-blank radius 2; P3 aims at range). The cooldown holds at 0 until the player is in range. */
static void aimed_shoot(Dungeon* env, int* timer, int count, float spread_deg, int cooldown, float acquire_radius) {
    if (*timer > 0) {
        (*timer)--;
        return;
    }
    if (dist_df(env->boss_x, env->boss_y, env->px, env->py) > acquire_radius) return;
    *timer = cooldown;
    double base = atan2((double)(env->py - env->boss_y), (double)(env->px - env->boss_x));
    spawn_burst(env, base, count, spread_deg * M_PI / 180.0);
}

/* Confused grenade: thrown at the player once the cooldown elapses AND the player is within the
 * boss's acquire range (matches the real Grenade gating on GetNearestEntity). */
static void throw_grenade_targeted(Dungeon* env, int* timer, int cooldown, float throw_range, float radius, float dmg, int status) {
    if (!env->cfg.enable_grenades) return;
    if (*timer > 0) {
        (*timer)--;
        return;
    }
    if (dist_df(env->boss_x, env->boss_y, env->px, env->py) > throw_range) return;
    *timer = cooldown;
    if (env->n_gren < MAX_GRENADES) {
        Grenade g = {env->px, env->py, (float)env->cfg.grenade_fuse, radius, dmg, (float)status};
        env->grenades[env->n_gren++] = g;
    }
}

/* P3 Petrify fan: 4 fixed-angle grenades thrown to a fixed distance from the boss. */
static void throw_fixed_grenades(Dungeon* env, int* timer, int cooldown, const float* angles_deg, int n_angles) {
    Config* c = &env->cfg;
    if (!c->enable_grenades) return;
    if (*timer > 0) {
        (*timer)--;
        return;
    }
    *timer = cooldown;
    for (int i = 0; i < n_angles; i++) {
        if (env->n_gren >= MAX_GRENADES) break;
        double a = angles_deg[i] * M_PI / 180.0;
        float tx = (float)(env->boss_x + c->grenade_petrify_dist * cos(a));
        float ty = (float)(env->boss_y + c->grenade_petrify_dist * sin(a));
        Grenade g = {tx, ty, (float)c->grenade_fuse, c->grenade_radius_petrify, c->grenade_dmg_petrify, 1.0f};
        env->grenades[env->n_gren++] = g;
    }
}

static int count_alive_snakes(Dungeon* env) {
    int n = 0;
    for (int i = 0; i < env->n_snake; i++)
        if (env->snakes[i].hp > 0.0f) n++;
    return n;
}

static void spawn_minions(Dungeon* env) {
    Config* c = &env->cfg;
    if (!c->enable_minions) return;
    if (env->minion_timer > 0) {
        env->minion_timer--;
        return;
    }
    env->minion_timer = c->minion_cd;
    if (count_alive_snakes(env) >= c->n_snakes + c->minion_max) return;
    for (int i = 0; i < c->minion_max; i++) {
        if (env->n_snake >= MAX_SNAKES) break;
        double ang = uniform_f(env, 0.0f, (float)(2.0 * M_PI));
        Snake s = {(float)(env->boss_x + cos(ang) * 3.0), (float)(env->boss_y + sin(ang) * 3.0), c->minion_hp, 0.0f, 0.0f};
        env->snakes[env->n_snake++] = s;  /* weak swarm: type 0 */
    }
}

static void boss_tick(Dungeon* env) {
    Config* c = &env->cfg;
    if (env->invuln_timer > 0) {
        env->invuln_timer--;
    } else {
        double frac = env->boss_hp / c->boss_hp_max;
        if (env->phase == 1 && frac <= 0.66f) {
            env->phase = 2;
            env->invuln_timer = c->invuln_ticks;
        } else if (env->phase == 2 && frac <= 0.33f) {
            env->phase = 3;
            env->invuln_timer = c->invuln_ticks;
        }
    }
    /* movement: Wander(0.3) random drift in P1, stationary in P2/P3. boss_pos stays double. */
    if (env->phase == 1 && c->boss_wander_speed > 0.0f) {
        double dx = randn(env) * c->boss_wander_speed, dy = randn(env) * c->boss_wander_speed;
        double cx = env->boss_x + dx, cy = env->boss_y + dy;
        if (walkable_at((float)cx, (float)cy)) {
            env->boss_x = cx;
            env->boss_y = cy;
        }
    }
    /* ReturnToSpawn(0.7, 1): gentle pull back toward spawn while wandering in P1, so the boss
     * can't random-walk out of the room. Only fires once drift exceeds the anchor radius. */
    if (env->phase == 1 && c->boss_return_speed > 0.0f) {
        double dx = env->boss_spawn_x - env->boss_x, dy = env->boss_spawn_y - env->boss_y;
        double d = sqrt(dx * dx + dy * dy);
        if (d > BOSS_RETURN_RADIUS) {
            double step = c->boss_return_speed < d ? c->boss_return_speed : d;
            double nx = env->boss_x + dx / d * step, ny = env->boss_y + dy / d * step;
            if (walkable_at((float)nx, (float)ny)) {
                env->boss_x = nx;
                env->boss_y = ny;
            }
        }
    }
    if (env->invuln_timer > 0) return;

    static const float CARDINALS[4] = {0.0f, 90.0f, 180.0f, 270.0f};
    static const float DIAGONALS[4] = {45.0f, 135.0f, 225.0f, 315.0f};
    if (env->phase == 1) {
        if (c->boss_shoots) aimed_shoot(env, &env->t_p1, 3, 15.0f, c->blade_cd, c->blade_radius_p1);
        spawn_minions(env);
        throw_grenade_targeted(env, &env->t_g1, c->grenade_cd_p1, c->grenade_range_confuse, c->grenade_radius_confuse, c->grenade_dmg_confuse, 0);
    } else if (env->phase == 2) {
        /* P2's 4-shot references a nonexistent projectile -> fires nothing; only the grenade threatens */
        throw_grenade_targeted(env, &env->t_g2, c->grenade_cd_p2, c->grenade_range_confuse, c->grenade_radius_confuse, c->grenade_dmg_confuse, 0);
    } else if (env->phase == 3) {
        if (c->boss_shoots) aimed_shoot(env, &env->t_p3a, 3, 15.0f, c->blade_cd, c->blade_radius_p3);
        throw_fixed_grenades(env, &env->t_g3card, c->grenade_cd_p1, CARDINALS, 4);
        throw_fixed_grenades(env, &env->t_g3diag, c->grenade_cd_p3_diag, DIAGONALS, 4);
    }
}

static double grenades_tick(Dungeon* env) {
    Config* c = &env->cfg;
    if (env->n_gren == 0) return 0.0;
    double reward = 0.0;
    int w = 0;
    for (int i = 0; i < env->n_gren; i++) {
        Grenade g = env->grenades[i];
        g.fuse -= 1.0f;
        if (g.fuse <= 0.0f) {
            if (dist_ff(env->px, env->py, g.x, g.y) <= g.rad) {
                float dmg = (float)defended(g.dmg, c->player_defense, c->damage_floor);
                env->player_hp -= dmg;
                reward -= (dmg / c->player_hp_max) * c->rew_damage_taken;
                if ((int)g.status == 0)
                    env->confused_timer = c->confused_ticks;
                else
                    env->petrify_timer = c->petrify_ticks;
            }
        } else {
            env->grenades[w++] = g;
        }
    }
    env->n_gren = w;
    return reward;
}

/* --- snakes --- */

static int sample_snake_type(Dungeon* env) {
    float r = frand(env), cum = 0.0f;
    for (int t = 0; t < N_SNAKE_TYPES; t++) {
        cum += SNAKE_WEIGHTS[t];
        if (r < cum) return t;
    }
    return N_SNAKE_TYPES - 1;
}

static void snakes_tick(Dungeon* env) {
    Config* c = &env->cfg;
    for (int i = 0; i < env->n_snake; i++) {
        if (env->snakes[i].hp <= 0.0f) continue;
        const float* st = SNAKE_TYPES[(int)env->snakes[i].type];
        double d = dist_ff(env->snakes[i].x, env->snakes[i].y, env->px, env->py);
        /* movement: Follow chase toward the player within acquire range, else Wander drift */
        float mvx, mvy;
        if (st[ST_FOLLOW] > 0.0f && d <= st[ST_ACQ]) {
            double inv = 1.0 / (d + 1e-6);
            mvx = (float)((env->px - env->snakes[i].x) * inv) * st[ST_FSPD];
            mvy = (float)((env->py - env->snakes[i].y) * inv) * st[ST_FSPD];
        } else {
            mvx = randn(env) * c->snake_speed;
            mvy = randn(env) * c->snake_speed;
        }
        float cx = env->snakes[i].x + mvx, cy = env->snakes[i].y + mvy;
        if (walkable_at(cx, cy)) {
            env->snakes[i].x = cx;
            env->snakes[i].y = cy;
        }
        env->snakes[i].timer -= 1.0f;
        if (env->snakes[i].timer <= 0.0f && d <= st[ST_SRANGE]) {
            env->snakes[i].timer = st[ST_CD];
            double base = atan2((double)(env->py - env->snakes[i].y), (double)(env->px - env->snakes[i].x));
            int cnt = (int)st[ST_CNT];
            for (int j = 0; j < cnt; j++) {
                double a = base + (j - (cnt - 1) / 2.0) * st[ST_ARC];
                Bullet b = {env->snakes[i].x, env->snakes[i].y, (float)cos(a) * st[ST_BVS],
                            (float)sin(a) * st[ST_BVS], st[ST_BLIFE], st[ST_DMG]};
                append_bullet(env->ebul, &env->n_ebul, MAX_EBULLETS, c->max_bullets, b);
            }
        }
    }
}

static void spawn_snakes(Dungeon* env) {
    Config* c = &env->cfg;
    env->n_snake = 0;
    int want = c->n_snakes;
    /* per-episode density jitter: spread the snake count in a band around the scheduled target so a
     * single difficulty d spans easier/harder episodes within a batch (no draw when jitter == 0). */
    if (c->n_snakes_jitter > 0) {
        int span = 2 * c->n_snakes_jitter + 1;
        want += (int)(rng_next(env) % (unsigned)span) - c->n_snakes_jitter;
        if (want < 0) want = 0;
    }
    if (want > g_n_walk) want = g_n_walk;
    /* sample distinct walkable tiles + a weighted archetype (training randomness; not parity-matched) */
    for (int s = 0; s < want; s++) {
        if (env->n_snake >= MAX_SNAKES) break;
        int idx = rng_next(env) % g_n_walk;
        int x = g_walk_x[idx], y = g_walk_y[idx];
        if (abs(x - ENTRANCE_X) + abs(y - ENTRANCE_Y) <= 6) continue;
        int type = sample_snake_type(env);
        Snake sn = {x + 0.5f, y + 0.5f, SNAKE_TYPES[type][ST_HP], (float)(rng_next(env) % SNAKE_TIMER_JITTER), (float)type};
        env->snakes[env->n_snake++] = sn;
    }
}

/* --- player movement --- */

static void try_move(Dungeon* env, float sx, float sy) {
    if (walkable_at(env->px + sx, env->py)) env->px = env->px + sx;
    if (walkable_at(env->px, env->py + sy)) env->py = env->py + sy;
}

/* --- observation --- */

static inline void set_cell(float* obs, int ch, int row, int col, float v) {
    obs[ch * GRID * GRID + row * GRID + col] = v;
}

static void scatter_f(float* obs, int ch, float relx, float rely, float v) {
    int col = (int)floorf(relx) + HALF;
    int row = (int)floorf(rely) + HALF;
    if (col >= 0 && col < GRID && row >= 0 && row < GRID) set_cell(obs, ch, row, col, v);
}

/* Fog of war: mark the disk of tiles within VIS_RADIUS of the player as discovered (integer tile
 * arithmetic, matching the numpy oracle), accumulate the minimap terrain pool for newly-seen tiles,
 * and flag whether the boss has been seen. The boss only enters the minimap once seen (no cheat). */
static void update_visibility(Dungeon* env) {
    int ipx = (int)env->px, ipy = (int)env->py;
    for (int dy = -VIS_RADIUS; dy <= VIS_RADIUS; dy++) {
        int y = ipy + dy;
        if (y < 0 || y >= MAP_H) continue;
        for (int dx = -VIS_RADIUS; dx <= VIS_RADIUS; dx++) {
            if (dx * dx + dy * dy > VIS_RADIUS * VIS_RADIUS) continue;
            int x = ipx + dx;
            if (x < 0 || x >= MAP_W) continue;
            int idx = y * MAP_W + x;
            if (env->discovered[idx]) continue;
            env->discovered[idx] = 1;
            int cell = (y * MM / MAP_H) * MM + (x * MM / MAP_W);
            /* priority: discovered walkable (+1) over discovered wall (-1) over fog (0) */
            if (MAP_WALKABLE[idx])
                env->mm_terr[cell] = 1.0f;
            else if (env->mm_terr[cell] != 1.0f)
                env->mm_terr[cell] = -1.0f;
        }
    }
    if (dist_df(env->boss_x, env->boss_y, env->px, env->py) <= VIS_RADIUS) env->boss_seen = 1;
}

static void compute_obs(Dungeon* env) {
    Config* c = &env->cfg;
    float* obs = env->observations;
    int ipx = (int)env->px, ipy = (int)env->py;
    update_visibility(env);

    /* Wall channel (0) depends only on the player's tile; the dynamic channels (1..6) + scalars
     * change every step. Clear/refill walls only when the tile changes; always clear the rest. */
    if (ipx != env->last_ipx || ipy != env->last_ipy) {
        memset(obs, 0, sizeof(float) * (GRID * GRID));  /* clear wall channel */
        for (int row = 0; row < GRID; row++) {
            int wy = ipy + row - HALF;
            if (wy < 0 || wy >= MAP_H) continue;
            for (int col = 0; col < GRID; col++) {
                int wx = ipx + col - HALF;
                if (wx >= 0 && wx < MAP_W && !MAP_WALKABLE[wy * MAP_W + wx])
                    set_cell(obs, CH_WALL, row, col, 1.0f);
            }
        }
        env->last_ipx = ipx;
        env->last_ipy = ipy;
    }
    memset(obs + GRID * GRID, 0, sizeof(float) * (OBS_SIZE - GRID * GRID));  /* channels 1..6 + scalars */

    for (int i = 0; i < env->n_snake; i++)
        if (env->snakes[i].hp > 0.0f)
            scatter_f(obs, CH_ENEMY, env->snakes[i].x - env->px, env->snakes[i].y - env->py, 0.6f);

    int boss_visible = env->fight_active && dist_df(env->boss_x, env->boss_y, env->px, env->py) <= VIS_RADIUS;
    if (boss_visible) {
        int col = (int)floor(env->boss_x - env->px) + HALF;
        int row = (int)floor(env->boss_y - env->py) + HALF;
        if (col >= 0 && col < GRID && row >= 0 && row < GRID) set_cell(obs, CH_ENEMY, row, col, 1.0f);
    }

    for (int i = 0; i < env->n_ebul; i++) {
        float relx = env->ebul[i].x - env->px, rely = env->ebul[i].y - env->py;
        scatter_f(obs, CH_EBULLET, relx, rely, 1.0f);
        float vn = sqrtf(env->ebul[i].vx * env->ebul[i].vx + env->ebul[i].vy * env->ebul[i].vy) + 1e-6f;
        scatter_f(obs, CH_EBVX, relx, rely, env->ebul[i].vx / vn);
        scatter_f(obs, CH_EBVY, relx, rely, env->ebul[i].vy / vn);
    }

    for (int i = 0; i < env->n_pbul; i++)
        scatter_f(obs, CH_PBULLET, env->pbul[i].x - env->px, env->pbul[i].y - env->py, 1.0f);

    for (int i = 0; i < env->n_gren; i++) {
        float urgency = 1.0f - env->grenades[i].fuse / (float)(c->grenade_fuse > 1 ? c->grenade_fuse : 1);
        if (urgency < 0.2f) urgency = 0.2f;
        if (urgency > 1.0f) urgency = 1.0f;
        scatter_f(obs, CH_GRENADE, env->grenades[i].x - env->px, env->grenades[i].y - env->py, urgency);
    }

    /* minimap (offset GRID_SIZE): terrain pool fresh from mm_terr, plus single player/boss cells. The
     * GRID_SIZE..OBS_SIZE region was already zeroed above, so player/boss channels start clean. */
    float* mmobs = obs + GRID_SIZE;
    for (int i = 0; i < MM * MM; i++) mmobs[i] = env->mm_terr[i];
    int pmx = (int)env->px * MM / MAP_W, pmy = (int)env->py * MM / MAP_H;
    mmobs[MM * MM + pmy * MM + pmx] = 1.0f;
    if (env->boss_seen) {
        int bmx = (int)env->boss_x * MM / MAP_W, bmy = (int)env->boss_y * MM / MAP_H;
        mmobs[2 * MM * MM + bmy * MM + bmx] = 1.0f;
    }

    int spell_ready = (env->player_mp >= c->spell_cost && env->spell_timer == 0);
    obs[SCALAR_OFF + 0] = env->player_hp / c->player_hp_max;
    obs[SCALAR_OFF + 1] = env->player_mp / c->player_mp_max;
    obs[SCALAR_OFF + 2] = spell_ready ? 1.0f : 0.0f;
    obs[SCALAR_OFF + 3] = boss_visible ? 1.0f : 0.0f;
    obs[SCALAR_OFF + 4] = env->confused_timer > 0 ? 1.0f : 0.0f;
    obs[SCALAR_OFF + 5] = env->petrify_timer > 0 ? 1.0f : 0.0f;
    obs[SCALAR_OFF + 6] = env->fight_active ? (float)((env->boss_hp > 0.0 ? env->boss_hp : 0.0) / c->boss_hp_max) : 0.0f;
    obs[SCALAR_OFF + 7] = env->invuln_timer > 0 ? 1.0f : 0.0f;
}

/* --- required Ocean API --- */

static void c_reset(Dungeon* env) {
    Config* c = &env->cfg;
    init_globals();
    env->steps = 0;
    env->last_ipx = env->last_ipy = -1000000; /* force a wall-channel rebuild on next obs */
    int bx, by, ex, ey;
    nearest_walkable(BOSS_X, BOSS_Y, &bx, &by);
    nearest_walkable(ENTRANCE_X, ENTRANCE_Y, &ex, &ey);

    float roll = frand(env);
    if (roll < c->random_spawn_prob) {
        int i = rng_next(env) % g_n_walk;
        env->px = g_walk_x[i] + 0.5f;
        env->py = g_walk_y[i] + 0.5f;
    } else if (roll < c->random_spawn_prob + c->spawn_in_room_prob) {
        double ang = uniform_f(env, 0.0f, (float)(2.0 * M_PI));
        int cx = (int)(bx + c->spawn_in_room_radius * cos(ang)), cy = (int)(by + c->spawn_in_room_radius * sin(ang));
        if (cx < 1) cx = 1;
        if (cx > MAP_W - 2) cx = MAP_W - 2;
        if (cy < 1) cy = 1;
        if (cy > MAP_H - 2) cy = MAP_H - 2;
        int sx, sy;
        nearest_walkable(cx, cy, &sx, &sy);
        env->px = sx + 0.5f;
        env->py = sy + 0.5f;
    } else {
        env->px = ex + 0.5f;
        env->py = ey + 0.5f;
    }
    env->player_hp = c->player_hp_max;
    env->player_mp = c->player_mp_max;
    env->staff_timer = 0.0;
    env->spell_timer = 0;
    env->boss_x = bx + 0.5;
    env->boss_y = by + 0.5;
    env->boss_spawn_x = bx + 0.5;
    env->boss_spawn_y = by + 0.5;
    env->prev_boss_dist = dist_df(env->boss_x, env->boss_y, env->px, env->py);
    env->boss_hp = c->boss_hp_max;
    env->phase = 0;
    env->fight_active = 0;
    env->invuln_timer = 0;
    env->t_p1 = env->t_p3a = env->t_g1 = env->t_g2 = env->t_g3card = env->t_g3diag = 0;
    env->confused_timer = env->petrify_timer = env->minion_timer = 0;
    env->n_pbul = env->n_ebul = env->n_gren = 0;
    memset(env->visited, 0, sizeof(env->visited));
    memset(env->discovered, 0, sizeof(env->discovered));
    memset(env->mm_terr, 0, sizeof(env->mm_terr));
    env->boss_seen = 0;
    spawn_snakes(env);
    compute_obs(env);
}

static void c_step(Dungeon* env) {
    Config* c = &env->cfg;
#ifdef PUFFER4
    int move_idx = (int)env->actions[0], aim_idx = (int)env->actions[1];
    int shoot = (int)env->actions[2], cast = (int)env->actions[3];
#else
    int move_idx = env->actions[0], aim_idx = env->actions[1];
    int shoot = env->actions[2], cast = env->actions[3];
#endif
    double reward = c->rew_step;

    if (move_idx > 0 && env->petrify_timer == 0) {
        float mvx = g_move_dx[move_idx - 1] * c->player_speed;
        float mvy = g_move_dy[move_idx - 1] * c->player_speed;
        if (env->confused_timer > 0) {
            mvx = -mvx;
            mvy = -mvy;
        }
        try_move(env, mvx, mvy);
    }
    if (env->confused_timer > 0) env->confused_timer--;
    if (env->petrify_timer > 0) env->petrify_timer--;

    int tx = (int)env->px, ty = (int)env->py;
    if (ty >= 0 && ty < MAP_H && tx >= 0 && tx < MAP_W && !env->visited[ty * MAP_W + tx]) {
        env->visited[ty * MAP_W + tx] = 1;
        reward += c->rew_explore;
    }

    env->staff_timer -= 1.0;  /* fractional cooldown: carried on fire, snapped up to 0 when idle below */
    if (env->spell_timer > 0) env->spell_timer--;
    env->player_mp = env->player_mp + c->mp_regen;
    if (env->player_mp > c->player_mp_max) env->player_mp = c->player_mp_max;
    /* HealthRegen (Player.cs HandleRegen): (1 + 0.36*VIT)/s, here a flat per-tick rate, capped at max */
    if (env->player_hp < c->player_hp_max) {
        env->player_hp = env->player_hp + c->hp_regen;
        if (env->player_hp > c->player_hp_max) env->player_hp = c->player_hp_max;
    }

    float aimx = g_aim_dx[aim_idx], aimy = g_aim_dy[aim_idx];
    if (shoot == 1 && env->staff_timer <= 0.0) {
        fire_staff(env, aimx, aimy);
        env->staff_timer += c->staff_cooldown;
    } else if (env->staff_timer < 0.0) {
        env->staff_timer = 0.0;
    }
    if (cast == 1 && env->spell_timer == 0 && env->player_mp >= c->spell_cost) {
        cast_spell(env);
        env->player_mp -= c->spell_cost;
        env->spell_timer = c->spell_cooldown;
    }

    if (!env->fight_active && c->rew_approach != 0.0f) {  /* dense gradient toward the boss while navigating */
        double cur_boss_dist = dist_df(env->boss_x, env->boss_y, env->px, env->py);
        reward += c->rew_approach * (env->prev_boss_dist - cur_boss_dist);
        env->prev_boss_dist = cur_boss_dist;
    }

    if (!env->fight_active && dist_df(env->boss_x, env->boss_y, env->px, env->py) <= c->activation_range) {
        env->fight_active = 1;
        env->phase = 1;
        env->invuln_timer = c->opening_invuln_ticks;  /* 1.0s invuln taunt before P1 acts */
        reward += c->rew_reach;
    }

    snakes_tick(env);
    if (env->fight_active) boss_tick(env);
    reward += grenades_tick(env);

    advance_bullets(env->pbul, &env->n_pbul);
    advance_bullets(env->ebul, &env->n_ebul);
    reward += resolve_collisions(env);

    reward += c->rew_survive;
    env->steps++;

    int terminated = 0, cleared = 0;
    if (env->boss_hp <= 0.0 && env->phase > 0) {
        terminated = 1;
        cleared = 1;
        reward += c->rew_clear;
    } else if (env->player_hp <= 0.0f) {
        terminated = 1;
        reward -= c->rew_death;
    }
    int truncated = (!terminated) && env->steps >= c->max_steps;

    env->rewards[0] = (float)reward;
    env->terminals[0] = terminated ? 1 : 0;

    /* Per-step metrics on the post-step (terminal) state, before any auto-reset, so they match
     * the numpy oracle's per-step info dict. On a clear step boss_hp<=0 -> boss_hp_frac=0,
     * cleared=1; on a normal step the boss's remaining HP fraction is recorded. */
    env->log.boss_hp_frac += (env->boss_hp > 0.0f ? env->boss_hp : 0.0f) / c->boss_hp_max;
    env->log.in_room += env->fight_active ? 1.0f : 0.0f;
    env->log.cleared += cleared ? 1.0f : 0.0f;
    env->log.snakes += (float)count_alive_snakes(env);
    env->log.player_hp_frac += env->player_hp / c->player_hp_max;
    env->log.reward += (float)reward;
    env->log.perf += cleared ? 1.0f : 0.0f;
    env->log.n += 1.0f;

    /* Per-episode score, recorded once at the episode boundary (before c_reset wipes boss_hp). */
    if (terminated || truncated) {
        float bhf_end = (env->boss_hp > 0.0 ? (float)env->boss_hp : 0.0f) / c->boss_hp_max;
        env->log.score += cleared ? 1.0f : (1.0f - bhf_end);
        env->log.clear_count += cleared ? 1.0f : 0.0f;
        env->log.player_hp_end_sum += (env->player_hp > 0.0f ? env->player_hp : 0.0f) / c->player_hp_max;
        env->log.death_count += (terminated && !cleared) ? 1.0f : 0.0f;  // terminated but boss alive = died
        env->log.episodes += 1.0f;
    }

    compute_obs(env);

    if (terminated || truncated) c_reset(env);
}

static void c_render(Dungeon* env) { (void)env; }

static void c_close(Dungeon* env) { (void)env; }

#endif
