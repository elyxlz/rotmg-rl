/* Faithful C port of rotmg_rl.sim.dungeon.DungeonEnv (the Snake Pit dungeon sim).
 *
 * The Python sim is the oracle; this matches its obs layout, action space, and dynamics so a
 * policy trained here transfers. Determinism notes for the parity test: with n_snakes=0 and
 * enable_minions=0 and point-mass damage ranges (lo==hi), the whole sim is RNG-free and matches
 * the Python oracle bit-for-faithfully (positions float32, distances float64, as in numpy<2.0).
 * Snake spawn/wander and minion placement use rand() and are not bit-matched (stochastic by design).
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
#define NUM_SCALARS 6
#define GRID_SIZE (NUM_CH * GRID * GRID)
#define OBS_SIZE (GRID_SIZE + NUM_SCALARS)

#define CH_WALL 0
#define CH_ENEMY 1
#define CH_EBULLET 2
#define CH_EBVX 3
#define CH_EBVY 4
#define CH_PBULLET 5
#define CH_GRENADE 6

#define MAX_PBULLETS 4096
#define MAX_EBULLETS 4096
#define MAX_SNAKES 512
#define MAX_GRENADES 64

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float cleared;
    float boss_hp_frac;
    float n;
} Log;

typedef struct {
    float x, y, vx, vy, life, dmg;
} Bullet;

typedef struct {
    float x, y, hp, timer;
} Snake;

typedef struct {
    float x, y, fuse, rad, dmg, status;
} Grenade;

typedef struct {
    /* --- DungeonConfig mirror --- */
    float player_speed, player_radius;
    int max_steps;
    float activation_range, spawn_in_room_prob, random_spawn_prob;
    float player_hp_max, player_mp_max, mp_regen;
    int staff_cooldown, staff_num;
    float staff_dmg_lo, staff_dmg_hi, staff_speed;
    int staff_life;
    float staff_radius, staff_offset, spell_cost;
    int spell_cooldown, spell_num;
    float spell_arc_deg, spell_dmg_lo, spell_dmg_hi, spell_speed;
    int spell_life;
    int n_snakes;
    float snake_hp, snake_speed, snake_shoot_range;
    int snake_cooldown;
    float snake_bullet_speed;
    int snake_bullet_life;
    float snake_bullet_dmg, snake_radius;
    float boss_hp_max, boss_radius, boss_speed;
    int boss_shoots, invuln_ticks;
    float ebullet_speed;
    int ebullet_life;
    float ebullet_dmg, ebullet_radius;
    int max_bullets;
    int grenade_fuse, grenade_cd_p1, grenade_cd_p2;
    float grenade_radius_confuse, grenade_dmg_confuse, grenade_radius_petrify, grenade_dmg_petrify;
    int confused_ticks, petrify_ticks;
    int minion_max, minion_cd;
    float minion_hp;
    int enable_grenades, enable_minions;
    float rew_explore, rew_kill, rew_boss_dmg, rew_reach, rew_survive;
    float rew_damage_taken, rew_clear, rew_death, rew_step;
} Config;

typedef struct {
    Log log;
    float* observations;  /* OBS_SIZE float32 */
    int* actions;         /* 4 ints: move, aim, shoot, cast */
    float* rewards;       /* 1 float */
    unsigned char* terminals;

    Config cfg;

    float px, py;          /* player pos (float32 in oracle) */
    float player_hp, player_mp;
    int staff_timer, spell_timer;
    double boss_x, boss_y; /* boss_pos becomes float64 in oracle after first move */
    float boss_hp;
    int phase, fight_active, invuln_timer;
    int confused_timer, petrify_timer, minion_timer;
    double rotate_angle;
    int t_p1, t_p2, t_p3a, t_p3b, t_g1, t_g2, t_g3;

    Bullet pbul[MAX_PBULLETS];
    int n_pbul;
    Bullet ebul[MAX_EBULLETS];
    int n_ebul;
    Snake snakes[MAX_SNAKES];
    int n_snake;
    Grenade grenades[MAX_GRENADES];
    int n_gren;

    unsigned char visited[MAP_H * MAP_W];
    int steps;
    double ep_return;
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
    double dx = ax - bx, dy = ay - by;  /* boss_pos is float64 once moved */
    return sqrt(dx * dx + dy * dy);
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

    /* player bullets vs snakes (index order; removing hits affects later snakes) */
    for (int s = 0; s < env->n_snake; s++) {
        if (env->snakes[s].hp <= 0.0f) continue;
        if (env->n_pbul == 0) break;
        float thr = c->snake_radius + c->staff_radius;
        double dmg = 0.0;
        int any = 0, w = 0;
        for (int i = 0; i < env->n_pbul; i++) {
            if (dist_ff(env->pbul[i].x, env->pbul[i].y, env->snakes[s].x, env->snakes[s].y) < thr) {
                dmg += env->pbul[i].dmg;
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

    /* player bullets vs boss */
    if (env->n_pbul > 0 && env->invuln_timer == 0 && env->phase > 0) {
        float thr = c->boss_radius + c->staff_radius;
        double dmg = 0.0;
        int any = 0, w = 0;
        for (int i = 0; i < env->n_pbul; i++) {
            if (dist_df(env->boss_x, env->boss_y, env->pbul[i].x, env->pbul[i].y) < thr) {
                dmg += env->pbul[i].dmg;
                any = 1;
            } else {
                env->pbul[w++] = env->pbul[i];
            }
        }
        if (any) {
            env->n_pbul = w;
            env->boss_hp -= (float)dmg;
            reward += dmg * c->rew_boss_dmg;
        }
    }

    /* enemy bullets vs player */
    if (env->n_ebul > 0) {
        float thr = c->player_radius + c->ebullet_radius;
        double dmg = 0.0;
        int any = 0, w = 0;
        for (int i = 0; i < env->n_ebul; i++) {
            if (dist_ff(env->ebul[i].x, env->ebul[i].y, env->px, env->py) < thr) {
                dmg += env->ebul[i].dmg;
                any = 1;
            } else {
                env->ebul[w++] = env->ebul[i];
            }
        }
        if (any) {
            env->n_ebul = w;
            env->player_hp -= (float)dmg;
            reward -= dmg * c->rew_damage_taken;
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

static void cast_spell(Dungeon* env, float dx, float dy) {
    Config* c = &env->cfg;
    double base = atan2((double)dy, (double)dx);
    double half_arc = c->spell_arc_deg * M_PI / 180.0 / 2.0;
    int n = c->spell_num;
    for (int i = 0; i < n; i++) {
        double t = (n == 1) ? 0.0 : (-half_arc + (2.0 * half_arc) * i / (n - 1));
        double a = base + t;
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

static void aimed_shoot(Dungeon* env, int* timer, int count, float spread_deg, int cooldown) {
    if (*timer > 0) {
        (*timer)--;
        return;
    }
    *timer = cooldown;
    double base = atan2((double)(env->py - env->boss_y), (double)(env->px - env->boss_x));
    spawn_burst(env, base, count, spread_deg * M_PI / 180.0);
}

static void rotating_shoot(Dungeon* env, int* timer, int count, float step_deg, int cooldown) {
    if (*timer > 0) {
        (*timer)--;
        return;
    }
    *timer = cooldown;
    env->rotate_angle += step_deg * M_PI / 180.0;
    spawn_burst(env, env->rotate_angle, count, 2.0 * M_PI / count);
}

static void throw_grenade(Dungeon* env, int* timer, int cooldown, float radius, float dmg, int status) {
    if (!env->cfg.enable_grenades) return;
    if (*timer > 0) {
        (*timer)--;
        return;
    }
    *timer = cooldown;
    if (env->n_gren < MAX_GRENADES) {
        Grenade g = {env->px, env->py, (float)env->cfg.grenade_fuse, radius, dmg, (float)status};
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
        Snake s = {(float)(env->boss_x + cos(ang) * 3.0), (float)(env->boss_y + sin(ang) * 3.0), c->minion_hp, 0.0f};
        env->snakes[env->n_snake++] = s;
    }
}

static void boss_tick(Dungeon* env) {
    Config* c = &env->cfg;
    if (env->invuln_timer > 0) {
        env->invuln_timer--;
    } else {
        float frac = env->boss_hp / c->boss_hp_max;
        if (env->phase == 1 && frac <= 0.66f) {
            env->phase = 2;
            env->invuln_timer = c->invuln_ticks;
        } else if (env->phase == 2 && frac <= 0.33f) {
            env->phase = 3;
            env->invuln_timer = c->invuln_ticks;
        }
    }
    /* boss_pos = boss_pos + unit(player - boss) * boss_speed  (becomes float64) */
    float vx = env->px - (float)env->boss_x, vy = env->py - (float)env->boss_y;
    double nrm = sqrt((double)vx * vx + (double)vy * vy) + 1e-6;
    env->boss_x = env->boss_x + ((double)vx / nrm) * c->boss_speed;
    env->boss_y = env->boss_y + ((double)vy / nrm) * c->boss_speed;
    if (env->invuln_timer > 0) return;

    if (env->phase == 1) {
        if (c->boss_shoots) aimed_shoot(env, &env->t_p1, 3, 15.0f, 15);
        spawn_minions(env);
        throw_grenade(env, &env->t_g1, c->grenade_cd_p1, c->grenade_radius_confuse, c->grenade_dmg_confuse, 0);
    } else if (env->phase == 2) {
        if (c->boss_shoots) rotating_shoot(env, &env->t_p2, 4, 15.0f, 3);
        throw_grenade(env, &env->t_g2, c->grenade_cd_p2, c->grenade_radius_confuse, c->grenade_dmg_confuse, 0);
    } else if (env->phase == 3) {
        if (c->boss_shoots) {
            aimed_shoot(env, &env->t_p3a, 3, 15.0f, 15);
            rotating_shoot(env, &env->t_p3b, 4, 15.0f, 5);
        }
        throw_grenade(env, &env->t_g3, c->grenade_cd_p1, c->grenade_radius_petrify, c->grenade_dmg_petrify, 1);
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
                env->player_hp -= g.dmg;
                reward -= g.dmg * c->rew_damage_taken;
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

static float randn(Dungeon* env) {
    /* Box-Muller; training-only randomness, not parity-matched */
    float u1 = frand(env), u2 = frand(env);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void snakes_tick(Dungeon* env) {
    Config* c = &env->cfg;
    for (int i = 0; i < env->n_snake; i++) {
        if (env->snakes[i].hp <= 0.0f) continue;
        float ddx = randn(env) * c->snake_speed, ddy = randn(env) * c->snake_speed;
        float cx = env->snakes[i].x + ddx, cy = env->snakes[i].y + ddy;
        if (walkable_at(cx, cy)) {
            env->snakes[i].x = cx;
            env->snakes[i].y = cy;
        }
        env->snakes[i].timer -= 1.0f;
        double d = dist_ff(env->snakes[i].x, env->snakes[i].y, env->px, env->py);
        if (env->snakes[i].timer <= 0.0f && d <= c->snake_shoot_range) {
            env->snakes[i].timer = (float)c->snake_cooldown;
            double ang = atan2((double)(env->py - env->snakes[i].y), (double)(env->px - env->snakes[i].x));
            Bullet b = {env->snakes[i].x, env->snakes[i].y, (float)cos(ang) * c->snake_bullet_speed,
                        (float)sin(ang) * c->snake_bullet_speed, (float)c->snake_bullet_life, c->snake_bullet_dmg};
            append_bullet(env->ebul, &env->n_ebul, MAX_EBULLETS, c->max_bullets, b);
        }
    }
}

static void spawn_snakes(Dungeon* env) {
    Config* c = &env->cfg;
    env->n_snake = 0;
    int want = c->n_snakes;
    if (want > g_n_walk) want = g_n_walk;
    /* sample distinct walkable tiles (training randomness; not parity-matched) */
    for (int s = 0; s < want; s++) {
        if (env->n_snake >= MAX_SNAKES) break;
        int idx = rng_next(env) % g_n_walk;
        int x = g_walk_x[idx], y = g_walk_y[idx];
        if (abs(x - ENTRANCE_X) + abs(y - ENTRANCE_Y) <= 6) continue;
        Snake sn = {x + 0.5f, y + 0.5f, c->snake_hp, (float)(rng_next(env) % c->snake_cooldown)};
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

static void compute_obs(Dungeon* env) {
    Config* c = &env->cfg;
    float* obs = env->observations;
    int ipx = (int)env->px, ipy = (int)env->py;

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

    int spell_ready = (env->player_mp >= c->spell_cost && env->spell_timer == 0);
    obs[GRID_SIZE + 0] = env->player_hp / c->player_hp_max;
    obs[GRID_SIZE + 1] = env->player_mp / c->player_mp_max;
    obs[GRID_SIZE + 2] = spell_ready ? 1.0f : 0.0f;
    obs[GRID_SIZE + 3] = boss_visible ? 1.0f : 0.0f;
    obs[GRID_SIZE + 4] = env->confused_timer > 0 ? 1.0f : 0.0f;
    obs[GRID_SIZE + 5] = env->petrify_timer > 0 ? 1.0f : 0.0f;
}

/* --- required Ocean API --- */

static void c_reset(Dungeon* env) {
    Config* c = &env->cfg;
    init_globals();
    env->steps = 0;
    env->ep_return = 0.0;
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
        int cx = (int)(bx + 6 * cos(ang)), cy = (int)(by + 6 * sin(ang));
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
    env->staff_timer = 0;
    env->spell_timer = 0;
    env->boss_x = bx + 0.5;
    env->boss_y = by + 0.5;
    env->boss_hp = c->boss_hp_max;
    env->phase = 0;
    env->fight_active = 0;
    env->invuln_timer = 0;
    env->rotate_angle = 0.0;
    env->t_p1 = env->t_p2 = env->t_p3a = env->t_p3b = env->t_g1 = env->t_g2 = env->t_g3 = 0;
    env->confused_timer = env->petrify_timer = env->minion_timer = 0;
    env->n_pbul = env->n_ebul = env->n_gren = 0;
    memset(env->visited, 0, sizeof(env->visited));
    spawn_snakes(env);
    compute_obs(env);
}

static void c_step(Dungeon* env) {
    Config* c = &env->cfg;
    int move_idx = env->actions[0], aim_idx = env->actions[1];
    int shoot = env->actions[2], cast = env->actions[3];
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

    if (env->staff_timer > 0) env->staff_timer--;
    if (env->spell_timer > 0) env->spell_timer--;
    env->player_mp = env->player_mp + c->mp_regen;
    if (env->player_mp > c->player_mp_max) env->player_mp = c->player_mp_max;

    float aimx = g_aim_dx[aim_idx], aimy = g_aim_dy[aim_idx];
    if (shoot == 1 && env->staff_timer == 0) {
        fire_staff(env, aimx, aimy);
        env->staff_timer = c->staff_cooldown;
    }
    if (cast == 1 && env->spell_timer == 0 && env->player_mp >= c->spell_cost) {
        cast_spell(env, aimx, aimy);
        env->player_mp -= c->spell_cost;
        env->spell_timer = c->spell_cooldown;
    }

    if (!env->fight_active && dist_df(env->boss_x, env->boss_y, env->px, env->py) <= c->activation_range) {
        env->fight_active = 1;
        env->phase = 1;
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
    if (env->boss_hp <= 0.0f && env->phase > 0) {
        terminated = 1;
        cleared = 1;
        reward += c->rew_clear;
    } else if (env->player_hp <= 0.0f) {
        terminated = 1;
        reward -= c->rew_death;
    }
    int truncated = (!terminated) && env->steps >= c->max_steps;

    env->rewards[0] = (float)reward;
    env->ep_return += reward;
    env->terminals[0] = terminated ? 1 : 0;
    compute_obs(env);

    if (terminated || truncated) {
        env->log.score += (float)env->ep_return;
        env->log.episode_return += (float)env->ep_return;
        env->log.episode_length += (float)env->steps;
        env->log.cleared += cleared ? 1.0f : 0.0f;
        env->log.boss_hp_frac += (env->boss_hp > 0 ? env->boss_hp : 0) / c->boss_hp_max;
        env->log.perf += cleared ? 1.0f : 0.0f;
        env->log.n += 1.0f;
        c_reset(env);
    }
}

static void c_render(Dungeon* env) { (void)env; }

static void c_close(Dungeon* env) { (void)env; }

#endif
