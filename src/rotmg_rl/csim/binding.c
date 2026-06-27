#include <Python.h>

#include "dungeon.h"

#define Env Dungeon
#define MY_PUT
#define MY_GET

#include "env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    (void)args;
    Config *c = &env->cfg;
    c->player_speed = unpack(kwargs, "player_speed");
    c->player_radius = unpack(kwargs, "player_radius");
    c->max_steps = (int)unpack(kwargs, "max_steps");
    c->activation_range = unpack(kwargs, "activation_range");
    c->spawn_in_room_prob = unpack(kwargs, "spawn_in_room_prob");
    c->random_spawn_prob = unpack(kwargs, "random_spawn_prob");
    c->spawn_in_room_radius = unpack(kwargs, "spawn_in_room_radius");
    c->player_hp_max = unpack(kwargs, "player_hp_max");
    c->player_mp_max = unpack(kwargs, "player_mp_max");
    c->player_defense = unpack(kwargs, "player_defense");
    c->damage_floor = unpack(kwargs, "damage_floor");
    c->mp_regen = unpack(kwargs, "mp_regen");
    c->hp_regen = unpack(kwargs, "hp_regen");
    c->staff_cooldown = unpack(kwargs, "staff_cooldown");
    c->staff_num = (int)unpack(kwargs, "staff_num");
    c->staff_dmg_lo = unpack(kwargs, "staff_dmg_lo");
    c->staff_dmg_hi = unpack(kwargs, "staff_dmg_hi");
    c->staff_speed = unpack(kwargs, "staff_speed");
    c->staff_life = unpack(kwargs, "staff_life");
    c->staff_radius = unpack(kwargs, "staff_radius");
    c->staff_offset = unpack(kwargs, "staff_offset");
    c->spell_cost = unpack(kwargs, "spell_cost");
    c->spell_cooldown = (int)unpack(kwargs, "spell_cooldown");
    c->spell_num = (int)unpack(kwargs, "spell_num");
    c->spell_dmg_lo = unpack(kwargs, "spell_dmg_lo");
    c->spell_dmg_hi = unpack(kwargs, "spell_dmg_hi");
    c->spell_speed = unpack(kwargs, "spell_speed");
    c->spell_life = (int)unpack(kwargs, "spell_life");
    c->n_snakes = (int)unpack(kwargs, "n_snakes");
    c->n_snakes_jitter = (int)unpack(kwargs, "n_snakes_jitter");
    c->snake_speed = unpack(kwargs, "snake_speed");
    c->snake_radius = unpack(kwargs, "snake_radius");
    c->boss_hp_max = unpack(kwargs, "boss_hp_max");
    c->boss_radius = unpack(kwargs, "boss_radius");
    c->boss_defense = unpack(kwargs, "boss_defense");
    c->boss_wander_speed = unpack(kwargs, "boss_wander_speed");
    c->boss_return_speed = unpack(kwargs, "boss_return_speed");
    c->boss_shoots = (int)unpack(kwargs, "boss_shoots");
    c->opening_invuln_ticks = (int)unpack(kwargs, "opening_invuln_ticks");
    c->invuln_ticks = (int)unpack(kwargs, "invuln_ticks");
    c->blade_cd = (int)unpack(kwargs, "blade_cd");
    c->blade_radius_p1 = unpack(kwargs, "blade_radius_p1");
    c->blade_radius_p3 = unpack(kwargs, "blade_radius_p3");
    c->ebullet_speed = unpack(kwargs, "ebullet_speed");
    c->ebullet_life = (int)unpack(kwargs, "ebullet_life");
    c->ebullet_dmg = unpack(kwargs, "ebullet_dmg");
    c->ebullet_radius = unpack(kwargs, "ebullet_radius");
    c->max_bullets = (int)unpack(kwargs, "max_bullets");
    c->grenade_fuse = (int)unpack(kwargs, "grenade_fuse");
    c->grenade_cd_p1 = (int)unpack(kwargs, "grenade_cd_p1");
    c->grenade_cd_p2 = (int)unpack(kwargs, "grenade_cd_p2");
    c->grenade_cd_p3_diag = (int)unpack(kwargs, "grenade_cd_p3_diag");
    c->grenade_range_confuse = unpack(kwargs, "grenade_range_confuse");
    c->grenade_petrify_dist = unpack(kwargs, "grenade_petrify_dist");
    c->grenade_radius_confuse = unpack(kwargs, "grenade_radius_confuse");
    c->grenade_dmg_confuse = unpack(kwargs, "grenade_dmg_confuse");
    c->grenade_radius_petrify = unpack(kwargs, "grenade_radius_petrify");
    c->grenade_dmg_petrify = unpack(kwargs, "grenade_dmg_petrify");
    c->confused_ticks = (int)unpack(kwargs, "confused_ticks");
    c->petrify_ticks = (int)unpack(kwargs, "petrify_ticks");
    c->minion_max = (int)unpack(kwargs, "minion_max");
    c->minion_cd = (int)unpack(kwargs, "minion_cd");
    c->minion_hp = unpack(kwargs, "minion_hp");
    c->enable_grenades = (int)unpack(kwargs, "enable_grenades");
    c->enable_minions = (int)unpack(kwargs, "enable_minions");
    c->rew_explore = unpack(kwargs, "rew_explore");
    c->rew_kill = unpack(kwargs, "rew_kill");
    c->rew_boss_dmg = unpack(kwargs, "rew_boss_dmg");
    c->rew_reach = unpack(kwargs, "rew_reach");
    c->rew_survive = unpack(kwargs, "rew_survive");
    c->rew_damage_taken = unpack(kwargs, "rew_damage_taken");
    c->rew_clear = unpack(kwargs, "rew_clear");
    c->rew_speed = unpack(kwargs, "rew_speed");
    c->rew_death = unpack(kwargs, "rew_death");
    c->rew_step = unpack(kwargs, "rew_step");
    c->rew_approach = unpack(kwargs, "rew_approach");
    env->rng_state = (uint64_t)(long)unpack(kwargs, "seed") * 2654435761ULL + 0x9E3779B97F4A7C15ULL;
    if (env->rng_state == 0)
        env->rng_state = 1;
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    /* vec_log already divided each field by n (total steps) -> per-step means matching numpy. */
    assign_to_dict(dict, "boss_hp_frac", log->boss_hp_frac);
    assign_to_dict(dict, "in_room", log->in_room);
    assign_to_dict(dict, "cleared", log->cleared);
    assign_to_dict(dict, "snakes", log->snakes);
    assign_to_dict(dict, "player_hp_frac", log->player_hp_frac);
    assign_to_dict(dict, "reward", log->reward);
    assign_to_dict(dict, "perf", log->perf);
    /* score/episodes are per-episode sums (both already divided by n); their ratio is the
     * per-episode mean end-of-episode score -- the metric the Protein sweep maximizes. */
    assign_to_dict(dict, "score", log->episodes > 0.0f ? log->score / log->episodes : 0.0f);
    /* clear_rate = fraction of episodes cleared = the TRUE per-episode clear rate (what we care
     * about), unlike `cleared` above which is a per-STEP rate (clears/steps, ~0.03 even at 100%). */
    assign_to_dict(dict, "clear_rate", log->episodes > 0.0f ? log->clear_count / log->episodes : 0.0f);
    assign_to_dict(dict, "episodes", log->episodes);
    return 0;
}

/* Scenario-test hook: inject deterministic player position + fight state, then refresh obs. */
static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    (void)args;
    PyObject *v;
    v = PyDict_GetItemString(kwargs, "player_x");
    if (v)
        env->px = (float)PyFloat_AsDouble(v);
    v = PyDict_GetItemString(kwargs, "player_y");
    if (v)
        env->py = (float)PyFloat_AsDouble(v);
    v = PyDict_GetItemString(kwargs, "fight_active");
    if (v)
        env->fight_active = (int)PyLong_AsLong(v);
    v = PyDict_GetItemString(kwargs, "phase");
    if (v)
        env->phase = (int)PyLong_AsLong(v);
    v = PyDict_GetItemString(kwargs, "boss_hp");
    if (v)
        env->boss_hp = PyFloat_AsDouble(v);
    compute_obs(env);
    return 0;
}

/* Allocate a contiguous (rows, cols) float32 numpy array, handing back its data pointer to fill. */
static PyObject *new_f32(npy_intp rows, npy_intp cols, float **data) {
    npy_intp dims[2] = {rows, cols};
    PyObject *arr = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (arr)
        *data = (float *)PyArray_DATA((PyArrayObject *)arr);
    return arr;
}

static void set_array(PyObject *dict, const char *key, PyObject *arr) {
    if (arr) {
        PyDict_SetItemString(dict, key, arr);
        Py_DECREF(arr);
    }
}

/* Expose internal state for the single-env eval/render wrapper + the info checks (env_get). The C env
 * is the only dynamics source: this is a read-only snapshot of its live entity buffers, never a
 * re-simulation. ep_* latch the just-ended episode's outcome (set before the in-place auto-reset). */
static PyObject *my_get(PyObject *dict, Env *env) {
    assign_to_dict(dict, "boss_hp", env->boss_hp);
    assign_to_dict(dict, "boss_hp_max", env->cfg.boss_hp_max);
    assign_to_dict(dict, "fight_active", (float)env->fight_active);
    assign_to_dict(dict, "phase", (float)env->phase);
    assign_to_dict(dict, "invuln_timer", (float)env->invuln_timer);
    assign_to_dict(dict, "player_hp", env->player_hp);
    assign_to_dict(dict, "player_hp_max", env->cfg.player_hp_max);
    assign_to_dict(dict, "player_mp", env->player_mp);
    assign_to_dict(dict, "player_mp_max", env->cfg.player_mp_max);
    assign_to_dict(dict, "confused", (float)(env->confused_timer > 0));
    assign_to_dict(dict, "petrified", (float)(env->petrify_timer > 0));
    assign_to_dict(dict, "steps", (float)env->steps);
    assign_to_dict(dict, "px", env->px);
    assign_to_dict(dict, "py", env->py);
    assign_to_dict(dict, "boss_x", (float)env->boss_x);
    assign_to_dict(dict, "boss_y", (float)env->boss_y);
    assign_to_dict(dict, "boss_seen", (float)env->boss_seen);
    assign_to_dict(dict, "ep_done", (float)env->ep_done);
    assign_to_dict(dict, "ep_cleared", (float)env->ep_cleared);
    assign_to_dict(dict, "ep_boss_hp_frac", (float)env->ep_boss_hp_frac);

    float *d;
    int ns = 0;
    for (int i = 0; i < env->n_snake; i++)
        if (env->snakes[i].hp > 0.0f)
            ns++;
    PyObject *snakes = new_f32(ns, 2, &d);
    if (snakes) {
        int k = 0;
        for (int i = 0; i < env->n_snake; i++)
            if (env->snakes[i].hp > 0.0f) {
                d[2 * k] = env->snakes[i].x;
                d[2 * k + 1] = env->snakes[i].y;
                k++;
            }
    }
    set_array(dict, "snakes", snakes);

    PyObject *ebul = new_f32(env->n_ebul, 2, &d);
    if (ebul)
        for (int i = 0; i < env->n_ebul; i++) {
            d[2 * i] = env->ebul[i].x;
            d[2 * i + 1] = env->ebul[i].y;
        }
    set_array(dict, "enemy_bullets", ebul);

    PyObject *pbul = new_f32(env->n_pbul, 2, &d);
    if (pbul)
        for (int i = 0; i < env->n_pbul; i++) {
            d[2 * i] = env->pbul[i].x;
            d[2 * i + 1] = env->pbul[i].y;
        }
    set_array(dict, "player_bullets", pbul);

    PyObject *gren = new_f32(env->n_gren, 3, &d); /* x, y, radius */
    if (gren)
        for (int i = 0; i < env->n_gren; i++) {
            d[3 * i] = env->grenades[i].x;
            d[3 * i + 1] = env->grenades[i].y;
            d[3 * i + 2] = env->grenades[i].rad;
        }
    set_array(dict, "grenades", gren);

    npy_intp ddims[1] = {MAP_H * MAP_W};
    PyObject *disc = PyArray_SimpleNew(1, ddims, NPY_UINT8);
    if (disc)
        memcpy(PyArray_DATA((PyArrayObject *)disc), env->discovered, sizeof(env->discovered));
    set_array(dict, "discovered", disc);
    return dict;
}
