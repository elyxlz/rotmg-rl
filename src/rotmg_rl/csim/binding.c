#include <Python.h>

#include "dungeon.h"

#define Env Dungeon
#define MY_PUT

#include "env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    Config* c = &env->cfg;
    c->player_speed = unpack(kwargs, "player_speed");
    c->player_radius = unpack(kwargs, "player_radius");
    c->max_steps = (int)unpack(kwargs, "max_steps");
    c->activation_range = unpack(kwargs, "activation_range");
    c->spawn_in_room_prob = unpack(kwargs, "spawn_in_room_prob");
    c->random_spawn_prob = unpack(kwargs, "random_spawn_prob");
    c->player_hp_max = unpack(kwargs, "player_hp_max");
    c->player_mp_max = unpack(kwargs, "player_mp_max");
    c->mp_regen = unpack(kwargs, "mp_regen");
    c->staff_cooldown = (int)unpack(kwargs, "staff_cooldown");
    c->staff_num = (int)unpack(kwargs, "staff_num");
    c->staff_dmg_lo = unpack(kwargs, "staff_dmg_lo");
    c->staff_dmg_hi = unpack(kwargs, "staff_dmg_hi");
    c->staff_speed = unpack(kwargs, "staff_speed");
    c->staff_life = (int)unpack(kwargs, "staff_life");
    c->staff_radius = unpack(kwargs, "staff_radius");
    c->staff_offset = unpack(kwargs, "staff_offset");
    c->spell_cost = unpack(kwargs, "spell_cost");
    c->spell_cooldown = (int)unpack(kwargs, "spell_cooldown");
    c->spell_num = (int)unpack(kwargs, "spell_num");
    c->spell_arc_deg = unpack(kwargs, "spell_arc_deg");
    c->spell_dmg_lo = unpack(kwargs, "spell_dmg_lo");
    c->spell_dmg_hi = unpack(kwargs, "spell_dmg_hi");
    c->spell_speed = unpack(kwargs, "spell_speed");
    c->spell_life = (int)unpack(kwargs, "spell_life");
    c->n_snakes = (int)unpack(kwargs, "n_snakes");
    c->snake_hp = unpack(kwargs, "snake_hp");
    c->snake_speed = unpack(kwargs, "snake_speed");
    c->snake_shoot_range = unpack(kwargs, "snake_shoot_range");
    c->snake_cooldown = (int)unpack(kwargs, "snake_cooldown");
    c->snake_bullet_speed = unpack(kwargs, "snake_bullet_speed");
    c->snake_bullet_life = (int)unpack(kwargs, "snake_bullet_life");
    c->snake_bullet_dmg = unpack(kwargs, "snake_bullet_dmg");
    c->snake_radius = unpack(kwargs, "snake_radius");
    c->boss_hp_max = unpack(kwargs, "boss_hp_max");
    c->boss_radius = unpack(kwargs, "boss_radius");
    c->boss_speed = unpack(kwargs, "boss_speed");
    c->boss_shoots = (int)unpack(kwargs, "boss_shoots");
    c->invuln_ticks = (int)unpack(kwargs, "invuln_ticks");
    c->ebullet_speed = unpack(kwargs, "ebullet_speed");
    c->ebullet_life = (int)unpack(kwargs, "ebullet_life");
    c->ebullet_dmg = unpack(kwargs, "ebullet_dmg");
    c->ebullet_radius = unpack(kwargs, "ebullet_radius");
    c->max_bullets = (int)unpack(kwargs, "max_bullets");
    c->grenade_fuse = (int)unpack(kwargs, "grenade_fuse");
    c->grenade_cd_p1 = (int)unpack(kwargs, "grenade_cd_p1");
    c->grenade_cd_p2 = (int)unpack(kwargs, "grenade_cd_p2");
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
    c->rew_death = unpack(kwargs, "rew_death");
    c->rew_step = unpack(kwargs, "rew_step");
    env->rng_state = (uint64_t)(long)unpack(kwargs, "seed") * 2654435761ULL + 0x9E3779B97F4A7C15ULL;
    if (env->rng_state == 0) env->rng_state = 1;
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "cleared", log->cleared);
    assign_to_dict(dict, "boss_hp_frac", log->boss_hp_frac);
    assign_to_dict(dict, "perf", log->perf);
    return 0;
}

/* Parity-test hook: inject deterministic player position + fight state, then refresh obs. */
static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    PyObject* v;
    v = PyDict_GetItemString(kwargs, "player_x");
    if (v) env->px = (float)PyFloat_AsDouble(v);
    v = PyDict_GetItemString(kwargs, "player_y");
    if (v) env->py = (float)PyFloat_AsDouble(v);
    v = PyDict_GetItemString(kwargs, "fight_active");
    if (v) env->fight_active = (int)PyLong_AsLong(v);
    v = PyDict_GetItemString(kwargs, "phase");
    if (v) env->phase = (int)PyLong_AsLong(v);
    v = PyDict_GetItemString(kwargs, "boss_hp");
    if (v) env->boss_hp = (float)PyFloat_AsDouble(v);
    compute_obs(env);
    return 0;
}
