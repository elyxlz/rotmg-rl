/* PufferLib 4.0 Ocean binding for the rotmg Snake Pit dungeon env.
 *
 * Copied to <PufferLib clone>/ocean/dungeon/binding.c by scripts/setup_box_puffer4.sh, alongside
 * dungeon.h + snakepit_map.h (copied verbatim from src/rotmg_rl/csim/, the single source of truth
 * for the env dynamics). dungeon.h must be compiled with -DPUFFER4 (float action/terminal buffers,
 * num_agents + rng fields) — build.sh passes our flag via the env's binding.
 *
 * Replaces the 3.0 standalone-extension binding (src/rotmg_rl/csim/binding.c + the vendored
 * env_binding.h). In 4.0 the env compiles into the monolithic _C backend via vecenv.h; there is no
 * per-env Python wrapper and actions/terminals are float* (vecenv.h owns the buffers).
 */
#define PUFFER4  /* dungeon.h: float action/terminal buffers + num_agents/rng fields (4.0 vecenv.h owns the buffers) */
#include "dungeon.h"  /* defines OBS_SIZE, NUM_CH, GRID, NUM_SCALARS, the Dungeon Env, c_step/c_reset/c_close, init_globals */

/* OBS_SIZE is already defined by dungeon.h (NUM_CH*GRID*GRID + NUM_SCALARS = 6733). */
#define NUM_ATNS 4
#define ACT_SIZES {9, 32, 2, 2}  /* MultiDiscrete: move, aim, shoot, cast */
#define OBS_TENSOR_T FloatTensor /* float32 obs in [-1, 1] */

#define Env Dungeon
#include "vecenv.h"

/* vecenv.h's default my_vec_init sets env->rng = env_index, then calls my_init (before the obs/
 * action buffers are wired and before c_reset). We only set config + seed the per-env RNG here. */
void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    Config* c = &env->cfg;
    c->player_speed = dict_get(kwargs, "player_speed")->value;
    c->player_radius = dict_get(kwargs, "player_radius")->value;
    c->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    c->activation_range = dict_get(kwargs, "activation_range")->value;
    c->spawn_in_room_prob = dict_get(kwargs, "spawn_in_room_prob")->value;
    c->random_spawn_prob = dict_get(kwargs, "random_spawn_prob")->value;
    c->spawn_in_room_radius = dict_get(kwargs, "spawn_in_room_radius")->value;
    c->player_hp_max = dict_get(kwargs, "player_hp_max")->value;
    c->player_mp_max = dict_get(kwargs, "player_mp_max")->value;
    c->player_defense = dict_get(kwargs, "player_defense")->value;
    c->damage_floor = dict_get(kwargs, "damage_floor")->value;
    c->mp_regen = dict_get(kwargs, "mp_regen")->value;
    c->hp_regen = dict_get(kwargs, "hp_regen")->value;
    c->staff_cooldown = (int)dict_get(kwargs, "staff_cooldown")->value;
    c->staff_num = (int)dict_get(kwargs, "staff_num")->value;
    c->staff_dmg_lo = dict_get(kwargs, "staff_dmg_lo")->value;
    c->staff_dmg_hi = dict_get(kwargs, "staff_dmg_hi")->value;
    c->staff_speed = dict_get(kwargs, "staff_speed")->value;
    c->staff_life = (int)dict_get(kwargs, "staff_life")->value;
    c->staff_radius = dict_get(kwargs, "staff_radius")->value;
    c->staff_offset = dict_get(kwargs, "staff_offset")->value;
    c->spell_cost = dict_get(kwargs, "spell_cost")->value;
    c->spell_cooldown = (int)dict_get(kwargs, "spell_cooldown")->value;
    c->spell_num = (int)dict_get(kwargs, "spell_num")->value;
    c->spell_dmg_lo = dict_get(kwargs, "spell_dmg_lo")->value;
    c->spell_dmg_hi = dict_get(kwargs, "spell_dmg_hi")->value;
    c->spell_speed = dict_get(kwargs, "spell_speed")->value;
    c->spell_life = (int)dict_get(kwargs, "spell_life")->value;
    c->n_snakes = (int)dict_get(kwargs, "n_snakes")->value;
    c->n_snakes_jitter = (int)dict_get(kwargs, "n_snakes_jitter")->value;
    c->snake_speed = dict_get(kwargs, "snake_speed")->value;
    c->snake_radius = dict_get(kwargs, "snake_radius")->value;
    c->boss_hp_max = dict_get(kwargs, "boss_hp_max")->value;
    c->boss_radius = dict_get(kwargs, "boss_radius")->value;
    c->boss_defense = dict_get(kwargs, "boss_defense")->value;
    c->boss_wander_speed = dict_get(kwargs, "boss_wander_speed")->value;
    c->boss_return_speed = dict_get(kwargs, "boss_return_speed")->value;
    c->boss_shoots = (int)dict_get(kwargs, "boss_shoots")->value;
    c->opening_invuln_ticks = (int)dict_get(kwargs, "opening_invuln_ticks")->value;
    c->invuln_ticks = (int)dict_get(kwargs, "invuln_ticks")->value;
    c->blade_cd = (int)dict_get(kwargs, "blade_cd")->value;
    c->blade_radius_p1 = dict_get(kwargs, "blade_radius_p1")->value;
    c->blade_radius_p3 = dict_get(kwargs, "blade_radius_p3")->value;
    c->ebullet_speed = dict_get(kwargs, "ebullet_speed")->value;
    c->ebullet_life = (int)dict_get(kwargs, "ebullet_life")->value;
    c->ebullet_dmg = dict_get(kwargs, "ebullet_dmg")->value;
    c->ebullet_radius = dict_get(kwargs, "ebullet_radius")->value;
    c->max_bullets = (int)dict_get(kwargs, "max_bullets")->value;
    c->grenade_fuse = (int)dict_get(kwargs, "grenade_fuse")->value;
    c->grenade_cd_p1 = (int)dict_get(kwargs, "grenade_cd_p1")->value;
    c->grenade_cd_p2 = (int)dict_get(kwargs, "grenade_cd_p2")->value;
    c->grenade_cd_p3_diag = (int)dict_get(kwargs, "grenade_cd_p3_diag")->value;
    c->grenade_range_confuse = dict_get(kwargs, "grenade_range_confuse")->value;
    c->grenade_petrify_dist = dict_get(kwargs, "grenade_petrify_dist")->value;
    c->grenade_radius_confuse = dict_get(kwargs, "grenade_radius_confuse")->value;
    c->grenade_dmg_confuse = dict_get(kwargs, "grenade_dmg_confuse")->value;
    c->grenade_radius_petrify = dict_get(kwargs, "grenade_radius_petrify")->value;
    c->grenade_dmg_petrify = dict_get(kwargs, "grenade_dmg_petrify")->value;
    c->confused_ticks = (int)dict_get(kwargs, "confused_ticks")->value;
    c->petrify_ticks = (int)dict_get(kwargs, "petrify_ticks")->value;
    c->minion_max = (int)dict_get(kwargs, "minion_max")->value;
    c->minion_cd = (int)dict_get(kwargs, "minion_cd")->value;
    c->minion_hp = dict_get(kwargs, "minion_hp")->value;
    c->enable_grenades = (int)dict_get(kwargs, "enable_grenades")->value;
    c->enable_minions = (int)dict_get(kwargs, "enable_minions")->value;
    c->rew_explore = dict_get(kwargs, "rew_explore")->value;
    c->rew_kill = dict_get(kwargs, "rew_kill")->value;
    c->rew_boss_dmg = dict_get(kwargs, "rew_boss_dmg")->value;
    c->rew_reach = dict_get(kwargs, "rew_reach")->value;
    c->rew_survive = dict_get(kwargs, "rew_survive")->value;
    c->rew_damage_taken = dict_get(kwargs, "rew_damage_taken")->value;
    c->rew_clear = dict_get(kwargs, "rew_clear")->value;
    c->rew_death = dict_get(kwargs, "rew_death")->value;
    c->rew_step = dict_get(kwargs, "rew_step")->value;
    c->rew_approach = dict_get(kwargs, "rew_approach")->value;

    env->rng_state = (uint64_t)env->rng * 2654435761ULL + 0x9E3779B97F4A7C15ULL;
    if (env->rng_state == 0) env->rng_state = 1;
    init_globals();  /* idempotent: build the shared map/direction tables once */
}

/* Legible END-OF-EPISODE metrics (NOT per-step means). Every field below is a per-episode sum;
 * vecenv.h divides all Log fields by n, so each sum and `episodes` arrive here both /n -> their
 * ratio recovers the true per-episode value (the n divisor cancels). This is the first-class place
 * to log: computed from the training rollouts the env already runs, surfaced under env/ for free. */
void my_log(Log* log, Dict* out) {
    float ep = log->episodes;
    dict_set(out, "clear_rate", ep > 0.0f ? log->clear_count / ep : 0.0f);          // fraction of episodes the boss dies
    dict_set(out, "boss_hp_remaining", ep > 0.0f ? 1.0f - log->score / ep : 0.0f);   // mean boss HP frac at episode end
    dict_set(out, "player_hp_remaining", ep > 0.0f ? log->player_hp_end_sum / ep : 0.0f);  // mean player HP frac at end
    dict_set(out, "death_rate", ep > 0.0f ? log->death_count / ep : 0.0f);           // fraction of episodes the player dies
    dict_set(out, "reward", log->reward);
    dict_set(out, "episodes", ep);
}
