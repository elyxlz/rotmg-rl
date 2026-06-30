/* PufferLib 4.0 Ocean binding for the server-as-sim env. Mirrors dungeon/binding.c's
 * surface (OBS_SIZE/NUM_ATNS/ACT_SIZES, my_init, my_log, c_step/c_reset via the header,
 * vecenv.h), but the whole env is a passthrough to the C# server over shm + the redis
 * lockstep gate (server_env.h). From pufferl's side it is indistinguishable from any
 * other Ocean env; the obs layout is identical to dungeon, so DungeonEncoder consumes it.
 *
 * ONE env owns all N agents (num_agents = N): my_vec_init (MY_VEC_INIT) builds exactly
 * one ServerEnv so env->observations/actions/rewards/terminals are the slot-0 bases with
 * per-agent stride — a 1:1 map onto the N shm slots SimShmBridge.cs writes.
 */
#define PUFFER4
#include "server_env.h" /* OBS_SIZE, NUM_CH, GRID, NUM_SCALARS, ServerEnv, c_step/c_reset/c_close, init_globals */

#define NUM_ATNS 4
#define ACT_SIZES {9, 32, 2, 2} /* MultiDiscrete: move, aim, shoot, cast (staff+spell share the aim) */
#define OBS_TENSOR_T FloatTensor /* float32 obs */

/* one env holds all N agents -> we own buffer layout */
#define MY_VEC_INIT

#include "vecenv.h"

/* Single env, num_agents = N. vecenv.h wires the contiguous N-agent vec-buffers onto it
 * in create_static_vec (one env, agents_per_buffer pointers). We seed cfg from kwargs +
 * the vec total so the env knows N, and connect lazily in c_reset (the C# server must be
 * up first). */
void my_init(Env *env, Dict *kwargs) {
    Config *c = &env->cfg;
    c->n_agents = (int)dict_get(kwargs, "n_agents")->value;
    c->max_steps = (int)dict_get(kwargs, "max_steps")->value;
    c->redis_port = (int)dict_get(kwargs, "redis_port")->value;
    c->redis_db = (int)dict_get(kwargs, "redis_db")->value;
    env->num_agents = c->n_agents;
    env->shm = NULL;
    env->shm_fd = 0;
    env->redis_fd = 0;
    env->tick = 0;
    env->use_barrier = 0;
    env->ctrl_req = NULL;
    env->ctrl_done = NULL;
    env->generation = 0;
    env->use_async = 0;
    env->act_seq = NULL;
    env->obs_seq = NULL;
    env->agent_seq = NULL;
    env->obs_idx = NULL;
    env->dbl_obs = NULL;
    env->dbl_rew = NULL;
    env->dbl_done = NULL;
    init_globals();
}

/* Build exactly ONE env owning all total_agents. num_buffers MUST be 1 (the lockstep gate
 * is a single global barrier; multiple buffers would each try to drive the same gate). */
Env *my_vec_init(int *num_envs_out, int *buffer_env_starts, int *buffer_env_counts,
                 Dict *vec_kwargs, Dict *env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    if (num_buffers != 1) {
        fprintf(stderr, "server_env: num_buffers must be 1 (one lockstep gate); got %d\n", num_buffers);
        exit(1);
    }

    Env *envs = (Env *)calloc(1, sizeof(Env));
    /* the env's agent count comes from the vec total, not the env ini, so a single
     * --vec.total-agents flag sizes everything (shm region, N pits, encoder batch). */
    dict_set(env_kwargs, "n_agents", (double)total_agents);
    envs[0].rng = 0;
    my_init(&envs[0], env_kwargs);

    *num_envs_out = 1;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 1;
    return envs;
}

void my_log(Log *log, Dict *out) {
    float ep = log->episodes;
    dict_set(out, "reward", log->reward);
    dict_set(out, "episodes", ep);
    dict_set(out, "done_rate", log->n > 0.0f ? log->done_count / log->n : 0.0f);
}
