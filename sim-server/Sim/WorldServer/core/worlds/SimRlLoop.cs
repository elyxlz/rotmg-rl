using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using WorldServer.core.objects;

namespace WorldServer.core.worlds
{
    // THROWAWAY in-process RL loop (sim-mode only, SIM_INPROC=1). The whole
    // observe -> act -> tick -> reward loop runs IN-PROCESS on the Snake Pit world
    // thread: NO nrelay client, NO packets, NO TCP, NO redis round-trip. This is the
    // path that removes the ~40ms agent-NewTick protocol hop the lockstep proof
    // measured as the wall (39.3ms of 41.5ms/step).
    //
    // Per logical tick, on the world thread, between WaitForGo and World.Update:
    //   1. build the 9807-float obs DIRECTLY from the live C# game objects
    //      (SimObsBuilder, bit-for-bit RealObsBuilder), into a reused buffer
    //   2. stub policy (random / scripted "shoot toward boss") picks {move,aim,shoot,cast}
    //   3. apply the action DIRECTLY to the SimAgent (SimActionApply)
    //   4. one gated World.Update advances exactly one 100ms logical tick
    //   5. reward (boss-HP delta / clear / death) + done read from live objects
    // and records the per-stage latency for the benchmark.
    //
    // One SimRlLoop per Snake Pit world. The agent spawns at the pit ENTRANCE
    // (training mode: it must navigate; no /tppos cheat from the lockstep proof).
    internal sealed class SimRlLoop
    {
        private const int OBS_LEN = SimObsBuilder.OBS_LEN; // 9807
        private const string BOSS_ID = "Stheno the Snake Queen";

        private readonly World _world;
        private readonly SimObsBuilder _obs = new SimObsBuilder();
        // Real in-process projectile collision (BOTH directions): the agent's aimed
        // shots step + collide against enemies (the boss dies from the agent's
        // bullets), enemy bursts step + collide against the agent (undodged bullets
        // cost HP). Replaces the old proximity-contact boss-damage shortcut.
        private readonly SimProjectiles _proj;
        private readonly SimObjects _objs = new SimObjects { Enemies = new System.Collections.Generic.List<Enemy>() };
        private readonly float[] _obsBuf = new float[OBS_LEN];
        private readonly Random _rng;
        private readonly string _policy; // "random" | "scripted"
        // Shared-memory slot for this agent (server-as-sim). >=0 == drive actions
        // from the shm region + write obs/reward/done there (the PufferLib C-shim is
        // the policy); -1 == the stand-alone in-proc benchmark (stub policy).
        private readonly int _slot;
        private readonly int _maxSteps;
        private readonly bool _invuln;
        private readonly bool _dumpObs;
        private readonly bool _dumpAtBoss;
        private readonly string _dumpPath;
        private readonly string _statePath;

        private SimAgent _agent;
        private long _tick;
        private int _step;
        private int _epStep; // steps in the CURRENT episode (for the timeout)
        private bool _done;

        private float _prevBossHp = -1f;
        private float _epReward;
        private float _totalReward;
        private int _episodes;
        private int _clears;
        private int _deaths;
        private int _inFightSteps;
        // RL-METRICS-PATCH per-episode reward-component sums + outcome telemetry (cheap, in-scope)
        private float _rApproach, _rBossDmg, _rClear, _rDeath, _rStep;

        // Geodesic-approach shaping: a BFS distance-to-boss field over the REAL pit
        // walkable grid, and the agent's distance at the previous tick. The approach
        // reward is the per-tick REDUCTION in geodesic distance -- the navigate-in
        // gradient. _prevGeo = -1 == not yet seeded (first tick of an episode).
        private readonly SimGeodesic _geo = new SimGeodesic();
        private float _prevGeo = -1f;

        // Effective per-episode difficulty knobs, RESOLVED each spawn/reset from the LIVE shm
        // config (the d-flow channel) with a fallback to the static SIM_* env defaults. This is
        // what makes d ramp LIVE: ResolveConfig() re-reads the trainer's current d-config every
        // episode, so the next spawn uses the new spawn distance / agent HP / agent DEF / boss HP
        // with NO server restart. -1 spawn == the real entrance (the SimMode.SpawnGeoDist sentinel).
        private int _cfgSpawnGeoDist;
        private int _cfgAgentHp;
        private int _cfgAgentDef;
        private int _cfgBossHp;
        private bool _cfgEverResolved;

        // ASYNC free-run handshake state (SIM_ASYNC, slot>=0 only). The world ticks
        // continuously; _asyncConsumedSeq is the last action sequence it has consumed and
        // published a transition for. Between fresh actions it re-applies the last action
        // (_lastMv/_lastAm/_lastSh/_lastCa) so the physics keeps advancing (enemies/bullets
        // move, matching the live deploy where the server keeps ticking between the policy
        // inputs), accumulating reward into _asyncRewardAcc. When a fresh action arrives it
        // applies it, and PostTick publishes ONE transition (the accumulated reward plus next
        // obs) tagged with that action seq, so the trainer sees exactly one consistent
        // transition per agent per step (a 1-tick-delayed-action MDP). _pendingPublishSeq
        // greater than consumed marks the tick that consumed a fresh action.
        private int _asyncConsumedSeq;
        private int _pendingPublishSeq;
        private float _asyncRewardAcc;
        private int _lastMv, _lastAm;
        private bool _lastSh, _lastCa;
        private bool _lastActionValid;
        private bool _asyncDonePending; // a terminal hit on a free-run in-between tick, reported on the next published step
        private int _asyncObsIdx;       // this slot's last-published double-buffer half (0/1); flips each publish

        // latency accumulators (stopwatch ticks)
        private long _tObs, _tPolicy, _tAction, _tTick, _tReward, _tColl, _tEnemyCopy;
        private long _profTicks;
        private static readonly bool Profile = Environment.GetEnvironmentVariable("SIM_RL_PROFILE") == "1";
        private readonly Stopwatch _sw = new Stopwatch();
        private double _loopWallStart;
        private static readonly double TicksToMs = 1000.0 / Stopwatch.Frequency;

        public SimRlLoop(World world) : this(world, -1)
        {
        }

        public SimRlLoop(World world, int slot)
        {
            _world = world;
            _slot = slot;
            _proj = new SimProjectiles(world.Id);
            _policy = Environment.GetEnvironmentVariable("SIM_RL_POLICY") ?? "scripted";
            _maxSteps = ReadInt("SIM_RL_STEPS", 2000);
            _invuln = Environment.GetEnvironmentVariable("SIM_RL_INVULN") == "1";
            _rng = new Random(ReadInt("SIM_RL_SEED", 0));
            _dumpObs = Environment.GetEnvironmentVariable("SIM_RL_DUMP_OBS") == "1";
            _dumpAtBoss = Environment.GetEnvironmentVariable("SIM_RL_DUMP_AT_BOSS") == "1";
            _dumpPath = Environment.GetEnvironmentVariable("SIM_RL_DUMP_PATH") ?? "./logs/inproc_obs.bin";
            _statePath = Environment.GetEnvironmentVariable("SIM_RL_STATE_PATH") ?? "./logs/inproc_state.jsonl";
            var drive = _slot >= 0 ? $"shm(slot={_slot})" : $"stub({_policy})";
            Console.WriteLine($"[SIM-RL] in-process loop world={world.Id} drive={drive} steps={_maxSteps} invuln={_invuln} dump_obs={_dumpObs}");
            if (SimMode.StateHashPath.Length > 0)
            {
                _stateHashWriter = new StreamWriter(SimMode.StateHashPath, false) { AutoFlush = false };
                _stateHashWriter.WriteLine("tick,enemy_count,agent_x,agent_y,agent_hp,state_hash");
                Console.WriteLine($"[SIM-RL] state-hash proof -> {SimMode.StateHashPath}");
            }
            _sw.Start();
            _loopWallStart = _sw.Elapsed.TotalSeconds;
        }

        // THROWAWAY byte-identical-fidelity proof writer (SIM_STATE_HASH_PATH).
        private readonly StreamWriter _stateHashWriter;

        // Hash the post-World.Update world state (every enemy's id/x/y/hp/state-name
        // plus the agent), one stable 64-bit FNV-1a per tick. Deterministic given the
        // seeded RNG, so the before/after optimization timelines must match exactly.
        private void DumpStateHash()
        {
            if (_stateHashWriter == null)
                return;
            ulong h = 1469598103934665603UL;
            void Mix(long v)
            {
                for (int b = 0; b < 8; b++)
                {
                    h ^= (byte)(v >> (b * 8));
                    h *= 1099511628211UL;
                }
            }
            foreach (var e in _objs.Enemies)
            {
                Mix(e.Id);
                Mix(BitConverter.SingleToInt32Bits(e.X));
                Mix(BitConverter.SingleToInt32Bits(e.Y));
                Mix(e.Health);
                // stable (process-independent) hash of the state name; string.GetHashCode
                // is randomized per process so it cannot be compared across runs.
                var sn = e.CurrentState != null ? e.CurrentState.Name : "";
                foreach (var ch in sn)
                    Mix(ch);
            }
            _stateHashWriter.WriteLine($"{_tick},{_objs.Enemies.Count},{_agent.X:R},{_agent.Y:R},{_agent.HP},{h:x16}");
        }

        // Resolve THIS episode's difficulty knobs from the LIVE shm config (the trainer's d-flow),
        // falling back to the static SIM_* env defaults when no live config has been written (a
        // fixed-config proof run, or the stand-alone in-proc benchmark with no shm). Called at every
        // spawn/reset so a d change applies to the next episode with no server restart.
        private void ResolveConfig()
        {
            var (valid, spawn, hp, def, bossHp) = _slot >= 0
                ? SimShmBridge.ReadConfig()
                : (false, 0, 0, 0, 0);
            var newSpawn = valid ? spawn : SimMode.SpawnGeoDist;
            var newHp = valid ? hp : SimMode.AgentHp;
            var newDef = valid ? def : SimMode.AgentDef;
            var newBoss = valid ? bossHp : SimMode.BossHp;
            // Log every time the trainer's live d-config CHANGES (the d-ramp signal): this is the
            // ground-truth proof that a d change reached the C# spawn per-episode with no restart.
            if (!_cfgEverResolved || newSpawn != _cfgSpawnGeoDist || newHp != _cfgAgentHp
                || newDef != _cfgAgentDef || newBoss != _cfgBossHp)
                Console.WriteLine($"[SIM-RL] CONFIG CHANGED world={_world.Id} ep={_episodes} -> spawn_geo={newSpawn} agent_hp={newHp} agent_def={newDef} boss_hp={newBoss}");
            _cfgSpawnGeoDist = newSpawn;
            _cfgAgentHp = newHp;
            _cfgAgentDef = newDef;
            _cfgBossHp = newBoss;
            _cfgEverResolved = true;
        }

        // Spawn the agent at the pit ENTRANCE (a Spawn-region tile), pin boss HP +
        // probe-hot once the boss exists. Called each tick until the agent is in.
        private void EnsureAgent()
        {
            if (_agent != null)
                return;

            // find the boss to pin HP (deterministic, like SimHarness) once spawned
            Enemy boss = null;
            foreach (var e in _world.Enemies.Values)
                if (e.ObjectDesc != null && e.ObjectDesc.IdName == BOSS_ID)
                {
                    boss = e;
                    break;
                }
            if (boss == null)
                return; // wait for the pit to populate

            ResolveConfig();
            if (_cfgBossHp > 0)
            {
                boss.MaxHealth = _cfgBossHp;
                boss.Health = _cfgBossHp;
            }

            // Build the geodesic field from the boss now (it anchors both the spawn
            // and the per-tick approach reward).
            _geo.EnsureField(_world, boss.X, boss.Y);

            // resolve the deploy loadout: T7 staff + T7 spell projectiles (by item id).
            var (proj, type, spellProj) = ResolveWeapon();

            _agent = new SimAgent(_world.GameServer, proj, type, spellProj);
            // Apply this episode's resolved HP/DEF (the d-flow knobs) over the agent's env
            // defaults so survivability ramps with d live.
            if (_cfgAgentHp > 0)
            {
                _agent.MaxHP = _cfgAgentHp;
                _agent.HP = _cfgAgentHp;
            }
            _agent.Def = _cfgAgentDef;
            _objs.Agent = _agent;
            _proj.Reset();
            // Spawn: the obs-MATCH PROOF (SIM_RL_DUMP_AT_BOSS=1) spawns ON the boss
            // (rich dumped obs) -- proof-only, never training. Training spawns at the
            // configured distance (ChooseSpawn): the real pit ENTRANCE for d=1, or a
            // walkable tile SIM_SPAWN_GEO_DIST geodesic-tiles from the boss for the
            // easy proof (a known short navigate-in path).
            var spawn = _dumpAtBoss ? (boss.X, boss.Y) : ChooseSpawn();
            _agent.Move(spawn.Item1, spawn.Item2);
            _prevGeo = _geo.HasField ? _geo.DistanceAt(spawn.Item1, spawn.Item2) : -1f;
            _world.EnterWorld(_agent);
            if (_invuln)
                _agent.ApplyConditionEffect(Shared.resources.ConditionEffectIndex.Invulnerable, int.MaxValue);
            Console.WriteLine($"[SIM-RL] agent spawned world={_world.Id} at ({spawn.Item1:F1},{spawn.Item2:F1}) geo_dist={_geo.DistanceAt(spawn.Item1, spawn.Item2):F0} (max={_geo.MaxReachable:F0}) | applied cfg: spawn_geo={_cfgSpawnGeoDist} agent_hp={_cfgAgentHp} agent_def={_cfgAgentDef} boss_hp={boss.Health}");
        }

        // The configured spawn position. Resolved spawn-geo-dist >= 0 spawns at a walkable
        // tile that geodesic distance from the boss; -1 == the real pit entrance.
        private (float, float) ChooseSpawn()
        {
            if (_cfgSpawnGeoDist >= 0 && _geo.HasField)
                return _geo.TileAtDistance(_cfgSpawnGeoDist);
            return PitEntrance();
        }

        private (float, float) PitEntrance()
        {
            // The Spawn region tiles are the pit entrance; pick the first
            // deterministically. Training: the agent navigates to the boss from here.
            foreach (var sp in _world.GetSpawnPoints())
                return (sp.Key.X + 0.5f, sp.Key.Y + 0.5f);
            // fallback: map centre
            return (_world.Map.Width / 2f, _world.Map.Height / 2f);
        }

        // Resolve the DEPLOY loadout's projectiles by item id: the T7 Staff of Destruction
        // (SimMode.WeaponItemType, the continuous-fire weapon) + the T7 Burning Retribution
        // Spell (its BulletNova Red Bolt). These are the EXACT items the live no-cheat test
        // char carries, so the agent's bullet speed/range/wave + damage match the real game.
        // Returns (staffProj, staffItemType, spellProj). A missing item hard-fails loudly so
        // a content mismatch can never silently fall back to an arbitrary weapon.
        private (Shared.resources.ProjectileDesc, int, Shared.resources.ProjectileDesc) ResolveWeapon()
        {
            var items = _world.GameServer.Resources.GameData.Items;
            var staffType = (ushort)SimMode.WeaponItemType;
            if (!items.ContainsKey(staffType) || items[staffType].Projectiles == null || items[staffType].Projectiles.Length == 0)
                throw new Exception($"[SIM-RL] deploy staff item 0x{staffType:x} has no projectile (check SIM_WEAPON_ITEM)");
            var staffProj = items[staffType].Projectiles[0];

            var spellType = (ushort)SimMode.SpellItemType;
            Shared.resources.ProjectileDesc spellProj = null;
            if (items.ContainsKey(spellType) && items[spellType].Projectiles != null && items[spellType].Projectiles.Length > 0)
                spellProj = items[spellType].Projectiles[0];
            else
                Console.WriteLine($"[SIM-RL] WARNING deploy spell item 0x{spellType:x} missing a projectile -> spell disabled");
            return (staffProj, staffType, spellProj);
        }

        // ASYNC free-run pacing (SIM_ASYNC, slot>=0). Park until the C-shim posts a fresh
        // action seq for this slot, so the world advances ONE policy tick per action (1:1,
        // matching the live deploy cadence) rather than ticking uncapped and outrunning the
        // policy. Per-slot park only -- it never blocks any other world or the c_step, so the
        // worlds and the GPU overlap. Before the agent exists (lazy spawn) there is no policy
        // step yet, so it returns immediately and the warm-up ticks populate the pit.
        public void WaitForAction()
        {
            if (_slot < 0 || _agent == null)
                return;
            SimShmAsync.WaitForAction(_slot, _asyncConsumedSeq);
        }

        // Called on the world thread AFTER WaitForGo, BEFORE World.Update. Reads the
        // pre-tick obs and applies the action for THIS tick.
        public void PreTick()
        {
            if (_done)
                return;
            // publish the logical tick so enemy bursts fired during this World.Update
            // are stamped with the same tick the obs ages them against.
            SimEnemyShoots.SetTick(_world.Id, _tick);
            EnsureAgent();
            if (_agent == null)
                return;

            // Per-tick combat upkeep BEFORE the action applies: HP/MP regen (real
            // HandleRegen rate) + staff/spell cooldown countdown, so a shoot/cast this
            // tick sees up-to-date MP + cooldown state.
            _agent.TickCombat();

            // Shared-memory (server-as-sim): the C-shim wrote THIS tick's action into
            // the slot before signalling the gate. Read + apply it; the resulting obs
            // is built post-tick (PostTick) -- the RL convention (obs[t+1] follows
            // action[t]). The very first action after a reset is the no-op the c_reset
            // gate-tick supplies, so obs[0] is the post-reset frame.
            if (_slot >= 0)
            {
                var ta0 = _sw.ElapsedTicks;
                if (SimMode.Async)
                {
                    // FREE-RUN: consume a fresh action only when the C-shim posted a new seq;
                    // otherwise re-apply the last action so the world keeps ticking (the physics
                    // advances between the policy steps, as in the live deploy).
                    var actSeq = SimShmBridge.ReadActSeq(_slot);
                    if (actSeq > _asyncConsumedSeq || !_lastActionValid)
                    {
                        var (amv, aam, ash, aca) = SimShmBridge.ReadAction(_slot);
                        _lastMv = amv; _lastAm = aam; _lastSh = ash; _lastCa = aca; _lastActionValid = true;
                        _pendingPublishSeq = actSeq;
                    }
                    var ta1 = _sw.ElapsedTicks;
                    _tPolicy += ta1 - ta0;
                    SimActionApply.Apply(_agent, _lastMv, _lastAm, _lastSh, _lastCa, _tick);
                    _tAction += _sw.ElapsedTicks - ta1;
                    return;
                }
                var (mv2, am2, sh2, ca2) = SimShmBridge.ReadAction(_slot);
                var ta1b = _sw.ElapsedTicks;
                _tPolicy += ta1b - ta0;
                SimActionApply.Apply(_agent, mv2, am2, sh2, ca2, _tick);
                _tAction += _sw.ElapsedTicks - ta1b;
                return;
            }

            // For the obs-match proof, snapshot the live bursts BEFORE Build prunes
            // them, so the dumped state hands RealObsBuilder the identical bursts.
            List<SimBurst> burstsForDump = _dumpObs ? SimEnemyShoots.Snapshot(_world.Id) : null;

            var t0 = _sw.ElapsedTicks;
            _obs.Build(_agent, _world, _tick, _obsBuf);
            var t1 = _sw.ElapsedTicks;
            _tObs += t1 - t0;

            if (_dumpObs)
                DumpObsAndState(burstsForDump);

            var (move, aim, shoot, cast) = Policy();
            var t2 = _sw.ElapsedTicks;
            _tPolicy += t2 - t1;

            SimActionApply.Apply(_agent, move, aim, shoot, cast, _tick);
            var t3 = _sw.ElapsedTicks;
            _tAction += t3 - t2;
        }

        // Called on the world thread AFTER World.Update completed this tick. Computes
        // reward/done from the post-tick live objects and advances counters.
        public void PostTick(ref TickTime time)
        {
            if (_agent == null)
                return;

            var t0 = _sw.ElapsedTicks;

            // REAL projectile collision (both directions), AFTER World.Update advanced
            // this tick: the agent's shots (fired in PreTick) step + collide against
            // the live enemies and reduce their HP (the boss dies from the agent's
            // bullets); the enemy bursts (fired during World.Update) step + collide
            // against the agent and reduce its HP. This MUST run before reading boss/
            // agent HP so the reward + death signals reflect this tick's hits.
            var tc0 = _sw.ElapsedTicks;
            _objs.Enemies.Clear();
            foreach (var e in _world.Enemies.Values)
                _objs.Enemies.Add(e);
            var tc1 = _sw.ElapsedTicks;
            _tEnemyCopy += tc1 - tc0;

            // THROWAWAY byte-identical proof: hash the post-World.Update enemy state
            // (BEFORE the Sim* collision, which is out of this effort's scope) so the
            // before/after optimization timelines diff exactly. _objs.Enemies is the
            // live world's enemy set in dictionary order (deterministic given the
            // seeded RNG), so the hash captures every enemy's id/pos/hp/state.
            DumpStateHash();

            _proj.AddAgentShots(_agent.DrainNewShots());
            _proj.StepAndCollide(_objs, _tick, ref time);
            _tColl += _sw.ElapsedTicks - tc1;

            Enemy boss = null;
            foreach (var e in _world.Enemies.Values)
                if (e.ObjectDesc != null && e.ObjectDesc.IdName == BOSS_ID)
                {
                    boss = e;
                    break;
                }
            var bossHp = boss != null ? (float)boss.Health : -1f;
            var bossHpMax = boss != null ? (float)boss.MaxHealth : -1f;

            var reward = 0f;
            var done = false;
            string reason = null;

            if (!_invuln && _agent.HP <= 0 && _prevBossHp >= 0)
            {
                reward -= 1f;
                _rDeath += -1f;
                done = true;
                reason = "death";
            }
            else if (bossHp >= 0 && bossHpMax > 0)
            {
                // The boss takes REAL projectile damage (SimProjectiles), clamped at 1
                // so the entity persists across episodes; bossHp<=1 == cleared. The
                // per-tick HP delta (driven by the agent's landed shots) is the dense
                // reward; the +5 is the clear bonus.
                if (_prevBossHp >= 0)
                {
                    var _dmg = Math.Max(0f, _prevBossHp - bossHp) / bossHpMax;
                    reward += _dmg; _rBossDmg += _dmg;
                }
                if (bossHp <= 1)
                {
                    reward += 5f;
                    _rClear += 5f;
                    done = true;
                    reason = "clear";
                }
            }

            // GEODESIC-APPROACH shaping + step penalty (skip on a terminal step so the
            // clear/death signal stays clean). approach = the per-tick REDUCTION in
            // geodesic-distance-to-boss (tiles) * scale -- the navigate-in gradient.
            // The boss is near-stationary, so EnsureField is a cached no-op after the
            // first build; if it wandered a tile the field re-anchors. _prevGeo<0
            // (first tick of an episode) only seeds the baseline, no reward.
            if (!done && boss != null)
            {
                _geo.EnsureField(_world, boss.X, boss.Y);
                var curGeo = _geo.DistanceAt(_agent.X, _agent.Y);
                if (_prevGeo >= 0f)
                {
                    var _appr = (_prevGeo - curGeo) * SimMode.ApproachScale;
                    reward += _appr; _rApproach += _appr;
                }
                _prevGeo = curGeo;
                reward -= SimMode.StepPenalty; _rStep += -SimMode.StepPenalty;
            }

            // Hard timeout: a stuck agent ends the episode (no terminal bonus) so it
            // can't run forever and mask the learning signal.
            _epStep++;
            if (!done && SimMode.EpisodeTimeout > 0 && _epStep >= SimMode.EpisodeTimeout)
            {
                done = true;
                reason = "timeout";
            }

            _totalReward += reward;
            _epReward += reward;
            if (bossHp > 0)
                _inFightSteps++;
            _prevBossHp = bossHp;
            _tick++;
            _step++;
            SimHarness.CountInProcTick(); // feed the aggregate-SPS reporter in in-proc mode

            _tReward += _sw.ElapsedTicks - t0;

            if (done)
            {
                _episodes++;
                if (reason == "clear") _clears++;
                else if (reason == "death") _deaths++;
                var _bossFrac = (bossHpMax > 0f) ? Math.Max(0f, bossHp) / bossHpMax : -1f;
                Console.WriteLine($"[SIM-RL] EPISODE DONE t={DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()} world={_world.Id} step={_step} reason={reason} ep_reward={_epReward:F3} ep_steps={_epStep} boss_hp_frac={_bossFrac:F4} agent_hp={_agent.HP} r_approach={_rApproach:F4} r_boss_dmg={_rBossDmg:F4} r_clear={_rClear:F4} r_death={_rDeath:F4} r_step={_rStep:F4}");
                _epReward = 0f;
                _rApproach = 0f; _rBossDmg = 0f; _rClear = 0f; _rDeath = 0f; _rStep = 0f;
                _prevBossHp = -1f;
                // reset: respawn the agent at the entrance (fresh episode). The pit
                // and boss persist (HP re-pinned); training would re-portal, but for
                // the in-proc benchmark a respawn keeps the loop hot.
                ResetEpisode();
            }

            if (_step % 250 == 0)
                Console.WriteLine($"[SIM-RL] step={_step} boss_hp={bossHp:F0} agent_hp={_agent.HP} ep_r={_epReward:F3}");

            // Shared-memory (server-as-sim): publish this tick's obs + reward + done
            // for the C-shim to read after the ack. On `done` ResetEpisode already ran
            // above, so the obs we build now is the FRESH-episode frame -- PufferLib's
            // auto-reset convention (terminal=1 carries the next episode's obs[0]).
            // The trainer owns run length, so the world NEVER self-terminates here.
            if (_slot >= 0)
            {
                var to0 = _sw.ElapsedTicks;
                _obs.Build(_agent, _world, _tick, _obsBuf);
                _tObs += _sw.ElapsedTicks - to0;
                if (SimMode.Async)
                {
                    // FREE-RUN publish: accumulate this tick's reward; a terminal on a free-run
                    // in-between tick is latched and reported on the next published step (the obs
                    // is already the fresh post-reset frame, since ResetEpisode ran above). Publish
                    // a transition ONLY on a tick that consumed a fresh action. The publish writes
                    // the WHOLE transition (obs + summed reward + latched done) into the slot's
                    // INACTIVE double-buffer half, then atomically flips obs_idx + bumps obs_seq --
                    // so the c_step always reads a complete, consistent frame, never a half that a
                    // later free-run tick is overwriting (the single-buffer tear is gone). Between
                    // published ticks the world does NOT touch shm at all (no wasted full-obs write
                    // into a buffer the trainer never reads).
                    _asyncRewardAcc += reward;
                    if (done)
                        _asyncDonePending = true;
                    if (_pendingPublishSeq > _asyncConsumedSeq)
                    {
                        _asyncConsumedSeq = _pendingPublishSeq;
                        _asyncObsIdx = SimShmBridge.PublishAsyncTransition(
                            _slot, _asyncObsIdx, _obsBuf, _asyncRewardAcc, _asyncDonePending, _asyncConsumedSeq);
                        _asyncRewardAcc = 0f;
                        _asyncDonePending = false;
                        // wake the C-shim parked in c_step on this slot's obs_seq (its spin
                        // budget may have expired before we published; without this wake it
                        // would park forever and deadlock the boundary).
                        SimShmAsync.WakeObs(_slot);
                    }
                    return;
                }
                SimShmBridge.WriteObs(_slot, _obsBuf);
                SimShmBridge.WriteRewardDone(_slot, reward, done);
                if (Profile)
                {
                    _profTicks++;
                    if (_profTicks % 1000 == 0)
                    {
                        var d = (double)_profTicks;
                        var enemyN = _objs.Enemies.Count;
                        Console.WriteLine($"[SIM-PROF] world={_world.Id} ticks={_profTicks} enemies={enemyN} | obs={_tObs * TicksToMs / d:F4} coll={_tColl * TicksToMs / d:F4} enemyCopy={_tEnemyCopy * TicksToMs / d:F4} worldUpd={_tTick * TicksToMs / d:F4} reward={(_tReward - _tColl - _tEnemyCopy) * TicksToMs / d:F4} action={_tAction * TicksToMs / d:F4} policy={_tPolicy * TicksToMs / d:F4} (ms/tick)");
                    }
                }
                return;
            }

            if (_step >= _maxSteps && !_done)
            {
                _done = true;
                Report();
            }
        }

        private void ResetEpisode()
        {
            _epStep = 0;
            // Re-resolve the live difficulty config: the trainer may have ramped d since the
            // last episode, so the NEW episode picks up the new spawn distance / HP / DEF / boss
            // HP. This is the per-episode d-ramp -- no server restart.
            ResolveConfig();
            if (_cfgAgentHp > 0)
                _agent.MaxHP = _cfgAgentHp;
            _agent.Def = _cfgAgentDef;
            _agent.HP = _agent.MaxHP;
            _agent.MP = _agent.MaxMP;
            // re-pin boss HP first so the geodesic field re-anchors on the live boss
            // before we place the agent at the configured distance from it.
            Enemy boss = null;
            foreach (var e in _world.Enemies.Values)
                if (e.ObjectDesc != null && e.ObjectDesc.IdName == BOSS_ID)
                {
                    boss = e;
                    break;
                }
            if (boss != null && _cfgBossHp > 0)
            {
                boss.MaxHealth = _cfgBossHp;
                boss.Health = _cfgBossHp;
            }
            if (boss != null)
                _geo.EnsureField(_world, boss.X, boss.Y);
            var spawn = ChooseSpawn();
            _agent.Move(spawn.Item1, spawn.Item2);
            // re-seed the geodesic baseline so the first post-reset tick does not bill
            // the spawn teleport as a giant approach reward.
            _prevGeo = _geo.HasField ? _geo.DistanceAt(spawn.Item1, spawn.Item2) : -1f;
            _obs.Reset();
            SimEnemyShoots.Clear(_world.Id);
            _proj.Reset(); // drop both bullet sets so a new episode starts bullet-free
        }

        private (int, int, bool, bool) Policy()
        {
            if (_policy == "random")
                return (_rng.Next(0, 9), _rng.Next(0, 32), _rng.Next(0, 2) == 1, false);

            // scripted: aim/shoot at boss (or nearest enemy), close distance.
            Enemy target = null;
            double best = double.MaxValue;
            Enemy boss = null;
            foreach (var e in _world.Enemies.Values)
            {
                if (e.ObjectDesc != null && e.ObjectDesc.IdName == BOSS_ID)
                    boss = e;
                var d = (e.X - _agent.X) * (e.X - _agent.X) + (e.Y - _agent.Y) * (e.Y - _agent.Y);
                if (d < best) { best = d; target = e; }
            }
            if (boss != null)
                target = boss;
            if (target == null)
                return (0, 0, false, false);

            var dx = target.X - _agent.X;
            var dy = target.Y - _agent.Y;
            var ang = Math.Atan2(dy, dx);
            var aim = (int)Math.Round(ang / (2 * Math.PI / 32)) & 31;
            var dist = Math.Sqrt(dx * dx + dy * dy);
            var move = 0;
            if (dist > 6.0)
                move = 1 + ((int)Math.Round(ang / (Math.PI / 4)) & 7);
            return (move, aim, true, false);
        }

        // Dump the in-proc obs vector (binary) + the RealObsBuilder-schema state
        // (JSONL) for the obs-MATCH PROOF. The Python proof replays the JSONL states
        // through the reference RealObsBuilder and compares to these obs vectors.
        // Full tiles are emitted once (step 0); RealObsBuilder persists them, and its
        // fog/latches then evolve from the per-step states exactly as the C# builder's
        // do, so the two stateful builders stay aligned.
        private void DumpObsAndState(List<SimBurst> bursts)
        {
            using (var fs = new FileStream(_dumpPath, FileMode.Append, FileAccess.Write))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(_step);
                bw.Write(_agent.X);
                bw.Write(_agent.Y);
                foreach (var v in _obsBuf)
                    bw.Write(v);
            }

            var sb = new System.Text.StringBuilder(1 << 16);
            sb.Append('{');
            sb.Append("\"step\":").Append(_step).Append(',');
            sb.Append("\"now_ms\":").Append(_tick * 100).Append(',');

            // player
            sb.Append("\"player\":{");
            J(sb, "x", _agent.X); sb.Append(',');
            J(sb, "y", _agent.Y); sb.Append(',');
            J(sb, "hp", _agent.HP); sb.Append(',');
            J(sb, "hp_max", _agent.MaxHP); sb.Append(',');
            J(sb, "mp", _agent.MP); sb.Append(',');
            J(sb, "mp_max", _agent.MaxMP); sb.Append(',');
            sb.Append("\"confused\":").Append(_agent.HasConditionEffect(Shared.resources.ConditionEffectIndex.Confused) ? "true" : "false").Append(',');
            sb.Append("\"petrified\":").Append(_agent.HasConditionEffect(Shared.resources.ConditionEffectIndex.Paralyzed) ? "true" : "false");
            sb.Append("},");

            // enemies (boss flagged)
            sb.Append("\"enemies\":[");
            var firstE = true;
            foreach (var e in _world.Enemies.Values)
            {
                var idName = e.ObjectDesc != null ? e.ObjectDesc.IdName : "";
                var isBoss = idName == BOSS_ID;
                if (!firstE) sb.Append(',');
                firstE = false;
                sb.Append('{');
                J(sb, "x", e.X); sb.Append(',');
                J(sb, "y", e.Y); sb.Append(',');
                J(sb, "hp", e.Health); sb.Append(',');
                J(sb, "hp_max", e.MaxHealth); sb.Append(',');
                sb.Append("\"is_boss\":").Append(isBoss ? "true" : "false").Append(',');
                var invuln = e.HasConditionEffect(Shared.resources.ConditionEffectIndex.Invulnerable)
                             || e.HasConditionEffect(Shared.resources.ConditionEffectIndex.Invincible);
                sb.Append("\"invuln\":").Append(invuln ? "true" : "false");
                sb.Append('}');
            }
            sb.Append("],");

            // player bullets (current positions, == obs PBULLET source)
            sb.Append("\"player_bullets\":[");
            var firstB = true;
            foreach (var pb in _agent.LiveProjectiles(_tick))
            {
                if (!firstB) sb.Append(',');
                firstB = false;
                sb.Append('{');
                J(sb, "x", pb.Item1); sb.Append(',');
                J(sb, "y", pb.Item2);
                sb.Append('}');
            }
            sb.Append("],");

            // enemy-shoot bursts (== RealObsBuilder.add_shots schema; spawn_ms in
            // logical ms so RealObsBuilder ages them by the same ticks).
            sb.Append("\"enemy_shots\":[");
            var firstS = true;
            foreach (var b in bursts)
            {
                if (!firstS) sb.Append(',');
                firstS = false;
                sb.Append('{');
                J(sb, "origin_x", b.OriginX); sb.Append(',');
                J(sb, "origin_y", b.OriginY); sb.Append(',');
                J(sb, "angle", b.Angle); sb.Append(',');
                J(sb, "count", b.Count); sb.Append(',');
                J(sb, "angle_inc", b.AngleInc); sb.Append(',');
                J(sb, "speed", b.Speed); sb.Append(',');
                J(sb, "lifetime", b.Lifetime); sb.Append(',');
                J(sb, "spawn_ms", b.SpawnTick * 100);
                sb.Append('}');
            }
            sb.Append(']');

            // full tiles + map dims once (step 0); RealObsBuilder persists them.
            if (_step == 0)
            {
                sb.Append(",\"map\":{");
                J(sb, "w", _world.Map.Width); sb.Append(',');
                J(sb, "h", _world.Map.Height);
                sb.Append("},\"tiles\":[");
                var gs = _world.GameServer;
                var map = _world.Map;
                var firstT = true;
                for (var y = 0; y < _world.Map.Height; y++)
                    for (var x = 0; x < _world.Map.Width; x++)
                    {
                        var tile = map[x, y];
                        bool walk;
                        if (tile == null) walk = true;
                        else
                        {
                            var td = gs.Resources.GameData.Tiles[tile.TileId];
                            walk = !td.NoWalk && !(tile.ObjType != 0 && tile.ObjDesc != null && (tile.ObjDesc.FullOccupy || tile.ObjDesc.EnemyOccupySquare));
                        }
                        if (!firstT) sb.Append(',');
                        firstT = false;
                        sb.Append("{\"x\":").Append(x).Append(",\"y\":").Append(y)
                          .Append(",\"walkable\":").Append(walk ? "true" : "false").Append('}');
                    }
                sb.Append(']');
            }
            sb.Append('}');

            File.AppendAllText(_statePath, sb.ToString() + "\n");
        }

        // append "key":number (invariant culture, round-trippable).
        private static void J(System.Text.StringBuilder sb, string key, float v)
        {
            sb.Append('"').Append(key).Append("\":")
              .Append(v.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        }

        private static void J(System.Text.StringBuilder sb, string key, int v)
        {
            sb.Append('"').Append(key).Append("\":").Append(v);
        }

        private static void J(System.Text.StringBuilder sb, string key, long v)
        {
            sb.Append('"').Append(key).Append("\":").Append(v);
        }

        private void Report()
        {
            _stateHashWriter?.Flush();
            var wall = _sw.Elapsed.TotalSeconds - _loopWallStart;
            var sps = _step / wall;
            var perStepMs = wall / _step * 1000;
            Console.WriteLine("\n================= IN-PROCESS RL RESULTS =================");
            Console.WriteLine($"steps={_step}  wall={wall:F2}s  steps/sec={sps:F1}  per_step={perStepMs:F4}ms");
            Console.WriteLine($"episodes={_episodes}  clears={_clears}  deaths={_deaths}  in_fight_steps={_inFightSteps}");
            Console.WriteLine($"total_reward={_totalReward:F3}");
            Console.WriteLine("--- per-step latency breakdown (mean ms/step) ---");
            Console.WriteLine($"  obs    (SimObsBuilder)   : {_tObs * TicksToMs / _step:F4}");
            Console.WriteLine($"  policy (stub)            : {_tPolicy * TicksToMs / _step:F4}");
            Console.WriteLine($"  action (apply)           : {_tAction * TicksToMs / _step:F4}");
            Console.WriteLine($"  tick   (World.Update)    : {_tTick * TicksToMs / _step:F4}");
            Console.WriteLine($"  reward                   : {_tReward * TicksToMs / _step:F4}");
            var perAgentSps = sps;
            var agentsTo22k = perAgentSps > 0 ? 22000.0 / perAgentSps : 0;
            Console.WriteLine($"--- extrapolation ---");
            Console.WriteLine($"  one-agent SPS={perAgentSps:F0}  agents-to-22K-SPS (perfect scaling) = {agentsTo22k:F1}");
            Console.WriteLine("=========================================================");
        }

        // Record the World.Update cost (called by the driver around the tick).
        public void AddTickTime(long swTicks) => _tTick += swTicks;

        public bool Done => _done;

        private static int ReadInt(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);
            if (raw == null || !int.TryParse(raw, out var v))
                return fallback;
            return v;
        }
    }
}
