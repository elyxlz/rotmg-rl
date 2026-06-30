using System;

namespace WorldServer.core.worlds
{
    // THROWAWAY sim-mode config. Reads env vars once so the rest of the
    // server can branch on them cheaply. With neither flag set the server
    // behaves exactly like the stock build.
    //
    // Two independent switches:
    //   SIM_HARNESS  -> spawn the Snake Pit measurement harness (probe + CSV).
    //                   Used by BOTH the fixed-dt and the real-time reference run.
    //   SIM_UNCAPPED -> use the uncapped fixed-dt loop instead of the stock
    //                   wall-clock-gated 10 TPS loop. Pure timing change.
    public static class SimMode
    {
        // Spawn the measurement harness. Implied by Uncapped so the fixed-dt run
        // never needs both set; the real-time reference run sets only this.
        // In-process RL loop: build obs + apply action + reward all IN-PROCESS on
        // the Snake Pit world thread (no nrelay, no packets, no redis). The path
        // that removes the ~40ms agent-NewTick protocol hop the lockstep proof
        // measured. Implies Harness (to spawn the pit) + Uncapped (fixed-dt). With
        // it unset the server is unchanged.
        public static readonly bool InProc =
            Environment.GetEnvironmentVariable("SIM_INPROC") == "1";

        // Server-as-sim: the in-process RL loop is driven by the PufferLib C-shim over
        // shared memory + the redis lockstep gate. Implies Harness (spawn N pits) +
        // Uncapped (fixed-dt), like InProc, but gated (the C-shim drives the ticks).
        public static readonly bool Shm =
            Environment.GetEnvironmentVariable("SIM_SHM") == "1";

        // ASYNC OVERLAP (SIM_ASYNC=1, server-as-sim only). The C# worlds FREE-RUN
        // (tick continuously, applying the latest shm action + publishing the latest
        // obs/reward) instead of taking strict lockstep turns with the policy. The
        // C-shim posts actions + collects transitions non-blocking, so the GPU and the
        // worlds OVERLAP and the GPU saturates. A per-world action/obs sequence handshake
        // keeps each (obs, action, reward, next_obs) transition consistent under the
        // overlap (a 1-tick delayed-action MDP, matching the live deploy timing). With it
        // unset the strict futex barrier (SimShmBarrier) runs unchanged.
        public static readonly bool Async =
            Environment.GetEnvironmentVariable("SIM_ASYNC") == "1";

        public static readonly bool Harness =
            Environment.GetEnvironmentVariable("SIM_HARNESS") == "1" ||
            Environment.GetEnvironmentVariable("SIM_UNCAPPED") == "1" ||
            InProc || Shm;

        // Use the uncapped fixed-dt loop. When false the stock real-time loop runs.
        public static readonly bool Uncapped =
            Environment.GetEnvironmentVariable("SIM_UNCAPPED") == "1" ||
            InProc || Shm;

        // Synthetic logical delta fed every tick in uncapped mode. 100ms == the
        // real-time 10 TPS step, so every dt-scaled mechanic advances exactly
        // one logical step per iteration regardless of wall-clock.
        public static readonly int FixedDtMs = ReadInt("SIM_FIXED_DT_MS", 100);

        // Number of Snake Pit worlds the harness spawns (parallel-scaling probe).
        public static readonly int Worlds = ReadInt("SIM_WORLDS", 1);

        // How many logical ticks the harness logs before it stops measuring.
        public static readonly int MeasureTicks = ReadInt("SIM_MEASURE_TICKS", 2000);

        // Deterministic boss HP the harness pins on spawn (overrides the stock
        // ClasifyEnemy RNG), so HP-gated phase transitions are comparable across
        // modes. 0 == leave the rolled HP untouched.
        public static readonly int BossHp = ReadInt("SIM_BOSS_HP", 7500);

        // Fixed synthetic damage the probe applies to the boss each logical tick
        // (a deterministic "scripted firing pattern" so the HP-gated phase
        // transitions also fire and can be compared across modes). 0 == no damage.
        public static readonly int ProbeDamagePerTick = ReadInt("SIM_PROBE_DPS_TICK", 200);

        // CSV path the harness writes the per-tick boss timeline to.
        public static readonly string LogPath =
            Environment.GetEnvironmentVariable("SIM_LOG_PATH") ?? "./logs/sim_timeline.csv";

        // ---- RL difficulty knobs (the curriculum will drive these later) ----
        // The real dungeon enemies + AI + boss mechanics stay UNMODIFIED; only the
        // agent spawn/stats + boss HP are controllable (legitimate training aids;
        // d=1 will restore the real conditions). Agent HP lives on SimAgent
        // (SIM_AGENT_HP); the rest live here so SimRlLoop reads one owner.

        // Flat damage reduction applied to every hit the agent takes (a DEF stat;
        // higher == the agent survives longer). 0 == raw damage (real conditions).
        public static readonly int AgentDef = ReadInt("SIM_AGENT_DEF", 0);

        // Agent max HP (the survivability knob -- the dominant difficulty gradient). The
        // static env fallback used when no LIVE difficulty config is in shm; SimRlLoop reads
        // the live d-config per episode and overrides it when present. 700 == the SimAgent
        // default (a mid-game char).
        public static readonly int AgentHp = ReadInt("SIM_AGENT_HP", 700);

        // Spawn the agent this many GEODESIC TILES from the boss (a walkable tile at
        // ~this BFS distance), so the navigate-in path is a known length. -1 == the
        // real pit ENTRANCE (the full maze, d=1). A small value gives the geodesic
        // a short clear path for the easy proof.
        public static readonly int SpawnGeoDist = ReadInt("SIM_SPAWN_GEO_DIST", -1);

        // Geodesic-approach reward scale: reward += (prevGeo - curGeo) * this, i.e.
        // the per-tick REDUCTION in geodesic-distance-to-boss (tiles). This is the
        // navigate-in gradient. 0 == no approach shaping (boss-HP-delta only).
        public static readonly float ApproachScale = ReadFloat("SIM_APPROACH_SCALE", 0.02f);

        // Small per-step time penalty (encourages finishing); subtracted each tick.
        public static readonly float StepPenalty = ReadFloat("SIM_STEP_PENALTY", 0.0005f);

        // Hard episode-step cap: end the episode (reason=timeout, no terminal bonus)
        // after this many ticks so a stuck agent can't run forever and mask learning.
        // The curriculum can shrink this as the pit gets harder. 0 == no timeout.
        public static readonly int EpisodeTimeout = ReadInt("SIM_EP_TIMEOUT", 1500);

        // Fixed per-bullet damage the agent's REAL projectiles deal on a collision
        // (SimProjectiles). 0 == roll Min..Max from the weapon desc (the faithful
        // default). A positive value pins damage so the boss-HP/weapon-DPS is a clean
        // curriculum knob: ticks-to-kill = boss_hp / (damage * fire_rate * hit_rate).
        public static readonly int AgentDamage = ReadInt("SIM_AGENT_DAMAGE", 0);

        // ---- Deploy loadout (the live no-cheat test char): a maxed Wizard carrying a
        // T7 Staff of Destruction + a T7 Burning Retribution Spell, NO armor + NO ring.
        // These calibrate the agent's combat to EXACTLY what it will have on the live
        // game (see SimAgent / SimActionApply). All env-overridable for tuning.

        // The deploy weapon: Staff of Destruction (item 0xa9e). The agent resolves THIS
        // item's projectile (not an arbitrary one) so the bullet speed/range/wave + the
        // shot pattern match the real staff.
        public static readonly int WeaponItemType = ReadInt("SIM_WEAPON_ITEM", 0xa9e);

        // The deploy ability: Burning Retribution Spell (item 0x2055). Its Red Bolt
        // projectile (Speed 160, Life 1000ms) drives the BulletNova bullets' motion.
        public static readonly int SpellItemType = ReadInt("SIM_SPELL_ITEM", 0x2055);

        // Staff fire pattern (Staff of Destruction): NumProjectiles=2, ArcGap=0 (two
        // parallel shots, offset perpendicular by StaffOffset). Damage is the raw item
        // roll [45,85] * the maxed-Wizard attack multiplier (0.5 + 75/75*1.5 = 2.0) =
        // [90,170]; AttackMult pins that 2.0 so the rolled desc damage is scaled to the
        // real on-hit value. The fire CADENCE is the real GetAttackFrequency at DEX 75
        // (0.008/ms -> 125ms/shot -> 1.25 ticks), carried as a fractional accumulator so
        // the staff fires ~8 shots/s across the 10Hz action grid (NOT one bullet/tick).
        public static readonly int StaffNumProjectiles = ReadInt("SIM_STAFF_NUM", 2);
        public static readonly float StaffArcGapDeg = ReadFloat("SIM_STAFF_ARCGAP", 0f);
        public static readonly float StaffOffset = ReadFloat("SIM_STAFF_OFFSET", 0.5f);
        public static readonly float StaffCooldownTicks = ReadFloat("SIM_STAFF_COOLDOWN", 1.25f);
        public static readonly float AttackMult = ReadFloat("SIM_ATTACK_MULT", 2.0f);

        // The deploy spell: Burning Retribution Spell (T7) -- a BulletNova. On cast it
        // emits SpellNumShots (20) bullets in a full 360 ring from the agent, each a Red
        // Bolt rolled [95,185] (NOT attack-scaled -- spells take no attack multiplier),
        // Speed 160 (1.6 t/tick), Life 1000ms (10 ticks). Cast costs SpellMpCost (90) MP
        // and is gated by SpellCooldownTicks. This is the Wizard's burst DPS; the policy
        // already has a `cast` head + sees MP in the obs.
        public static readonly int SpellNumShots = ReadInt("SIM_SPELL_NUM", 20);
        public static readonly int SpellMpCost = ReadInt("SIM_SPELL_MP_COST", 90);
        public static readonly int SpellCooldownTicks = ReadInt("SIM_SPELL_COOLDOWN", 0);
        public static readonly int SpellDmgLo = ReadInt("SIM_SPELL_DMG_LO", 95);
        public static readonly int SpellDmgHi = ReadInt("SIM_SPELL_DMG_HI", 185);
        public static readonly float SpellSpeed = ReadFloat("SIM_SPELL_SPEED", 1.6f);
        public static readonly int SpellLifeTicks = ReadInt("SIM_SPELL_LIFE", 10);

        // Maxed-Wizard MP pool + regen (the spell is MP-limited). Base Wizard MaxMP 385 +
        // the spell slot's ActivateOnEquip +40 (no ring) = 425. Regen is the real
        // HandleRegen rate (Player.cs): MP = (1 + 0.24*WIS)/s and HP = (1 + 0.36*VIT)/s,
        // per the maxed-Wizard stats from the class XML (WIS/MpRegen max 60, VIT/HpRegen
        // max 40, with NO ring). Both come to (1 + 0.24*60) = (1 + 0.36*40) = 15.4/s =
        // 1.54 per 100ms tick. Carried fractionally so the spell recharges at the real rate.
        public static readonly int AgentMp = ReadInt("SIM_AGENT_MP", 425);
        public static readonly float MpRegenPerTick = ReadFloat("SIM_MP_REGEN", 1.54f);
        public static readonly float HpRegenPerTick = ReadFloat("SIM_HP_REGEN", 1.54f);

        // THROWAWAY byte-identical-fidelity proof (SIM_RNG_SEED>=0). Seeds the
        // per-thread behavior/transition RNGs (Behavior.Random, Transition.Random)
        // deterministically so the WHOLE fight -- enemy spawns, bullet angles,
        // movement jitter, cooldown rolls -- is reproducible run-to-run. With this
        // set, a fixed scripted policy makes the per-tick boss/enemy timeline
        // identical across processes, which is what lets the before/after
        // optimization comparison prove behavior is unchanged. -1 (default) ==
        // unseeded (the stock non-deterministic RNG); never set in training.
        public static readonly int RngSeed = ReadInt("SIM_RNG_SEED", -1);
        public static bool RngSeeded => RngSeed >= 0;

        // THROWAWAY proof: dump a per-tick state hash (boss + every enemy: x,y,hp,
        // state) to this path so the before/after fight timelines can be diffed
        // byte-for-byte. Empty == no dump.
        public static readonly string StateHashPath =
            Environment.GetEnvironmentVariable("SIM_STATE_HASH_PATH") ?? "";

        private static int ReadInt(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);
            if (raw == null || !int.TryParse(raw, out var v))
                return fallback;
            return v;
        }

        private static float ReadFloat(string name, float fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);
            if (raw == null || !float.TryParse(raw, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var v))
                return fallback;
            return v;
        }
    }
}
