using System;
using System.Collections.Generic;
using Shared.resources;
using WorldServer.core.structures;
using WorldServer.core.worlds;

namespace WorldServer.core.objects
{
    // THROWAWAY controllable RL agent (sim-mode only). Like SimProbe it is a
    // StaticObject+IPlayer that lives in PlayersCollision (so the boss's chunk
    // stays hot and the boss TARGETS it), but unlike the stationary probe it is
    // DRIVEN by the in-process RL action path: move by a velocity vector, shoot
    // real projectiles into the world, take damage from enemy fire.
    //
    // It carries the player scalars the obs needs (HP/MaxHP/MP/MaxMP + conditions)
    // and tracks its own fired projectiles with the server's OWN
    // ValidatedProjectile.GetPosition math, so the CH_PBULLET channel is
    // server-faithful with no nrelay client. No Client, no Redis account, no packet
    // IO -- the whole observe/act loop is in-process.
    //
    // Decoy texture type 0x0715 (== SimProbe) resolves a valid ObjectDesc.
    internal sealed class SimAgent : StaticObject, IPlayer
    {
        private const ushort AGENT_TYPE = 0x0715;

        // Agent stats. Defaults roughly match a mid-game char; the RL obs only
        // needs them normalized, and the action path drives shoot cadence, not the
        // exact damage. Overridable via SIM_AGENT_* env if a run needs it.
        public int HP { get; set; }
        public int MaxHP { get; set; }
        public int MP { get; set; }
        public int MaxMP { get; set; }

        // Flat per-hit damage reduction (the DEF training aid). Defaults to the static
        // SIM_AGENT_DEF env so a fixed-config run is unchanged; SimRlLoop overwrites it
        // per episode from the LIVE difficulty config (the d-flow channel) so DEF ramps
        // with d without a server restart.
        public int Def { get; set; } = SimMode.AgentDef;

        // Staff fractional fire cooldown (ticks) + spell cooldown (ticks). The staff fires
        // ~8 shots/s on the 10Hz action grid by carrying the 1.25-tick cooldown as a
        // fractional accumulator (NOT one bullet/tick); the spell is gated by its own timer.
        private float _staffTimer;
        private int _spellTimer;
        // Fractional HP/MP regen carries (real HandleRegen ceils a per-tick rate).
        private float _hpRegenCarry;
        private float _mpRegenCarry;

        // The staff projectile desc (resolved from the deploy weapon item) + the spell
        // projectile desc (the BulletNova Red Bolt, from the deploy spell item).
        // Speed/lifetime feed the in-proc PBULLET forward-sim + the collision step.
        private readonly ProjectileDesc _weaponProj;
        private readonly int _weaponType;
        private readonly ProjectileDesc _spellProj;
        private int _nextBulletId;
        private readonly List<AgentProjectile> _projectiles = new List<AgentProjectile>();

        // New shots fired since the last drain, handed to SimProjectiles each tick so
        // its real swept-path collision can damage enemies (== the client emitting
        // EnemyHit for each landed bullet). One owner: SimRlLoop drains them in
        // PostTick after the action applied this tick's Shoot calls.
        private readonly List<SimShot> _pendingShots = new List<SimShot>();
        private readonly Random _dmgRng = new Random(ReadInt("SIM_RL_SEED", 0) ^ 0x5eed);

        // Per-tick movement target the action path sets (absolute world coords),
        // applied at the next world tick like a player's nextPos.
        private bool _hasMoveTarget;
        private float _moveTargetX;
        private float _moveTargetY;

        private sealed class AgentProjectile
        {
            public float StartX;
            public float StartY;
            public float Angle;
            public int BulletId;
            public long SpawnTick;
            public ProjectileDesc Desc;
        }

        // hittestable:true so the agent's collision-map position TRACKS as it moves
        // (a stationary SimProbe stays put; this agent navigates). Routed into
        // PlayersCollision via the SimAgent special-cases in Entity.Move + World
        // AddToWorld so enemy targeting (PlayersCollision.HitTest) sees it at its
        // current position. life:null -> never decays.
        public SimAgent(GameServer gameServer, ProjectileDesc weaponProj, int weaponType, ProjectileDesc spellProj)
            : base(gameServer, AGENT_TYPE, null, true, false, true)
        {
            _weaponProj = weaponProj;
            _weaponType = weaponType;
            _spellProj = spellProj;
            MaxHP = SimMode.AgentHp;
            HP = MaxHP;
            MaxMP = SimMode.AgentMp;
            MP = MaxMP;
        }

        public bool IsVisibleToEnemy() => true;

        // Enemy fire hits the agent. In-process the boss/enemy AI calls Damage on
        // its IPlayer target. The agent is mortal (HP can reach 0 -> done=death),
        // unless an invuln run pins it; the RL loop reads HP for the death signal.
        public void Damage(int dmg, Entity src)
        {
            if (HasConditionEffect(ConditionEffectIndex.Invulnerable) || HasConditionEffect(ConditionEffectIndex.Invincible))
                return;
            // DEF is a flat per-hit reduction, a legitimate training aid: a hit always
            // lands for at least 1 so the agent is never immortal. Def is the per-episode
            // live value (d-flow), seeded from SIM_AGENT_DEF.
            var taken = dmg - Def;
            if (taken < 1)
                taken = 1;
            HP -= taken;
            if (HP < 0)
                HP = 0;
        }

        // -------- action path (called by SimActionApply on the world thread) --------

        // Request a move to an absolute target. Applied (collision-checked) on Tick
        // so movement obeys the same walkability as a player's validated move.
        public void RequestMove(float targetX, float targetY)
        {
            _hasMoveTarget = true;
            _moveTargetX = targetX;
            _moveTargetY = targetY;
        }

        // Per-tick regen + cooldown tick. The real HandleRegen ceils a fractional
        // per-tick rate (carry the remainder); the staff/spell timers count down. Called
        // once per logical tick BEFORE the action applies so a shoot/cast this tick sees
        // the up-to-date MP + cooldown state.
        public void TickCombat()
        {
            _staffTimer -= 1f;
            if (_spellTimer > 0)
                _spellTimer--;
            if (HP < MaxHP)
            {
                _hpRegenCarry += SimMode.HpRegenPerTick;
                var r = (int)Math.Ceiling(_hpRegenCarry);
                if (r > 0) { HP = Math.Min(HP + r, MaxHP); _hpRegenCarry -= r; }
            }
            if (MP < MaxMP)
            {
                _mpRegenCarry += SimMode.MpRegenPerTick;
                var r = (int)Math.Ceiling(_mpRegenCarry);
                if (r > 0) { MP = Math.Min(MP + r, MaxMP); _mpRegenCarry -= r; }
            }
        }

        // Fire the STAFF along `angle`: the real Staff of Destruction pattern --
        // StaffNumProjectiles parallel shots (ArcGap 0, offset perpendicular by
        // StaffOffset), real projectile speed/range/wave (the resolved desc), damage
        // rolled [Min,Max] * the maxed-Wizard AttackMult (== PlayerShoot's
        // NextIntRange(Min,Max) * GetAttackMult). Fire is gated by the real fractional
        // attack-frequency cooldown so the staff makes ~8 shots/s across the 10Hz grid,
        // NOT one bullet/tick. Returns false (no shot) if still on cooldown.
        public bool Shoot(float angle, long nowTick)
        {
            if (_weaponProj == null)
                return false;
            if (_staffTimer > 0f)
                return false;
            _staffTimer += SimMode.StaffCooldownTicks;

            var n = SimMode.StaffNumProjectiles;
            var arc = SimMode.StaffArcGapDeg * (float)Math.PI / 180f;
            var perpAngle = angle + (float)Math.PI / 2f;
            var px = (float)Math.Cos(perpAngle) * SimMode.StaffOffset;
            var py = (float)Math.Sin(perpAngle) * SimMode.StaffOffset;
            for (var i = 0; i < n; i++)
            {
                var k = i - (n - 1) / 2.0f;
                // ArcGap 0 (staff) -> all shots share `angle`, only the perpendicular
                // offset separates them (the real two-shot staff). A nonzero ArcGap fans
                // them by k*arc (other weapons).
                var a = arc != 0f ? angle + k * arc : angle;
                var sx = X + (arc != 0f ? 0f : px * k);
                var sy = Y + (arc != 0f ? 0f : py * k);
                EmitWeaponBullet(sx, sy, a, nowTick, RollStaffDamage());
            }
            return true;
        }

        private int RollStaffDamage()
        {
            if (SimMode.AgentDamage > 0)
                return SimMode.AgentDamage;
            var raw = _dmgRng.Next(_weaponProj.MinDamage, _weaponProj.MaxDamage + 1);
            return (int)(raw * SimMode.AttackMult);
        }

        // Cast the SPELL (Burning Retribution -> BulletNova) at the aimed TARGET position:
        // emit SpellNumShots Red Bolts in a full 360 ring spawned AT (targetX,targetY) --
        // the nova explodes at the cast target and the bullets fly outward from there, so
        // the AoE is centered on the aimed point (== AEBulletNova, which spawns the burst
        // at `target`). Each bolt is rolled [SpellDmgLo,SpellDmgHi] (NOT attack-scaled),
        // SpellSpeed/SpellLife. Costs SpellMpCost MP, gated by the spell cooldown. Returns
        // false if MP-starved or on cooldown. The target is the SEPARATE spell-aim head,
        // so the spell can land away from the staff aim (the boss, or a swarm cluster).
        public bool CastSpell(float targetX, float targetY, long nowTick)
        {
            if (_spellProj == null)
                return false;
            if (_spellTimer > 0 || MP < SimMode.SpellMpCost)
                return false;
            MP -= SimMode.SpellMpCost;
            _spellTimer = SimMode.SpellCooldownTicks;
            var n = SimMode.SpellNumShots;
            for (var i = 0; i < n; i++)
            {
                var a = (float)(i * (2.0 * Math.PI / n));
                var dmg = _dmgRng.Next(SimMode.SpellDmgLo, SimMode.SpellDmgHi + 1);
                EmitWeaponBullet(targetX, targetY, a, nowTick, dmg, _spellProj);
            }
            return true;
        }

        // Spawn one player bullet: into the live set (obs PBULLET) AND queued for
        // SimProjectiles' real swept-path collision (== the client emitting EnemyHit).
        private void EmitWeaponBullet(float sx, float sy, float angle, long nowTick, int dmg, ProjectileDesc desc = null)
        {
            var d = desc ?? _weaponProj;
            var bulletId = _nextBulletId++;
            _projectiles.Add(new AgentProjectile
            {
                StartX = sx,
                StartY = sy,
                Angle = angle,
                BulletId = bulletId,
                SpawnTick = nowTick,
                Desc = d,
            });
            _pendingShots.Add(new SimShot
            {
                StartX = sx,
                StartY = sy,
                Angle = angle,
                BulletId = bulletId,
                SpawnTick = nowTick,
                Damage = dmg,
                Desc = d,
            });
        }

        // Drain the shots fired since the last call (SimRlLoop hands them to
        // SimProjectiles each tick). One owner: nothing else reads _pendingShots.
        public List<SimShot> DrainNewShots()
        {
            if (_pendingShots.Count == 0)
                return null;
            var drained = new List<SimShot>(_pendingShots);
            _pendingShots.Clear();
            return drained;
        }

        // -------- obs read path --------

        // Current positions of the agent's live (un-expired) projectiles, advanced by
        // the server's OWN ValidatedProjectile.GetPosition math (so CH_PBULLET matches
        // what the real client would render). Prunes expired bullets.
        public IEnumerable<Tuple<float, float>> LiveProjectiles(long nowTick)
        {
            var result = new List<Tuple<float, float>>();
            var kept = new List<AgentProjectile>(_projectiles.Count);
            foreach (var p in _projectiles)
            {
                var elapsedMs = (nowTick - p.SpawnTick) * 100; // 1 tick = 100ms
                if (elapsedMs < 0)
                {
                    kept.Add(p);
                    continue;
                }
                if (elapsedMs > p.Desc.LifetimeMS)
                    continue; // expired
                kept.Add(p);
                var rel = Player.ValidatedProjectile.GetPosition(elapsedMs, p.BulletId, p.Desc, p.Angle, 1.0f);
                result.Add(Tuple.Create(p.StartX + (float)rel.X, p.StartY + (float)rel.Y));
            }
            _projectiles.Clear();
            _projectiles.AddRange(kept);
            return result;
        }

        // -------- world tick --------

        public override void Tick(ref TickTime time)
        {
            // Apply the pending move with a player-style 3-way passable slide, so the
            // agent never walks through walls (== the bridge's applyIntent passable
            // check, but in-process against the live Wmap).
            if (_hasMoveTarget)
            {
                _hasMoveTarget = false;
                var tx = _moveTargetX;
                var ty = _moveTargetY;
                float gx = X, gy = Y;
                if (World.IsPassable(tx, ty)) { gx = tx; gy = ty; }
                else if (World.IsPassable(tx, Y)) { gx = tx; gy = Y; }
                else if (World.IsPassable(X, ty)) { gx = X; gy = ty; }
                if (gx != X || gy != Y)
                    Move(gx, gy);
            }
            base.Tick(ref time);
        }

        private static int ReadInt(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);
            if (raw == null || !int.TryParse(raw, out var v))
                return fallback;
            return v;
        }
    }

    // One agent shot handed to SimProjectiles for real swept-path collision against
    // enemies. Carries the rolled damage + the desc (speed/lifetime/multihit drive
    // the same GetPosition math the obs PBULLET channel uses).
    internal struct SimShot
    {
        public float StartX;
        public float StartY;
        public float Angle;
        public int BulletId;
        public long SpawnTick;
        public int Damage;
        public Shared.resources.ProjectileDesc Desc;
    }
}
