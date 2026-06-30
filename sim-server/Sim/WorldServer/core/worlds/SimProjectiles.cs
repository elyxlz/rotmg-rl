using System;
using System.Collections.Generic;
using Shared.resources;
using WorldServer.core.objects;
using WorldServer.core.structures;

namespace WorldServer.core.worlds
{
    // THROWAWAY in-process projectile-collision model (sim-mode only). Replaces the
    // old proximity-contact boss-damage shortcut (SIM_PROBE_DPS_TICK) with REAL
    // per-bullet collision in BOTH directions, replicating what the real game's
    // CLIENT computes (the real server is client-authoritative on hit detection:
    // the client steps bullets + emits PlayerHit/EnemyHit; here we step + collide
    // the same bullets in-process and apply the same EnemyHit/PlayerHit damage).
    //
    // GEOMETRY (faithful to the client):
    //   * a bullet's per-tick path is the SWEPT SEGMENT from its position at tick t
    //     to its position at tick t+1, advanced by the SAME ValidatedProjectile.
    //     GetPosition math the obs/PBULLET channel + the real client use.
    //   * a hit == that segment passes within (bullet_point) the target's hit
    //     radius. Enemy radius = Size/100 * ENEMY_RADIUS_SCALE; agent radius =
    //     AGENT_HIT_RADIUS (the standard ~0.5-tile RotMG player collision box).
    //   * on hit, damage == StatsManager.DamageWithDefense (the exact EnemyHit/
    //     PlayerHit formula); the bullet is consumed unless the desc is MultiHit.
    //
    // One SimProjectiles per SimRlLoop (per world). The agent's fired bullets live
    // here (moved off SimAgent's own list isn't needed -- SimAgent still tracks them
    // for the PBULLET obs); enemy bullets are read from the captured SimEnemyShoots
    // bursts. Both are stepped once per logical tick, AFTER the action is applied and
    // BEFORE/AROUND World.Update so positions match the obs the agent saw.
    internal sealed class SimProjectiles
    {
        // Standard RotMG player collision: a ~0.5-tile box; the client tests bullet
        // point vs this radius. Overridable for fidelity tuning.
        private static readonly float AGENT_HIT_RADIUS =
            ReadFloat("SIM_AGENT_HIT_RADIUS", 0.5f);
        // Enemy hit radius = Size/100 * this (Size 100 == 1 tile sprite -> 0.5 tile
        // radius, the client's enemy collision box). The boss has a large Size.
        private static readonly float ENEMY_RADIUS_SCALE =
            ReadFloat("SIM_ENEMY_RADIUS_SCALE", 0.5f);

        private const string BOSS_ID = "Stheno the Snake Queen";

        private readonly int _worldId;

        // One live agent bullet. Position is advanced by GetPosition (the real client
        // math); we keep the previous-tick position for the swept segment.
        private sealed class AgentBullet
        {
            public float StartX;
            public float StartY;
            public float Angle;
            public int BulletId;
            public long SpawnTick;
            public int Damage;
            public ProjectileDesc Desc;
            public bool Dead;
            public float PrevX;
            public float PrevY;
            public bool HasPrev;
            public readonly HashSet<int> HitEnemies = new HashSet<int>(); // MultiHit: one hit per enemy
        }

        // One live enemy bullet, materialised from a captured burst (each burst is
        // Count bullets at Angle + i*AngleInc). Same swept-segment collision vs the
        // agent.
        private sealed class EnemyBullet
        {
            public float OriginX;
            public float OriginY;
            public float Angle;
            public int Damage;
            public float Speed;     // tiles per tick (desc.Speed/100)
            public float Lifetime;  // ticks
            public long SpawnTick;
            public bool Dead;
            public float PrevX;
            public float PrevY;
            public bool HasPrev;
            public bool Multi;
        }

        private readonly List<AgentBullet> _agentBullets = new List<AgentBullet>();
        private readonly List<EnemyBullet> _enemyBullets = new List<EnemyBullet>();
        private long _enemyBurstCursor = -1; // SpawnTick high-water mark already materialised

        // Broad-phase: a uniform spatial grid over the live enemies, rebuilt once per
        // tick, so an agent bullet's swept segment tests ONLY the enemies in the cells it
        // crosses (not all ~440). Each enemy is inserted into every cell its (centre ±
        // radius) box covers, so a query by the segment's box can never miss a real hit.
        // Candidates are tested in ASCENDING enemy index -- the same order the old linear
        // objs.Enemies scan used -- so the non-MultiHit "first overlapped enemy wins"
        // tie-break is byte-identical to the pre-broad-phase collision.
        private const float GRID_CELL = 4.0f; // tiles per cell (a few enemy diameters)
        private readonly Dictionary<long, List<int>> _enemyGrid = new Dictionary<long, List<int>>();
        private readonly List<List<int>> _cellPool = new List<List<int>>();
        private int _cellPoolUsed;
        private readonly List<int> _candidates = new List<int>();
        private readonly HashSet<int> _candidateSeen = new HashSet<int>();

        public SimProjectiles(int worldId)
        {
            _worldId = worldId;
        }

        public void Reset()
        {
            _agentBullets.Clear();
            _enemyBullets.Clear();
            _enemyBurstCursor = -1;
        }

        // Register the shots the agent fired this tick (drained from SimAgent by the
        // RL loop). Each carries the rolled weapon damage + desc.
        public void AddAgentShots(List<WorldServer.core.objects.SimShot> shots)
        {
            if (shots == null)
                return;
            foreach (var s in shots)
            {
                if (s.Desc == null)
                    continue;
                _agentBullets.Add(new AgentBullet
                {
                    StartX = s.StartX,
                    StartY = s.StartY,
                    Angle = s.Angle,
                    BulletId = s.BulletId,
                    SpawnTick = s.SpawnTick,
                    Damage = s.Damage,
                    Desc = s.Desc,
                });
            }
        }

        // Step + collide BOTH bullet sets for the tick that just advanced to nowTick.
        // Called from SimRlLoop after the action is applied (which captures new agent
        // shots) and the enemy bursts for this tick are recorded. Mutates enemy HP
        // (agent bullets) and agent HP (enemy bullets) exactly like EnemyHit/PlayerHit.
        public void StepAndCollide(SimObjects objs, long nowTick, ref TickTime time)
        {
            MaterialiseEnemyBursts(nowTick);
            // Only build the broad-phase grid when there is at least one live agent bullet
            // to query it (the common case is zero, e.g. between shots), so an idle tick
            // pays nothing.
            if (_agentBullets.Count > 0)
                BuildEnemyGrid(objs);
            StepAgentBullets(objs, nowTick, ref time);
            StepEnemyBullets(objs, nowTick);
        }

        private static long CellKey(int cx, int cy) => ((long)cx << 32) ^ (uint)cy;

        // Rebuild the enemy grid for this tick. Each enemy is inserted into every cell its
        // bounding box (centre ± radius) overlaps, so the segment-box query is conservative
        // (never misses a hit). Cell lists are pooled to avoid per-tick allocations.
        private void BuildEnemyGrid(SimObjects objs)
        {
            foreach (var kv in _enemyGrid)
                kv.Value.Clear();
            _cellPoolUsed = 0;
            var enemies = objs.Enemies;
            for (var i = 0; i < enemies.Count; i++)
            {
                var e = enemies[i];
                if (e.Dead)
                    continue;
                var r = EnemyRadius(e);
                var minCx = (int)Math.Floor((e.X - r) / GRID_CELL);
                var maxCx = (int)Math.Floor((e.X + r) / GRID_CELL);
                var minCy = (int)Math.Floor((e.Y - r) / GRID_CELL);
                var maxCy = (int)Math.Floor((e.Y + r) / GRID_CELL);
                for (var cy = minCy; cy <= maxCy; cy++)
                    for (var cx = minCx; cx <= maxCx; cx++)
                        CellList(CellKey(cx, cy)).Add(i);
            }
        }

        private List<int> CellList(long key)
        {
            if (_enemyGrid.TryGetValue(key, out var list))
                return list;
            if (_cellPoolUsed < _cellPool.Count)
                list = _cellPool[_cellPoolUsed];
            else
            {
                list = new List<int>();
                _cellPool.Add(list);
            }
            _cellPoolUsed++;
            _enemyGrid[key] = list;
            return list;
        }

        // Gather the unique enemy indices whose grid cells overlap the segment's bounding
        // box, in ASCENDING index order (== the old linear objs.Enemies scan order).
        private void GatherCandidates(float ax, float ay, float bx, float by)
        {
            _candidates.Clear();
            _candidateSeen.Clear();
            var minX = Math.Min(ax, bx);
            var maxX = Math.Max(ax, bx);
            var minY = Math.Min(ay, by);
            var maxY = Math.Max(ay, by);
            var minCx = (int)Math.Floor(minX / GRID_CELL);
            var maxCx = (int)Math.Floor(maxX / GRID_CELL);
            var minCy = (int)Math.Floor(minY / GRID_CELL);
            var maxCy = (int)Math.Floor(maxY / GRID_CELL);
            for (var cy = minCy; cy <= maxCy; cy++)
                for (var cx = minCx; cx <= maxCx; cx++)
                    if (_enemyGrid.TryGetValue(CellKey(cx, cy), out var list))
                        foreach (var idx in list)
                            if (_candidateSeen.Add(idx))
                                _candidates.Add(idx);
            _candidates.Sort();
        }

        // Pull any newly-captured enemy bursts into per-bullet records (one burst ->
        // Count bullets). Bursts are stamped with the tick they fired; we only
        // materialise bursts at or after our cursor so each is taken exactly once.
        private void MaterialiseEnemyBursts(long nowTick)
        {
            // Read the live burst list directly (no copy); we only READ it and track our
            // own high-water cursor, the obs builder owns pruning it.
            var bursts = SimEnemyShoots.Live(_worldId);
            long maxTick = _enemyBurstCursor;
            for (var bi = 0; bi < bursts.Count; bi++)
            {
                var b = bursts[bi];
                if (b.SpawnTick > maxTick)
                    maxTick = b.SpawnTick;
                if (b.SpawnTick <= _enemyBurstCursor)
                    continue;
                for (var i = 0; i < b.Count; i++)
                {
                    var a = b.Angle + i * b.AngleInc;
                    _enemyBullets.Add(new EnemyBullet
                    {
                        OriginX = b.OriginX,
                        OriginY = b.OriginY,
                        Angle = a,
                        Damage = b.Damage,
                        Speed = b.Speed,
                        Lifetime = b.Lifetime,
                        SpawnTick = b.SpawnTick,
                        Multi = b.MultiHit,
                    });
                }
            }
            _enemyBurstCursor = maxTick;
        }

        private void StepAgentBullets(SimObjects objs, long nowTick, ref TickTime time)
        {
            foreach (var p in _agentBullets)
            {
                if (p.Dead)
                    continue;
                var elapsedMs = (nowTick - p.SpawnTick) * 100;
                if (elapsedMs < 0)
                    continue;
                if (elapsedMs > p.Desc.LifetimeMS)
                {
                    p.Dead = true;
                    continue;
                }
                var rel = Player.ValidatedProjectile.GetPosition(elapsedMs, p.BulletId, p.Desc, p.Angle, 1.0f);
                var curX = p.StartX + (float)rel.X;
                var curY = p.StartY + (float)rel.Y;
                var fromX = p.HasPrev ? p.PrevX : p.StartX;
                var fromY = p.HasPrev ? p.PrevY : p.StartY;
                p.PrevX = curX;
                p.PrevY = curY;
                p.HasPrev = true;

                // Broad-phase: only the enemies whose grid cells the swept segment crosses,
                // visited in ascending index order (== the old full objs.Enemies scan).
                GatherCandidates(fromX, fromY, curX, curY);
                var enemies = objs.Enemies;
                foreach (var ei in _candidates)
                {
                    var e = enemies[ei];
                    if (e.Dead)
                        continue;
                    if (p.HitEnemies.Contains(e.Id))
                        continue;
                    var radius = EnemyRadius(e);
                    if (!SegmentHitsCircle(fromX, fromY, curX, curY, e.X, e.Y, radius))
                        continue;
                    // EnemyHit: the exact in-process damage application
                    // (StatsManager.DamageWithDefense == the real EnemyHit formula).
                    if (!e.HasConditionEffect(ConditionEffectIndex.Invulnerable)
                        && !e.HasConditionEffect(ConditionEffectIndex.Invincible))
                    {
                        var dmg = WorldServer.core.net.stats.StatsManager.DamageWithDefense(e, p.Damage, p.Desc.ArmorPiercing, e.Defense);
                        if (e.ObjectDesc != null && e.ObjectDesc.IdName == BOSS_ID)
                        {
                            // The boss takes REAL projectile damage but is CLAMPED at 1
                            // (not despawned): the RL loop reads bossHp<=1 as the clear
                            // and re-pins HP next episode, so the throwaway pit + boss
                            // persist across episodes (Enemy.Death would remove the boss
                            // for good). The HP drop is genuinely from the agent's shots.
                            var hp = e.Health - dmg;
                            e.Health = hp < 1 ? 1 : hp;
                        }
                        else
                        {
                            e.Health -= dmg;
                            if (e.Health < 0 && e.World != null)
                                e.Death(ref time);
                        }
                    }
                    p.HitEnemies.Add(e.Id);
                    if (!p.Desc.MultiHit)
                    {
                        p.Dead = true;
                        break;
                    }
                }
            }
            _agentBullets.RemoveAll(p => p.Dead);
        }

        private void StepEnemyBullets(SimObjects objs, long nowTick)
        {
            var agent = objs.Agent;
            foreach (var p in _enemyBullets)
            {
                if (p.Dead)
                    continue;
                var age = nowTick - p.SpawnTick; // ticks
                if (age < 0)
                    continue;
                if (age > p.Lifetime)
                {
                    p.Dead = true;
                    continue;
                }
                var ca = (float)Math.Cos(p.Angle);
                var sa = (float)Math.Sin(p.Angle);
                var curX = p.OriginX + ca * p.Speed * age;
                var curY = p.OriginY + sa * p.Speed * age;
                var fromX = p.HasPrev ? p.PrevX : p.OriginX;
                var fromY = p.HasPrev ? p.PrevY : p.OriginY;
                p.PrevX = curX;
                p.PrevY = curY;
                p.HasPrev = true;

                if (SegmentHitsCircle(fromX, fromY, curX, curY, agent.X, agent.Y, AGENT_HIT_RADIUS))
                {
                    // PlayerHit: SimAgent.Damage applies DEF + invuln, == the real
                    // PlayerHit damage application against the agent.
                    agent.Damage(p.Damage, null);
                    if (!p.Multi)
                        p.Dead = true;
                }
            }
            _enemyBullets.RemoveAll(p => p.Dead);
        }

        private static float EnemyRadius(Entity e)
        {
            var size = e.Size <= 0 ? 100 : e.Size;
            return size / 100f * ENEMY_RADIUS_SCALE;
        }

        // True if the segment (ax,ay)->(bx,by) passes within `radius` of (cx,cy)
        // (point-bullet vs circle-target, the client's collision test).
        private static bool SegmentHitsCircle(float ax, float ay, float bx, float by, float cx, float cy, float radius)
        {
            var dx = bx - ax;
            var dy = by - ay;
            var lenSq = dx * dx + dy * dy;
            float t;
            if (lenSq <= 1e-9f)
                t = 0f;
            else
            {
                t = ((cx - ax) * dx + (cy - ay) * dy) / lenSq;
                if (t < 0f) t = 0f;
                else if (t > 1f) t = 1f;
            }
            var px = ax + t * dx;
            var py = ay + t * dy;
            var ex = px - cx;
            var ey = py - cy;
            return ex * ex + ey * ey <= radius * radius;
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

    // Light value carrier so the collision step doesn't re-scan the world's enemy
    // dict per bullet: the RL loop snapshots the live enemies + agent once per tick.
    internal sealed class SimObjects
    {
        public WorldServer.core.objects.SimAgent Agent;
        public List<WorldServer.core.objects.Enemy> Enemies;
    }
}
