using System.Collections.Generic;
using WorldServer.networking.packets.outgoing;

namespace WorldServer.core.worlds
{
    // THROWAWAY in-process enemy-bullet collector (sim-mode only). The nrelay obs
    // path reconstructs enemy bullets from the EnemyShoot PACKET stream (the server
    // streams firing BURSTS, not per-frame positions; the client forward-simulates
    // each burst to the current positions). To build the SAME obs IN-PROCESS with no
    // nrelay client, we capture those same bursts directly at the server-side
    // broadcast point and forward-simulate them ourselves, identically.
    //
    // One burst record == one EnemyShootMessage, plus the spawn tick so the in-proc
    // obs builder can age it by logical ticks (1 tick = SIM_FIXED_DT_MS = 100ms),
    // matching RealObsBuilder._active_bullets exactly.
    //
    // Keyed by world id so parallel sim worlds don't cross-pollinate. Bursts are
    // appended on broadcast and pruned by the obs builder once they expire.
    internal struct SimBurst
    {
        public float OriginX;
        public float OriginY;
        public float Angle;
        public int Count;
        public float AngleInc;
        public float Speed;     // tiles per 100ms (ProjectileDesc.Speed / 100)
        public float Lifetime;  // ticks (ProjectileDesc.LifetimeMS / 100)
        public long SpawnTick;  // logical tick at which the burst was fired
        public int Damage;      // per-bullet damage (EnemyShootMessage.Damage)
        public bool MultiHit;   // desc.MultiHit -> a bullet can hit through the agent
    }

    internal static class SimEnemyShoots
    {
        private static readonly Dictionary<int, List<SimBurst>> _byWorld = new Dictionary<int, List<SimBurst>>();
        private static readonly Dictionary<int, long> _tickByWorld = new Dictionary<int, long>();
        private static readonly object _lock = new object();

        // The in-proc RL loop publishes the current logical tick for a world before
        // its World.Update, so bursts fired during that tick are stamped with the
        // same tick the obs builder ages them against (== the bridge stamping a
        // burst with the now-tick it arrived on).
        public static void SetTick(int worldId, long tick)
        {
            lock (_lock)
                _tickByWorld[worldId] = tick;
        }

        // Capture a burst at the server-side broadcast. Speed/Lifetime are pulled
        // from the ProjectileDesc exactly as the nrelay bridge does
        // (speed = desc.Speed/100, lifetime = desc.LifetimeMS/100). spawnTick comes
        // from the current per-world logical tick.
        public static void Record(int worldId, EnemyShootMessage pkt)
        {
            long spawnTick;
            lock (_lock)
                spawnTick = _tickByWorld.TryGetValue(worldId, out var t) ? t : 0;
            if (pkt.ProjectileDesc == null)
                return;
            var burst = new SimBurst
            {
                OriginX = pkt.StartingPos.X,
                OriginY = pkt.StartingPos.Y,
                Angle = pkt.Angle,
                Count = pkt.NumShots,
                AngleInc = pkt.AngleInc,
                Speed = pkt.ProjectileDesc.Speed / 100f,
                Lifetime = pkt.ProjectileDesc.LifetimeMS / 100f,
                SpawnTick = spawnTick,
                Damage = pkt.Damage,
                MultiHit = pkt.ProjectileDesc.MultiHit,
            };
            lock (_lock)
            {
                if (!_byWorld.TryGetValue(worldId, out var list))
                {
                    list = new List<SimBurst>();
                    _byWorld[worldId] = list;
                }
                list.Add(burst);
            }
        }

        // The live burst list for a world, returned DIRECTLY (no copy). Append (Record)
        // and read (obs/collision) both run on the same world thread, so the caller may
        // iterate and prune it in place without copying ~hundreds of structs twice every
        // tick (the old Snapshot allocated a fresh List on every call -- twice per tick).
        public static List<SimBurst> Live(int worldId)
        {
            lock (_lock)
            {
                if (!_byWorld.TryGetValue(worldId, out var list))
                {
                    list = new List<SimBurst>();
                    _byWorld[worldId] = list;
                }
                return list;
            }
        }

        // Copy of the live bursts (used only by the obs-MATCH dump path, which must not
        // mutate the list the obs builder then prunes).
        public static List<SimBurst> Snapshot(int worldId)
        {
            lock (_lock)
            {
                if (_byWorld.TryGetValue(worldId, out var list))
                    return new List<SimBurst>(list);
                return new List<SimBurst>();
            }
        }

        public static void Clear(int worldId)
        {
            lock (_lock)
            {
                _byWorld.Remove(worldId);
                _tickByWorld.Remove(worldId);
            }
        }
    }
}
