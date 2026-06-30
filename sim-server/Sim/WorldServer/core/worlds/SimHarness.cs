using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using WorldServer.core.objects;

namespace WorldServer.core.worlds
{
    // THROWAWAY measurement harness (sim-mode only). One instance per Snake Pit
    // world, owned by that world's RootWorldThread, so all per-tick work runs on
    // the world thread (deterministic, no cross-thread races).
    //
    // Per logical tick it: (1) lazily injects a stationary SimProbe at the boss's
    // coords once the boss has spawned (keeps the chunk hot + gives the boss a
    // target), (2) applies a fixed scripted damage to the boss (so HP-gated phase
    // transitions fire and can be compared across modes), (3) logs the boss's
    // CurrentState name + HP keyed by logical tick number to a CSV.
    internal sealed class SimHarness
    {
        private const string BOSS_ID = "Stheno the Snake Queen";

        private readonly World _world;
        private readonly int _index;
        private readonly int _measureTicks;
        private readonly int _damagePerTick;
        private readonly TextWriter _log;

        private SimProbe _probe;
        private Enemy _boss;
        private long _logicalTick;
        private string _lastState = "";
        private bool _done;

        // Shared across all worlds: total logical ticks + the wall-clock window,
        // so the parent can compute aggregate ticks/sec across parallel worlds.
        private static long _totalTicks;
        private static readonly Stopwatch _wall = Stopwatch.StartNew();
        public static long TotalTicks => Interlocked.Read(ref _totalTicks);
        public static double WallSeconds => _wall.Elapsed.TotalSeconds;
        // In-proc throughput probe: the SimRlLoop drives ticks itself (no harness),
        // so it counts into the same global tally that StartReporter prints.
        public static void CountInProcTick() => Interlocked.Increment(ref _totalTicks);

        public SimHarness(World world, int index, TextWriter log)
        {
            _world = world;
            _index = index;
            _log = log;
            _measureTicks = SimMode.MeasureTicks;
            _damagePerTick = SimMode.ProbeDamagePerTick;
        }

        public void OnTick()
        {
            Interlocked.Increment(ref _totalTicks);
            if (_done)
                return;

            EnsureBoss();
            if (_boss == null)
                return; // boss not yet pulled from EntitiesToAdd

            EnsureProbe();

            // Scripted firing pattern: a fixed HP bite each logical tick. Skipped
            // while the boss is Invulnerable (Start / phase-start windows) so the
            // timed transitions there are observed cleanly. Deterministic and
            // identical across fixed-dt and real-time modes.
            if (_damagePerTick > 0 && !_boss.HasConditionEffect(Shared.resources.ConditionEffectIndex.Invulnerable))
            {
                var hp = _boss.Health - _damagePerTick;
                _boss.Health = hp < 1 ? 1 : hp; // keep it alive; we measure phases not the kill
            }

            var state = _boss.CurrentState != null ? _boss.CurrentState.Name : "(null)";
            // Log every tick for the boss-HP trajectory; also flag the tick where
            // the state name changes (the phase-transition tick number). enemy_count
            // captures the cooldown-driven Reproduce("Stheno Swarm") spawn cadence,
            // a direct check that cooldown-scaled behaviors advance per logical tick
            // identically across modes.
            var transition = state != _lastState ? 1 : 0;
            _log.WriteLine($"{_index},{_logicalTick},{state},{_boss.Health},{_boss.MaxHealth},{transition},{_world.Enemies.Count}");
            _lastState = state;

            _logicalTick++;
            if (_logicalTick >= _measureTicks)
            {
                _done = true;
                _log.Flush();
            }
        }

        private void EnsureBoss()
        {
            if (_boss != null)
                return;
            foreach (var e in _world.Enemies.Values)
                if (e.ObjectDesc != null && e.ObjectDesc.IdName == BOSS_ID)
                {
                    _boss = e;
                    // Pin the boss HP to a deterministic value. The stock spawn
                    // path rolls ClasifyEnemy() (20% chance of Rare/Epic/Legendary,
                    // MaxHealth *= 1..3), which is RNG independent of the dt
                    // decouple and would otherwise move the HP-gated phase
                    // transitions between runs. Pinning makes the two modes start
                    // from identical HP so the fidelity comparison is clean.
                    if (SimMode.BossHp > 0)
                    {
                        _boss.MaxHealth = SimMode.BossHp;
                        _boss.Health = SimMode.BossHp;
                    }
                    // Lockstep: publish the boss coords so the harness can /tppos the
                    // agent onto the boss (the policy doesn't model the ~80-tile maze).
                    SimStepGate.PublishBossCoords(_world.Id, _boss.X, _boss.Y);
                    return;
                }
        }

        private void EnsureProbe()
        {
            if (_probe != null)
                return;
            _probe = new SimProbe(_world.GameServer);
            // Place on the boss so PlayerWithinTransition(20) fires immediately.
            _probe.Move(_boss.X, _boss.Y);
            _world.EnterWorld(_probe);
        }
    }
}
