using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace WorldServer.core.worlds
{
    // THROWAWAY sim-mode boot harness. Spawns SIM_WORLDS Snake Pit worlds, owns
    // the shared CSV timeline writer, and (per-world) hands a SimHarness to the
    // world's RootWorldThread so all measurement runs on the world thread.
    //
    // Only the Snake Pit worlds get a harness; the Nexus/Arena that boot alongside
    // are left untouched. The per-world index is assigned at attach time so there
    // is no ordering dependency with world creation (CreateNewWorld spawns the
    // RootWorldThread synchronously, which calls AttachHarness immediately).
    public static class SimRunner
    {
        private static TextWriter _log;
        private static readonly ConcurrentDictionary<int, byte> _attached = new ConcurrentDictionary<int, byte>();
        private static int _nextIndex;
        private static int _nextShmSlot;

        // Server-as-sim: hand each Snake Pit world a unique shm agent slot 0..N-1, in
        // RootWorldThread-start order. SimMode.Worlds == N is the agent count the shm
        // region was sized for, so slots never exceed N (one pit per slot).
        public static int NextShmSlot() => Interlocked.Increment(ref _nextShmSlot) - 1;

        public static void Start(GameServer gameServer)
        {
            if (!SimMode.Harness)
                return;

            // Allocate the shared-memory region for N agents BEFORE the worlds spawn
            // (their RootWorldThreads claim slots as they boot).
            if (SimShmBridge.Enabled)
                SimShmBridge.Init(SimMode.Worlds);

            var logPath = SimMode.LogPath;
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(logPath)));
            _log = TextWriter.Synchronized(new StreamWriter(logPath, false) { AutoFlush = false });
            _log.WriteLine("world_index,logical_tick,boss_state,boss_hp,boss_maxhp,transition,enemy_count");

            Console.WriteLine($"[SIM] harness=on uncapped={SimMode.Uncapped} fixed_dt={SimMode.FixedDtMs}ms worlds={SimMode.Worlds} measure_ticks={SimMode.MeasureTicks} dps_tick={SimMode.ProbeDamagePerTick}");
            Console.WriteLine($"[SIM] timeline -> {Path.GetFullPath(logPath)}");

            for (var i = 0; i < SimMode.Worlds; i++)
            {
                var world = gameServer.WorldManager.CreateNewWorld("Snake Pit", null, null);
                if (world == null)
                {
                    Console.WriteLine("[SIM] FAILED to create Snake Pit world");
                    continue;
                }
                Console.WriteLine($"[SIM] created Snake Pit world id={world.Id}");
            }

            StartReporter();
        }

        // Called by RootWorldThread for each world it owns. Returns a harness only
        // for Snake Pit worlds (the only worlds SimRunner creates), assigning a
        // stable per-world index once. _attached guards against double-attach.
        internal static SimHarness AttachHarness(World world)
        {
            if (_log == null || world.IdName != "Snake Pit")
                return null;
            if (!_attached.TryAdd(world.Id, 0))
                return null;
            var index = Interlocked.Increment(ref _nextIndex) - 1;
            Console.WriteLine($"[SIM] harness attached world id={world.Id} index={index}");
            return new SimHarness(world, index, _log);
        }

        // Periodically prints aggregate ticks/sec across all measured worlds so the
        // speedup factor vs the 10 TPS baseline is visible live in the log.
        private static void StartReporter()
        {
            _ = Task.Factory.StartNew(() =>
            {
                long lastTicks = 0;
                var lastWall = SimHarness.WallSeconds;
                while (true)
                {
                    Thread.Sleep(2000);
                    var ticks = SimHarness.TotalTicks;
                    var wall = SimHarness.WallSeconds;
                    var dTicks = ticks - lastTicks;
                    var dWall = wall - lastWall;
                    var tps = dWall > 0 ? dTicks / dWall : 0;
                    var perWorld = SimMode.Worlds > 0 ? tps / SimMode.Worlds : tps;
                    Console.WriteLine($"[SIM] aggregate_tps={tps:F0} per_world_tps={perWorld:F0} speedup_vs_10tps={(tps / 10.0):F1}x total_ticks={ticks}");
                    lastTicks = ticks;
                    lastWall = wall;
                }
            }, TaskCreationOptions.LongRunning);
        }
    }
}
