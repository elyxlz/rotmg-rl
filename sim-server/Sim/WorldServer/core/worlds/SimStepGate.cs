using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using StackExchange.Redis;

namespace WorldServer.core.worlds
{
    // THROWAWAY step-on-command gate (sim-mode only, SIM_STEP_DRIVEN=1). Turns the
    // free-running uncapped fixed-dt loop into a LOCKSTEP one: every sim world
    // advances EXACTLY ONE logical 100ms tick per external "tick" command, and the
    // gate acks once all worlds have completed that tick. This is what aligns the
    // RL obs and action to the same tick (read obs -> act -> step one tick -> ack).
    //
    // Control channel: the dedicated sim redis (SIM_STEP_REDIS_* env, defaults to
    // the same 127.0.0.1:6390 db5 the server already uses). The harness LPUSHes a
    // token onto SIM_STEP_CMD_KEY; the gate BLPOPs it, releases all world threads
    // for one tick, waits for them, then LPUSHes onto SIM_STEP_ACK_KEY. Redis lists
    // give a blocking, language-agnostic rendezvous (the Python harness speaks it
    // with plain redis-py) with no busy-polling.
    //
    // Dynamic worlds: the agent portals into its OWN Snake Pit world AFTER login, so
    // the participant set grows over time. Each tick snapshots the currently
    // registered worlds, so a world that appears between ticks simply joins on the
    // next command -- no fixed participant count, no Barrier reconfiguration.
    public static class SimStepGate
    {
        // True only when SIM_STEP_DRIVEN=1. The uncapped loop branches on this to
        // wait at the gate instead of free-running.
        public static readonly bool Enabled =
            Environment.GetEnvironmentVariable("SIM_STEP_DRIVEN") == "1";

        private static readonly string CmdKey =
            Environment.GetEnvironmentVariable("SIM_STEP_CMD_KEY") ?? "sim:step:cmd";
        private static readonly string AckKey =
            Environment.GetEnvironmentVariable("SIM_STEP_ACK_KEY") ?? "sim:step:ack";

        // Per-world rendezvous: the controller sets Go (release one tick), the world
        // sets Done (tick complete). Auto-reset so each pair drives exactly one tick.
        private sealed class WorldGate
        {
            public readonly AutoResetEvent Go = new AutoResetEvent(false);
            public readonly AutoResetEvent Done = new AutoResetEvent(false);
        }

        private static readonly ConcurrentDictionary<int, WorldGate> _gates =
            new ConcurrentDictionary<int, WorldGate>();
        private static IDatabase _redis;
        private static int _started;

        // Boots the controller loop once. Reuses the server's redis endpoint (the
        // sim db5) over a dedicated connection so the lockstep traffic never shares
        // the gameplay multiplexer. No-op unless SIM_STEP_DRIVEN=1.
        public static void Start()
        {
            if (!Enabled)
                return;
            if (Interlocked.Exchange(ref _started, 1) == 1)
                return;

            var host = Environment.GetEnvironmentVariable("SIM_STEP_REDIS_HOST") ?? "127.0.0.1";
            var port = ReadInt("SIM_STEP_REDIS_PORT", 6390);
            var db = ReadInt("SIM_STEP_REDIS_DB", 5);

            var cfg = new ConfigurationOptions
            {
                EndPoints = { { host, port } },
                AbortOnConnectFail = false,
                ConnectTimeout = 5000,
            };
            var mux = ConnectionMultiplexer.Connect(cfg);
            _redis = mux.GetDatabase(db);

            // Drain any stale tokens so a fresh run starts clean.
            _redis.KeyDelete(CmdKey);
            _redis.KeyDelete(AckKey);

            Console.WriteLine($"[SIM] step-driven lockstep ON: redis {host}:{port} db{db} cmd='{CmdKey}' ack='{AckKey}'");

            Task.Factory.StartNew(ControllerLoop, TaskCreationOptions.LongRunning);
        }

        // Each uncapped world thread calls this once to join the lockstep set.
        internal static void Register(int worldId) =>
            _gates.TryAdd(worldId, new WorldGate());

        internal static void Unregister(int worldId) =>
            _gates.TryRemove(worldId, out _);

        // World thread: block until the controller releases this world for one tick.
        internal static void WaitForGo(int worldId)
        {
            if (_gates.TryGetValue(worldId, out var g))
                g.Go.WaitOne();
        }

        // World thread: signal the tick is complete.
        internal static void SignalDone(int worldId)
        {
            if (_gates.TryGetValue(worldId, out var g))
                g.Done.Set();
        }

        // Publish the boss spawn coords for a world so the harness can /tppos the
        // agent onto the boss (bridges the unmodelled-maze nav gap). Per-world key
        // plus a "latest" key for the single-agent case. No-op if the gate's redis
        // isn't up (free-running, non-lockstep runs).
        internal static void PublishBossCoords(int worldId, float x, float y)
        {
            if (_redis == null)
                return;
            var val = $"{x},{y}";
            _redis.StringSet($"sim:boss:coords:{worldId}", val);
            _redis.StringSet("sim:boss:coords", val);
        }

        // Controller: one external "tick" command -> release every registered world
        // for exactly one tick -> wait for all -> ack. The snapshot is taken per
        // command so late-joining worlds are picked up on the next tick, never
        // mid-tick.
        private static void ControllerLoop()
        {
            while (true)
            {
                // Genuinely blocking pop (no busy-poll): BLPOP returns [key, value]
                // the instant the harness pushes a tick command, with a 1s safety
                // timeout so the loop can never wedge. The value is opaque (the
                // harness sends the tick number for its own logging).
                var res = _redis.Execute("BLPOP", CmdKey, "1");
                if (res.IsNull || res.Type != ResultType.MultiBulk)
                    continue; // timed out with no command -> re-arm
                var arr = (RedisResult[])res;
                if (arr == null || arr.Length < 2)
                    continue;
                var token = (RedisValue)arr[1];

                var snapshot = new List<WorldGate>(_gates.Values);
                foreach (var g in snapshot)
                    g.Go.Set();
                foreach (var g in snapshot)
                    g.Done.WaitOne();

                _redis.ListRightPush(AckKey, token);
            }
        }

        private static int ReadInt(string name, int fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);
            if (raw == null || !int.TryParse(raw, out var v))
                return fallback;
            return v;
        }
    }
}
