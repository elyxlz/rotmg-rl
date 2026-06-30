using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace WorldServer.core.worlds
{
    // THROWAWAY pure-shared-memory lockstep barrier (server-as-sim, SIM_SHM_BARRIER=1).
    // Replaces the redis LPUSH/BLPOP step gate (SimStepGate) with atomic generation counters
    // + Linux futexes, on BOTH edges:
    //
    //   cross-process (C-shim <-> server), shm region tail:
    //     ctrl[0] = req  (C-shim bumps to gen G == actions ready, tick now)
    //     ctrl[1] = done (this side bumps to G == obs/reward/done for G in shm)
    //   in-process (controller <-> N world threads), managed words:
    //     _goGen     : controller bumps + ONE futex broadcast wakes ALL N worlds at once
    //     _doneCount : each world Interlocked.Increment after its tick; the controller
    //                  futex-waits on it until it reaches N
    //
    // The key over a per-world AutoResetEvent pair: releasing/collecting N worlds is O(1)
    // syscalls (one broadcast wake + one wait), not O(N) kernel wakeups -- so the per-tick
    // cost stays flat as N grows, which is what lets aggregate SPS scale with N. All waits
    // are futex-parks (cooperative; the server runs at low priority, so busy-spinning is
    // counterproductive). Same clean lockstep contract: ONE c_step == one tick on all worlds,
    // no torn obs (the generation counter prevents the C-shim reading a stale frame).
    public static class SimShmBarrier
    {
        public static readonly bool Enabled =
            Environment.GetEnvironmentVariable("SIM_SHM_BARRIER") == "1";

        private const int SYS_futex = 202;     // x86-64
        private const int FUTEX_WAIT = 0;
        private const int FUTEX_WAKE = 1;

        [DllImport("libc", SetLastError = true)]
        private static extern unsafe long syscall(long number, void *uaddr, int futex_op, int val,
            void *timeout, void *uaddr2, int val3);

        // cross-process control words (shm tail)
        private static unsafe int *_reqPtr;
        private static unsafe int *_donePtr;
        private static int _generation;

        // in-process world fan-out (managed words; pinned so the futex address is stable)
        private static int[] _goBox = new int[1];     // _goBox[0] == release generation
        private static int[] _doneBox = new int[1];   // _doneBox[0] == completed-world count
        private static GCHandle _goHandle;
        private static GCHandle _doneHandle;
        private static unsafe int *_goPtr;
        private static unsafe int *_donePtrLocal;

        private static readonly ConcurrentDictionary<int, byte> _worlds = new ConcurrentDictionary<int, byte>();
        private static int _started;

        public static unsafe void Init(IntPtr reqPtr, IntPtr donePtr)
        {
            _reqPtr = (int *)reqPtr;
            _donePtr = (int *)donePtr;
            _generation = 0;
            _goHandle = GCHandle.Alloc(_goBox, GCHandleType.Pinned);
            _doneHandle = GCHandle.Alloc(_doneBox, GCHandleType.Pinned);
            _goPtr = (int *)_goHandle.AddrOfPinnedObject();
            _donePtrLocal = (int *)_doneHandle.AddrOfPinnedObject();
            _goBox[0] = 0;
            _doneBox[0] = 0;
        }

        public static unsafe void Start()
        {
            if (!Enabled)
                return;
            if (Interlocked.Exchange(ref _started, 1) == 1)
                return;
            if (_reqPtr == null)
            {
                Console.WriteLine("[SIM-BARRIER] ERROR: control pointers not initialized (SIM_SHM must be on)");
                return;
            }
            Console.WriteLine("[SIM-BARRIER] pure-shm futex lockstep ON (no redis on the hot path, O(1) fan-out)");
            Task.Factory.StartNew(ControllerLoop, TaskCreationOptions.LongRunning);
        }

        // Returns the current release generation so a freshly-registered world starts its
        // local counter at "now" and joins on the NEXT tick, instead of fast-forwarding
        // through every generation it missed before it existed.
        internal static int Register(int worldId)
        {
            _worlds.TryAdd(worldId, 0);
            return Volatile.Read(ref _goBox[0]);
        }
        internal static void Unregister(int worldId) => _worlds.TryRemove(worldId, out _);

        // World thread: futex-park on the shared go word until the controller broadcasts the
        // next release generation. One wake covers all parked worlds. The world tracks its own
        // last-served generation via the ref so a stale wake never double-counts a tick.
        internal static unsafe void WaitForGo(int worldId, ref int lastGen)
        {
            var target = lastGen + 1;
            while (true)
            {
                var cur = Volatile.Read(ref _goBox[0]);
                if (cur >= target)
                    break;
                syscall(SYS_futex, _goPtr, FUTEX_WAIT, cur, null, null, 0);
            }
            lastGen = target;
        }

        // World thread: this tick is complete. Increment the shared count; the last world to
        // finish wakes the controller parked on it.
        internal static unsafe void SignalDone(int worldId)
        {
            Interlocked.Increment(ref _doneBox[0]);
            syscall(SYS_futex, _donePtrLocal, FUTEX_WAKE, int.MaxValue, null, null, 0);
        }

        private static unsafe int LoadAcquire(int *p) => Volatile.Read(ref *p);
        private static unsafe void StoreRelease(int *p, int v) => Volatile.Write(ref *p, v);

        private static unsafe void WaitForRequest()
        {
            var target = _generation + 1;
            for (var spin = 0; spin < 1000; spin++)
            {
                if (LoadAcquire(_reqPtr) >= target)
                    return;
                Thread.SpinWait(16);
            }
            while (true)
            {
                var cur = LoadAcquire(_reqPtr);
                if (cur >= target)
                    return;
                syscall(SYS_futex, _reqPtr, FUTEX_WAIT, cur, null, null, 0);
            }
        }

        private static unsafe void PublishDone(int generation)
        {
            StoreRelease(_donePtr, generation);
            syscall(SYS_futex, _donePtr, FUTEX_WAKE, int.MaxValue, null, null, 0);
        }

        // One request -> broadcast-release all worlds for one tick -> futex-wait for the
        // done count to reach the snapshot count -> publish done. The participant count is
        // snapshotted per tick, so a lazily-spawned pit joins on the next tick.
        private static unsafe void ControllerLoop()
        {
            while (true)
            {
                WaitForRequest();
                _generation++;

                var nworlds = _worlds.Count;
                Volatile.Write(ref _doneBox[0], 0);
                // release: bump the go generation, then ONE broadcast wakes every parked world.
                Volatile.Write(ref _goBox[0], _generation);
                syscall(SYS_futex, _goPtr, FUTEX_WAKE, int.MaxValue, null, null, 0);

                while (true)
                {
                    var cur = Volatile.Read(ref _doneBox[0]);
                    if (cur >= nworlds)
                        break;
                    syscall(SYS_futex, _donePtrLocal, FUTEX_WAIT, cur, null, null, 0);
                }

                PublishDone(_generation);
            }
        }
    }
}
