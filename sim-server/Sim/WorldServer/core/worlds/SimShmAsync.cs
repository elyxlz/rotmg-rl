using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace WorldServer.core.worlds
{
    // THROWAWAY async-overlap futex helper (server-as-sim, SIM_ASYNC=1). Owns a stable
    // native pointer into the shm act_seq[N] block so a free-run world thread can PARK on
    // its own slot's action-sequence word until the C-shim posts a fresh action, instead of
    // busy-spinning a core while it waits for the GPU policy. This is the only sync the async
    // path needs -- there is NO global barrier: each world waits ONLY on its own act_seq, and
    // the C-shim collects each transition via obs_seq non-blocking, so the worlds and the GPU
    // overlap (the pipeline) instead of taking strict lockstep turns.
    //
    // The C-shim, after writing the action + bumping act_seq[slot] for all N, issues ONE
    // shared FUTEX_WAKE on the act_seq base address (it bumps a contiguous run, so a single
    // wake on the block covers every parked world -- they each re-check their own slot). A
    // short adaptive spin first (the next action usually lands within microseconds), then the
    // futex park, mirrors the barrier's cooperative wait so the server never hot-spins a core.
    public static class SimShmAsync
    {
        public static readonly bool Enabled =
            Environment.GetEnvironmentVariable("SIM_ASYNC") == "1";

        private const int SYS_futex = 202; // x86-64
        private const int FUTEX_WAIT = 0;
        private const int FUTEX_WAKE = 1;
        private const int SRV_SPIN_BUDGET = 2000;

        [DllImport("libc", SetLastError = true)]
        private static extern unsafe long syscall(long number, void *uaddr, int futex_op, int val,
            void *timeout, void *uaddr2, int val3);

        private static unsafe int *_actSeqBase; // &act_seq[0] in the shared page
        private static unsafe int *_obsSeqBase; // &obs_seq[0] in the shared page

        // Hand the helper stable native pointers to act_seq[0] (the world parks here for a
        // fresh action; the C-shim FUTEX_WAKEs it) and obs_seq[0] (the world FUTEX_WAKEs here
        // after publishing a transition so the C-shim parked in c_step wakes). Called once from
        // SimShmBridge.Init after the region is mapped.
        public static unsafe void Init(IntPtr actSeqPtr, IntPtr obsSeqPtr)
        {
            _actSeqBase = (int *)actSeqPtr;
            _obsSeqBase = (int *)obsSeqPtr;
        }

        // Wake the C-shim parked in c_step waiting for this slot to publish its transition. The
        // C-shim futex-waits on obs_seq[slot], so the wake MUST target that per-slot address.
        // Called AFTER SimShmBridge.WriteObsSeq, so the value the woken C-shim reads is current.
        public static unsafe void WakeObs(int slot)
        {
            if (_obsSeqBase == null)
                return;
            syscall(SYS_futex, _obsSeqBase + slot, FUTEX_WAKE, int.MaxValue, null, null, 0);
        }

        // Park the calling world thread until act_seq[slot] exceeds lastConsumed (a fresh
        // action is in shm). Adaptive spin first, then a futex park on the slot word. Returns
        // immediately if a fresh action is already pending (the common case once the pipeline
        // is full). Never blocks any OTHER world -- the wait is per-slot.
        public static unsafe void WaitForAction(int slot, int lastConsumed)
        {
            if (_actSeqBase == null)
                return;
            var p = _actSeqBase + slot;
            for (var spin = 0; spin < SRV_SPIN_BUDGET; spin++)
            {
                if (Volatile.Read(ref *p) > lastConsumed)
                    return;
                Thread.SpinWait(16);
            }
            while (true)
            {
                var cur = Volatile.Read(ref *p);
                if (cur > lastConsumed)
                    return;
                syscall(SYS_futex, p, FUTEX_WAIT, cur, null, null, 0);
            }
        }
    }
}
