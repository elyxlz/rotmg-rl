using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Threading;

namespace WorldServer.core.worlds
{
    // THROWAWAY shared-memory bridge (server-as-sim, SIM_SHM=1). A fixed-layout
    // region shared with the PufferLib C-shim env (_pufferlib/ocean/server_env).
    // The lockstep gate (SimStepGate) already serializes access: the C-shim writes
    // actions + signals the gate, the C# worlds advance one tick, write obs/reward/
    // done, then ack. So there are NO concurrent readers/writers of any slot across
    // the barrier -- the gate IS the memory barrier. No locks needed here.
    //
    // Layout (slot i == agent i, contiguous so a single mmap maps it):
    //   [N * OBS_LEN]  obs       float32 (C# writes, C-shim reads)
    //   [N * N_ATNS]   actions   float32 (C-shim writes, C# reads): move, aim, shoot, cast
    //   [N]            rewards   float32 (C# writes, C-shim reads)
    //   [N]            dones     float32 (C# writes, C-shim reads)
    //   [CTRL_INTS]    barrier   int32   (req/done generation counters; SimShmBarrier)
    //   [CONFIG_INTS]  config    int32   (C-shim writes, C# reads -- the d-flow channel)
    // matching the PufferLib vec-buffer dtype (float32) and per-agent stride. The
    // C-shim mmaps the SAME path with the identical offsets (see server_env.h). The
    // difficulty-config tail is the d-flow: the trainer writes the d->knobs each episode,
    // the C# SimRlLoop reads them at every spawn/reset so d ramps live (no restart).
    //
    // The path defaults to /dev/shm/rotmg_sim_shm (a POSIX shm-backed file). Both
    // sides agree on N via SIM_AGENTS / the env config; a header word stores N so a
    // mismatched launch fails loudly instead of corrupting memory.
    public static class SimShmBridge
    {
        public const int OBS_LEN = SimObsBuilder.OBS_LEN; // 9807
        public const int N_ATNS = 4;                      // move, aim, shoot, cast (staff+spell share the aim)

        // 4 header int32s: magic, n_agents, obs_len, n_atns. Lets the C-shim verify
        // the C# side launched with a matching layout before it touches a slot.
        public const int HEADER_INTS = 4;
        public const int MAGIC = 0x52544D47;              // 'RTMG'

        // Pure-shm barrier control words (server-as-sim, SIM_SHM_BARRIER=1). Two atomic
        // int32 generation counters at the TAIL of the region (after [obs][act][rew][done])
        // so every data-region offset above is UNCHANGED and the C-shim + verify scripts keep
        // their layout. ctrl[0]=req (C-shim bumps), ctrl[1]=done (controller bumps). See
        // SimShmBarrier.cs for the futex protocol.
        public const int CTRL_INTS = 2;

        // Live difficulty-config block at the VERY tail (after the ctrl words, so the barrier
        // ctrl offset the C-shim already computes -- right after the done array -- is UNCHANGED).
        // The d-flow channel: the Python trainer derives the d->config knobs and writes them here
        // each episode; the C# SimRlLoop reads them at every spawn/reset so a d change applies LIVE
        // (no server restart). One GLOBAL block (not per-slot): the trainer ramps ONE d across all
        // N worlds. Five int32s: [valid, spawn_geo_dist, agent_hp, agent_def, boss_hp]. valid==MAGIC
        // means "the trainer has written a live config"; until then (zeroed) the server falls back
        // to the static SIM_* env defaults, so the existing fixed-config proof path is unchanged.
        public const int CONFIG_INTS = 5;
        // sub-offsets within the config block
        public const int CFG_VALID = 0;
        public const int CFG_SPAWN_GEO = 1;
        public const int CFG_AGENT_HP = 2;
        public const int CFG_AGENT_DEF = 3;
        public const int CFG_BOSS_HP = 4;

        // ASYNC OVERLAP block at the VERY tail (after the config block, so every existing
        // offset -- header, obs, act, rew, done, ctrl, config -- is UNCHANGED and the lockstep
        // path + verify scripts keep their layout). Per-slot int32 sequence counters for the
        // free-run consistency handshake (SIM_ASYNC=1): act_seq[i] (the C-shim bumps when it
        // posts a fresh action for slot i) and obs_seq[i] (the world bumps when it has consumed
        // a fresh action AND published the resulting obs/reward). c_step posts action seq S then
        // waits for obs_seq[i] >= S, so it collects exactly ONE consistent transition per agent
        // per step (the reward+next_obs that resulted from the action it sent). Sized 2*N int32s.
        private static long _actSeqBase; // act_seq[N] int32 (C-shim writes, C# reads)
        private static long _obsSeqBase; // obs_seq[N] int32 (C# writes, C-shim reads)

        // ASYNC DOUBLE-BUFFER block (SIM_ASYNC). The fix for the free-run obs race: the single
        // [obs][rew][done] slots above are torn-prone under async (a free-run world keeps
        // overwriting a slot while the c_step copies it). Instead, async writes the WHOLE
        // transition (obs + reward + done) into a per-slot DOUBLE BUFFER and publishes which
        // half is live via obs_idx[i] (a release store the c_step reads with acquire). The
        // world writes the NON-current half, then flips obs_idx[i]; the c_step reads obs_idx[i]
        // then copies that half -- always a complete, consistent frame, no retry, no block.
        // Two halves, each [N*OBS_LEN obs][N rew][N done], plus obs_idx[N]. The lockstep path
        // never touches these (it uses the primary [obs][rew][done] slots), so its layout +
        // the verify scripts are UNCHANGED; this block only exists when SimMode.Async.
        private static long _obsIdxBase;        // obs_idx[N] int32 (C# publishes, C-shim reads)
        private static long _dblObsBase;        // obsB[2*N*OBS_LEN] float32 (two halves)
        private static long _dblRewBase;        // rewB[2*N] float32
        private static long _dblDoneBase;       // doneB[2*N] float32

        public static readonly bool Enabled =
            Environment.GetEnvironmentVariable("SIM_SHM") == "1";

        private static readonly string ShmPath =
            Environment.GetEnvironmentVariable("SIM_SHM_PATH") ?? "/dev/shm/rotmg_sim_shm";

        private static int _n;
        private static MemoryMappedFile _mmf;
        private static MemoryMappedViewAccessor _view;

        // Byte offsets into the region (after the header).
        private static long _obsBase;
        private static long _actBase;
        private static long _rewBase;
        private static long _doneBase;
        private static long _cfgBase; // live difficulty-config block (after the ctrl words)

        // Total bytes: header + obs + actions + rewards + dones + ctrl + config + async tail.
        // The async tail (seq handshake + per-slot obs/rew/done double buffer) is ALWAYS sized,
        // so the C# region and the C-shim mmap agree on bytes regardless of which sync mode the
        // run uses -- the mmap-size-vs-pointer-offset mismatch that wrote past the mapped page
        // and corrupted the heap is gone by construction. The lockstep path simply never reads
        // the tail. The double buffer holds TWO halves of [obs][rew][done] per slot.
        private static long RegionBytes(int n) =>
            (long)HEADER_INTS * sizeof(int)
            + (long)n * OBS_LEN * sizeof(float)
            + (long)n * N_ATNS * sizeof(float)
            + (long)n * sizeof(float)
            + (long)n * sizeof(float)
            + (long)CTRL_INTS * sizeof(int)      // barrier req/done
            + (long)CONFIG_INTS * sizeof(int)    // live difficulty config
            + (long)2 * n * sizeof(int)          // async act_seq[N] + obs_seq[N] (SIM_ASYNC)
            + (long)n * sizeof(int)              // async obs_idx[N] (published double-buffer half)
            + (long)2 * n * OBS_LEN * sizeof(float) // async obs double buffer (two halves)
            + (long)2 * n * sizeof(float)        // async reward double buffer
            + (long)2 * n * sizeof(float);       // async done double buffer

        // Create + zero the region for n agents and stamp the header. Called once on
        // boot (server side OWNS creation so the file always matches the C# layout;
        // the C-shim opens it after the server is up). Truncating the backing file
        // to the exact size makes the mmap deterministic across runs.
        public static void Init(int n)
        {
            if (!Enabled)
                return;
            _n = n;
            var bytes = RegionBytes(n);

            // Back the mmap with a real file under /dev/shm (tmpfs) so it is a normal
            // POSIX shared region the C-shim can mmap by path. Recreate fresh each run.
            using (var fs = new FileStream(ShmPath, FileMode.Create, FileAccess.ReadWrite, FileShare.ReadWrite))
                fs.SetLength(bytes);

            _mmf = MemoryMappedFile.CreateFromFile(ShmPath, FileMode.Open, null, bytes, MemoryMappedFileAccess.ReadWrite);
            _view = _mmf.CreateViewAccessor(0, bytes, MemoryMappedFileAccess.ReadWrite);

            var headerBytes = (long)HEADER_INTS * sizeof(int);
            _obsBase = headerBytes;
            _actBase = _obsBase + (long)n * OBS_LEN * sizeof(float);
            _rewBase = _actBase + (long)n * N_ATNS * sizeof(float);
            _doneBase = _rewBase + (long)n * sizeof(float);
            // config block sits after the done array AND the 2 ctrl words (the tail).
            _cfgBase = _doneBase + (long)n * sizeof(float) + (long)CTRL_INTS * sizeof(int);
            // async sequence block sits after the config block (region tail).
            _actSeqBase = _cfgBase + (long)CONFIG_INTS * sizeof(int);
            _obsSeqBase = _actSeqBase + (long)n * sizeof(int);
            // async double-buffer block sits after the seq block: obs_idx[N], then two halves
            // of [obs][rew][done]. obs_idx publishes which half is the current consistent frame.
            _obsIdxBase = _obsSeqBase + (long)n * sizeof(int);
            _dblObsBase = _obsIdxBase + (long)n * sizeof(int);
            _dblRewBase = _dblObsBase + (long)2 * n * OBS_LEN * sizeof(float);
            _dblDoneBase = _dblRewBase + (long)2 * n * sizeof(float);

            // zero the whole region, then stamp the header (magic last so a reader
            // that polls magic only proceeds once the layout words are written).
            for (long i = 0; i < bytes; i += sizeof(int))
                _view.Write(i, 0);
            _view.Write(1 * sizeof(int), n);
            _view.Write(2 * sizeof(int), OBS_LEN);
            _view.Write(3 * sizeof(int), N_ATNS);
            _view.Write(0 * sizeof(int), MAGIC);

            // Pure-shm barrier: hand the controller stable native pointers to the two
            // control words at the region tail. AcquirePointer pins the SafeBuffer for the
            // process lifetime (the region lives until shutdown), so the futex addresses stay
            // valid. ctrl[0]=req, ctrl[1]=done, laid out right after the done array.
            if (SimShmBarrier.Enabled)
            {
                var ctrlBase = _doneBase + (long)n * sizeof(float);
                unsafe
                {
                    byte* basePtr = null;
                    _view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
                    var reqPtr = (IntPtr)(basePtr + ctrlBase);
                    var donePtr = (IntPtr)(basePtr + ctrlBase + sizeof(int));
                    SimShmBarrier.Init(reqPtr, donePtr);
                }
                Console.WriteLine($"[SIM-SHM] barrier ctrl words at byte offset {ctrlBase} (req,done)");
            }

            if (SimMode.Async)
            {
                unsafe
                {
                    byte* asyncBase = null;
                    _view.SafeMemoryMappedViewHandle.AcquirePointer(ref asyncBase);
                    SimShmAsync.Init((IntPtr)(asyncBase + _actSeqBase), (IntPtr)(asyncBase + _obsSeqBase));
                }
            }
            Console.WriteLine($"[SIM-SHM] live difficulty-config block ({CONFIG_INTS} ints) at byte offset {_cfgBase} (valid,spawn_geo,agent_hp,agent_def,boss_hp)");
            if (SimMode.Async)
            {
                Console.WriteLine($"[SIM-SHM] async sequence block (2*{n} ints) act_seq@{_actSeqBase} obs_seq@{_obsSeqBase} (SIM_ASYNC free-run handshake)");
                Console.WriteLine($"[SIM-SHM] async double-buffer obs_idx@{_obsIdxBase} obsB@{_dblObsBase} rewB@{_dblRewBase} doneB@{_dblDoneBase} (tear-free per-slot publish)");
            }
            Console.WriteLine($"[SIM-SHM] region '{ShmPath}' n={n} obs_len={OBS_LEN} n_atns={N_ATNS} bytes={bytes}");
        }

        // C# writes the slot's obs vector (called in PreTick after SimObsBuilder).
        public static void WriteObs(int slot, float[] obs)
        {
            var off = _obsBase + (long)slot * OBS_LEN * sizeof(float);
            _view.WriteArray(off, obs, 0, OBS_LEN);
        }

        // C# reads the slot's action (called in PreTick to drive the agent). The
        // C-shim wrote it before signalling the gate, so it is the action for THIS
        // tick. Floats cast to int per-dim, exactly like the dungeon binding.
        public static (int move, int aim, bool shoot, bool cast) ReadAction(int slot)
        {
            var off = _actBase + (long)slot * N_ATNS * sizeof(float);
            var move = (int)_view.ReadSingle(off + 0 * sizeof(float));
            var aim = (int)_view.ReadSingle(off + 1 * sizeof(float));
            var shoot = (int)_view.ReadSingle(off + 2 * sizeof(float));
            var cast = (int)_view.ReadSingle(off + 3 * sizeof(float));
            // staff + spell share this single aim (one mouse): the cast direction is the
            // staff aim, so there is no separate spell-aim slot to read.
            return (move, aim, shoot != 0, cast != 0);
        }

        // C# writes the slot's reward + done (called in PostTick).
        public static void WriteRewardDone(int slot, float reward, bool done)
        {
            _view.Write(_rewBase + (long)slot * sizeof(float), reward);
            _view.Write(_doneBase + (long)slot * sizeof(float), done ? 1.0f : 0.0f);
        }

        // Read the LIVE difficulty config the trainer wrote (the d-flow channel). Returns
        // (valid, spawnGeoDist, agentHp, agentDef, bossHp). valid==false means the trainer has
        // not written one yet (block still zeroed) -> the caller falls back to the SIM_* env
        // defaults, so a fixed-config proof run (no live writes) behaves exactly as before. The
        // gate already serialized this: the C-shim wrote the block BEFORE bumping req, so the
        // value the world reads at its spawn is the trainer's current d-config. No lock needed.
        public static (bool valid, int spawnGeoDist, int agentHp, int agentDef, int bossHp) ReadConfig()
        {
            if (_view == null)
                return (false, 0, 0, 0, 0);
            var valid = _view.ReadInt32(_cfgBase + (long)CFG_VALID * sizeof(int)) == MAGIC;
            if (!valid)
                return (false, 0, 0, 0, 0);
            var spawn = _view.ReadInt32(_cfgBase + (long)CFG_SPAWN_GEO * sizeof(int));
            var hp = _view.ReadInt32(_cfgBase + (long)CFG_AGENT_HP * sizeof(int));
            var def = _view.ReadInt32(_cfgBase + (long)CFG_AGENT_DEF * sizeof(int));
            var bossHp = _view.ReadInt32(_cfgBase + (long)CFG_BOSS_HP * sizeof(int));
            return (true, spawn, hp, def, bossHp);
        }

        // ASYNC handshake (SIM_ASYNC). The C-shim bumps act_seq[slot] when it posts a fresh
        // action; the free-run world reads it to detect a new policy action this tick. Volatile
        // read: the C-shim wrote it from another process via the shared MAP_SHARED page.
        public static int ReadActSeq(int slot) =>
            _view.ReadInt32(_actSeqBase + (long)slot * sizeof(int));

        // The world publishes obs_seq[slot] = the action seq it just consumed, AFTER it has
        // written the resulting obs/reward/done -- so a C-shim that sees obs_seq[slot] >= S is
        // guaranteed the obs for action S is already in shm (write obs/reward FIRST, then seq).
        public static void WriteObsSeq(int slot, int seq) =>
            _view.Write(_obsSeqBase + (long)slot * sizeof(int), seq);

        // ASYNC double-buffer publish (SIM_ASYNC). Replaces the torn-prone single-buffer write
        // under free-run: the world writes the WHOLE transition (obs + reward + done) into the
        // NON-current half of this slot's double buffer, full-fences, then flips obs_idx[slot]
        // to point at the half it just filled. The c_step reads obs_idx[slot] (acquire) and
        // copies that half -- always a complete, consistent frame with NO retry and NO block,
        // because the world is never writing the half the c_step is reading (it writes the other
        // half). obs_seq[slot] still carries the published action-seq for the pipeline collect
        // ordering; it is bumped here too, AFTER the index flip, so a c_shim that waits on
        // obs_seq then reads obs_idx sees the matching buffer. cur is this slot's last-published
        // index (0/1); returns the new current index so the caller can track it.
        public static int PublishAsyncTransition(int slot, int cur, float[] obs, float reward, bool done, int seq)
        {
            var next = 1 - cur;
            var obsOff = _dblObsBase + ((long)next * _n + slot) * OBS_LEN * sizeof(float);
            _view.WriteArray(obsOff, obs, 0, OBS_LEN);
            _view.Write(_dblRewBase + ((long)next * _n + slot) * sizeof(float), reward);
            _view.Write(_dblDoneBase + ((long)next * _n + slot) * sizeof(float), done ? 1.0f : 0.0f);
            // full fence: every byte of the inactive half is visible BEFORE the index flip, so a
            // c_shim that reads the flipped index (acquire) sees a complete frame.
            Thread.MemoryBarrier();
            _view.Write(_obsIdxBase + (long)slot * sizeof(int), next);
            // bump obs_seq AFTER the flip so the pipeline-collect wait (obs_seq >= S) only releases
            // once the matching buffer is published.
            Thread.MemoryBarrier();
            _view.Write(_obsSeqBase + (long)slot * sizeof(int), seq);
            return next;
        }
    }
}
