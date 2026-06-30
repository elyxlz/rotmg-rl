using System;
using System.Collections.Generic;
using WorldServer.core.objects;
using WorldServer.core.terrain;

namespace WorldServer.core.worlds
{
    // THROWAWAY in-process observation builder (sim-mode only). Ports
    // rotmg_rl/deploy/obs.py:RealObsBuilder BIT-FOR-BIT, but sources every value
    // from the live C# game objects (the agent entity, the world's enemies, the
    // boss, the captured enemy-shoot bursts, the player's live projectiles, the
    // Wmap tiles) instead of from decoded nrelay packets. No nrelay, no TCP, no
    // double serialization.
    //
    // Layout (== RealObsBuilder): a flat float[9807] =
    //   grid    7 x 31 x 31   egocentric (CH_WALL/ENEMY/EBULLET/EBVX/EBVY/PBULLET/GRENADE)
    //   minimap 3 x 32 x 32   fog-of-war (TERRAIN/PLAYER/BOSS)
    //   scalars 8             hp,mp,spell_ready,boss_visible,confused,petrified,boss_hp_frac,boss_invuln
    //
    // Stateful (like RealObsBuilder): the explored walkable map, the fog-of-war
    // discovered mask, the boss-seen latch, the fight-active latch. One instance
    // per agent. Constants mirror obs.py exactly (note obs.py uses SPELL_COST=100
    // and ACTIVATION_RANGE=20, NOT the DungeonConfig values).
    internal sealed class SimObsBuilder
    {
        private const int VIS_RADIUS = 15;
        private const int GRID = 2 * VIS_RADIUS + 1; // 31
        private const int HALF = GRID / 2;            // 15
        private const int MM = 32;
        private const int NUM_CH = 7;
        private const int NUM_MM_CH = 3;
        private const int NUM_SCALARS = 8;
        public const int OBS_LEN = NUM_CH * GRID * GRID + NUM_MM_CH * MM * MM + NUM_SCALARS; // 9807

        private const int CH_WALL = 0;
        private const int CH_ENEMY = 1;
        private const int CH_EBULLET = 2;
        private const int CH_EBVX = 3;
        private const int CH_EBVY = 4;
        private const int CH_PBULLET = 5;
        // CH_GRENADE = 6 left zero (boss grenade telegraphs not decoded), == RealObsBuilder.

        private const int MM_CH_TERRAIN = 0;
        private const int MM_CH_PLAYER = 1;
        private const int MM_CH_BOSS = 2;

        private const float ACTIVATION_RANGE = 20.0f; // obs.py ACTIVATION_RANGE
        private const float SPELL_COST = 100.0f;      // obs.py SPELL_COST (NOT config.spell_cost)

        private const string BOSS_ID = "Stheno the Snake Queen";

        private int _w;
        private int _h;
        private bool[] _walkable;    // [h*w] True == walkable; unmapped defaults walkable
        private bool[] _discovered;  // [h*w] fog-of-war
        private bool _bossSeen;
        private bool _fightActive;

        // The walls are STATIC (tiles + blocking objects never change in the pit), so
        // the walkable grid is synced from the Wmap exactly ONCE per episode instead of
        // every tick. Re-reading ~22k tiles + tile descs each tick was the dominant obs
        // cost; the grid is identical every tick, so this is bit-for-bit equivalent.
        private bool _walkableSynced;
        // Persisted minimap terrain layer. It only ever GROWS as the discovered mask
        // grows (fog reveals more tiles), so it is maintained INCREMENTALLY over the
        // tiles newly discovered this tick instead of re-scanning the whole map twice
        // every build. Precedence (floor +1 wins over wall -1 within a cell, and +1 is
        // permanent) is preserved, so the produced minimap is identical to the old
        // two-full-pass version. _terrDirty rebuilds it once after a Reset.
        private float[] _terrCell;   // [MM*MM] persisted minimap terrain (-1/0/+1)
        private int[] _newTilesX;    // scratch: tiles discovered this tick
        private int[] _newTilesY;
        private int _newTileCount;

        public void Reset()
        {
            _w = 0;
            _h = 0;
            _walkable = null;
            _discovered = null;
            _bossSeen = false;
            _fightActive = false;
            _walkableSynced = false;
            _terrCell = null;
        }

        // Mirror RealObsBuilder.set_map: idempotent, defaults all-walkable.
        public void SetMap(int w, int h)
        {
            if (_w == w && _h == h && _walkable != null)
                return;
            _w = w;
            _h = h;
            _walkable = new bool[h * w];
            _discovered = new bool[h * w];
            for (var i = 0; i < _walkable.Length; i++)
                _walkable[i] = true;
            _walkableSynced = false;
            _terrCell = new float[MM * MM];
            _newTilesX = new int[h * w];
            _newTilesY = new int[h * w];
        }

        // Pull the static walkability for the egocentric window from the live Wmap.
        // RealObsBuilder learns walkability incrementally from Update/GroundTile
        // packets; in-process we read the authoritative Wmap directly, which is the
        // SAME ground truth those packets carry (tile NoWalk OR a blocking object).
        private bool TileWalkable(Wmap map, GameServer gs, int x, int y)
        {
            if (x < 0 || y < 0 || x >= _w || y >= _h)
                return true; // out-of-bounds defaults walkable (== unmapped default)
            var tile = map[x, y];
            if (tile == null)
                return true;
            var tileDesc = gs.Resources.GameData.Tiles[tile.TileId];
            if (tileDesc.NoWalk)
                return false;
            if (tile.ObjType != 0 && tile.ObjDesc != null)
                if (tile.ObjDesc.FullOccupy || tile.ObjDesc.EnemyOccupySquare)
                    return false;
            return true;
        }

        private void SyncWalkable(World world)
        {
            // Sync the walkable grid from the Wmap ONCE (the walls are static). The
            // discovered mask still gates what feeds the minimap, so reading the whole
            // map here does not leak un-explored terrain into the obs (the CH_WALL grid
            // is egocentric within VIS, always within the explored disk anyway).
            if (_walkableSynced)
                return;
            var map = world.Map;
            var gs = world.GameServer;
            for (var y = 0; y < _h; y++)
                for (var x = 0; x < _w; x++)
                    _walkable[y * _w + x] = TileWalkable(map, gs, x, y);
            _walkableSynced = true;
        }

        private void UpdateVisibility(int ipx, int ipy, bool hasBoss, float bx, float by, float px, float py)
        {
            var y0 = Math.Max(0, ipy - VIS_RADIUS);
            var y1 = Math.Min(_h, ipy + VIS_RADIUS + 1);
            var x0 = Math.Max(0, ipx - VIS_RADIUS);
            var x1 = Math.Min(_w, ipx + VIS_RADIUS + 1);
            var r2 = VIS_RADIUS * VIS_RADIUS;
            _newTileCount = 0;
            for (var y = y0; y < y1; y++)
            {
                var dy = y - ipy;
                for (var x = x0; x < x1; x++)
                {
                    var dx = x - ipx;
                    if (dy * dy + dx * dx <= r2)
                    {
                        var idx = y * _w + x;
                        if (!_discovered[idx])
                        {
                            _discovered[idx] = true;
                            // record the freshly-discovered tile so the minimap terrain
                            // layer is updated only over the new tiles, not the whole map.
                            _newTilesX[_newTileCount] = x;
                            _newTilesY[_newTileCount] = y;
                            _newTileCount++;
                        }
                    }
                }
            }
            if (hasBoss)
            {
                var d = (float)Math.Sqrt((bx - px) * (bx - px) + (by - py) * (by - py));
                if (d <= VIS_RADIUS)
                    _bossSeen = true;
            }
        }

        // grid[ch, cy, cx] = value at floor(rel)+HALF, in-bounds only. cells[:,0]==x
        // (col), cells[:,1]==y (row), == RealObsBuilder._scatter.
        private static void Scatter(float[] grid, int ch, float relX, float relY, float value)
        {
            var cx = (int)Math.Floor(relX) + HALF;
            var cy = (int)Math.Floor(relY) + HALF;
            if (cx < 0 || cx >= GRID || cy < 0 || cy >= GRID)
                return;
            grid[ch * GRID * GRID + cy * GRID + cx] = value;
        }

        // Build the 9807-float obs into `outBuf`. `agent` is the controllable sim
        // agent; `world` is its Snake Pit. nowTick is the current logical tick (for
        // forward-simulating enemy bullets, == RealObsBuilder now_ms / 100ms).
        public void Build(SimAgent agent, World world, long nowTick, float[] outBuf)
        {
            Array.Clear(outBuf, 0, outBuf.Length);
            SetMap(world.Map.Width, world.Map.Height);
            SyncWalkable(world);

            var px = agent.X;
            var py = agent.Y;
            var ipx = (int)px;
            var ipy = (int)py;

            // boss = the first Stheno enemy; non-boss = every other enemy. No per-tick
            // list: find the boss in one pass, then scatter the rest inline below
            // (skipping that single boss reference) == the old nonBoss split.
            Enemy boss = null;
            foreach (var e in world.Enemies.Values)
                if (e.ObjectDesc != null && e.ObjectDesc.IdName == BOSS_ID)
                {
                    boss = e;
                    break;
                }
            var hasBoss = boss != null;
            var bx = hasBoss ? boss.X : 0f;
            var by = hasBoss ? boss.Y : 0f;

            if (hasBoss)
            {
                var d = (float)Math.Sqrt((bx - px) * (bx - px) + (by - py) * (by - py));
                if (d <= ACTIVATION_RANGE)
                    _fightActive = true;
            }
            UpdateVisibility(ipx, ipy, hasBoss, bx, by, px, py);

            // --- grid ---
            // CH_WALL: egocentric (~walkable). wx=px+col-HALF, wy=py+row-HALF, clamped
            // read but only written in-bounds, == RealObsBuilder.
            var ggrid = NUM_CH * GRID * GRID;
            for (var row = 0; row < GRID; row++)
                for (var col = 0; col < GRID; col++)
                {
                    var wx = ipx + col - HALF; // (px + xs - half).astype(int): int(px) then offset
                    var wy = ipy + row - HALF;
                    if (wx < 0 || wx >= _w || wy < 0 || wy >= _h)
                        continue;
                    if (!_walkable[wy * _w + wx])
                        outBuf[CH_WALL * GRID * GRID + row * GRID + col] = 1.0f;
                }

            // CH_ENEMY: non-boss at 0.6, boss at 1.0 when fight-active and visible.
            // Iterate all enemies, skipping the single boss reference (== the old
            // nonBoss split, no per-tick list allocation).
            foreach (var e in world.Enemies.Values)
                if (!ReferenceEquals(e, boss))
                    Scatter(outBuf, CH_ENEMY, e.X - px, e.Y - py, 0.6f);

            var bossVisible = false;
            if (hasBoss && _fightActive)
            {
                var d = (float)Math.Sqrt((bx - px) * (bx - px) + (by - py) * (by - py));
                bossVisible = d <= VIS_RADIUS;
            }
            if (bossVisible)
                Scatter(outBuf, CH_ENEMY, bx - px, by - py, 1.0f);

            // CH_EBULLET/EBVX/EBVY: forward-simulate the captured bursts to nowTick,
            // == RealObsBuilder._active_bullets. unit velocity = (cos a, sin a). The live
            // burst list is iterated (FORWARD, so last-write-wins on a shared cell is
            // unchanged) + pruned in place (no per-tick Snapshot/Replace copy).
            var bursts = SimEnemyShoots.Live(world.Id);
            for (var bi = 0; bi < bursts.Count; bi++)
            {
                var b = bursts[bi];
                var age = nowTick - b.SpawnTick; // ticks (integer, but RealObsBuilder uses float ms/100)
                if (age < 0 || age > b.Lifetime)
                    continue; // not yet live, or expired (pruned below)
                for (var i = 0; i < b.Count; i++)
                {
                    var a = b.Angle + i * b.AngleInc;
                    var ca = (float)Math.Cos(a);
                    var sa = (float)Math.Sin(a);
                    var x = b.OriginX + ca * b.Speed * age;
                    var y = b.OriginY + sa * b.Speed * age;
                    if ((int)x >= 0 && (int)x < _w && (int)y >= 0 && (int)y < _h)
                    {
                        var relX = x - px;
                        var relY = y - py;
                        Scatter(outBuf, CH_EBULLET, relX, relY, 1.0f);
                        Scatter(outBuf, CH_EBVX, relX, relY, ca);
                        Scatter(outBuf, CH_EBVY, relX, relY, sa);
                    }
                }
            }
            bursts.RemoveAll(b => nowTick - b.SpawnTick > b.Lifetime);

            // CH_PBULLET: the agent's live projectiles.
            foreach (var pb in agent.LiveProjectiles(nowTick))
                Scatter(outBuf, CH_PBULLET, pb.Item1 - px, pb.Item2 - py, 1.0f);

            // --- minimap ---
            BuildMinimap(outBuf, ggrid, px, py, ipx, ipy, hasBoss, bx, by);

            // --- scalars ---
            var so = ggrid + NUM_MM_CH * MM * MM;
            var hp = agent.HP;
            var maxHp = agent.MaxHP;
            var mp = agent.MP;
            var maxMp = agent.MaxMP;
            var bossHpFrac = 0f;
            var bossInvuln = 0f;
            if (_fightActive && hasBoss)
            {
                bossHpFrac = Math.Max(boss.Health, 0f) / Math.Max(boss.MaxHealth, 1f);
                bossInvuln = boss.HasConditionEffect(Shared.resources.ConditionEffectIndex.Invulnerable)
                             || boss.HasConditionEffect(Shared.resources.ConditionEffectIndex.Invincible) ? 1f : 0f;
            }
            outBuf[so + 0] = hp / Math.Max(maxHp, 1f);
            outBuf[so + 1] = mp / Math.Max(maxMp, 1f);
            outBuf[so + 2] = mp >= SPELL_COST ? 1f : 0f;
            outBuf[so + 3] = bossVisible ? 1f : 0f;
            outBuf[so + 4] = agent.HasConditionEffect(Shared.resources.ConditionEffectIndex.Confused) ? 1f : 0f;
            outBuf[so + 5] = agent.HasConditionEffect(Shared.resources.ConditionEffectIndex.Paralyzed) ? 1f : 0f;
            outBuf[so + 6] = bossHpFrac;
            outBuf[so + 7] = bossInvuln;
        }

        private void BuildMinimap(float[] outBuf, int gridOffset, float px, float py, int ipx, int ipy, bool hasBoss, float bx, float by)
        {
            // terrain: discovered & ~walkable -> -1, discovered & walkable -> +1.
            // cell index uses the SAME downsample as RealObsBuilder:
            //   cx = (x*MM)//w, cy = (y*MM)//h, cell = cy*MM + cx.
            // The reference is TWO PASSES (walls then floors): a cell is +1 if ANY
            // discovered FLOOR maps to it, else -1 if any discovered WALL maps to it,
            // else 0; +1 is permanent (floors win and the discovered mask only grows).
            // Equivalently, maintain _terrCell INCREMENTALLY over the tiles discovered
            // THIS tick: a floor always sets its cell to +1, a wall sets -1 only when the
            // cell is not already +1. This is bit-identical to the two full passes but
            // touches only the new tiles, not the whole map every build.
            var terrBase = gridOffset + MM_CH_TERRAIN * MM * MM;
            for (var i = 0; i < _newTileCount; i++)
            {
                var x = _newTilesX[i];
                var y = _newTilesY[i];
                var cell = ((y * MM) / _h) * MM + (x * MM) / _w;
                if (_walkable[y * _w + x])
                    _terrCell[cell] = 1.0f;
                else if (_terrCell[cell] != 1.0f)
                    _terrCell[cell] = -1.0f;
            }
            Array.Copy(_terrCell, 0, outBuf, terrBase, MM * MM);
            // player: int(px),int(py) downsampled.
            var pmx = (ipx * MM) / _w;
            var pmy = (ipy * MM) / _h;
            outBuf[gridOffset + MM_CH_PLAYER * MM * MM + pmy * MM + pmx] = 1.0f;
            // boss: only when boss_seen latched.
            if (_bossSeen && hasBoss)
            {
                var bmx = ((int)bx * MM) / _w;
                var bmy = ((int)by * MM) / _h;
                outBuf[gridOffset + MM_CH_BOSS * MM * MM + bmy * MM + bmx] = 1.0f;
            }
        }
    }
}
