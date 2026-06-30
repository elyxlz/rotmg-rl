using System;
using System.Collections.Generic;
using WorldServer.core.terrain;

namespace WorldServer.core.worlds
{
    // THROWAWAY geodesic distance field (sim-mode only). BFS from the boss tile over
    // the REAL Snake Pit walkable grid (the live Wmap), so the navigate-in reward
    // can be shaped as the per-tick REDUCTION in geodesic-distance-to-boss -- the
    // signal the C-sim baked as MAP_GEODESIC. Without it the agent has no gradient to
    // cross the maze from the entrance to the boss room.
    //
    // Walkability mirrors SimObsBuilder.TileWalkable / World.IsPassable EXACTLY (tile
    // NoWalk OR a FullOccupy/EnemyOccupySquare object blocks), so the field is over
    // the same cells the agent can actually stand on. 4-connected BFS in TILE units;
    // unreachable tiles stay int.MaxValue. One field per boss tile, recomputed only
    // when the boss tile changes (the Snake Queen is near-stationary, so it is built
    // once per episode and cached).
    internal sealed class SimGeodesic
    {
        private int _w;
        private int _h;
        private int[] _dist;       // [h*w] geodesic tile-distance to the boss tile; int.MaxValue == unreachable
        private int _bossTileX = -1;
        private int _bossTileY = -1;
        private float _maxReachable = 1f; // largest finite distance, for normalization

        public bool HasField => _dist != null;
        public int BossTileX => _bossTileX;
        public int BossTileY => _bossTileY;
        public float MaxReachable => _maxReachable;

        // Rebuild the field if the boss tile moved (or no field yet). Cheap no-op when
        // the boss has not moved a whole tile since the last build.
        public void EnsureField(World world, float bossX, float bossY)
        {
            var bx = (int)bossX;
            var by = (int)bossY;
            if (_dist != null && bx == _bossTileX && by == _bossTileY)
                return;
            Build(world, bx, by);
        }

        public void Reset()
        {
            _dist = null;
            _bossTileX = -1;
            _bossTileY = -1;
            _maxReachable = 1f;
        }

        private bool TileWalkable(Wmap map, GameServer gs, int x, int y)
        {
            if (x < 0 || y < 0 || x >= _w || y >= _h)
                return false;
            var tile = map[x, y];
            if (tile == null)
                return false;
            var tileDesc = gs.Resources.GameData.Tiles[tile.TileId];
            if (tileDesc.NoWalk)
                return false;
            if (tile.ObjType != 0 && tile.ObjDesc != null)
                if (tile.ObjDesc.FullOccupy || tile.ObjDesc.EnemyOccupySquare)
                    return false;
            return true;
        }

        private void Build(World world, int bx, int by)
        {
            var map = world.Map;
            var gs = world.GameServer;
            _w = map.Width;
            _h = map.Height;
            _bossTileX = bx;
            _bossTileY = by;
            _dist = new int[_h * _w];
            for (var i = 0; i < _dist.Length; i++)
                _dist[i] = int.MaxValue;

            // The boss tile itself may be non-walkable (the boss can stand on an
            // occupied square); seed BFS from the nearest walkable tile in a small
            // neighbourhood so the field still anchors on the boss room.
            var seedX = bx;
            var seedY = by;
            if (!TileWalkable(map, gs, bx, by))
            {
                var found = false;
                for (var r = 1; r <= 4 && !found; r++)
                    for (var dy = -r; dy <= r && !found; dy++)
                        for (var dx = -r; dx <= r && !found; dx++)
                            if (TileWalkable(map, gs, bx + dx, by + dy))
                            {
                                seedX = bx + dx;
                                seedY = by + dy;
                                found = true;
                            }
                if (!found)
                    return; // boss walled off (should not happen); leave field all-unreachable
            }

            var q = new Queue<int>();
            var seedIdx = seedY * _w + seedX;
            _dist[seedIdx] = 0;
            q.Enqueue(seedIdx);
            var maxReach = 0;
            while (q.Count > 0)
            {
                var idx = q.Dequeue();
                var cx = idx % _w;
                var cy = idx / _w;
                var d = _dist[idx];
                if (d > maxReach)
                    maxReach = d;
                TryStep(map, gs, q, cx + 1, cy, d + 1);
                TryStep(map, gs, q, cx - 1, cy, d + 1);
                TryStep(map, gs, q, cx, cy + 1, d + 1);
                TryStep(map, gs, q, cx, cy - 1, d + 1);
            }
            _maxReachable = Math.Max(1, maxReach);
        }

        private void TryStep(Wmap map, GameServer gs, Queue<int> q, int x, int y, int nd)
        {
            if (x < 0 || y < 0 || x >= _w || y >= _h)
                return;
            var idx = y * _w + x;
            if (_dist[idx] != int.MaxValue)
                return;
            if (!TileWalkable(map, gs, x, y))
                return;
            _dist[idx] = nd;
            q.Enqueue(idx);
        }

        // Geodesic tile-distance at a world position. Unreachable / off-map tiles
        // fall back to the max reachable distance so the shaping is still finite (a
        // far-from-boss penalty rather than an infinity that poisons the delta).
        public float DistanceAt(float worldX, float worldY)
        {
            if (_dist == null)
                return _maxReachable;
            var x = (int)worldX;
            var y = (int)worldY;
            if (x < 0 || y < 0 || x >= _w || y >= _h)
                return _maxReachable;
            var d = _dist[y * _w + x];
            return d == int.MaxValue ? _maxReachable : d;
        }

        // A walkable tile whose geodesic distance to the boss is closest to
        // targetDist (used to spawn the agent a controllable distance from the boss).
        // Returns world-centre coords, or the boss seed if no field. Deterministic
        // (first tile in row-major order at the best |dist-target|).
        public (float, float) TileAtDistance(float targetDist)
        {
            if (_dist == null)
                return (_bossTileX + 0.5f, _bossTileY + 0.5f);
            var bestIdx = -1;
            var bestErr = float.MaxValue;
            for (var i = 0; i < _dist.Length; i++)
            {
                if (_dist[i] == int.MaxValue)
                    continue;
                var err = Math.Abs(_dist[i] - targetDist);
                if (err < bestErr)
                {
                    bestErr = err;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0)
                return (_bossTileX + 0.5f, _bossTileY + 0.5f);
            return (bestIdx % _w + 0.5f, bestIdx / _w + 0.5f);
        }
    }
}
