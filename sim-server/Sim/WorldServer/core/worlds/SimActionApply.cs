using System;
using WorldServer.core.objects;

namespace WorldServer.core.worlds
{
    // THROWAWAY in-process action apply (sim-mode only). Decodes the 4-head
    // MultiDiscrete action {move,aim,shoot,cast} into a game intent and applies it
    // DIRECTLY to the SimAgent -- no PlayerShoot/Move/UseItem packets, no nrelay.
    //
    // Mirrors rotmg_rl/deploy/obs.py:action_to_intent + the lockstep bridge's
    // applyIntent: move 0=stand / 1..8 = MOVE_DIRS*player_speed (a velocity vector
    // added to the agent's position), aim = AIM_DIRS unit vector, shoot/cast bools.
    internal static class SimActionApply
    {
        private const int N_AIM = 32;
        // player_speed: the obs/action decode default (action_to_intent default 0.55).
        private const float PLAYER_SPEED = 0.55f;

        // 8 MOVE_DIRS = unit vectors at k*pi/4, k=0..7 (== config.MOVE_DIRS).
        private static readonly (float dx, float dy)[] MoveDirs = BuildMoveDirs();
        // 32 AIM_DIRS = unit vectors at k*2pi/32 (== config.AIM_DIRS).
        private static readonly (float ax, float ay)[] AimDirs = BuildAimDirs();

        private static (float, float)[] BuildMoveDirs()
        {
            var dirs = new (float, float)[8];
            for (var k = 0; k < 8; k++)
            {
                var a = k * Math.PI / 4;
                dirs[k] = ((float)Math.Cos(a), (float)Math.Sin(a));
            }
            return dirs;
        }

        private static (float, float)[] BuildAimDirs()
        {
            var dirs = new (float, float)[N_AIM];
            for (var k = 0; k < N_AIM; k++)
            {
                var a = k * 2 * Math.PI / N_AIM;
                dirs[k] = ((float)Math.Cos(a), (float)Math.Sin(a));
            }
            return dirs;
        }

        // SPELL_CAST_RANGE: the BulletNova lands at this many tiles from the agent along
        // the SHARED staff aim direction (one mouse: staff + spell point the same way at
        // any instant; only the cast DISTANCE differs). Overridable via SIM_SPELL_RANGE.
        private static readonly float SPELL_CAST_RANGE = ReadFloat("SIM_SPELL_RANGE", 4.0f);

        private static float ReadFloat(string name, float fallback)
        {
            var raw = Environment.GetEnvironmentVariable(name);
            if (raw == null || !float.TryParse(raw, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var v))
                return fallback;
            return v;
        }

        public static void Apply(SimAgent agent, int move, int aim, bool shoot, bool cast, long nowTick)
        {
            // move -> velocity vector -> requested absolute target (collision-checked
            // in SimAgent.Tick, == bridge applyIntent passable slide).
            if (move != 0)
            {
                var (dx, dy) = MoveDirs[move - 1];
                agent.RequestMove(agent.X + dx * PLAYER_SPEED, agent.Y + dy * PLAYER_SPEED);
            }

            var (ax, ay) = AimDirs[aim % N_AIM];
            var angle = (float)Math.Atan2(ay, ax);
            // shoot -> the STAFF (real 2-shot pattern, fractional attack-freq cooldown).
            if (shoot)
                agent.Shoot(angle, nowTick);

            // cast -> the SPELL (BulletNova) along the SHARED staff aim: the nova lands
            // SPELL_CAST_RANGE tiles from the agent ALONG THE SAME aim direction the staff
            // fires (one mouse -> one direction), then explodes there (AoE centered on the
            // target). The agent owns MP cost + cooldown gating. To drop the spell on a
            // different target the policy must turn the aim between ticks, like a human
            // moving the mouse -- staff and spell can never point different ways the same tick.
            if (cast)
            {
                var targetX = agent.X + ax * SPELL_CAST_RANGE;
                var targetY = agent.Y + ay * SPELL_CAST_RANGE;
                agent.CastSpell(targetX, targetY, nowTick);
            }
        }
    }
}
