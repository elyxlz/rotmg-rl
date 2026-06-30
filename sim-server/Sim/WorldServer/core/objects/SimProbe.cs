namespace WorldServer.core.objects
{
    // THROWAWAY measurement probe (sim-mode only). A stationary IPlayer that
    // lives in PlayersCollision so the boss's chunk stays active (enemies only
    // tick in chunks near a PlayersCollision occupant) and so the boss's
    // GetNearestEntity()-based targeting / PlayerWithinTransition see a target.
    //
    // life:null  -> StaticObject.Vulnerable == false -> Health never decays.
    // dying:false -> never expires. hittestable:false -> no collision blocking.
    // It is NOT a real Player: no Client, no Redis account, no packet IO.
    internal sealed class SimProbe : StaticObject, IPlayer
    {
        // Decoy texture type (0x0715) exists in ObjectDescs, so the base Entity
        // ctor resolves a valid ObjectDesc.
        private const ushort PROBE_TYPE = 0x0715;

        public SimProbe(GameServer gameServer)
            : base(gameServer, PROBE_TYPE, null, true, false, false)
        {
        }

        public bool IsVisibleToEnemy() => true;
        public void Damage(int dmg, Entity src) { }
    }
}
