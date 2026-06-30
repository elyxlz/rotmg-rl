import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/audiogen/rotmg-rl")
import obs_reader as O
nw=O.load_no_walk(Path("/home/audiogen/rotmg-realgame/nrelay/resources/GroundTypes.json"))
pj=O.load_projectiles(Path("/home/audiogen/rotmg-realgame/nrelay/resources/Objects.json"))
state=O.GameState(nw,pj); ob=O.RealObsBuilder(); asm=O.FrameAssembler()
out=Path("/dev/shm/rotmg_obs.f32"); tmp=out.with_suffix(".f32.tmp")
map_set=False; n=0
print("[obs] consumer up",file=sys.stderr,flush=True)
while True:
    chunk=sys.stdin.buffer.read(65536)
    if not chunk: break
    for ptype,body in asm.feed(chunk):
        r=O.Reader(body); now=time.time()*1000
        if ptype==O.PKT_MAPINFO:
            p=O.parse_map_info(r); state.on_map_info(p); ob.__init__(); ob.set_map(p["width"],p["height"]); map_set=True
            print("[obs] MAPINFO name=%s %dx%d"%(p.get("name"),p["width"],p["height"]),file=sys.stderr,flush=True)
        elif ptype==O.PKT_UPDATE:
            t=state.on_update(O.parse_update(r))
            if map_set: ob.update_tiles(t)
        elif ptype==O.PKT_ENEMYSHOOT:
            state.on_enemy_shoot(O.parse_enemy_shoot(r),now)
        elif ptype==O.PKT_NEWTICK:
            state.on_new_tick(O.parse_new_tick(r))
            if not map_set: continue
            built=state.build_tick(now)
            if built is None: continue
            tick,shots=built
            if shots: ob.add_shots(shots)
            obs=ob.build(tick); tmp.write_bytes(obs.tobytes()); tmp.replace(out); n+=1
            if n%15==0:
                p=tick["player"]
                print("[obs] tick=%d player=(%.1f,%.1f) hp=%d/%d mp=%d enemies=%d obs_nonzero=%d finite=%s"%(
                    n,p["x"],p["y"],p["hp"],p["hp_max"],p["mp"],len(tick["enemies"]),int((obs!=0).sum()),bool(np.isfinite(obs).all())),
                    file=sys.stderr,flush=True)
