"""Parity + gradient check: native CUDA dungeon encoder (puffer4/dungeon_encoder.cu, compiled into
.pufferlib4) vs the torch DungeonEncoder (pufferlib.models). Writes identical weights+input, runs the
standalone harness (puffer4/test_encoder.cu), and compares the encoder output + weight grads.

    .venv4/bin/python scripts/check_encoder_parity.py        # forward + backward parity

Run from repo root on the box (needs .venv4 with torch + pufferlib, nvcc, the built clone src/).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
CLONE = REPO / ".pufferlib4"
CUDA = os.environ.get("CUDA_HOME", "/usr/local/cuda-12.4")

B, H = 4, 128
GRID, NUM_CH, NUM_SCALARS = 31, 7, 6
OBS = NUM_CH * GRID * GRID + NUM_SCALARS


def build_harness(tmp: pathlib.Path) -> pathlib.Path:
    exe = tmp / "test_encoder"
    cmd = [f"{CUDA}/bin/nvcc", "-DPRECISION_FLOAT", "-O2", "-arch=sm_86", f"-I{CLONE}/src",
           str(REPO / "puffer4" / "test_encoder.cu"), "-lcudart", "-lcublas", "-o", str(exe)]
    subprocess.run(cmd, check=True, cwd=str(CLONE))
    return exe


def flat_weights(enc) -> np.ndarray:
    # reg_params order in dungeon_encoder.cu: c1w,c1b,c2w,c2b,gfw,gfb,sfw,sfb,fw,fb
    parts = [
        enc.cnn[0].weight.reshape(32, -1), enc.cnn[0].bias,        # conv1 (32,7,3,3)->(32,63)
        enc.cnn[2].weight.reshape(32, -1), enc.cnn[2].bias,        # conv2 (32,32,3,3)->(32,288)
        enc.grid_fc[0].weight, enc.grid_fc[0].bias,                # (256,30752)
        enc.scalar_fc[0].weight, enc.scalar_fc[0].bias,            # (64,6)
        enc.fuse[0].weight, enc.fuse[0].bias,                      # (H,320)
    ]
    return np.concatenate([p.detach().float().cpu().numpy().ravel() for p in parts])


def flat_grads(enc) -> np.ndarray:
    parts = [
        enc.cnn[0].weight.grad.reshape(32, -1), enc.cnn[0].bias.grad,
        enc.cnn[2].weight.grad.reshape(32, -1), enc.cnn[2].bias.grad,
        enc.grid_fc[0].weight.grad, enc.grid_fc[0].bias.grad,
        enc.scalar_fc[0].weight.grad, enc.scalar_fc[0].bias.grad,
        enc.fuse[0].weight.grad, enc.fuse[0].bias.grad,
    ]
    return np.concatenate([p.detach().float().cpu().numpy().ravel() for p in parts])


def main() -> None:
    sys.path.insert(0, str(CLONE))
    import pufferlib.models as models

    torch.manual_seed(0)
    enc = models.DungeonEncoder(OBS, H).cuda().float()
    obs = (torch.rand(B, OBS, device="cuda") * 2 - 1).float()  # in [-1,1] like the env
    obs.requires_grad_(False)

    out = enc(obs)                       # [B, H]
    dout = torch.randn(B, H, device="cuda").float()
    out.backward(dout)

    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        exe = build_harness(tmp)
        (tmp / "w.bin").write_bytes(flat_weights(enc).astype(np.float32).tobytes())
        (tmp / "in.bin").write_bytes(obs.detach().cpu().numpy().astype(np.float32).tobytes())
        (tmp / "dout.bin").write_bytes(dout.cpu().numpy().astype(np.float32).tobytes())
        subprocess.run([str(exe), str(B), str(H), str(tmp / "w.bin"), str(tmp / "in.bin"),
                        str(tmp / "out.bin"), str(tmp / "dout.bin"), str(tmp / "grads.bin")], check=True)

        nat_out = np.frombuffer((tmp / "out.bin").read_bytes(), dtype=np.float32).reshape(B, H)
        nat_grad = np.frombuffer((tmp / "grads.bin").read_bytes(), dtype=np.float32)

    ref_out = out.detach().cpu().numpy()
    ref_grad = flat_grads(enc)

    fwd_err = np.abs(nat_out - ref_out).max()
    fwd_rel = fwd_err / (np.abs(ref_out).max() + 1e-9)
    print(f"FORWARD  max_abs_err={fwd_err:.3e}  rel={fwd_rel:.3e}  (torch range [{ref_out.min():.3f},{ref_out.max():.3f}])")
    grad_err = np.abs(nat_grad - ref_grad).max()
    grad_rel = grad_err / (np.abs(ref_grad).max() + 1e-9)
    print(f"BACKWARD max_abs_err={grad_err:.3e}  rel={grad_rel:.3e}  (torch grad range [{ref_grad.min():.3f},{ref_grad.max():.3f}])")
    # per-layer grad breakdown to localize a backward bug
    sizes = [32 * 63, 32, 32 * 288, 32, 256 * 30752, 256, 64 * 6, 64, H * 320, H]
    names = ["c1w", "c1b", "c2w", "c2b", "gfw", "gfb", "sfw", "sfb", "fw", "fb"]
    off = 0
    for nm, sz in zip(names, sizes):
        a, b = nat_grad[off:off + sz], ref_grad[off:off + sz]
        e = np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
        print(f"   {nm:4s} rel_err={e:.3e}")
        off += sz

    ok = fwd_rel < 2e-3 and grad_rel < 5e-3
    print("PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
