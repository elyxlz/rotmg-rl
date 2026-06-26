# --- rotmg-rl DungeonEncoder (appended to pufferlib/models.py by scripts/setup_box_puffer4.sh) ---
# Mirrors rotmg_rl.csim.policy.CDungeonPolicy exactly so the --slowly 4.0 path uses our real
# architecture. Flat float obs layout [grid (7,31,31), minimap (3,32,32), scalars (8)] = 9807.
#   grid CNN:    Conv(7->32,k3) + GELU + MaxPool2 + Conv(32->32,k3) + GELU + MaxPool2 -> Linear(32*7*7=1568 -> 256)
#   minimap CNN: Conv(3->16,k3) + GELU + MaxPool2 + Conv(16->16,k3) + GELU + MaxPool2 -> Linear(16*8*8=1024 -> 128)
#   scalars:     Linear(8 -> 64) + GELU
#   fuse:        Linear(256+128+64=448 -> hidden) + GELU
# The two MaxPool2 cut the grid_fc from 30752->256 (~7.9M params) to 1568->256, keeping spatial
# structure for aim/dodge; the minimap branch gives global fog-of-war navigation context.
class DungeonEncoder(nn.Module):
    GRID = 31
    NUM_CH = 7
    MM = 32
    NUM_MM_CH = 3
    NUM_SCALARS = 8

    def __init__(self, obs_size, hidden_size=256):
        super().__init__()
        grid_flat = self.NUM_CH * self.GRID * self.GRID
        mm_flat = self.NUM_MM_CH * self.MM * self.MM
        assert obs_size == grid_flat + mm_flat + self.NUM_SCALARS, (obs_size, grid_flat + mm_flat + self.NUM_SCALARS)
        self._grid_flat = grid_flat
        self._mm_flat = mm_flat
        gp = (self.GRID // 2) // 2  # 31 -> 15 -> 7
        mp = (self.MM // 2) // 2    # 32 -> 16 -> 8
        self.cnn = nn.Sequential(
            nn.Conv2d(self.NUM_CH, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2), nn.Flatten(),
        )
        self.grid_fc = nn.Sequential(nn.Linear(32 * gp * gp, 256), nn.GELU())
        self.mm_cnn = nn.Sequential(
            nn.Conv2d(self.NUM_MM_CH, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2), nn.Flatten(),
        )
        self.mm_fc = nn.Sequential(nn.Linear(16 * mp * mp, 128), nn.GELU())
        self.scalar_fc = nn.Sequential(nn.Linear(self.NUM_SCALARS, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(256 + 128 + 64, hidden_size), nn.GELU())

    def forward(self, observations):
        x = observations.view(observations.shape[0], -1).float()
        b = x.shape[0]
        grid = x[:, :self._grid_flat].view(b, self.NUM_CH, self.GRID, self.GRID)
        minimap = x[:, self._grid_flat:self._grid_flat + self._mm_flat].view(b, self.NUM_MM_CH, self.MM, self.MM)
        scalars = x[:, self._grid_flat + self._mm_flat:]
        g = self.grid_fc(self.cnn(grid))
        m = self.mm_fc(self.mm_cnn(minimap))
        s = self.scalar_fc(scalars)
        return self.fuse(torch.cat([g, m, s], dim=1))
