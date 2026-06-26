# --- rotmg-rl DungeonEncoder (appended to pufferlib/models.py by scripts/setup_box_puffer4.sh) ---
# CNN over the 7x31x31 grid + MLP over the 6 scalars, fused to hidden_size. Mirrors the 3.0
# rotmg_rl.csim.policy.CDungeonPolicy encoder so the spatial grid isn't flattened away (the stock
# DefaultEncoder is a single Linear over the flat obs). Used via config: [torch] encoder = DungeonEncoder.
# The flat float obs is laid out [grid (NUM_CH*GRID*GRID), scalars (NUM_SCALARS)], matching the C env.
class DungeonEncoder(nn.Module):
    GRID = 31
    NUM_CH = 7
    NUM_SCALARS = 6

    def __init__(self, obs_size, hidden_size=256):
        super().__init__()
        grid_flat = self.NUM_CH * self.GRID * self.GRID
        assert obs_size == grid_flat + self.NUM_SCALARS, (obs_size, grid_flat + self.NUM_SCALARS)
        self._grid_flat = grid_flat
        self.cnn = nn.Sequential(
            nn.Conv2d(self.NUM_CH, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Flatten(),
        )
        self.grid_fc = nn.Sequential(nn.Linear(32 * self.GRID * self.GRID, 256), nn.GELU())
        self.scalar_fc = nn.Sequential(nn.Linear(self.NUM_SCALARS, 64), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(256 + 64, hidden_size), nn.GELU())

    def forward(self, observations):
        x = observations.view(observations.shape[0], -1).float()
        b = x.shape[0]
        grid = x[:, :self._grid_flat].view(b, self.NUM_CH, self.GRID, self.GRID)
        scalars = x[:, self._grid_flat:]
        g = self.grid_fc(self.cnn(grid))
        s = self.scalar_fc(scalars)
        return self.fuse(torch.cat([g, s], dim=1))
