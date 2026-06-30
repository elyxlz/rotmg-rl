"""Single source of truth: every path, coordinate, port, timing constant, and checkpoint
setting for the Snake Pit harness lives here. Nothing else hard-codes a path or a magic number.

The pipeline is no-root: a real Flash client runs under Xvfb :99, its outbound :2050 socket is
rerouted by LD_PRELOAD=redirect.so to the local tee proxy on :2052, the proxy relays bytes to the
real server on :2050 and tees the server->client direction to the obs consumer, which frames the
@realmlib/net packets and writes the 9807-float observation to OBS_PATH. The policy reads OBS_PATH
and drives WASD/mouse/space via XTest into :99.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.path.expanduser("~"))

# ---- display / X ---------------------------------------------------------------------------
DISPLAY = ":99"
SCREEN = "1024x768x24"
SCREEN_SIZE = "1024x768"  # ffmpeg -video_size

# ---- ports ---------------------------------------------------------------------------------
GAME_PORT = 2050   # real betterSkillys server (loopback)
PROXY_PORT = 2052  # tee proxy the redirected client connects to; redirect.so maps 2050 -> 2052

# ---- repo / venv -----------------------------------------------------------------------------
RL_DIR = HOME / "rotmg-rl"
VENV_PY = RL_DIR / ".venv" / "bin" / "python"
OBS_READER_DIR = RL_DIR              # obs_reader.py lives at the repo root
POLICY_SRC = RL_DIR / "src"         # rotmg_rl package

# ---- checkpoint / policy ---------------------------------------------------------------------
CKPT = RL_DIR / "checkpoints" / "longrun" / "final.pt"
HIDDEN = 2048
NUM_LAYERS = 2
POLICY_GPU = "1"     # CUDA_VISIBLE_DEVICES; the trainer owns GPU 0
OBS_FLOATS = 9807    # expected obs vector length

# ---- flash client ----------------------------------------------------------------------------
FLASH_BIN = HOME / "flash_client_build" / "proj" / "flashplayer"
CLIENT_SWF = HOME / "flash_client_build" / "client.swf"
REDIRECT_SO = HOME / "snakepit-harness" / "redirect.so"
MINIWM_BIN = HOME / "snakepit-harness" / "miniwm"

# ---- resource tables (static, read-only) -----------------------------------------------------
GROUND_TYPES = HOME / "rotmg-realgame" / "nrelay" / "resources" / "GroundTypes.json"
OBJECTS_JSON = HOME / "rotmg-realgame" / "nrelay" / "resources" / "Objects.json"

# ---- obs sink --------------------------------------------------------------------------------
OBS_PATH = Path("/dev/shm/rotmg_obs.f32")

# ---- log files -------------------------------------------------------------------------------
LOG_DIR = Path("/tmp/snakepit")
PROXY_LOG = LOG_DIR / "proxy.log"
OBS_LOG = LOG_DIR / "obs.log"
XVFB_LOG = LOG_DIR / "xvfb.log"
MINIWM_LOG = LOG_DIR / "miniwm.log"
FLASH_LOG = LOG_DIR / "flash.log"
POLICY_LOG = LOG_DIR / "policy.log"
RECORD_LOG = LOG_DIR / "record.log"

# ---- recording -------------------------------------------------------------------------------
VIDEO_DIR = RL_DIR / "videos"
VIDEO_NAME = "snakepit_run.mp4"
RECORD_FPS = 30
RECORD_MAX_SECS = 600  # ffmpeg -t hard cap so a forgotten recorder always self-terminates

# ---- coordinates on the :99 stage (1024x768) -------------------------------------------------
# All verified against the running client layout. The stage is camera-centered on the player.
PLAY_BUTTON = (592, 718)       # "Play" on the account screen
CHAR_SELECT = (450, 314)       # first character slot -> enters Nexus as "Wizardbot"
STAGE_FOCUS = (490, 400)       # a click here grabs Flash keyboard focus (needs miniwm)
PORTAL_KEY_SLOT = (826, 575)   # first inventory slot: where the pre-provisioned Snake Pit Key sits (shift+click = use)
PORTAL_ENTER = (890, 748)      # the opened Snake Pit portal's enter hotspot

# ---- action -> input mapping -----------------------------------------------------------------
AIM_CENTER = (490, 400)  # player is camera-centered; aim is a mouse offset from here
MOVE_FRAME_ROT_DEG = 45.0  # real client WASD frame is rotated 45 deg vs the sim world frame the policy trained in
AIM_TILE = 36
AIM_RADIUS_TILES = 4     # cast/aim lands ~4 tiles out along the aim angle

# ---- timing (tunable) ------------------------------------------------------------------------
FLASH_BOOT_SECS = 22        # client launch -> account screen ready
LOGIN_RETRY_MAX = 6         # play->char attempts before giving up
LOGIN_STEP_PLAY_SECS = 5    # wait after clicking Play
LOGIN_STEP_CHAR_SECS = 8    # wait after clicking the character slot
GIVE_RETRY_MAX = 6          # /give + use-key attempts before the portal is open
GIVE_SETTLE_SECS = 3        # wait after sending the /give chat line
KEY_USE_SETTLE_SECS = 4     # wait after shift+clicking the key (portal spawns)
ENTER_SETTLE_SECS = 6       # wait after clicking the portal before checking the dungeon gate

# ---- verification gates ----------------------------------------------------------------------
OBS_FRESH_SECS = 3.0        # obs mtime must be younger than this to count as "live"
GATE_POLL_SECS = 0.5        # poll interval while waiting on a gate
INGAME_GATE_TIMEOUT = 30.0  # max wait for the in-game gate per login attempt
DUNGEON_OBJ_JUMP = 1.5      # dungeon object-count must be >= this * nexus baseline to count as "entered"
DUNGEON_GATE_TIMEOUT = 25.0 # max wait for the dungeon gate after Enter

# ---- monitor ---------------------------------------------------------------------------------
MONITOR_POLL_SECS = 5.0     # how often to sample obs while the policy plays
DEATH_STALE_SECS = 4.0      # obs older than this == death / clear / world-transition -> run over


def env_for_subprocess() -> dict:
    """Base environment for the X-driving / pipeline subprocesses: DISPLAY set, import paths wired."""
    env = dict(os.environ)
    env["DISPLAY"] = DISPLAY
    pythonpath = [str(OBS_READER_DIR), str(POLICY_SRC), str(HOME / "snakepit-harness")]
    existing = env["PYTHONPATH"] if "PYTHONPATH" in env else ""
    if existing:
        pythonpath.append(existing)
    env["PYTHONPATH"] = ":".join(pythonpath)
    return env
