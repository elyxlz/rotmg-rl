"""The env Config struct is the single source of truth; every binding + ini must stay in lockstep.

Plain-text parse (no compilation), so it runs in the cheap test tier. This is the guard that would
have caught the stale-_C bug: the 4.0 binding/ini still listed the OLD Config fields (spell_arc_deg,
snake_hp, boss_speed, ...) and were missing the faithful ones, so build.sh failed and every retrain
silently ran the previous _C. A field added to or removed from dungeon.h's Config now fails here
until both 4.0 surfaces (and, as a bonus, the 3.0 parity binding) are updated to match.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DUNGEON_H = REPO / "src" / "rotmg_rl" / "csim" / "dungeon.h"
BINDING_4 = REPO / "puffer4" / "binding.c"
INI_4 = REPO / "puffer4" / "dungeon.ini"
BINDING_3 = REPO / "src" / "rotmg_rl" / "csim" / "binding.c"

# kwargs a binding reads that are NOT Config fields (env wiring, not stored on cfg).
NON_CONFIG_KWARGS = {"seed"}


def _config_fields() -> list[str]:
    """Field names declared in the `typedef struct { ... } Config;` block of dungeon.h."""
    text = DUNGEON_H.read_text()
    end = re.search(r"\}\s*Config\s*;", text)
    assert end is not None, "no `} Config;` found in dungeon.h"
    opens = [m.end() for m in re.finditer(r"typedef\s+struct\s*\{", text) if m.end() <= end.start()]
    assert opens, "no `typedef struct {` opening the Config block"
    body = text[opens[-1] : end.start()]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)  # drop block comments
    body = re.sub(r"//[^\n]*", "", body)  # drop line comments
    fields: list[str] = []
    for stmt in body.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        head, _, rest = stmt.partition(" ")
        if head not in ("int", "float") or not rest:
            continue
        for name in rest.split(","):
            name = name.strip()
            if name:
                fields.append(name)
    return fields


def _kwargs_read(path: Path, accessor: str) -> set[str]:
    """All `<accessor>(kwargs, "X")` field names a binding reads, minus the non-Config wiring keys."""
    names = set(re.findall(accessor + r'\(\s*kwargs\s*,\s*"(\w+)"\s*\)', path.read_text()))
    return names - NON_CONFIG_KWARGS


def _ini_env_keys() -> set[str]:
    """Keys defined in the [env] section of dungeon.ini."""
    keys: set[str] = set()
    in_env = False
    for line in INI_4.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_env = stripped == "[env]"
            continue
        if in_env and "=" in stripped and not stripped.startswith(("#", ";")):
            keys.add(stripped.split("=", 1)[0].strip())
    return keys


def test_config_fields_parse_is_sane():
    fields = _config_fields()
    # no duplicates, and a couple of faithful anchors that must exist (regression guards).
    assert len(fields) == len(set(fields)), "duplicate field parsed from Config"
    for anchor in ("player_defense", "boss_defense", "spell_num", "blade_cd", "rew_approach"):
        assert anchor in fields, f"expected faithful field {anchor!r} missing from Config parse"
    # the stale pre-faithful fields must be gone (this exact drift caused the bug).
    for stale in ("spell_arc_deg", "snake_hp", "boss_speed", "snake_cooldown", "snake_bullet_dmg"):
        assert stale not in fields, f"stale field {stale!r} unexpectedly present in Config"


def test_puffer4_binding_reads_exactly_the_config_fields():
    config = set(_config_fields())
    read = _kwargs_read(BINDING_4, "dict_get")
    assert config - read == set(), f"puffer4/binding.c my_init MISSING Config fields: {sorted(config - read)}"
    assert read - config == set(), f"puffer4/binding.c my_init reads STALE non-Config fields: {sorted(read - config)}"


def test_puffer4_dungeon_ini_defines_exactly_the_config_fields():
    config = set(_config_fields())
    keys = _ini_env_keys()
    assert config - keys == set(), f"puffer4/dungeon.ini [env] MISSING Config fields: {sorted(config - keys)}"
    assert keys - config == set(), f"puffer4/dungeon.ini [env] has STALE non-Config keys: {sorted(keys - config)}"


def test_csim_3_0_binding_stays_in_lockstep():
    """Bonus: the 3.0 parity-harness binding must cover the same fields, so all bindings agree."""
    config = set(_config_fields())
    read = _kwargs_read(BINDING_3, "unpack")
    assert config - read == set(), f"csim/binding.c my_init MISSING Config fields: {sorted(config - read)}"
    assert read - config == set(), f"csim/binding.c my_init reads STALE non-Config fields: {sorted(read - config)}"
