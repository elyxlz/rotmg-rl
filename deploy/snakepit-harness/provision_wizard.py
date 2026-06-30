#!/usr/bin/env python3
"""
provision_wizard.py - idempotently (re)provision account 1's first char (char.1.2)
into the deliverable-ready maxed Wizard on this self-hosted betterSkillys server.

WHY THIS EXISTS
---------------
Setting a deliverable char up through the Flash UI is flaky. The char lives entirely
as a redis HASH (key `char.<accountId>.<charId>`) in the game server's redis at
127.0.0.1:6379. This script edits that hash directly + the account's char-select set,
so we never have to touch the Flash UI again. Re-runnable: running it twice is a no-op.

CACHING / SAFETY  (read before running while anyone is logged in)
-----------------------------------------------------------------
The WorldServer keeps a loaded char in memory and writes it back to redis on save.
If the target char is *in-world* (its client past char-select), an edit here will be
clobbered by the next server save. So: only run while the target char is NOT in-world
(client at char-select or logged out). The account-level `lock:<acc>` key only tells
you the account has an active session; what matters is whether *this charId* is the one
the client has loaded. The training run plays char 7 (alive.1 = {7}); char 1 is dormant,
so editing char.1.* is safe even while the training session holds the account lock.
After this runs, the deliverable char appears the next time the client hits char-select.

DATA FORMAT (reverse-engineered from betterSkillys source, see report)
----------------------------------------------------------------------
Hash key: char.1.2  (account 1 "Wizardbot", char slot ... see TARGET_CHAR_KEY)
  stats : int32[8] little-endian  = [MaxHP, MaxMP, ATK, DEF, SPD, DEX, VIT, WIS]
          (RedisObject GetValue<int[]> -> Buffer.BlockCopy of raw LE int32s)
  items : ushort[28] little-endian = [weapon, ability, armor, ring,
          inv0..inv7 (slots 4-11), backpack 12-27]; empty slot = 0xffff.
          (RedisObject GetValue<ushort[]> -> raw LE uint16; Inventory is 28 slots:
           NUM_EQUIPMENT_SLOTS=4 + NUM_INVENTORY_SLOTS=8 + backpack region.)
  dead  : 1 byte, 0x00 = alive, 0x01 = dead  (bool -> single byte)
  hp/mp/level/exp/fame/charType : ASCII decimal strings (UTF-8)
  charType : ushort as string; 782 (0x30e) == Wizard on this server's Players.xml.

The char-select list is driven by the SET `alive.<acc>` whose members are the charId
as a 4-byte LE int32. Death pushes the id onto the LIST `dead.<acc>` and removes it
from `alive.<acc>`. Reviving = add to alive set + remove (all copies) from dead list +
clear the `dead` field. (Database.cs Death/GetAliveCharacters/IsAlive.)

Run:  ssh ripbox; cd ~/rotmg-rl && .venv/bin/python ~/provision_wizard.py
"""

import socket
import struct
import sys

# ---- connection ----
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379

# ---- target char (account 1 "Wizardbot", first char slot) ----
ACCOUNT_ID = 1
CHAR_ID = 2
TARGET_CHAR_KEY = f"char.{ACCOUNT_ID}.{CHAR_ID}"

# ---- class ----
WIZARD_CHAR_TYPE = 782  # 0x30e, "Wizard" in Players.xml on this server

# ---- equipped + inventory item types (from the Object xml the names came from) ----
STAFF_OF_DESTRUCTION = 0xA9E       # 2718  -> WEAPON slot (slot 0)
BURNING_RETRIBUTION_SPELL = 0x2055  # 8277  -> ABILITY slot (slot 1)
SNAKE_PIT_KEY = 0x70B              # 1803  -> first backpack/inventory slot
EMPTY = 0xFFFF                     # empty equip/inventory slot

# ---- inventory geometry (InventoryConstants) ----
NUM_SLOTS = 28
SLOT_WEAPON = 0
SLOT_ABILITY = 1
SLOT_ARMOR = 2
SLOT_RING = 3
FIRST_INVENTORY_SLOT = 4  # NUM_EQUIPMENT_SLOTS

# ---- maxed Wizard stats, AUTHORITATIVE for THIS server ----
# Players.xml: <Object type="0x030e" id="Wizard"> MaxHitPoints max=670, MaxMagicPoints
# max=385, Attack max=75, Defense max=25, Speed max=50, Dexterity max=75,
# HpRegen(VIT) max=40, MpRegen(WIS) max=60. These match the existing alive maxed
# Wizards on this server (char.1.1 / char.2.1) which store exactly these values and
# hp=670. (The prompt's 710/425 are from a different RotMG build; this server caps a
# maxed Wizard at 670/385, so those are the real 8/8 values here.)
STAT_MAX_HP = 670
STAT_MAX_MP = 385
STAT_ATK = 75
STAT_DEF = 25
STAT_SPD = 50
STAT_DEX = 75
STAT_VIT = 40
STAT_WIS = 60
STATS = [STAT_MAX_HP, STAT_MAX_MP, STAT_ATK, STAT_DEF, STAT_SPD, STAT_DEX, STAT_VIT, STAT_WIS]

FULL_HP = STAT_MAX_HP
FULL_MP = STAT_MAX_MP
CHAR_LEVEL = 20

ITEM_NAMES = {
    STAFF_OF_DESTRUCTION: "Staff of Destruction",
    BURNING_RETRIBUTION_SPELL: "Burning Retribution Spell",
    SNAKE_PIT_KEY: "Snake Pit Key",
    0xA97: "Energy Staff",
    0xA2E: "Fire Spray Spell",
    0xA22: "Health Potion",
    EMPTY: "(empty)",
}
STAT_NAMES = ["MaxHP", "MaxMP", "ATK", "DEF", "SPD", "DEX", "VIT", "WIS"]


# ----------------------------------------------------------------------------
# Minimal RESP client (raw socket, redis_char.py style) so this has zero deps.
# ----------------------------------------------------------------------------
def _read_reply(sock):
    buf = b""

    def read_line():
        nonlocal buf
        while b"\r\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("redis closed")
            buf += chunk
        line, buf = buf.split(b"\r\n", 1)
        return line

    def read_n(n):
        nonlocal buf
        while len(buf) < n + 2:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("redis closed")
            buf += chunk
        data = buf[:n]
        buf = buf[n + 2:]
        return data

    def parse():
        line = read_line()
        t, rest = line[:1], line[1:]
        if t == b"+":
            return rest
        if t == b"-":
            raise RuntimeError("redis error: " + rest.decode())
        if t == b":":
            return int(rest)
        if t == b"$":
            n = int(rest)
            if n == -1:
                return None
            return read_n(n)
        if t == b"*":
            n = int(rest)
            if n == -1:
                return None
            return [parse() for _ in range(n)]
        raise RuntimeError("bad RESP type: " + repr(line))

    return parse()


class Redis:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port))

    def cmd(self, *args):
        out = [b"*%d\r\n" % len(args)]
        for a in args:
            b = a if isinstance(a, bytes) else str(a).encode()
            out.append(b"$%d\r\n" % len(b) + b + b"\r\n")
        self.sock.sendall(b"".join(out))
        return _read_reply(self.sock)

    def hget(self, key, field):
        return self.cmd("HGET", key, field)

    def hset(self, key, field, value):
        return self.cmd("HSET", key, field, value)

    def exists(self, key):
        return self.cmd("EXISTS", key) == 1

    def sadd(self, key, member):
        return self.cmd("SADD", key, member)

    def sismember(self, key, member):
        return self.cmd("SISMEMBER", key, member) == 1

    def lrem(self, key, count, member):
        return self.cmd("LREM", key, count, member)

    def smembers(self, key):
        return self.cmd("SMEMBERS", key) or []

    def lrange(self, key, start, stop):
        return self.cmd("LRANGE", key, start, stop) or []


# ----------------------------------------------------------------------------
# encode / decode
# ----------------------------------------------------------------------------
def decode_stats(blob):
    if blob is None or len(blob) != 32:
        return None
    return list(struct.unpack("<8i", blob))


def encode_stats(stats):
    return struct.pack("<8i", *stats)


def decode_items(blob):
    if blob is None:
        return []
    n = len(blob) // 2
    return list(struct.unpack("<%dH" % n, blob[: n * 2]))


def encode_items(items):
    return struct.pack("<%dH" % len(items), *items)


def item_name(t):
    return ITEM_NAMES[t] if t in ITEM_NAMES else f"type 0x{t:x} ({t})"


def build_target_items():
    items = [EMPTY] * NUM_SLOTS
    items[SLOT_WEAPON] = STAFF_OF_DESTRUCTION
    items[SLOT_ABILITY] = BURNING_RETRIBUTION_SPELL
    items[SLOT_ARMOR] = EMPTY
    items[SLOT_RING] = EMPTY
    items[FIRST_INVENTORY_SLOT] = SNAKE_PIT_KEY
    return items


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def print_char_state(label, dead, hp, mp, level, char_type, stats, items):
    print(f"--- {label} ---")
    print(f"  charType = {char_type}  level = {level}  hp = {hp}  mp = {mp}")
    dead_str = "?" if dead is None else ("ALIVE" if dead == b"\x00" else f"DEAD ({dead!r})")
    print(f"  dead = {dead_str}")
    if stats is None:
        print("  stats = <missing>")
    else:
        print("  stats: " + ", ".join(f"{n}={v}" for n, v in zip(STAT_NAMES, stats)))
    if not items:
        print("  items = <missing>")
    else:
        equip = ["weapon", "ability", "armor", "ring"]
        for i, t in enumerate(items):
            if i < 4:
                print(f"    slot {i:2d} ({equip[i]:7s}): {item_name(t)}")
            elif t != EMPTY:
                print(f"    slot {i:2d} (backpack): {item_name(t)}")
    print()


def member_le(char_id):
    return struct.pack("<i", char_id)


def main():
    r = Redis(REDIS_HOST, REDIS_PORT)
    if not r.exists(TARGET_CHAR_KEY):
        print(f"ERROR: {TARGET_CHAR_KEY} does not exist in redis", file=sys.stderr)
        sys.exit(1)

    # ---- BEFORE ----
    before_stats = decode_stats(r.hget(TARGET_CHAR_KEY, "stats"))
    before_items = decode_items(r.hget(TARGET_CHAR_KEY, "items"))
    print_char_state(
        "BEFORE  " + TARGET_CHAR_KEY,
        r.hget(TARGET_CHAR_KEY, "dead"),
        r.hget(TARGET_CHAR_KEY, "hp"),
        r.hget(TARGET_CHAR_KEY, "mp"),
        r.hget(TARGET_CHAR_KEY, "level"),
        r.hget(TARGET_CHAR_KEY, "charType"),
        before_stats,
        before_items,
    )

    # ---- shortcut probe: does it already encode the target gear? ----
    already_geared = (
        len(before_items) >= 2
        and before_items[SLOT_WEAPON] == STAFF_OF_DESTRUCTION
        and before_items[SLOT_ABILITY] == BURNING_RETRIBUTION_SPELL
    )
    print(f"[shortcut] target gear already equipped on this char: {already_geared}")
    print("[shortcut] (no dead char on this account had the T7 deliverable gear; re-encoding items)\n"
          if not already_geared else "[shortcut] reusing existing gear blob, only refreshing life state\n")

    # ---- write deliverable spec (idempotent: SetValue/HSET no-op if unchanged) ----
    target_items = build_target_items()
    r.hset(TARGET_CHAR_KEY, "charType", WIZARD_CHAR_TYPE)
    r.hset(TARGET_CHAR_KEY, "stats", encode_stats(STATS))
    r.hset(TARGET_CHAR_KEY, "items", encode_items(target_items))
    r.hset(TARGET_CHAR_KEY, "dead", b"\x00")
    r.hset(TARGET_CHAR_KEY, "hp", FULL_HP)
    r.hset(TARGET_CHAR_KEY, "mp", FULL_MP)
    r.hset(TARGET_CHAR_KEY, "level", CHAR_LEVEL)

    # ---- make it selectable: in alive set, not in dead list ----
    alive_key = f"alive.{ACCOUNT_ID}"
    dead_key = f"dead.{ACCOUNT_ID}"
    member = member_le(CHAR_ID)
    if not r.sismember(alive_key, member):
        r.sadd(alive_key, member)
    # remove ALL copies of this char id from the dead list (count=0 removes all)
    r.lrem(dead_key, 0, member)

    # ---- AFTER + verify ----
    after_dead = r.hget(TARGET_CHAR_KEY, "dead")
    after_hp = r.hget(TARGET_CHAR_KEY, "hp")
    after_mp = r.hget(TARGET_CHAR_KEY, "mp")
    after_level = r.hget(TARGET_CHAR_KEY, "level")
    after_ct = r.hget(TARGET_CHAR_KEY, "charType")
    after_stats = decode_stats(r.hget(TARGET_CHAR_KEY, "stats"))
    after_items = decode_items(r.hget(TARGET_CHAR_KEY, "items"))
    print_char_state("AFTER   " + TARGET_CHAR_KEY, after_dead, after_hp, after_mp, after_level, after_ct, after_stats, after_items)

    ok = True
    if after_items[SLOT_WEAPON] != STAFF_OF_DESTRUCTION:
        ok = False
        print("VERIFY FAIL: weapon slot is not Staff of Destruction")
    if after_items[SLOT_ABILITY] != BURNING_RETRIBUTION_SPELL:
        ok = False
        print("VERIFY FAIL: ability slot is not Burning Retribution Spell")
    if after_items[SLOT_ARMOR] != EMPTY or after_items[SLOT_RING] != EMPTY:
        ok = False
        print("VERIFY FAIL: armor/ring slot is not empty")
    if SNAKE_PIT_KEY not in after_items[FIRST_INVENTORY_SLOT:]:
        ok = False
        print("VERIFY FAIL: no Snake Pit Key in backpack")
    if after_dead != b"\x00":
        ok = False
        print("VERIFY FAIL: dead flag not cleared")
    if after_stats != STATS:
        ok = False
        print(f"VERIFY FAIL: stats {after_stats} != {STATS}")
    if int(after_hp) != FULL_HP:
        ok = False
        print(f"VERIFY FAIL: hp {after_hp!r} != {FULL_HP}")
    if not r.sismember(alive_key, member):
        ok = False
        print(f"VERIFY FAIL: char not in {alive_key} (won't show in char-select)")
    remaining_dead = [struct.unpack("<i", m)[0] for m in r.lrange(dead_key, 0, -1)]
    if CHAR_ID in remaining_dead:
        ok = False
        print(f"VERIFY FAIL: char still present in {dead_key}")

    print()
    print(f"alive.{ACCOUNT_ID} = {[struct.unpack('<i', m)[0] for m in r.smembers(alive_key)]}")
    print(f"dead.{ACCOUNT_ID}  = {remaining_dead}")
    print()
    print("RESULT:", "OK - deliverable maxed Wizard provisioned and selectable" if ok else "FAILED - see VERIFY lines above")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
