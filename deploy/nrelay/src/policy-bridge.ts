import { CreateSuccessPacket, EnemyShootPacket, MapInfoPacket, NewTickPacket, PlayerTextPacket, UpdatePacket, UseItemPacket, UsePortalPacket, WorldPosData } from "@realmlib/net";
import { spawn } from "child_process";
import * as readline from "readline";
import { Client } from "./core/client";
import { Library, PacketHook } from "./decorators";
import { ConditionEffect, hasEffect } from "./models";
import { Runtime } from "./runtime/runtime";
import { Logger, LogLevel } from "./services";

const RL_DIR = "/home/audiogen/rotmg-rl";
const PYTHON = `${RL_DIR}/.venv/bin/python`;  // the scripts/setup.sh-provisioned venv (torch + vendored pufferlib)
const CKPT = "checkpoints/curriculum/finish.pt";  // the 98%-faithful-clear policy (trial-10 arch: server.py defaults hidden 256 / num_layers 2)
const INVINCIBLE_MASK = ConditionEffect.INVINCIBLE | ConditionEffect.INVULNERABLE;
const SNAKE_PIT_PORTAL_TYPE = 0x0718;
const SPELL_MP_COST = 20;
const CAST_COOLDOWN_MS = 350;

interface Intent { dx: number; dy: number; aim_x: number; aim_y: number; shoot: boolean; cast: boolean; }
interface Reply { intent?: Intent; reset_ok?: boolean; ok?: boolean; ready?: boolean; }

@Library({ name: "policy-bridge", author: "bot" })
class PolicyBridge {
  private proc: any;
  private ready = false;
  private busy = false;
  private mapSent = false;
  private mapName = "";
  private commandsSent = false;
  private seekPortal = false;
  private portalId: number | undefined;
  private lastPortalAttempt = -100;
  private resolvers: Array<(r: Reply) => void> = [];
  private tilesBuf: Array<{ x: number; y: number; walkable: boolean }> = [];
  private shotsBuf: object[] = [];
  private shootCount = 0;
  private castCount = 0;
  private lastCastTime = -100000;
  private firstPos: { x: number; y: number } | undefined;
  private tickCount = 0;
  private bossHp = -1;
  private bossHpMax = -1;
  private bossMinHp = -1;
  private bossX = -1;
  private bossY = -1;
  private teleportedToBoss = false;
  private lastTpTime = -100000;

  constructor(private runtime: Runtime) {
    Logger.log("PolicyBridge", "Spawning policy server...", LogLevel.Info);
    this.proc = spawn(PYTHON, ["-m", "rotmg_rl.deploy.server", "--checkpoint", CKPT], { cwd: RL_DIR });
    const rl = readline.createInterface({ input: this.proc.stdout });
    rl.on("line", (line: string) => this.onLine(line));
    this.proc.stderr.on("data", (d: Buffer) => { const s = d.toString().trim(); if (s.length > 0 && /error|traceback|exception/i.test(s)) Logger.log("PolicyBridge", "py: " + s.split("\n")[0], LogLevel.Warning); });
    this.proc.on("exit", (code: any) => Logger.log("PolicyBridge", "policy server exited code=" + code, LogLevel.Error));
  }

  private onLine(line: string): void {
    let msg: Reply;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.ready) { this.ready = true; Logger.log("PolicyBridge", "Policy server READY", LogLevel.Success); return; }
    const res = this.resolvers.shift();
    if (res) res(msg);
  }

  private send(obj: object): Promise<Reply> {
    return new Promise((resolve) => { this.resolvers.push(resolve); this.proc.stdin.write(JSON.stringify(obj) + "\n"); });
  }

  private sendPacket(client: Client, pkt: any): void { (client as any).send(pkt); }

  @PacketHook()
  onMapInfo(client: Client, p: MapInfoPacket): void {
    this.mapName = p.name; this.mapSent = false; this.bossHp = -1; this.bossMinHp = -1; this.teleportedToBoss = false;
    Logger.log("PolicyBridge", "MAPINFO name=" + p.name + " size=" + p.width + "x" + p.height, LogLevel.Success);
  }

  @PacketHook()
  async onCreateSuccess(client: Client, p: CreateSuccessPacket): Promise<void> {
    client.autoAim = false;
    client.autoNexusThreshold = 0;
    this.mapSent = false; this.tilesBuf = []; this.shotsBuf = []; this.firstPos = undefined;
    if (this.mapName === "Nexus") { this.commandsSent = false; this.seekPortal = false; this.portalId = undefined; this.lastPortalAttempt = -100; }
    await this.send({ reset: true });
    Logger.log("PolicyBridge", "reset sent (map=" + this.mapName + ")", LogLevel.Info);
  }

  @PacketHook()
  onUpdate(client: Client, p: UpdatePacket): void {
    const tiles = this.runtime.resources.tiles;
    for (const t of p.tiles) { const info = tiles[t.type]; this.tilesBuf.push({ x: t.x, y: t.y, walkable: !(info && info.noWalk) }); }
    if (this.seekPortal && this.portalId === undefined) {
      for (const o of p.newObjects) { if (o.objectType === SNAKE_PIT_PORTAL_TYPE) { this.portalId = o.status.objectId; Logger.log("PolicyBridge", "Found Snake Pit Portal objectId=" + o.status.objectId, LogLevel.Success); break; } }
    }
  }

  @PacketHook()
  onEnemyShoot(client: Client, p: EnemyShootPacket): void {
    const ownerObj = (client as any).enemies ? (client as any).enemies.get(p.ownerId) : undefined;
    let speed = 0.6, lifetime = 30;
    if (ownerObj && ownerObj.properties) { const objDef = this.runtime.resources.objects[ownerObj.properties.type]; if (objDef && objDef.projectiles && objDef.projectiles[p.bulletType]) { const pr = objDef.projectiles[p.bulletType]; speed = pr.speed / 100; lifetime = pr.lifetimeMS / 100; } }
    this.shotsBuf.push({ origin_x: p.startingPos.x, origin_y: p.startingPos.y, angle: p.angle, count: p.numShots, angle_inc: p.angleInc, speed, lifetime, spawn_ms: Date.now() });
  }

  @PacketHook()
  async onNewTick(client: Client, p: NewTickPacket): Promise<void> {
    if (!this.ready || this.busy || !client.worldPos || !client.mapInfo) return;
    this.busy = true;
    try {
      if (this.mapName === "Nexus" && !this.commandsSent) {
        this.commandsSent = true; this.seekPortal = true;
        for (const cmd of ["/max", "/spawn Snake Pit Portal"]) { const pt = new PlayerTextPacket(); pt.text = cmd; this.sendPacket(client, pt); }
        Logger.log("PolicyBridge", "sent /max + /spawn Snake Pit Portal", LogLevel.Success);
      }
      if (this.mapName === "Nexus" && this.portalId !== undefined && this.tickCount - this.lastPortalAttempt >= 10) {
        this.lastPortalAttempt = this.tickCount;
        const up = new UsePortalPacket(); up.objectId = this.portalId; this.sendPacket(client, up);
        Logger.log("PolicyBridge", "UsePortal sent objectId=" + this.portalId, LogLevel.Info);
      }

      const pd = client.playerData;
      const wp = client.worldPos;
      const enemies = (client as any).enemies as Map<number, any>;
      const projectiles = (client as any).projectiles as any[];
      const enemyArr: object[] = [];
      let bossSeen = false; this.bossHp = -1; this.bossHpMax = -1;
      if (enemies) for (const e of enemies.values()) {
        const od = e.objectData;
        const idName: string = e.properties && e.properties.id ? e.properties.id : "";
        const isBoss = /Stheno|Snake Queen/i.test(idName);
        if (isBoss) { bossSeen = true; this.bossHp = od.hp; this.bossHpMax = od.maxHP; this.bossX = od.worldPos.x; this.bossY = od.worldPos.y; if (this.bossMinHp < 0 || od.hp < this.bossMinHp) this.bossMinHp = od.hp; }
        enemyArr.push({ x: od.worldPos.x, y: od.worldPos.y, hp: od.hp, hp_max: od.maxHP, is_boss: isBoss, invuln: hasEffect(od.condition, INVINCIBLE_MASK) });
      }
      // Teleport to the boss to bridge the sim-to-real navigation gap (boss is ~80 tiles away in the real maze).
      if (this.mapName === "Snake Pit" && this.bossHp > 0) {
        const bd = Math.hypot(this.bossX - wp.x, this.bossY - wp.y);
        if (!this.teleportedToBoss || (bd > 40 && Date.now() - this.lastTpTime > 5000)) {
          this.teleportedToBoss = true; this.lastTpTime = Date.now();
          const pt = new PlayerTextPacket(); pt.text = "/tppos " + Math.floor(this.bossX) + " " + Math.floor(this.bossY); this.sendPacket(client, pt);
          Logger.log("PolicyBridge", "TP to boss at (" + Math.floor(this.bossX) + "," + Math.floor(this.bossY) + ")", LogLevel.Success);
        }
      }
      const bullets: object[] = [];
      if (projectiles) for (const pr of projectiles) if (pr.damageEnemies && pr.currentPosition) bullets.push({ x: pr.currentPosition.x, y: pr.currentPosition.y });

      const tick: any = {
        player: { x: wp.x, y: wp.y, hp: pd.hp, hp_max: pd.maxHP, mp: pd.mp, mp_max: pd.maxMP, confused: hasEffect(pd.condition, ConditionEffect.CONFUSED), petrified: hasEffect(pd.condition, ConditionEffect.PARALYZED) },
        enemies: enemyArr, player_bullets: bullets, now_ms: Date.now(),
      };
      if (!this.mapSent) { tick.map = { w: client.mapInfo.width, h: client.mapInfo.height }; this.mapSent = true; }
      if (this.tilesBuf.length > 0) { tick.tiles = this.tilesBuf; this.tilesBuf = []; }
      if (this.shotsBuf.length > 0) { tick.enemy_shots = this.shotsBuf; this.shotsBuf = []; }

      const reply = await this.send(tick);
      if (reply.intent) this.applyIntent(client, reply.intent);
      this.tickCount++;
      if (this.tickCount % 20 === 0 && this.firstPos) {
        const bdist = this.bossHp >= 0 ? Math.hypot(this.bossX - client.worldPos.x, this.bossY - client.worldPos.y) : -1;
        const bossStr = this.bossHp >= 0 ? " bossHP=" + this.bossHp + "/" + this.bossHpMax + " bossDist=" + bdist.toFixed(1) : " NO_BOSS_VIS";
        Logger.log("PolicyBridge", "[" + this.mapName + "] tick=" + this.tickCount + " pos=(" + client.worldPos.x.toFixed(1) + "," + client.worldPos.y.toFixed(1) + ") hp=" + pd.hp + "/" + pd.maxHP + " mp=" + pd.mp + " shots=" + this.shootCount + " casts=" + this.castCount + " enemies=" + enemyArr.length + (bossSeen ? " BOSS" : "") + bossStr, LogLevel.Success);
      }
    } finally { this.busy = false; }
  }

  private passable(client: Client, x: number, y: number): boolean {
    const w = client.mapInfo.width, h = client.mapInfo.height;
    const ix = Math.floor(x), iy = Math.floor(y);
    if (ix < 0 || iy < 0 || ix >= w || iy >= h) return false;
    const tile = (client as any).mapTiles[iy * w + ix];
    if (!tile) return false;
    if (tile.occupied) return false;
    const def = this.runtime.resources.tiles[tile.type];
    if (def && def.noWalk) return false;
    const enemies = (client as any).enemies as Map<number, any>;
    if (enemies) for (const e of enemies.values()) { const od = e.objectData; if (Math.floor(od.worldPos.x) === ix && Math.floor(od.worldPos.y) === iy && e.properties && (e.properties.occupySquare || e.properties.fullOccupy)) return false; }
    return true;
  }

  private applyIntent(client: Client, intent: Intent): void {
    const wp = client.worldPos;
    if (this.firstPos === undefined) this.firstPos = { x: wp.x, y: wp.y };
    if (intent.dx !== 0 || intent.dy !== 0) {
      const tx = wp.x + intent.dx, ty = wp.y + intent.dy;
      let gx = wp.x, gy = wp.y;
      if (this.passable(client, tx, ty)) { gx = tx; gy = ty; }
      else if (this.passable(client, tx, wp.y)) { gx = tx; gy = wp.y; }
      else if (this.passable(client, wp.x, ty)) { gx = wp.x; gy = ty; }
      client.nextPos.length = 0;
      if (gx !== wp.x || gy !== wp.y) client.nextPos.push(new WorldPosData(gx, gy));
    } else { client.nextPos.length = 0; }

    const inv = client.playerData.inventory;
    const angle = Math.atan2(intent.aim_y, intent.aim_x);
    if (intent.shoot) {
      const weapon = inv && inv.length > 0 ? inv[0] : -1;
      if (weapon !== -1 && weapon !== undefined && this.runtime.resources.items[weapon]) if (client.shoot(angle)) this.shootCount++;
    }
    // cast = activate the slot-1 ability (Wizard spell, BulletNova) via UseItem
    const spell = inv && inv.length > 1 ? inv[1] : -1;
    if (intent.cast && spell !== -1 && spell !== undefined && client.playerData.mp >= SPELL_MP_COST && (Date.now() - this.lastCastTime) >= CAST_COOLDOWN_MS) {
      this.lastCastTime = Date.now();
      const ui = new UseItemPacket();
      ui.time = (client as any).getTime();
      ui.slotObject.objectId = client.objectId;
      ui.slotObject.slotId = 1;
      ui.slotObject.objectType = spell;
      ui.itemUsePos = new WorldPosData(wp.x + intent.aim_x, wp.y + intent.aim_y);
      ui.useType = 0;
      this.sendPacket(client, ui);
      this.castCount++;
    }
  }
}

export { PolicyBridge };
