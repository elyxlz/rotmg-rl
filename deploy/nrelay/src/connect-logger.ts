import { CreateSuccessPacket, FailurePacket, MapInfoPacket, NewTickPacket, ObjectData, ObjectStatusData, StatData, StatType, UpdatePacket } from "@realmlib/net";
import { Client } from "./core/client";
import { Library, PacketHook } from "./decorators";
import { Logger, LogLevel } from "./services";

function statSummary(stats: StatData[]): string {
  const parts: string[] = [];
  for (const s of stats) {
    let label = String(s.statType);
    if (s.statType === StatType.HP_STAT) label = "HP";
    else if (s.statType === StatType.MAX_HP_STAT) label = "MAXHP";
    else if (s.statType === StatType.NAME_STAT) label = "NAME";
    else if (s.statType === StatType.LEVEL_STAT) label = "LVL";
    else if (s.statType === StatType.DEFENSE_STAT) label = "DEF";
    const val = s.stringStatValue && s.stringStatValue.length > 0 ? s.stringStatValue : s.statValue;
    parts.push(`${label}=${val}`);
  }
  return parts.join(",");
}

@Library({ name: "connect-logger", author: "bot" })
class ConnectLogger {

  @PacketHook()
  onMapInfo(client: Client, p: MapInfoPacket): void {
    Logger.log("DECODE", `MAPINFO name=${p.name} size=${p.width}x${p.height} fp=${p.fp} difficulty=${p.difficulty}`, LogLevel.Success);
  }

  @PacketHook()
  onCreateSuccess(client: Client, p: CreateSuccessPacket): void {
    Logger.log("DECODE", `CREATE_SUCCESS objectId=${p.objectId} charId=${p.charId}`, LogLevel.Success);
  }

  @PacketHook()
  onFailure(client: Client, p: FailurePacket): void {
    Logger.log("DECODE", `FAILURE id=${p.errorId} desc="${p.errorDescription}"`, LogLevel.Error);
  }

  @PacketHook()
  onUpdate(client: Client, p: UpdatePacket): void {
    let players = 0;
    let objects = 0;
    let me: ObjectData | undefined;
    for (const o of p.newObjects) {
      if (o.status.objectId === client.objectId) me = o;
      if (o.objectType >= 0x300 && o.objectType <= 0x330) players++;
      else objects++;
    }
    Logger.log("DECODE", `UPDATE tiles=${p.tiles.length} newObjects=${p.newObjects.length} (playerLike=${players} other=${objects}) drops=${p.drops.length}`, LogLevel.Success);
    if (p.tiles.length > 0) {
      const t = p.tiles[0];
      Logger.log("DECODE", `  tile[0] x=${t.x} y=${t.y} type=${t.type}`, LogLevel.Info);
    }
    if (me) {
      Logger.log("DECODE", `  PLAYER objectId=${me.status.objectId} type=0x${me.objectType.toString(16)} pos=(${me.status.pos.x.toFixed(2)},${me.status.pos.y.toFixed(2)}) stats[${statSummary(me.status.stats)}]`, LogLevel.Success);
    } else if (p.newObjects.length > 0) {
      const o = p.newObjects[0];
      Logger.log("DECODE", `  obj[0] objectId=${o.status.objectId} type=0x${o.objectType.toString(16)} pos=(${o.status.pos.x.toFixed(2)},${o.status.pos.y.toFixed(2)})`, LogLevel.Info);
    }
  }

  @PacketHook()
  onNewTick(client: Client, p: NewTickPacket): void {
    let myPos = "";
    for (const s of p.statuses) {
      if (s.objectId === client.objectId) {
        myPos = ` myPos=(${s.pos.x.toFixed(2)},${s.pos.y.toFixed(2)}) stats[${statSummary(s.stats)}]`;
        break;
      }
    }
    Logger.log("DECODE", `NEWTICK tickId=${p.tickId} tickTime=${p.tickTime} statuses=${p.statuses.length}${myPos}`, LogLevel.Info);
  }
}

export { ConnectLogger };
