require("reflect-metadata");
const { Environment, Runtime } = require("./lib");
const env = new Environment(__dirname);
const runtime = new Runtime(env);
runtime.run({ update: false, debug: true, plugins: true, log: false }).catch((e) => {
  console.error("FATAL", e);
  process.exit(1);
});
// auto-exit after 30s so the run is bounded
setTimeout(() => { console.log("=== run window elapsed, exiting ==="); process.exit(0); }, 120000);
