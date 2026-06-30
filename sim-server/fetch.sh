#!/usr/bin/env bash
# Fetch the pinned upstream betterSkillys checkout, apply the engine-seam patch,
# drop in the 15 Sim*.cs files, and build the WorldServer. Idempotent: re-running
# resets the upstream tree to pristine before re-applying the overlay.
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"

# The dotnet SDK on this box lives under ~/.dotnet and is not on the default
# non-interactive PATH; add it without relying on a login shell.
export PATH="$HOME/.dotnet:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1

# pin.txt provides UPSTREAM_REPO / UPSTREAM_BRANCH / UPSTREAM_COMMIT.
# shellcheck disable=SC1091
source ./pin.txt

UPSTREAM_DIR="$HERE/upstream"
BUILD_TARGET="WorldServer/WorldServer.csproj"

echo ">> Pin: $UPSTREAM_REPO @ $UPSTREAM_COMMIT"

# 1. Clone upstream if absent.
if [ ! -d "$UPSTREAM_DIR/.git" ]; then
  echo ">> Cloning upstream into ./upstream ..."
  git clone "$UPSTREAM_REPO" "$UPSTREAM_DIR"
else
  echo ">> ./upstream already present, reusing."
fi

cd "$UPSTREAM_DIR"

# 2. Fetch + check out the exact pinned commit.
echo ">> Fetching + checking out $UPSTREAM_COMMIT ..."
git fetch --quiet origin
git checkout --quiet "$UPSTREAM_COMMIT"

# 3. Reset to a pristine tree so the overlay always applies onto stock.
echo ">> Resetting upstream to pristine ..."
git clean -fdq
git checkout -- .

# 4. Apply the 11-file engine-seam patch (paths are a/WorldServer/...; -p1 strips a/).
echo ">> Applying overlay/seam.patch ..."
git apply -p1 "$HERE/overlay/seam.patch"
echo ">> Patch applied with no conflict."

# 5. Drop in the 15 Sim*.cs files at their UPSTREAM-relative paths.
echo ">> Copying Sim files into upstream ..."
cp -r "$HERE/Sim/." "$UPSTREAM_DIR/"

# 6. Build the WorldServer (Release).
echo ">> Building $BUILD_TARGET (Release) ..."
dotnet build -c Release "$BUILD_TARGET"

echo ""
echo "=================================================================="
echo "  SIM-SERVER OVERLAY BUILD SUCCEEDED"
echo "  upstream: $UPSTREAM_REPO @ $UPSTREAM_COMMIT"
echo "  target:   $UPSTREAM_DIR/$BUILD_TARGET (Release)"
echo "=================================================================="
