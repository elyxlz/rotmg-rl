# Build/run env for the server_env C-shim (no system ccache: ~/bin/ccache is a passthrough shim).
export PATH=$HOME/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LIBRARY_PATH=$HOME/.local/nvlib:${LIBRARY_PATH:-}
