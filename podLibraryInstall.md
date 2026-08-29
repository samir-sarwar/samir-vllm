cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build
./build/samir-vllm

apt-get update
apt-get install -y gh

cd /workspace/samir-vllm
git config user.name samir-sarwar
git config user.email samirsarwaremail@gmail.com

gh auth login

export PATH="/usr/local/cuda/bin:$PATH"

DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  git \
  ninja-build \
  nlohmann-json3-dev \
  pkg-config

python3 -m pip install --no-cache-dir --break-system-packages --upgrade "cmake>=3.24"