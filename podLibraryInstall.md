export PATH="/usr/local/cuda/bin:$PATH"

apt-get update
apt-get install -y build-essential git ninja-build nlohmann-json3-dev

python3 -m pip install --no-cache-dir --upgrade cmake

cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build
./build/samir-vllm