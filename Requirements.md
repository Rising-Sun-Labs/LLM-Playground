### 1.1 Host OS, Runtimes, and Base Tooling

- **OS:** Ubuntu 22.04 LTS (or Rocky/Alma 9.x).
- **Drivers/Runtime:** NVIDIA Driver (matching CUDA), **CUDA 12.1/12.2+**, **cuDNN**. Optional **ROCm** if targeting AMD.
- **Compilers/Build:** `gcc/g++ (>=10)`, `clang` (opt), `cmake`, `ninja`, `make`, `git`, `llvm-dev` (for Triton).
- **Python:** 3.10 or 3.11; env via **conda** or **uv/venv**.
- **Optional JVM stack:** JDK 17 if using Spark/Flink.

### Set up your virtual environment

1. Create a new virtual environment
   `python3 -m venv ~/venv`

2. Activate it
   `source ~/venv/bin/activate`

3. Install the pytorch etc
   `pip install pytorch torchvision torchaudio`

4. When done
   `deactivate`
