# ── Fly brain-body simulation — Linux container ───────────────────────────────
# MuJoCo runs headless via EGL (no display needed).
# Brian2 C++ backend uses GCC (no Visual Studio needed on Linux).
# First run adds ~13 min for Brian2 Cython compilation; subsequent runs use cache.

FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Brian2 C++ / Cython compilation
    build-essential \
    # flybody git install
    git \
    # video encoding (imageio-ffmpeg)
    ffmpeg \
    # MuJoCo EGL headless rendering
    libegl1 \
    libgl1 \
    libgles2 \
    libglu1-mesa \
    # h5py native build
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (own layer for caching) ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Project files ─────────────────────────────────────────────────────────────
COPY brain_model/ brain_model/
COPY fruitfly/    fruitfly/
COPY fly_brain_body_simulation.py .
COPY generate_plots.py .

# ── Output directories ────────────────────────────────────────────────────────
RUN mkdir -p simulations logs plots

# ── MuJoCo headless rendering via EGL — no DISPLAY required ──────────────────
ENV MUJOCO_GL=egl
ENV MUJOCO_EGL_DEVICE_ID=0

# ── Brian2 C++ cache lives here — mount a volume to persist across runs ───────
# ~/.cython/brian_extensions/  (populated on first net.run())

# ── Mount these to retrieve outputs after the run ────────────────────────────
VOLUME ["/app/simulations", "/app/logs", "/app/plots"]

CMD ["python", "fly_brain_body_simulation.py"]
