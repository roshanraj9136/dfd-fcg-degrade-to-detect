# Deepfake video checker: self-calibrated DFD-FCG on CPU (Hugging Face Space, Docker SDK).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python 2>/dev/null; pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless

# upstream model code (the released checkpoint itself is downloaded at startup from a private repo)
RUN git clone --depth 1 https://github.com/aiiu-lab/DFD-FCG /opt/DFD-FCG && rm -rf /opt/DFD-FCG/.git

COPY --chown=user qa ./qa
COPY --chown=user app ./app

USER user
ENV HOME=/home/user \
    PYTHONPATH=/home/user:/home/user/app \
    DFD_FCG_UPSTREAM=/opt/DFD-FCG \
    DFD_DATA=/tmp/dfd-data \
    MODEL_DIR=/tmp/models \
    TORCH_HOME=/home/user/.cache/torch \
    TORCHDYNAMO_DISABLE=1 \
    OMP_NUM_THREADS=2

EXPOSE 7860
CMD ["uvicorn", "main:app", "--app-dir", "app", "--host", "0.0.0.0", "--port", "7860"]
