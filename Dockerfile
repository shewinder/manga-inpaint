FROM python:3.12-slim

ENV LAMA_MODEL=/app/models/lama/big-lama.pt

# 系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx libglib2.0-0 tini curl && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
ADD https://astral.sh/uv/install.sh /tmp/uv-install.sh
RUN sh /tmp/uv-install.sh && rm /tmp/uv-install.sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# 依赖文件层
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# 应用代码 + 字体
COPY main.py render.py ./
COPY fonts/ fonts/

# 下载 CRAFT + LaMa 模型
RUN mkdir -p models/craft models/lama && \
    curl -sSL --retry 3 "https://huggingface.co/Manbehindthemadness/craft_mlt_25k/resolve/main/craft_mlt_25k.pth" -o models/craft/craft_mlt_25k.pth && \
    curl -sSL --retry 3 "https://huggingface.co/Manbehindthemadness/craft_mlt_25k/resolve/main/craft_refiner_CTW1500.pth" -o models/craft/craft_refiner_CTW1500.pth && \
    curl -sSL --retry 3 "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt" -o models/lama/big-lama.pt

# 下载 manga-ocr 模型
RUN uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('kha-white/manga-ocr-base', local_dir='models/manga-ocr-flat')"

EXPOSE 8899
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uv", "run", "python", "main.py"]
