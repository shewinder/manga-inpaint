FROM python:3.12-slim

# apt 阿里源 + 系统依赖
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx libglib2.0-0 tini && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
ADD https://astral.sh/uv/install.sh /tmp/uv-install.sh
RUN sh /tmp/uv-install.sh && rm /tmp/uv-install.sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# 依赖文件层
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# 应用代码 + 模型 + 字体
COPY main.py render.py test.py ./
COPY models/ models/
COPY fonts/ fonts/

EXPOSE 8899
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "main.py"]
