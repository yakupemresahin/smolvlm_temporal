# ベースイメージとしてCUDA 12.1を使用
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libssl-dev \
    libffi-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# デフォルトのpythonをpython3に設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# pipとsetuptoolsをアップグレード
RUN pip install --upgrade pip setuptools

# CUDA 12.1 に対応したPyTorchのnightlyビルドをインストール
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu121

# requirements.txtをコピーして依存関係をインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# アプリケーションのコードをコピー
COPY . .

# 必要に応じて環境変数を設定（オプション）
ENV PYTHONUNBUFFERED=1

# コンテナ起動時にbashを実行
CMD ["bash"]