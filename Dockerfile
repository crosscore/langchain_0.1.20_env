# ベースイメージとしてPython 3.9を使用
FROM python:3.11

# 作業ディレクトリを設定
WORKDIR /app

# requirements.txtを作業ディレクトリにコピー
COPY requirements.txt .

# パッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# エントリーポイントを設定
ENTRYPOINT ["/bin/bash"]
