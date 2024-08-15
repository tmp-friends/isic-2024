FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 as builder

ARG WORKDIR=/kaggle
ENV PYTHONPATH $WORKDIR
WORKDIR $WORKDIR

# @ref: https://qiita.com/naozo-se/items/17cb127fab3783361ca4
ARG WORKDIR=/kaggle \
    USERNAME=kaggler \
    GROUPNAME=kaggler \
    UID=1000 \
    GID=1000

# ユーザ追加
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID $USERNAME

RUN mkdir -p $WORKDIR
RUN chown -R $UID:$GID $WORKDIR

RUN apt update \
    && apt install -yq software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt install -yq --no-install-recommends \
    # for open-cv
    libgl1-mesa-glx \
    curl \
    python3.10 \
    python3-distutils \
    # Python version登録
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

USER $USERNAME

# Poetryのインストールと設定
ENV PATH /home/$USERNAME/.local/bin:$PATH
RUN curl -sSL https://install.python-poetry.org | python3.10 -
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./

# requirements.txtの生成
# Poetryだとuser installができないので、requirements.txtを生成してpip installする
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes \
    && chmod 644 requirements.txt \
    && chown $UID:$GID requirements.txt

# poetry exportでCACHEを効かせるための2stage build
FROM builder

USER $USERNAME
COPY --from=builder $WORKDIR/requirements.txt .

RUN python -m pip install --user -r requirements.txt

# jupyter notebookの起動
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token='kaggle'", "--notebook-dir=/kaggle"]
