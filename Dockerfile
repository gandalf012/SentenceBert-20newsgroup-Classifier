FROM python:3.7-slim-buster
LABEL maintainer cletadjos@gmail.com

RUN useradd worker

WORKDIR /worker

COPY --chown=worker:worker requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=worker:worker app.py .
COPY --chown=worker:worker output/ output/
COPY --chown=worker:worker models/ models/

USER worker
