# ReclamoRuta — imagen de la app (CPU). Los pesos NO se copian a la imagen:
# se montan en tiempo de ejecución en /app/artifacts (solo lectura).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# PyTorch CPU desde su índice oficial (más liviano que la build CUDA).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app,model]"

COPY .streamlit ./.streamlit
COPY configs/routes.json configs/decision.json ./configs/
COPY reports/BANKING77_LABEL_TO_ID.json ./reports/

RUN useradd --create-home --uid 10001 app && mkdir -p /app/artifacts && chown -R app /app/artifacts
USER app

ENV RECLAMORUTA_ARTIFACT_DIR=/app/artifacts \
    RECLAMORUTA_DECISION=/app/configs/decision.json \
    RECLAMORUTA_DEVICE=cpu \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "src/reclamoruta/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
