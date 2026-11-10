FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False HF_HOME=/tmp/hf

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
COPY --chown=user app ./app
RUN pip install --no-cache-dir --no-deps .

USER user
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
