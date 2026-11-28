# One image for the whole live system: the scoring API, the dashboard and the log feed
# run the same code with different commands (see docker-compose.yml).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. The libraries first: this slow step is then reused when only the code changes.
#    PyTorch comes from its CPU-only index (same version as requirements.txt); the default
#    one adds gigabytes of GPU libraries that the CPU-only models never use.
COPY requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# 2. The code. Data, models and results are NOT copied in: they are made by scripts 01-06
#    and attached read-only when the containers start (docker-compose.yml).
COPY src/ src/
COPY dashboard/ dashboard/
COPY .streamlit/ .streamlit/

# 3. Run as a normal user, not as root
RUN useradd --create-home app
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
