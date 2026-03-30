FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose HF Spaces port
EXPOSE 7860

# Validate environment on build
RUN python -c "from environment import CodeReviewEnv; env = CodeReviewEnv(); obs = env.reset(); print('Environment OK')"

# Health check — tests the live HTTP server, not just Python imports
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Run CodeCrack dashboard (serves on port 7860 for Hugging Face Spaces)
CMD ["python", "app.py"]
