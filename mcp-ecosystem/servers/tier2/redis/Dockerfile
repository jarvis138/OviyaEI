FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .

# Expose port
EXPOSE 3009

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "
import asyncio
import httpx
async def check():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get('http://localhost:3009/health', timeout=5.0)
            return response.status_code == 200
        except:
            return False
asyncio.run(check())
" || exit 1

# Run the server
CMD ["python", "server.py"]
