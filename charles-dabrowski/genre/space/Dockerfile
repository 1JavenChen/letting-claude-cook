# Hugging Face Space (Docker, free CPU) for the genre demo in src/app.py
FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /app
ENV HF_HOME=/app/hf_cache HF_HUB_DISABLE_XET=1 MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
# bake the CLAP weights into the image so the Space starts fast
RUN python -c "from transformers import ClapModel, ClapProcessor; ClapModel.from_pretrained('laion/larger_clap_general'); ClapProcessor.from_pretrained('laion/larger_clap_general')"
COPY . .
RUN chown -R user /app
USER user
EXPOSE 7860
CMD ["python", "serve.py"]
