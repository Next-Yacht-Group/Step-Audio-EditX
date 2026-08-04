# Step-Audio-EditX su NVIDIA DGX Spark

Guida completa per eseguire **Step-Audio-EditX** su **NVIDIA DGX Spark / ARM64** usando il container NVIDIA vLLM.

Questa procedura è stata verificata il **4 agosto 2026** su DGX Spark con:

- architettura host: `aarch64`
- container: `nvcr.io/nvidia/vllm:26.04-py3`
- vLLM: `0.19.0+...nv26.04`
- CUDA del container: `13.2`
- kernel driver host: ramo `580`
- Python: `3.12`
- modello: `stepfun-ai/Step-Audio-EditX` completo, non quantizzato
- tokenizer: `stepfun-ai/Step-Audio-Tokenizer`

La configurazione finale usa:

- backend attenzione vLLM: `TRITON_ATTN`
- vLLM in eager mode
- KV cache limitata per uso single-user
- CosyVoice in `float32`
- CUDA Graph disabilitati per CosyVoice
- TorchAudio compilato localmente contro il PyTorch NVIDIA del container
- ONNX Runtime GPU per CUDA 13 su ARM64

---

## Perché la procedura standard non funziona su Spark

Il setup ufficiale del repository è pensato principalmente per Linux `x86_64`.

Su DGX Spark si incontrano diversi problemi:

1. il wheel vLLM configurato nel progetto è `x86_64`;
2. i wheel PyPI di `onnxruntime-gpu` non coprono la combinazione ARM64/Python/CUDA necessaria;
3. il container NVIDIA impone versioni compatibili di NumPy, PyTorch e librerie CUDA;
4. TorchAudio installato da PyPI può essere compilato per una versione CUDA diversa da quella di PyTorch;
5. Step1 richiede `use_alibi_sqrt`, non supportato dal backend `FLASH_ATTN`;
6. CosyVoice può bloccarsi o fallire durante l'inizializzazione dei CUDA Graph;
7. la configurazione vLLM predefinita alloca una KV cache eccessiva per una UI usata da una sola persona.

Per questo si usa un Dockerfile dedicato:

```text
Dockerfile.spark
```

e una lista di dipendenze dedicata:

```text
requirements.spark.txt
```

---

## File Spark presenti nel repository

Questa guida assume che siano già stati committati:

```text
Dockerfile.spark
requirements.spark.txt
```

e le modifiche a:

```text
app.py
model_loader.py
```

### `model_loader.py`

La creazione di `vllm.LLM` deve includere:

```python
"attention_backend": "TRITON_ATTN",
```

Il modello Step1 usa `use_alibi_sqrt`, che non è supportato da `FLASH_ATTN`.

La vecchia variabile:

```python
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
```

non è sufficiente con la versione vLLM usata dal container NVIDIA. Il backend deve essere passato direttamente a `LLM(...)`.

### Configurazione finale dell'app

Il comando predefinito del container deve usare questi parametri:

```text
--model-path /models/Step-Audio-EditX
--tokenizer-path /models/Step-Audio-Tokenizer
--model-source local
--server-name 0.0.0.0
--server-port 7860
--gpu-memory-utilization 0.20
--max-num-seqs 1
--enforce-eager
--cosyvoice-dtype float32
--no-cosyvoice-cuda-graph
```

---

## 1. Prerequisiti

Verificare che Docker possa vedere la GPU:

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 \
  nvidia-smi
```

Su DGX Spark l'output di `nvidia-smi` può essere diverso da quello delle GPU discrete tradizionali. Il controllo importante è che il container parta e rilevi CUDA.

Verificare l'architettura:

```bash
uname -m
```

Output atteso:

```text
aarch64
```

Verificare Docker:

```bash
docker version
```

---

## 2. Login al registry NVIDIA

NGC CLI non è necessario per eseguire il container.

Serve soltanto il login Docker, se il registry richiede autenticazione:

```bash
docker login nvcr.io
```

Usare:

```text
Username: $oauthtoken
Password: <NGC API key>
```

Verificare il pull:

```bash
docker pull nvcr.io/nvidia/vllm:26.04-py3
```

---

## 3. Clonare il repository

```bash
git clone https://github.com/stepfun-ai/Step-Audio-EditX.git
cd Step-Audio-EditX
```

Se si usa un fork con il supporto Spark già committato:

```bash
git clone <URL_DEL_FORK>
cd Step-Audio-EditX
```

Controllare che i file siano presenti:

```bash
test -f Dockerfile.spark
test -f requirements.spark.txt
grep -n 'TRITON_ATTN' model_loader.py
```

---

## 4. Controllare le dipendenze Spark

`requirements.spark.txt` non deve reinstallare i componenti fondamentali già forniti dal container NVIDIA.

Controllare:

```bash
grep -Ei \
  '^(torch|torchaudio|torchvision|vllm|cuda-toolkit|nvidia-cuda-nvrtc-cu12|onnxruntime-gpu|numpy|deepspeed|trl|llmcompressor|datasets|wandb|spaces|whisper)([<>=!~;[:space:]]|$)' \
  requirements.spark.txt || echo "OK: dipendenze incompatibili escluse"
```

`openai-whisper` può essere presente. È un pacchetto distinto da quello chiamato esattamente `whisper`.

Pacchetti come questi possono restare:

```text
torch-complex
torchcodec
```

Non sono il pacchetto `torch`.

### Non aggiornare indiscriminatamente pip e setuptools

Il container NVIDIA vLLM porta con sé un ambiente già coerente.

In particolare, evitare una riga generica come:

```bash
python3 -m pip install --upgrade pip setuptools wheel
```

Una versione troppo recente di `setuptools` può entrare in conflitto con vLLM.

---

## 5. Costruire l'immagine

Dalla root del repository:

```bash
docker build --progress=plain \
  -f Dockerfile.spark \
  -t step-audio-editx:spark .
```

La prima build può richiedere parecchio tempo perché:

- installa le dipendenze Python;
- installa ONNX Runtime GPU per CUDA 13;
- clona e compila TorchAudio;
- prepara l'immagine applicativa.

Le build successive riutilizzano la cache Docker.

### Verificare l'immagine

```bash
docker image inspect step-audio-editx:spark >/dev/null &&
echo "Immagine creata correttamente"
```

---

## 6. Perché TorchAudio viene compilato localmente

Il container NVIDIA usa una build custom di PyTorch compilata con CUDA 13.2.

Un TorchAudio scaricato da PyPI può invece essere compilato, per esempio, con CUDA 13.0 e produrre:

```text
RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions.
PyTorch has CUDA version 13.2 whereas TorchAudio has CUDA version 13.0.
```

Il Dockerfile Spark rimuove quel wheel e compila TorchAudio contro il PyTorch già presente nel container.

La build usata è CPU-only per le operazioni TorchAudio:

```text
USE_CUDA=0
BUILD_SOX=0
USE_FFMPEG=0
BUILD_RNNT=0
BUILD_CTC_DECODER=0
```

Questo non disabilita la GPU per:

- PyTorch
- vLLM
- ONNX Runtime
- Step-Audio-EditX
- CosyVoice

TorchAudio viene usato principalmente per caricamento, salvataggio, resampling e trasformazioni audio.

---

## 7. Verificare PyTorch, vLLM, ONNX Runtime e TorchAudio

Eseguire:

```bash
docker run --rm --gpus all \
  --ipc=host \
  step-audio-editx:spark \
  python3 -c '
import torch
import torchaudio
import vllm
import onnxruntime as ort

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("TorchAudio:", torchaudio.__version__)
print("vLLM:", vllm.__version__)
print("ONNX Runtime:", ort.__version__)
print("ONNX providers:", ort.get_available_providers())

assert torch.cuda.is_available()
assert "CUDAExecutionProvider" in ort.get_available_providers()

audio = torch.randn(1, 16000)
resampled = torchaudio.functional.resample(audio, 16000, 24000)
print("Resample:", tuple(audio.shape), "->", tuple(resampled.shape))
print("OK")
'
```

L'output deve includere:

```text
CUDAExecutionProvider
OK
```

---

## 8. Scaricare i modelli

Creare una directory persistente sull'host:

```bash
mkdir -p "$HOME/models/step-audio-editx"
```

Scaricare il tokenizer e il modello completo usando `huggingface_hub` già installato nell'immagine:

```bash
docker run --rm -i \
  -v "$HOME/models/step-audio-editx:/models" \
  step-audio-editx:spark \
  python3 - <<'PY'
from huggingface_hub import snapshot_download

models = [
    (
        "stepfun-ai/Step-Audio-Tokenizer",
        "/models/Step-Audio-Tokenizer",
    ),
    (
        "stepfun-ai/Step-Audio-EditX",
        "/models/Step-Audio-EditX",
    ),
]

for repo_id, destination in models:
    print(f"\nDownloading {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=destination,
    )

print("\nDownload completato.")
PY
```

Il comando è riprendibile: se viene rilanciato, Hugging Face riusa i file già presenti.

### Struttura attesa

```text
$HOME/models/step-audio-editx/
├── Step-Audio-EditX/
└── Step-Audio-Tokenizer/
```

Controllare:

```bash
du -sh "$HOME/models/step-audio-editx/"*
```

Controllare alcuni file essenziali:

```bash
find "$HOME/models/step-audio-editx" \
  -maxdepth 5 \
  -type f \
  \( \
    -name 'model*.safetensors' \
    -o -name 'speech_tokenizer_v1.onnx' \
    -o -name 'linguistic_tokenizer.npy' \
    -o -name 'model.pt' \
  \) \
  -print
```

### Modello completo o AWQ

Per il primo avvio usare il modello completo:

```text
stepfun-ai/Step-Audio-EditX
```

La variante AWQ aggiunge un'altra variabile di compatibilità. Non è necessaria sullo Spark per questo modello e non fa parte della configurazione verificata in questa guida.

---

## 9. Avviare l'applicazione

Comando interattivo:

```bash
docker run --rm -it \
  --name step-audio-editx \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 7860:7860 \
  -v "$HOME/models/step-audio-editx:/models" \
  step-audio-editx:spark
```

I flag importanti sono:

```text
--gpus all
--ipc=host
--ulimit memlock=-1
--ulimit stack=67108864
```

`--ipc=host` evita il limite Docker predefinito di 64 MB per shared memory, insufficiente per alcuni workload vLLM.

### Output finale atteso

Dopo il caricamento:

```text
Successfully loaded vLLM model
CosyVoice model loaded successfully
StepCommonAudioTTS loaded successfully
Running on local URL: http://0.0.0.0:7860
```

Aprire dal computer locale:

```text
http://spark-bd59.local:7860
```

Oppure usare l'indirizzo IP dello Spark:

```text
http://<IP_DELLO_SPARK>:7860
```

Verificare dalla shell dello Spark:

```bash
curl -I http://127.0.0.1:7860
```

---

## 10. Primo test nella UI

Per un test di voice cloning:

1. caricare un audio pulito di circa 5-15 secondi;
2. inserire in `Prompt Text` la trascrizione esatta dell'audio;
3. inserire in `Target Text` la frase da generare;
4. selezionare `clone`;
5. premere `CLONE`.

La trascrizione automatica è disabilitata per impostazione predefinita.

Per il primo test:

- usare una sola voce;
- evitare musica e rumore;
- usare una frase breve;
- non superare circa 30 secondi di audio;
- lasciare il log Docker visibile.

---

## 11. Avvio in background

Per eseguire il servizio senza terminale aperto:

```bash
docker run -d \
  --name step-audio-editx \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 7860:7860 \
  -v "$HOME/models/step-audio-editx:/models" \
  step-audio-editx:spark
```

Seguire i log:

```bash
docker logs -f step-audio-editx
```

Fermare:

```bash
docker stop step-audio-editx
```

Riavviare:

```bash
docker start step-audio-editx
```

Eliminare:

```bash
docker rm -f step-audio-editx
```

---

## 12. Tempi di avvio

Su DGX Spark, un avvio completo può impiegare alcuni minuti.

Sequenza tipica:

1. caricamento FunASR;
2. inizializzazione ONNX Runtime;
3. inizializzazione vLLM;
4. caricamento dei pesi Step-Audio-EditX;
5. creazione della KV cache;
6. caricamento CosyVoice;
7. avvio Gradio.

Il caricamento dei pesi vLLM può richiedere circa 30-60 secondi.

Non considerare bloccato il processo finché il log continua ad avanzare o la CPU/GPU mostrano attività.

Il caso osservato di blocco reale era invece:

```text
Loading CosyVoice with dtype=bfloat16, cuda_graph=False
```

rimasto fermo per circa un'ora con CPU al 100%.

La configurazione funzionante usa:

```text
--cosyvoice-dtype float32
--no-cosyvoice-cuda-graph
```

---

## 13. Configurazione vLLM scelta

La web app è normalmente usata da una sola persona. La configurazione predefinita di vLLM allocava una KV cache enorme, con centinaia di richieste teoricamente concorrenti.

La configurazione verificata limita l'allocazione:

```text
--gpu-memory-utilization 0.20
--max-num-seqs 1
--enforce-eager
```

### Perché `--enforce-eager`

Disabilita:

- Torch Inductor per il modello;
- CUDA Graph di vLLM.

Questo riduce la complessità e previene problemi sulla piattaforma Spark.

Il log può mostrare:

```text
Inductor compilation was disabled by user settings
Cudagraph is disabled under eager mode
```

Sono messaggi attesi.

### Aumentare le prestazioni in seguito

Solo dopo avere verificato stabilità e qualità si può sperimentare rimuovendo:

```text
--enforce-eager
```

oppure aumentando:

```text
--gpu-memory-utilization
```

Questa guida documenta la configurazione stabile, non quella più aggressiva.

---

## 14. Warning attesi

### CUDA Forward Compatibility

Possibile messaggio:

```text
NOTE: CUDA Forward Compatibility mode ENABLED.
Using CUDA 13.2 driver version ... with kernel driver version 580...
```

Nella configurazione testata è normale.

Significa che il container CUDA 13.2 usa il meccanismo NVIDIA Forward Compatibility sul driver kernel installato.

Non è lo stesso problema dell'errore Torch/TorchAudio.

### ONNX Runtime Memcpy

Possibili warning:

```text
Memcpy nodes are added to the graph
Some nodes were not assigned to the preferred execution providers
```

Sono warning di performance.

ONNX Runtime può assegnare alla CPU alcune operazioni di forma o controllo anche quando usa:

```text
CUDAExecutionProvider
```

### Variabili vLLM sconosciute

Possibili warning:

```text
Unknown vLLM environment variable detected: VLLM_VERSION
Unknown vLLM environment variable detected: VLLM_FLASH_ATTN_SRC_DIR
```

Provengono dall'ambiente del container NVIDIA e non impediscono l'avvio.

### Spawn multiprocessing

Possibile messaggio:

```text
We must use the spawn multiprocessing start method
```

È atteso quando CUDA è già stata inizializzata nel processo principale.

---

## 15. Troubleshooting

### `uv sync` cerca un wheel vLLM ARM64 inesistente

Errore tipico:

```text
Failed to download vllm ... manylinux ... aarch64.whl
404 Not Found
```

Causa:

- la configurazione del repository punta a un wheel vLLM non disponibile per ARM64.

Soluzione:

- non usare `uv sync` sul sistema host;
- usare `Dockerfile.spark`;
- partire da `nvcr.io/nvidia/vllm:26.04-py3`.

---

### `onnxruntime-gpu` non ha una distribuzione compatibile

Errore:

```text
Could not find a version that satisfies the requirement onnxruntime-gpu
No matching distribution found
```

Causa:

- PyPI non offre il wheel adatto alla combinazione ARM64, Python e CUDA del container.

Soluzione:

- non lasciare `onnxruntime-gpu` in `requirements.spark.txt`;
- installare la build CUDA 13 ARM64 dal feed Microsoft previsto nel Dockerfile.

---

### Conflitto NumPy

Errore:

```text
The user requested numpy>=2.2.6
The user requested (constraint) numpy<=2.1
ResolutionImpossible
```

Causa:

- il container NVIDIA impone `numpy<=2.1`;
- il progetto richiede esplicitamente una versione successiva.

Soluzione:

- rimuovere il requisito esplicito `numpy>=2.2.6` da `requirements.spark.txt`;
- mantenere la versione scelta dal container NVIDIA.

Non rimuovere i constraint NVIDIA globalmente.

---

### Conflitto setuptools

Warning o errore:

```text
vllm ... requires setuptools<81
but setuptools 81.0.0 is installed
```

Causa:

```bash
pip install --upgrade pip setuptools wheel
```

Soluzione:

- non aggiornare indiscriminatamente setuptools;
- usare le versioni già presenti nel container.

---

### PyTorch e TorchAudio compilati con CUDA diverse

Errore:

```text
PyTorch has CUDA version 13.2
TorchAudio has CUDA version 13.0
```

Soluzione:

- usare il `Dockerfile.spark` che compila TorchAudio localmente;
- non reinstallare TorchAudio da PyPI dopo la build.

Verificare:

```bash
docker run --rm --gpus all \
  step-audio-editx:spark \
  python3 -c '
import torch, torchaudio
print(torch.__version__)
print(torch.version.cuda)
print(torchaudio.__version__)
'
```

---

### `use_alibi_sqrt` non supportato da FLASH_ATTN

Errore:

```text
ValueError: use_alibi_sqrt is not supported by backend FLASH_ATTN
```

Soluzione in `model_loader.py`:

```python
llm_kwargs = {
    ...
    "attention_backend": "TRITON_ATTN",
}
```

L'output corretto contiene:

```text
Using AttentionBackendEnum.TRITON_ATTN backend
```

Non affidarsi soltanto a:

```text
VLLM_ATTENTION_BACKEND
```

perché nella configurazione vLLM usata è stata ignorata.

---

### CosyVoice: `GET was unable to find an engine`

Errore osservato:

```text
Loading CosyVoice with dtype=float32, cuda_graph=True
CUDA Graph initialized successfully for chunk decoder
GET was unable to find an engine to execute this computation
```

Soluzione:

```text
--cosyvoice-dtype float32
--no-cosyvoice-cuda-graph
```

---

### CosyVoice fermo con CPU al 100%

Caso osservato:

```text
Loading CosyVoice with dtype=bfloat16, cuda_graph=False
```

senza ulteriori output per circa un'ora.

Soluzione:

- interrompere il container;
- usare CosyVoice in `float32`;
- disabilitare i CUDA Graph;
- ridurre la memoria riservata da vLLM;
- usare eager mode.

Comando:

```bash
docker rm -f step-audio-editx
```

Poi avviare con la configurazione predefinita del Dockerfile Spark.

---

### Shared memory Docker di 64 MB

Warning:

```text
The SHMEM allocation limit is set to the default of 64MB
```

Soluzione:

```text
--ipc=host
--ulimit memlock=-1
--ulimit stack=67108864
```

Durante il solo download dei modelli il warning può essere ignorato.

---

### Porta 7860 già occupata

Controllare:

```bash
sudo ss -ltnp | grep ':7860'
```

Controllare i container:

```bash
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Ports}}'
```

Fermare il vecchio container:

```bash
docker rm -f step-audio-editx
```

Oppure usare un'altra porta host:

```bash
-p 7861:7860
```

e aprire:

```text
http://spark-bd59.local:7861
```

---

### Container apparentemente bloccato

Controllare i processi:

```bash
docker top step-audio-editx
```

Controllare le risorse:

```bash
docker stats step-audio-editx
```

Controllare i log:

```bash
docker logs --tail 200 step-audio-editx
```

Controllare GPU e memoria:

```bash
watch -n 1 nvidia-smi
```

Interrompere forzatamente:

```bash
docker rm -f step-audio-editx
```

---

## 16. Ricostruire dopo modifiche al codice

Dopo modifiche a:

```text
app.py
model_loader.py
tts.py
Dockerfile.spark
requirements.spark.txt
```

ricostruire:

```bash
docker build \
  -f Dockerfile.spark \
  -t step-audio-editx:spark .
```

Docker riutilizzerà i layer non modificati.

Per una build completamente pulita:

```bash
docker build --no-cache --progress=plain \
  -f Dockerfile.spark \
  -t step-audio-editx:spark .
```

Usare `--no-cache` solo quando si sospetta che un layer precedente contenga una dipendenza sbagliata.

---

## 17. Aggiornare il repository

Prima di fare pull, fermare il container:

```bash
docker rm -f step-audio-editx 2>/dev/null || true
```

Aggiornare:

```bash
git pull --ff-only
```

Ricostruire:

```bash
docker build \
  -f Dockerfile.spark \
  -t step-audio-editx:spark .
```

I modelli restano fuori dall'immagine e non devono essere riscaricati, perché sono montati da:

```text
$HOME/models/step-audio-editx
```

---

## 18. File da non committare

Non aggiungere al repository:

```text
models/
.cache/
.venv/
__pycache__/
*.wav
*.mp3
```

I modelli devono restare fuori da Git.

Esempio `.gitignore`:

```gitignore
models/
.cache/
.venv/
__pycache__/
*.pyc
*.wav
*.mp3
```

È utile anche un `.dockerignore`:

```dockerignore
.git
.venv
__pycache__
*.pyc
models
.cache
```

---

## 19. Comandi rapidi

### Build

```bash
docker build \
  -f Dockerfile.spark \
  -t step-audio-editx:spark .
```

### Avvio

```bash
docker run --rm -it \
  --name step-audio-editx \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 7860:7860 \
  -v "$HOME/models/step-audio-editx:/models" \
  step-audio-editx:spark
```

### Stop

```bash
docker rm -f step-audio-editx
```

### Log

```bash
docker logs -f step-audio-editx
```

### UI

```text
http://spark-bd59.local:7860
```

---

## 20. Configurazione finale verificata

Riassunto della configurazione stabile:

```text
Base image:
  nvcr.io/nvidia/vllm:26.04-py3

Architecture:
  ARM64 / aarch64

Model:
  stepfun-ai/Step-Audio-EditX

Tokenizer:
  stepfun-ai/Step-Audio-Tokenizer

vLLM:
  attention_backend=TRITON_ATTN
  gpu_memory_utilization=0.20
  max_num_seqs=1
  enforce_eager=True

CosyVoice:
  dtype=float32
  cuda_graph=False

Docker:
  --gpus all
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864

Port:
  7860
```

Questa configurazione ha completato correttamente:

```text
StepAudioTokenizer loaded successfully
Successfully loaded vLLM model
CosyVoice model loaded successfully
StepCommonAudioTTS loaded successfully
Running on local URL: http://0.0.0.0:7860
```
