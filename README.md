# OpenEMR LLM Module v2.0

AI-powered medical assistant integrating local LLM inference with OpenEMR for patient data-aware clinical decision support.

> **Note**: This module is for research and educational purposes. Always verify AI-generated information and consult healthcare professionals for clinical decisions.

## Features

- **Multiple LLM Backends**: Support for llama.cpp, Ollama, OpenAI-compatible APIs, and HuggingFace
- **OpenEMR Integration**: FHIR R4 API integration for patient data context
- **Privacy-First**: Local inference keeps patient data on-premises
- **Fine-Tuning Support**: Unsloth integration for creating medical domain-adapted models
- **Modern UI**: Responsive chat interface with real-time status updates
- **Security**: Rate limiting, input sanitization, PHI anonymization, and audit logging

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenEMR Web Interface                        │
│                     (llm.php frontend)                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP/JSON
┌─────────────────────▼───────────────────────────────────────────┐
│                  Python Flask Server                            │
│                   (llm_server.py)                               │
│  ┌─────────────┬─────────────┬──────────────┬────────────────┐ │
│  │ llama.cpp   │   Ollama    │   OpenAI     │  HuggingFace   │ │
│  │  Backend    │   Backend   │   Compat     │   Backend      │ │
│  └──────┬──────┴──────┬──────┴──────┬───────┴───────┬────────┘ │
└─────────┼─────────────┼─────────────┼───────────────┼──────────┘
          │             │             │               │
    ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐ ┌──────▼──────┐
    │ llama.cpp │ │  Ollama   │ │  vLLM/    │ │ Transformers │
    │  Server   │ │  Server   │ │  LocalAI  │ │   Library    │
    └───────────┘ └───────────┘ └───────────┘ └──────────────┘
```

## Quick Start

### 1. Prerequisites

- OpenEMR 7.0+ installed
- Python 3.10+
- One of the following LLM backends:
  - **llama.cpp** (recommended for local inference)
  - **Ollama** (easiest setup)
  - HuggingFace Transformers (fallback)

### 2. Install the Module

```bash
# Clone into OpenEMR custom modules directory
cd /var/www/html/openemr/interface/modules/custom_modules/
git clone https://github.com/SolshineCode/OpenEMR_LLM_Module.git llm
cd llm

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
nano .env  # Edit settings
```

### 3. Set Up an LLM Backend

#### Option A: llama.cpp Server (Recommended)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# Download a medical model (example: Llama 3.2)
./llama-cli --hf-repo bartowski/Llama-3.2-1B-Instruct-GGUF \
            --hf-file Llama-3.2-1B-Instruct-Q4_K_M.gguf \
            --model-only

# Start the server
./llama-server -m Llama-3.2-1B-Instruct-Q4_K_M.gguf \
               --port 8080 --ctx-size 4096
```

Configure in `.env`:
```env
LLM_BACKEND=llamacpp
LLAMACPP_SERVER_URL=http://localhost:8080
```

#### Option B: Ollama (Easiest)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2
# or for medical: ollama pull medllama2
```

Configure in `.env`:
```env
LLM_BACKEND=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### 4. Start the Python Server

```bash
cd /var/www/html/openemr/interface/modules/custom_modules/llm
source venv/bin/activate
python llm_server.py
```

For production, use Gunicorn:
```bash
gunicorn -w 4 -b 127.0.0.1:5000 llm_server:app
```

### 5. Enable in OpenEMR

1. Go to **Administration > Modules**
2. Find "Medical Assistant LLM" and click **Enable**
3. Navigate to the new "Medical Assistant LLM" tab

## Configuration

All configuration is done via environment variables. Copy `.env.example` to `.env` and edit:

### Server Settings
```env
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
FLASK_DEBUG=false
```

### LLM Backend
```env
# Options: llamacpp, ollama, openai, huggingface
LLM_BACKEND=llamacpp

# llama.cpp settings
LLAMACPP_SERVER_URL=http://localhost:8080

# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### OpenEMR Integration
```env
OPENEMR_BASE_URL=https://localhost:9300
OPENEMR_CLIENT_ID=your-client-id
OPENEMR_CLIENT_SECRET=your-secret
```

### Generation Parameters
```env
MAX_TOKENS=512
TEMPERATURE=0.7
SYSTEM_PROMPT=You are a helpful medical assistant...
```

## OpenEMR API Integration

To enable patient data context, register an OAuth2 client in OpenEMR:

1. Go to **Administration > System > API Clients**
2. Create a new client with these scopes:
   - `user/Patient.rs`
   - `user/Encounter.rs`
   - `user/Observation.rs`
   - `user/Condition.rs`
   - `user/AllergyIntolerance.rs`
   - `user/MedicationRequest.rs`
3. Copy the Client ID and Secret to your `.env` file

## Fine-Tuning with Unsloth

Create a custom medical model optimized for your use case:

```bash
cd fine_tuning
pip install -r requirements-unsloth.txt

# Train on medical QA data
python train_medical_llm.py \
    --base_model unsloth/Llama-3.2-1B-Instruct \
    --dataset medmcqa \
    --output_dir ./models/medical_adapter \
    --epochs 1
```

See [fine_tuning/README.md](fine_tuning/README.md) for detailed instructions.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/config` | GET | Get current configuration |
| `/generate` | POST | Generate LLM response |
| `/models` | GET | List available models |
| `/feedback` | POST | Submit response feedback |
| `/patient/<id>/summary` | GET | Get patient summary |

### Generate Request Example

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the common symptoms of diabetes?",
    "patient_id": "12345",
    "include_patient_data": true,
    "max_tokens": 512
  }'
```

## Project Structure

```
llm/
├── llm_server.py           # Main Flask server
├── config.py               # Configuration management
├── openemr_client.py       # OpenEMR FHIR API client
├── llm.php                 # Frontend interface
├── module.info             # OpenEMR module metadata
├── module.config.php       # OpenEMR module configuration
├── requirements.txt        # Python dependencies
├── .env.example           # Environment configuration template
├── llm_backends/          # LLM backend implementations
│   ├── __init__.py
│   ├── base.py            # Abstract base class
│   ├── llamacpp.py        # llama.cpp server backend
│   ├── ollama.py          # Ollama backend
│   ├── openai_compat.py   # OpenAI-compatible backend
│   └── huggingface.py     # HuggingFace Transformers backend
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── logging_config.py  # Structured logging setup
│   └── security.py        # Input sanitization, PHI anonymization
└── fine_tuning/           # Model fine-tuning tools
    ├── README.md
    ├── requirements-unsloth.txt
    ├── train_medical_llm.py
    └── sample_training_data.json
```

## Security Considerations

- **PHI Anonymization**: Patient data is anonymized before being sent to the LLM
- **Rate Limiting**: Prevents abuse with configurable rate limits
- **Input Sanitization**: All user input is sanitized to prevent injection attacks
- **Audit Logging**: All generations are logged for compliance
- **Local Inference**: Keeps patient data on-premises

### HIPAA Compliance Notes

- Run the LLM server on the same secure network as OpenEMR
- Enable SSL/TLS for all connections
- Configure proper access controls in OpenEMR
- Review audit logs regularly
- Consider using `ANONYMIZE_PATIENT_DATA=true` in production

## Recommended Medical Models

| Model | Source | Size | Notes |
|-------|--------|------|-------|
| Llama 3.2 | Ollama/llama.cpp | 1-3B | Good general model |
| Mistral 7B | Ollama/llama.cpp | 7B | High quality |
| MedLlama2 | Ollama | 7B | Medical fine-tuned |
| BioGPT | HuggingFace | 1.5B | Biomedical domain |
| Meditron | HuggingFace | 7-70B | Clinical domain |

## Troubleshooting

### Server won't start
- Check that the LLM backend (llama.cpp/Ollama) is running
- Verify `.env` configuration is correct
- Check logs in `logs/app.log`

### "Disconnected" status in UI
- Ensure Flask server is running on configured port
- Check CORS settings if running on different domains
- Verify firewall allows the connection

### Slow responses
- Consider using a smaller/quantized model
- Increase `MAX_TOKENS` gradually
- Enable GPU acceleration for llama.cpp/Ollama

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

GNU General Public License v3 (GPL-3.0)

## Disclaimer

This module is for research and educational purposes only. It should not be used as the sole basis for medical decisions. Always consult qualified healthcare professionals for clinical decisions.

The AI-generated responses may contain errors or outdated information. Users are responsible for verifying all information before clinical use.
