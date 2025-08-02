# College Admission Assistant

The College Admission Assistant is an intelligent conversational AI system designed to provide comprehensive guidance on college admissions processes, academic programs, tuition information, and campus life. The system leverages advanced natural language processing capabilities to deliver accurate, contextual responses based on real institutional data.

## Features

### Core Functionality
- **Intelligent Query Processing**: Advanced semantic search using Azure OpenAI embeddings
- **Multi-Modal Interface**: Hybrid button-based and free-text input system
- **Real-Time Information Retrieval**: Dynamic access to up-to-date college information
- **Contextual Response Generation**: GPT-powered responses grounded in verified institutional data


## Architecture

### Technology Stack
- **Frontend**: Streamlit web application framework
- **AI Services**: Azure OpenAI (`gpt-35-turbo`, `text-embedding-ada-002`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Backend**: Python 3.8+ with asyncio support
- **Deployment**: Azure App Service / Streamlit Cloud compatible

---

## 🗂 Project Structure

```
.
├── app.py                # Streamlit chatbot UI
├── embed_mit.py          # Embedding + FAISS index builder
├── .env                  # Secrets for Azure OpenAI
├── data/                 # Scraped MIT pages
├── index.faiss           # Vector search index
├── texts.txt             # Cleaned + chunked documents
├── requirements.txt      # Python dependencies
├── startup.sh            # (for Azure App Service)
└── README.md
```

### Azure Services
- **Azure OpenAI Service** with the following deployments:
  - `gpt-4.1` (Chat completion model)
  - `text-embedding-ada-002` (Embedding model)
- **Azure App Service** (for cloud deployment)


## Installation

### 1. Repository Setup

```bash
mkdir college-admission-assistant
cd college-admission-assistant
```

### 2. Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Dependencies

```bash
pip install -r config/requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2023-12-01-preview

# Model Deployments
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-35-turbo
```

### Document Processing

1. **Data Collection**
   ```bash
   python scrape_mit.py
   ```

2. **Embedding Generation**
   ```bash
   python embeed_mit.py
   ```


### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

Access the application at `http://localhost:8501`


---

