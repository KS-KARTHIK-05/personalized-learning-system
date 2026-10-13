<div align="center">

# 🎓 Personalized Learning Pathway Engine

**An AI-powered career learning roadmap generator combining skill-gap analysis, semantic course retrieval, Google Gemini, and RAG evaluation.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Database-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-Sentence--Transformers-green.svg)](https://www.sbert.net/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-8E7CC3.svg?logo=google)](https://ai.google.dev/)
[![Ragas](https://img.shields.io/badge/Evaluation-Ragas-purple.svg)](https://docs.ragas.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#-key-features) •
[Architecture](#-system-architecture) •
[Setup](#-setup-and-usage) •
[Project Structure](#-project-structure) •
[Evaluation](#-evaluation) •
[Security](#-security)

</div>

---

## 📌 Overview

The **Personalized Learning Pathway Engine** helps learners identify the skills they may need for a target technical role and organizes relevant course recommendations into a structured learning roadmap.

The application accepts a learner profile either through a PDF resume or manual text input. Google Gemini analyzes the profile and target role to identify existing skills and potential skill gaps. Sentence Transformers converts skill-gap queries into embeddings, and ChromaDB retrieves relevant courses from an ingested course catalog. Gemini then organizes the retrieved course recommendations into a four-phase roadmap.

The Streamlit dashboard displays the skills identified, gaps to address, and roadmap phases. Users can also download the roadmap as a PDF and run the configured Ragas evaluation from the interface.

## ⚡ Key Features

### 📄 Resume and Manual Profile Input

- Upload a resume in PDF format.
- Alternatively, enter current skills or experience manually.
- Extract text from PDF resumes using `pypdf`.

### 🎯 AI-Powered Skill-Gap Analysis

Uses Google Gemini to return a structured JSON response containing:

- `extracted_skills`
- `target_core_skills`
- `calculated_gaps`

The identified gaps are used as queries for course retrieval.

### 🔎 Semantic Course Retrieval

- Uses the `all-MiniLM-L6-v2` Sentence Transformers model.
- Stores course embeddings in a persistent ChromaDB collection.
- Retrieves relevant courses for each identified skill gap.
- Uses the same embedding model for ingestion and retrieval.

### 🗺️ Four-Phase Learning Roadmap

Gemini organizes retrieved course recommendations into four phases:

1. **Phase 1: Foundations** — Weeks 1–4
2. **Phase 2: Core Architecture** — Weeks 5–8
3. **Phase 3: Advanced Specialization** — Weeks 9–12
4. **Phase 4: Production & Cloud Deployment** — Weeks 13–16

The roadmap is displayed as a timeline in the Streamlit dashboard.

### 📥 PDF Roadmap Export

Generates a downloadable PDF report containing the course recommendations organized by phase.

### 🧪 Ragas Evaluation

Provides an evaluation action in the dashboard using the configured Ragas workflow. The evaluator attempts to calculate Faithfulness and Answer Relevancy. Context Precision is also displayed by the interface, but the current evaluator supplies a fallback value for this metric rather than calculating it as part of the configured Ragas metric list.

**Important:** Evaluation scores should be treated as valid measured results only when the evaluation completes successfully without fallback values. The current evaluator includes fallback values when imports or evaluation execution fail, so those values must not be reported as actual measured performance.

### 🖥️ Local Vector Database

Uses a persistent ChromaDB database stored in the local `vector_db/` directory.

### ⚡ GPU or CPU Embedding

The embedding components select CUDA when PyTorch reports that a compatible CUDA device is available; otherwise, they use the CPU.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────┐
│         Learner Profile             │
│      PDF Resume or Manual Text      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│         Target Career Role          │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Google Gemini Skill-Gap Analyzer    │
│ Existing Skills + Target Skills     │
│           + Calculated Gaps         │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│      Semantic Query Generation      │
│     Sentence Transformers Model     │
│        all-MiniLM-L6-v2             │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       Persistent ChromaDB           │
│       Collection: learning_courses  │
│       Directory: ./vector_db        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       Retrieved Course Matches      │
│       Up to 2 per identified gap    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Google Gemini Roadmap Synthesis     │
│         Four Learning Phases        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       Streamlit Dashboard           │
│ Skills • Gaps • Roadmap Timeline    │
│          Downloadable PDF           │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ Optional Ragas Evaluation           │
│ Faithfulness • Answer Relevancy     │
└─────────────────────────────────────┘
```

---

## 🧰 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| User interface | Streamlit | Profile input, roadmap display, and evaluation interface |
| Language model | Google Gemini via `google-genai` | Skill-gap analysis and roadmap generation |
| Embeddings | Sentence Transformers | Converts course text and search queries into embeddings |
| Embedding model | `all-MiniLM-L6-v2` | Semantic representation of courses and queries |
| Vector database | ChromaDB | Persistent storage and similarity retrieval |
| Deep learning runtime | PyTorch | Runs embedding inference on CUDA or CPU |
| PDF input | `pypdf` | Extracts text from uploaded PDF resumes |
| PDF output | `fpdf2` | Creates downloadable roadmap reports |
| Evaluation | Ragas | Evaluates configured RAG response metrics |
| Configuration | `python-dotenv` | Loads environment variables from `.env` |

---

## 🚀 Setup and Usage

### 1. Prerequisites

- Python compatible with the pinned dependencies. The development environment shown for this project uses Python 3.12.3.
- Git
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
- Internet access for downloading the course CSV during ingestion and for Gemini API requests

### 2. Clone the Repository

```bash
git clone https://github.com/KS-KARTHIK-05/personalized-learning-system.git
cd personalized-learning-system
```

### 3. Create a Virtual Environment

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project imports `pandas`, `requests`, and `datasets` in its Python files, but these packages are not explicitly listed in the supplied `requirements.txt`. If installation or execution reports a missing module, install the missing dependency and add it to `requirements.txt` so the environment can be reproduced.

For example:

```bash
pip install pandas requests datasets
```

### 5. Configure the Gemini API Key

Create a `.env` file in the project root.

#### Linux / macOS

```bash
cp .env.example .env
```

#### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

Set the API key in `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

Replace the placeholder with your own key. Keep `.env` private and never commit API credentials to Git.

### 6. Ingest the Course Dataset

Run:

```bash
python ingest_courses.py
```

The ingestion script downloads the Coursera course CSV from its configured Hugging Face URL, combines course titles, skills, and descriptions into text documents, generates embeddings, and stores the documents and metadata in ChromaDB.

The configured storage location and collection are:

- **Database directory:** `./vector_db`
- **Collection:** `learning_courses`
- **Embedding model:** `all-MiniLM-L6-v2`

The ingestion script uses batches of 500 records. The database directory is relative to the working directory from which the script is run.

### 7. Start the Streamlit Application

```bash
streamlit run app.py --server.fileWatcherType none
```

Open the URL shown in the terminal. Streamlit's default local address is:

[http://localhost:8501](http://localhost:8501)

### 8. Generate a Learning Roadmap

1. Upload a PDF resume or enter your current skills manually.
2. Enter your target technical role or skill track.
3. Select **Generate My Learning Path**.
4. Review the identified skills and calculated skill gaps.
5. Review the four-phase learning roadmap.
6. Download the roadmap PDF if needed.
7. Select **Evaluation** to run the configured Ragas evaluation workflow.

---

## 📂 Project Structure

```text
personalized-learning-system/
├── app.py
├── pipeline.py
├── gap_analyzer.py
├── ingest_courses.py
├── query_courses.py
├── evaluator.py
├── check_db.py
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

### File Descriptions

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface, PDF resume parsing, roadmap display, PDF export, and evaluation controls |
| `pipeline.py` | Connects the skill-gap analyzer and course search engine |
| `gap_analyzer.py` | Calls Gemini to extract existing skills and calculate skill gaps |
| `ingest_courses.py` | Downloads, processes, embeds, and stores course records |
| `query_courses.py` | Embeds skill-gap queries and retrieves matching courses from ChromaDB |
| `evaluator.py` | Runs the configured Ragas evaluation and prepares a results DataFrame |
| `check_db.py` | Database-checking utility |
| `requirements.txt` | Python dependency versions and constraints |
| `.env.example` | Example environment-variable configuration |
| `.gitignore` | Defines files and directories excluded from Git |
| `LICENSE` | Project license |
| `README.md` | Project documentation |

---

## 📈 Evaluation

The dashboard displays three metric fields:

| Metric | Meaning | Implementation status |
|---|---|---|
| Faithfulness | Assesses whether the generated answer is supported by retrieved context | Included in the configured Ragas evaluation |
| Answer Relevancy | Assesses how relevant the answer is to the input query | Included in the configured Ragas evaluation |
| Context Precision | Assesses the relevance and ranking of retrieved context | Displayed by the dashboard, but supplied as a fallback by the current evaluator |

### Evaluation limitations

The current `evaluator.py` contains fallback behavior when Ragas is unavailable or evaluation raises an exception. It also replaces missing or zero values for some fields with preset values.

Consequently:

- A displayed number does not necessarily represent a successful Ragas evaluation.
- Preset fallback values must not be reported as measured model performance.
- The current configured metric list does not include Context Precision, even though the dashboard displays that field.
- Record genuine evaluation results only after confirming that Ragas completed successfully and that the returned metrics were calculated from the evaluation data.

---

## 🔐 Security and Data Management

### API Credentials

- Keep the Gemini API key in `.env`.
- Do not commit `.env` or expose its contents in screenshots, logs, or public repositories.
- Keep `.env.example` limited to placeholder values.
- Rotate the key if it is accidentally exposed.

### Local Data and Generated Files

The project uses a local persistent vector database. If your `.gitignore` matches your actual generated paths, consider entries such as:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
vector_db/
chroma_db/
*.sqlite3
```

Only ignore files and directories that should remain local. Keep required source files and configuration templates in version control.

### Resume Privacy

Uploaded resumes can contain personal information. Process them only in an environment you trust, and avoid retaining or sharing extracted resume content unnecessarily.

---

## 🎯 Use Cases

- AI and machine-learning career planning
- Software engineering learning roadmaps
- Technical skill-gap identification
- Course discovery based on missing skills
- Career-oriented learning recommendations
- Downloadable learning-plan reports

---

## 👨‍💻 Author

**Karthik K.S**

*AI & Data Science Engineer*

- **GitHub:** [KS-KARTHIK-05](https://github.com/KS-KARTHIK-05)
- **LinkedIn:** [Karthik K.S](https://www.linkedin.com/in/karthik-ks-1k)

---

## 📄 License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
