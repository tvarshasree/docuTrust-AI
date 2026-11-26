# DocuTrust AI

### Evidence-Grounded Enterprise RAG with Self-Correction, Relevance Verification, and Audit Citations

DocuTrust AI is an evidence-grounded enterprise Retrieval-Augmented Generation (RAG) system designed to answer questions from organisation-specific documents without relying on unsupported external knowledge.

Instead of directly passing retrieved chunks to an LLM, DocuTrust introduces a verification stage that grades retrieved evidence using a Cross-Encoder. If the retrieved evidence is insufficient, the system automatically broadens the search and retries before returning a controlled failure response.

The result is a RAG pipeline that prioritises **grounded answers over plausible answers**.

---

## Key Features

* **Multi-format document ingestion** for PDF, CSV and supported office documents
* **Semantic document chunking** for context-aware retrieval
* **Dense vector retrieval** using BGE embeddings and ChromaDB
* **Cross-Encoder relevance grading** to verify retrieved evidence before generation
* **Self-correcting retrieval loop** that retries with a broader query when evidence is insufficient
* **Evidence-grounded generation** using Gemini 2.5 Flash
* **Source citations** containing document names, pages/rows and relevance scores
* **Controlled fallback behaviour** when no sufficiently relevant evidence is found
* **LangGraph orchestration** for explicit retrieval, verification, generation and failure states
* **FastAPI backend** with WebSocket-based conversational interaction
* **Evaluation framework** measuring answer correctness, groundedness, hallucination rate and pipeline success

---
## Architecture

### 1. Document ingestion

Uploaded documents are loaded and converted into LangChain document objects.

The system supports document ingestion through loaders including:

* PDF
* CSV
* DOCX
* PPTX
* Markdown
* other supported text-based formats

Documents are then split using a semantic chunking strategy.

### 2. Embedding and vector storage

Document chunks are embedded using:

**BAAI/bge-small-en-v1.5**

The embeddings are stored in a Chroma vector database for similarity retrieval.

### 3. Retrieval

For each user query, DocuTrust retrieves the top 7 candidate chunks from the vector store using BAAI asymmetric query instructions (`"Represent this sentence for searching relevant passages: "`).

Each retrieved chunk retains source metadata such as:

* document filename
* page number / row information

This metadata is later used to construct the audit evidence attached to the generated response.

### 4. Relevance verification

Retrieved chunks are passed through:

**cross-encoder/ms-marco-MiniLM-L-6-v2**

The Cross-Encoder scores query/document pairs for relevance and computes a calibrated Sigmoid confidence score.

The pipeline applies a calibrated relevance threshold of **0.80** (~69% confidence). For multi-part and comparative queries, the pipeline also supports **cumulative multi-chunk verification** (passing if the top two chunks jointly demonstrate strong positive support). If retrieved evidence does not meet these criteria, the system does not immediately generate an answer.

### 5. Self-correction

When retrieval fails verification, DocuTrust performs a second retrieval attempt using intelligent query reformulation. In online mode, Gemini 2.5 Flash reformulates the query with enterprise synonyms; in offline evaluation mode, stop-word stripping extracts core entity keywords to align query vectors with document chunks.

```text
Original query
      ↓
Dense Retrieval (Top 7)
      ↓
Cross-Encoder Relevance Grading
      ↓
Below threshold?
      ↓
Query Reformulation (LLM / Entity Keywords)
      ↓
Second retrieval
      ↓
Re-grade evidence
```

If the second attempt also fails, the system returns a controlled failure message instead of fabricating an answer.

### 6. Evidence-grounded generation

When sufficient evidence is found, the top-ranked chunks are passed to Gemini 2.5 Flash.

The generation prompt explicitly instructs the model to:

* use only the verified context
* avoid outside knowledge
* state when verification evidence is unavailable
* produce a concise professional response

The final response includes an audit section containing the source document, page/row and relevance score.

### 7. Application layer

The backend is implemented using **FastAPI**.

The application provides:

* document upload
* document processing
* knowledge-base reset
* WebSocket chat
* real-time pipeline logs
* generated answers with evidence citations

The LangGraph workflow explicitly models retrieval, grading, generation and failure states.

---

## Technology Stack

| Component               | Technology                              |
| ----------------------- | --------------------------------------- |
| Backend                 | FastAPI                                 |
| Workflow orchestration  | LangGraph                               |
| RAG framework           | LangChain                               |
| Vector database         | ChromaDB                                |
| Embeddings              | BAAI/bge-small-en-v1.5                  |
| Relevance model         | MS MARCO MiniLM Cross-Encoder           |
| LLM                     | Gemini 2.5 Flash                        |
| Document processing     | PyPDF, python-docx, python-pptx, pandas |
| Real-time communication | WebSockets                              |
| Language                | Python                                  |

---
## Evaluation

DocuTrust includes a dedicated evaluation pipeline rather than relying only on qualitative examples.

The evaluation framework runs queries against organisation-specific document collections and records:

* Answer Correctness
* Groundedness
* Hallucination Rate
* Pipeline Pass Rate
* Number of Retrieval Attempts
* Results by organisation
* Results by file type
* Results by query

The generated evaluation artefacts are stored under `evaluation_results/`.

### Evaluation outputs

```text
evaluation_results/
├── evaluation_report.txt
├── full_results.csv
├── overall_metrics.csv
├── summary_by_filetype.csv
├── summary_by_org.csv
└── summary_by_org_filetype.csv
```

### Benchmark Results (5,008 Evaluated Queries)

#### Overall System Performance

| Metric | Score | Target Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Answer Correctness** | **89.6%** (0.8959) | $\ge 70.0\%$ | ✅ PASS |
| **Groundedness** | **67.3%** (0.6732) | $\ge 60.0\%$ | ✅ PASS |
| **Hallucination Rate** | **10.4%** (0.1041) | $\le 20.0\%$ | ✅ PASS |
| **Pipeline Pass Rate** | **93.7%** (0.9367) | $\ge 80.0\%$ | ✅ PASS |
| **Retry Rate** | **6.5%** (0.0653) | $\le 30.0\%$ | ✅ PASS |
| **Fail Rate** | **6.3%** (0.0633) | $\le 10.0\%$ | ✅ PASS |

#### Performance by Document Format

| Format | Total Queries | Answer Correctness | Groundedness | Hallucination Rate |
| :--- | :---: | :---: | :---: | :---: |
| **DOCX** | 1,252 | **91.1%** | 68.7% | **8.9%** |
| **Markdown (.md)** | 1,252 | **90.2%** | 64.3% | **9.8%** |
| **HTML** | 1,252 | **89.0%** | 67.8% | **11.0%** |
| **PDF** | 1,252 | **88.0%** | 68.5% | **12.0%** |

#### Performance by Organization

| Organization | Answer Correctness | Groundedness | Hallucination Rate |
| :--- | :---: | :---: | :---: |
| **Cloudway 24** | **96.6%** | 73.9% | **3.4%** |
| **Velvera Technologies** | **90.6%** | 61.5% | **9.4%** |
| **Aventro Motors** | **89.4%** | 67.5% | **10.6%** |
| **ZX Bank** | **88.5%** | 74.0% | **11.5%** |
| **Cendara University** | **83.7%** | 60.8% | **16.3%** |

For reproducibility, the evaluation script can be run directly:

```bash
python evaluate.py --no-llm --skip-pptx
```

---

## Installation

### Prerequisites

* Python 3.10+
* Google Gemini API key

Clone the repository:

```bash
git clone https://github.com/tvarshasree/docuTrust-AI.git
cd docuTrust-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` or API credentials to the repository.

---

## Running the Application

Start the FastAPI server:

```bash
python app.py
```

The application runs on:

```text
http://localhost:8000
```

Upload the documents you want the system to use as its knowledge base, then submit questions through the application.

---

## Example Workflow

### Upload

Upload organisation documents such as:

```text
HR_Policy.pdf
Leave_Policy.pdf
Employee_Handbook.docx
Benefits.csv
```

### Ask

```text
What is the maximum number of annual leave days an employee can carry forward?
```

### Pipeline

```text
Query
  ↓
Vector Retrieval
  ↓
Top 5 Chunks
  ↓
Cross-Encoder Relevance Grading
  ↓
Evidence Verified
  ↓
Gemini 2.5 Flash
  ↓
Answer + Evidence Sources
```

Example evidence format:

```text
Audit Evidence Sources:

[1] Leave_Policy.pdf — Page 12 (score: ...)
[2] Employee_Handbook.docx — Page 7 (score: ...)
[3] Benefits.csv — Row ... (score: ...)
```

If sufficient evidence cannot be found, DocuTrust returns a controlled failure response rather than generating an unsupported answer.

---

## Design Decisions

### Why use a Cross-Encoder?

Vector similarity is useful for candidate retrieval but does not guarantee that a retrieved chunk directly supports the user's question.

DocuTrust therefore uses a Cross-Encoder as a second-stage relevance verifier before generation.

### Why use LangGraph?

The workflow contains conditional behaviour:

```text
retrieve → grade
              │
       ┌──────┼──────┐
       ↓      ↓      ↓
   generate retrieve fail
```

LangGraph makes these states and transitions explicit and allows the retry/failure behaviour to be represented as part of the application workflow.

### Why return a failure response?

For enterprise policy questions, an incorrect confident answer can be worse than no answer.

DocuTrust therefore treats insufficient evidence as a valid system outcome.

---

## Limitations

This project is a research/engineering prototype rather than a production enterprise deployment.

Current limitations include:

* the vector store is configured using an ephemeral Chroma client for the application
* the relevance threshold is currently configured as a fixed value
* the system is designed around a single active knowledge-base session
* authentication and authorisation are not implemented
* uploaded files are handled locally by the application
* evaluation quality depends on the supplied test dataset and reference facts
* LLM output is still dependent on the underlying model and prompt

These limitations would need to be addressed before production deployment.

---

## Future Improvements

* Persistent multi-tenant vector storage
* Authentication and role-based access control
* Configurable relevance thresholds
* Hybrid BM25 + dense retrieval
* Reranker benchmarking
* Query classification and routing
* Better citation validation
* Automated regression evaluation in CI/CD
* Docker deployment
* Observability and latency tracking
* Evaluation against larger enterprise datasets
* Human feedback loop for retrieval failures

---

## Project Highlights

DocuTrust demonstrates an end-to-end approach to building a more reliable RAG system:

```text
Document Processing
        ↓
Semantic Chunking
        ↓
Dense Retrieval
        ↓
Cross-Encoder Verification
        ↓
Self-Correction
        ↓
Evidence-Grounded Generation
        ↓
Citations + Audit Trail
        ↓
Quantitative Evaluation
```

The primary design goal is not simply to generate an answer, but to **generate an answer only when the system can identify sufficiently relevant supporting evidence**.

## Dataset Attribution

This project uses the **RAG-Multi-Corpus** benchmark dataset for document retrieval and RAG evaluation.

> Uday Allu, Sonu Kedia, and Tanmay Odapally, “RAG-Multi-Corpus: A Multi-Organisation Multi-Format RAG Benchmark Dataset,” 2025. Available at: https://github.com/udayallu/RAG-Multi-Corpus

The dataset is a synthetic, multi-format, multi-domain enterprise corpus designed for evaluating RAG systems across document parsing, retrieval quality, answer correctness, hallucination detection, and grounding. It contains documents from five fictional organisations across multiple industry domains and includes a curated set of 786 evaluation queries.

### Dataset Usage

The dataset is used in DocuTrust AI to evaluate:

* Document ingestion across multiple file formats
* Semantic retrieval quality
* Cross-Encoder relevance verification
* RAG answer correctness
* Groundedness and hallucination behaviour
* Retrieval performance across different organisations and document types

The dataset itself was **not created as part of this project**. DocuTrust AI's contribution is the RAG pipeline, retrieval verification, self-correction workflow, evidence-grounded generation, and evaluation framework built on top of the benchmark dataset.

### Citation

```bibtex
@misc{Allu2025RAGMultiCorpus,
  author       = {Uday Allu and Sonu Kedia and Tanmay Odapally},
  title        = {RAG-Multi-Corpus: A Multi-Organisation Multi-Format RAG Benchmark Dataset},
  year         = {2025},
  howpublished = {\url{https://github.com/udayallu/RAG-Multi-Corpus}}
}
```

Dataset repository: https://github.com/udayallu/RAG-Multi-Corpus

### Author

Varsha Sree

GitHub: https://github.com/tvarshasree

