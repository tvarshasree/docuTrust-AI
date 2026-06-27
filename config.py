import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

COLLECTION_NAME = "docutrust"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")


def _make_fresh_store() -> Chroma:
    import chromadb
    raw = chromadb.EphemeralClient()
    return Chroma(client=raw, collection_name=COLLECTION_NAME, embedding_function=embeddings)


vector_store = _make_fresh_store()


def load_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()  # use splitext, not string .lower() on full path

    # PDF
    if ext == ".pdf":
        try:
            return PyPDFLoader(file_path).load()
        except Exception as e:
            raise RuntimeError(f"PDF failed: {e}") from e

    # CSV
    elif ext == ".csv":
        try:
            return CSVLoader(file_path, encoding="utf-8").load()
        except Exception as e:
            raise RuntimeError(f"CSV failed: {e}") from e

    # DOCX / DOC
    elif ext in (".docx", ".doc"):
        try:
            from docx import Document as DocxDoc   # python-docx
            doc = DocxDoc(file_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
            if not text:
                raise RuntimeError("No text found in DOCX — file may be empty or image-only.")
            return [Document(page_content=text, metadata={"source": file_path, "page": 1})]
        except ImportError:
            raise RuntimeError("DOCX failed: python-docx not installed — run: pip install python-docx")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"DOCX failed: {e}") from e

    # PPTX / PPT
    elif ext in (".pptx", ".ppt"):
        try:
            from pptx import Presentation          # python-pptx
            prs = Presentation(file_path)
            docs = []
            for i, slide in enumerate(prs.slides, start=1):
                parts = [
                    shape.text.strip()
                    for shape in slide.shapes
                    if hasattr(shape, "text") and shape.text.strip()
                ]
                if parts:
                    docs.append(Document(
                        page_content="\n".join(parts),
                        metadata={"source": file_path, "page": i}
                    ))
            return docs
        except ImportError:
            raise RuntimeError("PPTX failed: python-pptx not installed — run: pip install python-pptx")
        except Exception as e:
            raise RuntimeError(f"PPTX failed: {e}") from e

    # MARKDOWN
    elif ext == ".md":
        try:
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            return [Document(page_content=text, metadata={"source": file_path, "page": 1})]
        except Exception as e:
            raise RuntimeError(f"Markdown failed: {e}") from e

    # HTML / HTM
    elif ext in (".html", ".htm"):
        try:
            from bs4 import BeautifulSoup
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                soup = BeautifulSoup(fh, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if not text:
                raise RuntimeError("No text extracted from HTML.")
            return [Document(page_content=text, metadata={"source": file_path, "page": 1})]
        except ImportError:
            raise RuntimeError("HTML failed: beautifulsoup4 not installed — run: pip install beautifulsoup4")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"HTML failed: {e}") from e

    # UNSUPPORTED 
    else:
        raise RuntimeError(f"Unsupported file type: '{ext}'. Supported: pdf, csv, docx, pptx, md, html")


def _clear_documents_folder():
    doc_dir = "./documents"
    if os.path.exists(doc_dir):
        for fname in os.listdir(doc_dir):
            try:
                os.remove(os.path.join(doc_dir, fname))
            except Exception as e:
                print(f"⚠️  Could not delete {fname}: {e}")
    os.makedirs(doc_dir, exist_ok=True)
    print("  Cleared ./documents folder on startup.")


_clear_documents_folder()