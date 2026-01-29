import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "rag_docs"

pdf_files = [
    DATA_DIR / "2040_report.pdf",
    DATA_DIR / "OneNYC-2050-Summary.pdf"
]

for p in pdf_files:
    if not p.exists():
        raise FileNotFoundError(f"PDF 파일이없습니다: {p}")
    
documents = []
for pdf_path in pdf_files:
    documents.extend(PyPDFLoader(str(pdf_path)).load())

print("로드된 페이지수: ", len(documents))
if len(documents) == 0:
    raise RuntimeError("PDF 로딩 결과가 0페이지 입니다.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

embeddings.embed_query("test")
print("임베딩 호출 OK")

if PERSIST_DIR.exists():
    shutil.rmtree(PERSIST_DIR)

PERSIST_DIR.mkdir(parents=True, exist_ok=True)

db = Chroma.from_documents(
    documents=splits,
    embedding = embeddings,
    persist_directory = str(PERSIST_DIR),
    collection_name = COLLECTION_NAME,
)

count = db._collection.count()
print("문서 수: ", count)

if count == 0:
    raise RuntimeError("Chroma에 문서가 저장되지 않았습니다. (count=0)")

print("Chroma DB 생성 완료")
print("저장 경로: ", PERSIST_DIR)
print("컬렉션 이름: ", COLLECTION_NAME)