import os
import streamlit as st

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title='RAG 챗봇', layout='wide')
st.title("RAG 문서 기반 챗봇")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY 환경변수가 필요합니다.")
    st.stop()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory = PERSIST_DIR,
    collection_name = COLLECTION_NAME,
    embedding_function = embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":4})

def format_docs(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source")
        page = d.metadata.get("page")
        lines.append(f"(source={src}, page={page}) {d.page_content}")
        return "\n\n".join(lines)
    
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
parser = StrOutputParser()

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "문서 기반 질의응답 도우미입니다. 제공된 컨텍스트에 근거하여 답변합니다. "
     "컨텍스트에 없는 내용은 '문서에서 확인되지 않았습니다.'라고 답변합니다. "
     "답변은 8줄 이내로 작성합니다."),
    ("user", "질문: {question}\n\n컨텍스트:\n{context}\n\n답변:")
])

answer_chain = answer_prompt | llm | parser

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("질문을 입력하세요.")

if user_q:
    st.session_state.messages.append({"role":"user", "content":user_q})
    with st.chat_message("user"):
        st.write(user_q)

    docs = retriever.invoke(user_q)
    context = format_docs(docs)
    answer = answer_chain.invoke({"question":user_q, "context":context})
    
    st.session_state.messages.append({"role":"assistant", "content":answer})
    with st.chat_message("assistant"):
        st.write(answer)

st.caption(f"컬렉션 이름 : {vectorstore._collection.name}")
st.caption(f"컬렉션 문서 수: {vectorstore._collection.count()}")
st.caption(f"persist 경로(절대): {os.path.abspath(PERSIST_DIR)}")