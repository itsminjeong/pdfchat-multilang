import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile
import os

# 페이지 설정
st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")

# 환경 변수 불러오기
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 언어 선택 UI 추가 (사이드바)
lang = st.sidebar.selectbox("Language / 언어 선택", ("한국어", "English"))

# 다국어 메시지 딕셔너리 정의
messages = {
    "한국어": {
        "page_title": "PDF 기반 GPT 챗봇",
        "page_markdown": "PDF 파일을 업로드하고 자유롭게 질문해보세요!",
        "file_uploader": "PDF 파일 업로드",
        "input_label": "질문을 입력하세요:",
        "input_placeholder": "예: 2페이지 요약해줘",
        "send_button": "보내기",
        "init_greeting_user": "안녕하세요!",
        "init_greeting_bot": "안녕하세요! 업로드한 문서에 대해 질문해보세요."
    },
    "English": {
        "page_title": "GPT-powered PDF Chatbot",
        "page_markdown": "Upload a PDF and ask questions freely!",
        "file_uploader": "Upload PDF file",
        "input_label": "Enter your question:",
        "input_placeholder": "e.g., Summarize page 2",
        "send_button": "Send",
        "init_greeting_user": "Hello!",
        "init_greeting_bot": "Hello! Ask me anything about the uploaded document."
    }
}

# 선택한 언어에 따른 메시지 불러오기
msg = messages[lang]

# 페이지 제목과 설명 출력
st.title(msg["page_title"])
st.markdown(msg["page_markdown"])

# PDF 파일 업로드 받기
uploaded_file = st.sidebar.file_uploader(msg["file_uploader"], type="pdf")

# 파일 업로드가 완료되었을 때
if uploaded_file:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # 1. PDF 문서 로드
    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    # 2. 문서 임베딩 후 벡터 저장소 생성
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectors = FAISS.from_documents(data, embeddings)

    # 3. ConversationalRetrievalChain 구성
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0.0, api_key=openai_api_key),
        retriever=vectors.as_retriever()
    )

    # 4. 사용자 질문 처리 함수 정의
    def conversational_chat(query):
        # 선택한 언어에 따라 GPT 응답 언어를 명확히 지시
        if lang == "English":
            prompt = f"Regardless of the question language, please answer ONLY in English.\n\nUser: {query}"
        elif lang == "한국어":
            prompt = f"질문이 어떤 언어든 관계없이 반드시 한국어로만 답변해 주세요.\n\n사용자 질문: {query}"
        else:
            prompt = query  # 기타 처리 없이 그대로 전달

        result = chain({
            "question": prompt,
            "chat_history": st.session_state["history"]
        })

        # 원래 사용자 입력과 GPT 응답을 대화 기록에 저장
        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]

    # 세션 상태 초기화
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = [msg["init_greeting_bot"]]
    if "past" not in st.session_state:
        st.session_state["past"] = [msg["init_greeting_user"]]

    # 사용자 입력 영역
    with st.container():
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(msg["input_label"], placeholder=msg["input_placeholder"])
            submit_button = st.form_submit_button(label=msg["send_button"])

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    # 대화 히스토리 출력
    if st.session_state["generated"]:
        with st.container():
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="fun-emoji")
                message(st.session_state["generated"][i], key=f"{i}_bot", avatar_style="bottts")
