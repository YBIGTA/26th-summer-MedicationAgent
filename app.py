import os
import time
import streamlit as st
from dotenv import load_dotenv

from agent import build_agent


st.set_page_config(page_title="복약 알림 & 부작용 안내 Agent", page_icon="💊")


def ensure_api_key() -> bool:
	load_dotenv()
	api_key = os.getenv("OPENAI_API_KEY")
	with st.sidebar:
		st.header("환경 설정")
		key_input = st.text_input(
			"OpenAI API Key",
			value=api_key or "",
			type="password",
			help=".env 또는 환경변수 OPENAI_API_KEY로도 설정할 수 있습니다.",
		)
		if key_input:
			os.environ["OPENAI_API_KEY"] = key_input.strip()
			return True
		return api_key is not None and len(api_key) > 0


st.title("💊 복약 알림 & 부작용 안내 Agent")
st.caption("LangChain + GPT-3.5-turbo 기반 약물 안내 챗봇")


if "messages" not in st.session_state:
	st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str}]

if "agent" not in st.session_state:
	st.session_state.agent = None


has_key = ensure_api_key()
if not has_key:
	st.warning("OpenAI API Key를 설정해 주세요.")
	st.stop()


# Build agent once
if st.session_state.agent is None:
	with st.spinner("에이전트를 초기화하는 중..."):
		st.session_state.agent = build_agent()


# Display chat history
for msg in st.session_state.messages:
	with st.chat_message(msg["role"]):
		st.markdown(msg["content"])


user_input = st.chat_input("약 관련 질문을 입력하세요 (예: 이부프로펜 복용법)")
if user_input:
	st.session_state.messages.append({"role": "user", "content": user_input})
	with st.chat_message("user"):
		st.markdown(user_input)

	with st.chat_message("assistant"):
		placeholder = st.empty()
		placeholder.markdown("생각 중...")
		try:
			agent = st.session_state.agent
			# Run the agent
			result = agent.invoke({"input": user_input})
			answer = result.get("output", "") if isinstance(result, dict) else str(result)
			placeholder.markdown(answer)
			st.session_state.messages.append({"role": "assistant", "content": answer})
		except Exception as e:
			placeholder.markdown(f"오류가 발생했습니다: {e}")

