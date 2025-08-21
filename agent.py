import os
from dotenv import load_dotenv

from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI

from tools import get_tools


def get_llm() -> ChatOpenAI:
	"""Return an OpenAI chat model configured for Korean medication guidance."""
	load_dotenv()
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise EnvironmentError(
			"OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 환경변수를 확인하세요."
		)
	# gpt-3.5-turbo as requested
	return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)


def build_agent():
	"""Create and return an AgentExecutor ready to answer medication questions in Korean."""
	llm = get_llm()
	tools = get_tools()
	system_message = (
		"너는 '복약 알림 & 부작용 안내' 전문가 에이전트야. 한국어로만 답변해. "
		"약 이름이나 성분이 언급되면, 필요시 제공된 도구를 사용해 약물 정보를 조회한 뒤, "
		"안전하고 실용적인 복용 안내/주의사항/부작용/상호작용을 간결히 제공해. "
		"추측하지 말고, 정보가 불충분하면 간단히 확인 질문을 먼저 해."
	)

	agent = initialize_agent(
		tools=tools,
		llm=llm,
		agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
		verbose=True,
		handle_parsing_errors=True,
		agent_kwargs={"system_message": system_message},
	)
	return agent


__all__ = ["build_agent", "get_llm"]

