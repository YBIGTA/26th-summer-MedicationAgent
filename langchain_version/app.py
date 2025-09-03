#!/usr/bin/env python3
"""
💊 Medication Agent - Tool 기반 버전
LangChain Tool을 활용한 약물 정보 챗봇
"""

import streamlit as st

from ui_components import (
    initialize_system, render_sidebar, render_chat_interface,
    render_medication_management, render_calendar_export
)


def main():
    """메인 애플리케이션 함수"""
    # 시스템 초기화
    initialize_system()
    
    # 메인 페이지 제목
    st.title("💊 Medication Agent - AI 기반 약물 정보 챗봇")
    st.markdown("---")
    
    # 사이드바 렌더링
    render_sidebar()
    
    # 메인 콘텐츠 영역
    tab1, tab2, tab3 = st.tabs(["💬 챗봇", "💊 복약 관리", "📅 캘린더"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_medication_management()
    
    with tab3:
        render_calendar_export()
    
    # 페이지 하단 정보
    st.markdown("---")
    st.markdown("**YBIGTA 26기 여름방학 프로젝트** - AI 기반 약물 정보 챗봇")
    st.markdown("*이 애플리케이션은 교육 및 참고 목적으로만 제공됩니다. 의학적 조언이나 진단을 대체할 수 없습니다.*")


if __name__ == "__main__":
    main()
