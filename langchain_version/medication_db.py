"""
 복약 정보 데이터베이스 관리 모듈
Supabase PostgreSQL을 직접 연결하여 사용자별 복약 체크리스트 관리
"""

import os
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

class MedicationDatabase:
    def __init__(self):
        """PostgreSQL 데이터베이스 연결 초기화"""
        self.db_url = os.getenv("SUPABASE_DB_URL")
        
        if not self.db_url:
            st.error("❌ SUPABASE_DB_URL 환경변수가 설정되지 않았습니다!")
            st.info("💡 .env 파일에 SUPABASE_DB_URL을 추가해주세요.")
            return
        
        try:
            # URL 파싱하여 개별 파라미터로 연결
            from urllib.parse import urlparse
            
            parsed = urlparse(self.db_url)
            
            # 연결 파라미터 구성
            conn_params = {
                'host': parsed.hostname,
                'port': parsed.port or 5432,
                'database': parsed.path[1:],  # 앞의 '/' 제거
                'user': parsed.username,
                'password': parsed.password
            }
            
            self.conn = psycopg2.connect(**conn_params)
            st.success("✅ Supabase PostgreSQL 연결 성공!")
        except Exception as e:
            st.error(f"❌ 데이터베이스 연결 실패: {e}")
            return
    
    def get_or_create_user(self, username: str, email: str = None) -> Optional[Dict]:
        """사용자 조회 또는 생성"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # 기존 사용자 조회
            cursor.execute("SELECT * FROM users WHERE name = %s", (username,))
            user = cursor.fetchone()
            
            if user:
                cursor.close()
                return dict(user)
            
            # 새 사용자 생성
            cursor.execute(
                "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING *",
                (username, email)
            )
            new_user = cursor.fetchone()
            self.conn.commit()
            cursor.close()
            
            if new_user:
                st.success(f"✅ 새 사용자 '{username}' 생성됨!")
                return dict(new_user)
            
        except Exception as e:
            st.error(f"❌ 사용자 생성/조회 실패: {e}")
        
        return None

    def add_medication(self, user_id: int, medication_data: Dict) -> Optional[Dict]:
        """복약 정보 추가"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                INSERT INTO user_medications (
                    user_id, medication_name, morning, lunch, dinner,
                    before_meal, after_meal, start_date, end_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """, (
                user_id, medication_data['medication_name'], 
                medication_data.get('morning', False), medication_data.get('lunch', False),
                medication_data.get('dinner', False), medication_data.get('before_meal', False),
                medication_data.get('after_meal', False), medication_data['start_date'],
                medication_data.get('end_date')
            ))
            
            new_medication = cursor.fetchone()
            self.conn.commit()
            cursor.close()
            
            if new_medication:
                st.success(f"✅ '{medication_data['medication_name']}' 복약 정보가 추가되었습니다!")
                return dict(new_medication)
            
        except Exception as e:
            st.error(f"❌ 복약 정보 추가 실패: {e}")
        
        return None
    
    def get_user_medications(self, user_id: int) -> List[Dict]:
        """사용자의 복약 정보 조회"""
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM user_medications 
                WHERE user_id = %s
                ORDER BY id DESC
            """, (user_id,))
            
            medications = cursor.fetchall()
            cursor.close()
            
            return [dict(med) for med in medications]
            
        except Exception as e:
            st.error(f"❌ 복약 정보 조회 실패: {e}")
            return []
    
    def update_medication(self, medication_id: int, update_data: Dict) -> bool:
        """복약 정보 수정"""
        try:
            cursor = self.conn.cursor()
            
            # 업데이트할 필드들
            set_fields = []
            values = []
            
            for key, value in update_data.items():
                if key != 'id':  # id는 제외
                    set_fields.append(f"{key} = %s")
                    values.append(value)
            
            values.append(medication_id)
            
            sql = f"""
                UPDATE user_medications 
                SET {', '.join(set_fields)}
                WHERE id = %s
            """
            
            cursor.execute(sql, values)
            self.conn.commit()
            cursor.close()
            
            st.success("✅ 복약 정보가 수정되었습니다!")
            return True
        
        except Exception as e:
            st.error(f"❌ 복약 정보 수정 실패: {e}")
            return False
    
    def delete_medication(self, medication_id: int) -> bool:
        """복약 정보 삭제"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                DELETE FROM user_medications 
                WHERE id = %s
            """, (medication_id,))
            
            self.conn.commit()
            cursor.close()
            
            st.success("✅ 복약 정보가 삭제되었습니다!")
            return True
            
        except Exception as e:
            st.error(f"❌ 복약 정보 삭제 실패: {e}")
            return False
    
    def close_connection(self):
        """데이터베이스 연결 종료"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

# 사용 예시 및 테스트 함수
def test_database():
    """데이터베이스 테스트"""
    db = MedicationDatabase()
    
    if not hasattr(db, 'conn') or not db.conn:
        return
    
    try:
        # 테스트 사용자 생성
        user = db.get_or_create_user("test_user", "test@example.com")
        
        if user:
            # 테스트 복약 정보 추가
            medication_data = {
                'medication_name': '판테놀',
                'morning': True,
                'lunch': True,
                'dinner': True,
                'before_meal': True,
                'start_date': datetime.now().date().isoformat(),
                'end_date': None
            }
            
            med = db.add_medication(user['id'], medication_data)
            
            if med:
                st.write("✅ 테스트 복약 정보 추가 성공!")
                
                # 사용자 복약 정보 조회
                medications = db.get_user_medications(user['id'])
                st.write(f"📋 사용자 복약 정보: {len(medications)}개")
    
    finally:
        db.close_connection()

if __name__ == "__main__":
    test_database()