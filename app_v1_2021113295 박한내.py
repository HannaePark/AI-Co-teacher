import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# 1. 설정 및 교육학 지표 로직 (SRS FR4 반영)
st.set_page_config(page_title="AI Co-teacher", layout="wide")

def calculate_metrics(df):
    # SRL(자기조절학습) 지수
    df['SRL_Index'] = (df['time_spent']/120 + df['interactions'] + df['success']) / 3
    # ZPD(근접발달영역) 계산: 평균 ± 표준편차
    avg = df['quiz_score'].mean()
    std = df['quiz_score'].std()
    zpd_range = (avg - std, avg + std)
    # VARK 유형 분류
    df['VARK'] = np.where(df['clicks'] > df['clicks'].median(), 'Visual', 'Read/Write')
    return df, zpd_range

st.title("🍎 AI Co-teacher: AI 학습 분석 대시보드")
st.markdown("영어영문학과 박한내 (2021113295)")

# 데이터 로드 (SRS 변수 포함)
data = {
    'student_id': [f'STU_{i:03d}' for i in range(1, 11)],
    'quiz_score': [85, 42, 90, 35, 77, 55, 48, 92, 60, 38], # 의도적 위험군 포함
    'clicks': np.random.randint(20, 150, 10),
    'time_spent': np.random.randint(30, 180, 10),
    'interactions': np.random.randint(1, 15, 10),
    'success': [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)
df, (zpd_low, zpd_high) = calculate_metrics(df)

# --- [ 1: ZPD 구간 시각화] ---
st.subheader("📊 1. 성과 분석 및 ZPD 구간 (Vygotsky 이론 적용)")
fig = px.bar(df, x='student_id', y='quiz_score', color='quiz_score', 
             title="학생별 성적 (점선 사이: ZPD 적정 난이도 구간)")
# ZPD 라인 추가 (이 부분이 안 보였을 수 있습니다)
fig.add_hline(y=zpd_low, line_dash="dash", line_color="green", annotation_text=f"ZPD 하한 ({zpd_low:.1f})")
fig.add_hline(y=zpd_high, line_dash="dash", line_color="red", annotation_text=f"ZPD 상한 ({zpd_high:.1f})")
st.plotly_chart(fig, use_container_width=True)

# --- [ 2: 위험 학생 및 개인화 추천 (FR5/UC-002)] ---
st.divider()
st.subheader("🚨 2. 위험 학생 자동 감지 및 개인화 추천")
# SRS 기준: 점수 < 50 또는 참여도(SRL) < 0.4
df['Status'] = np.where((df['quiz_score'] < 50) | (df['SRL_Index'] < 0.4), '⚠️ 고위험', '✅ 정상')

col1, col2 = st.columns([1, 1])
with col1:
    st.write("**실시간 위험 학생 명단**")
    risk_df = df[df['Status'] == '⚠️ 고위험']
    st.dataframe(risk_df[['student_id', 'quiz_score', 'SRL_Index', 'Status']])

with col2:
    st.write("**개인화 추천 (UC-003)**")
    target = st.selectbox("학생 선택", df['student_id'].unique())
    info = df[df['student_id'] == target].iloc[0]
    if info['Status'] == '⚠️ 고위험':
        st.error(f"[{target}] 학생은 {info['VARK']}형 보충 자료와 1:1 면담이 시급합니다.")
    else:
        st.success(f"[{target}] 학생은 정상 궤도입니다. {info['VARK']}형 심화 과제를 추천합니다.")

# --- [ 3: AI 예측 분석 (FR3/RandomForest)] ---
st.divider()
st.subheader("🤖 3. AI 학습 성공 요인 예측 분석")
X = df[['quiz_score', 'clicks', 'time_spent', 'interactions', 'SRL_Index']]
y = df['success']
rf = RandomForestClassifier(n_estimators=50).fit(X, y)

importance_df = pd.DataFrame({
    '특성': X.columns,
    '중요도': rf.feature_importances_
}).sort_values('중요도', ascending=False)

fig_ai = px.bar(importance_df, x='중요도', y='특성', orientation='h', 
                title="AI가 분석한 성공 기여도 (어떤 데이터가 성패를 결정하는가?)")
st.plotly_chart(fig_ai, use_container_width=True)
