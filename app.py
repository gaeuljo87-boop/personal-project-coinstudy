import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from openai import OpenAI  # OpenAI 라이브러리 추가

# 1. 데이터 수집 (최대치 로드 및 캐싱)
@st.cache_data
def get_crypto_data(symbol='BTC/USDT', timeframe='1d', limit=2000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# 2. 변곡점 탐지 엔진
def find_pivots(df, order=10):
    if len(df) < (order * 2 + 1): return pd.DataFrame(), pd.DataFrame()
    peak_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
    valley_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
    return df.iloc[peak_idx], df.iloc[valley_idx]

# 3. 10대 기술적 패턴 판독
def check_patterns(p, v):
    patterns = []
    if len(v) >= 2 and len(p) >= 1:
        v1, v2 = v.iloc[-2]['low'], v.iloc[-1]['low']
        diff = abs(v1 - v2) / v1
        if diff <= 0.05:
            patterns.append({"name": "역대급 이중 바닥 (W)", "score": (1-diff)*100, "type": "Strong Bullish"})
    
    if len(p) >= 3:
        p1, p2, p3 = p.iloc[-3]['high'], p.iloc[-2]['high'], p.iloc[-1]['high']
        if p2 > p1 and p2 > p3: 
            patterns.append({"name": "거시적 헤드 앤 숄더", "score": 90.0, "type": "Strong Bearish"})

    if not patterns:
        patterns.append({"name": "장기 추세 진행 중", "score": 50.0, "type": "Trend"})
    return sorted(patterns, key=lambda x: x['score'], reverse=True)[:3]

# ==========================================
# 4. 웹 인터페이스 및 시각화
# ==========================================

st.set_page_config(page_title="BTC History Master", layout="wide")
st.title("📜 비트코인 연대기: 전체 역사 탐색 및 패턴 분석")

# --- [수정 포인트] 사이드바 설정에 API 키 입력란 추가 ---
st.sidebar.header("🔑 설정")
api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password", help="sk-...로 시작하는 키를 입력하세요.")

st.sidebar.header("📊 데이터 설정")
symbol = st.sidebar.selectbox("코인", ['BTC/USDT', 'ETH/USDT'])
timeframe = st.sidebar.selectbox("시간 단위", ['1h', '4h', '1d', '1w'], index=2)
use_log = st.sidebar.checkbox("로그 차트로 보기", value=True)

full_df = get_crypto_data(symbol, timeframe, limit=3000)

st.subheader(f"1️⃣ {symbol} 전체 역사 차트")
# --- Plotly 차트 (생략 없이 사용자님 코드 유지) ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=full_df['timestamp'], y=full_df['close'], mode='lines', name='Price', line=dict(color='orange', width=1.5)))
if use_log: fig.update_yaxes(type="log")
fig.update_layout(height=500, template="plotly_dark", xaxis=dict(rangeslider=dict(visible=True), type="date"), yaxis_title="Price (USDT)", dragmode='pan')
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

st.divider()
st.subheader("2️⃣ 역사적 특정 구간 집중 분석")

min_dt = full_df['timestamp'].min().to_pydatetime()
max_dt = full_df['timestamp'].max().to_pydatetime()

analysis_range = st.slider("분석할 범위를 드래그하세요", min_value=min_dt, max_value=max_dt, value=(max_dt - timedelta(days=365), max_dt), format="YYYY/MM/DD")

# --- 5. 분석 및 실시간 AI 리포트 생성 ---
if st.button("✨ 선택 구간 기술적 분석 실행"):
    if not api_key:
        st.error("❌ 왼쪽 사이드바에 OpenAI API Key를 먼저 입력해주세요!")
    else:
        sel_df = full_df[(full_df['timestamp'] >= pd.Timestamp(analysis_range[0])) & (full_df['timestamp'] <= pd.Timestamp(analysis_range[1]))].copy()
        peaks, valleys = find_pivots(sel_df, order=15)
        top_3 = check_patterns(peaks, valleys)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("### 🔍 감지된 거시 패턴")
            for i, pat in enumerate(top_3, 1):
                st.success(f"**{i}. {pat['name']}**")
                st.caption(f"신뢰도: {pat['score']:.1f}% | 관점: {pat['type']}")

        with col2:
            st.write("### 🤖 AI 역사적 관점 해설")
            high, low = sel_df['close'].max(), sel_df['close'].min()
            fibo_618 = low + (high - low) * 0.618
            
            # --- 실시간 AI 호출 로직 ---
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
            당신은 Investopedia 스타일의 기술적 분석 전문가입니다.
            분석 구간: {analysis_range[0]} ~ {analysis_range[1]}
            감지된 패턴: {top_3[0]['name']}
            구간 최고가: {high:,.0f} USDT, 최저가: {low:,.0f} USDT, 0.618 피보나치: {fibo_618:,.0f} USDT.
            
            이 데이터를 바탕으로 상승과 하락의 관점을 균형 있게 설명하고, 초보 투자자를 위한 교육적 멘트를 한국어로 작성해주세요.
            """
            
            with st.spinner('AI가 거대한 역사를 읽고 있습니다...'):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": "친절한 기술적 분석 강사입니다."},
                                  {"role": "user", "content": prompt}]
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"AI 호출 중 오류 발생: {e}")
            
            st.warning("⚠️ 본 서비스는 교육용이며 투자의 책임은 사용자 본인에게 있습니다.")
