import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from openai import OpenAI
import requests
import time

# 1. 데이터 수집 (CoinGecko API 사용 - 지역 제한 없음)
@st.cache_data(ttl=600) # 10분마다 캐시 갱신
def get_crypto_data(symbol='bitcoin', timeframe='1d', limit=365):
    try:
        # CoinGecko API 사용 (무료, 지역 제한 없음)
        coin_id = 'bitcoin' if symbol == 'BTC/USDT' else 'ethereum'
        
        # 과거 데이터 조회 (마켓 차트)
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': limit,
            'interval': 'daily'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 데이터 변환
        prices = data.get('prices', [])
        if not prices:
            st.error("⚠️ 데이터를 가져올 수 없습니다.")
            return pd.DataFrame()
        
        df = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # OHLCV 근사값 생성 (일일 데이터의 경우)
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1)
        df['low'] = df[['open', 'close']].min(axis=1)
        df['volume'] = 0  # CoinGecko에서 직접 제공하지 않으므로 0으로 설정
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        st.error(f"⚠️ 데이터를 가져오는데 실패했습니다: {e}")
        return pd.DataFrame()

# 2. 변곡점 탐지 엔진
def find_pivots(df, order=10):
    if len(df) < (order * 2 + 1): return pd.DataFrame(), pd.DataFrame()
    peak_idx = argrelextrema(df['high'].values, np.greater, order=order)[0]
    valley_idx = argrelextrema(df['low'].values, np.less, order=order)[0]
    return df.iloc[peak_idx], df.iloc[valley_idx]

# 3. 패턴 판독
def check_patterns(p, v):
    patterns = []
    if len(v) >= 2 and len(p) >= 1:
        v1, v2 = v.iloc[-2]['low'], v.iloc[-1]['low']
        diff = abs(v1 - v2) / v1
        if diff <= 0.05:
            patterns.append({"name": "장기 이중 바닥 (W)", "score": (1-diff)*100, "type": "Strong Bullish"})
    if not patterns:
        patterns.append({"name": "추세 탐색 중", "score": 50.0, "type": "Neutral"})
    return patterns[:3]

# ==========================================
# 4. UI 및 시각화
# ==========================================
st.set_page_config(page_title="BTC Chronicle Master", layout="wide")
st.title("📜 비트코인 연대기: 전체 역사 탐색 및 패턴 분석")

# 사이드바
st.sidebar.header("🔑 설정")
api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")

st.sidebar.header("📊 데이터 설정")
symbol = st.sidebar.selectbox("코인", ['BTC/USDT', 'ETH/USDT'])
timeframe = st.sidebar.selectbox("시간 단위", ['1h', '4h', '1d', '1w'], index=2)
use_log = st.sidebar.checkbox("로그 차트로 보기", value=True)

# 데이터 로드
full_df = get_crypto_data(symbol, timeframe)

if not full_df.empty:
    # --- 차트 영역 ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full_df['timestamp'], y=full_df['close'], mode='lines', name='Price', line=dict(color='orange', width=1.5)))
    if use_log: fig.update_yaxes(type="log")
    fig.update_layout(height=500, template="plotly_dark", xaxis=dict(rangeslider=dict(visible=True), type="date"), yaxis_title="Price (USDT)")
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

    st.divider()
    st.subheader("2️⃣ 역사적 구간 분석")

    # --- ⚠️ 날짜 에러 해결 구간 ---
    # 모든 날짜 계산 후 즉시 .to_pydatetime()으로 변환하여 순수 파이썬 객체로 만듭니다.
    min_dt = full_df['timestamp'].min().to_pydatetime()
    max_dt = full_df['timestamp'].max().to_pydatetime()
    
    # 초기 선택 범위 (최근 1년)
    default_start_dt = (full_df['timestamp'].max() - timedelta(days=365)).to_pydatetime()

    analysis_range = st.slider(
        "분석할 범위를 선택하세요",
        min_value=min_dt,
        max_value=max_dt,
        value=(default_start_dt, max_dt), # 여기서 max_dt는 이미 pydatetime임
        format="YYYY/MM/DD"
    )

    if st.button("✨ 분석 리포트 생성"):
        if not api_key:
            st.warning("먼저 OpenAI API Key를 입력해주세요.")
        else:
            sel_df = full_df[(full_df['timestamp'] >= pd.Timestamp(analysis_range[0])) & (full_df['timestamp'] <= pd.Timestamp(analysis_range[1]))].copy()
            peaks, valleys = find_pivots(sel_df)
            top_3 = check_patterns(peaks, valleys)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("### 🔍 감지된 패턴")
                st.success(f"**{top_3[0]['name']}**")
            with col2:
                st.write("### 🤖 AI 해설")
                client = OpenAI(api_key=api_key)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": f"{symbol} {top_3[0]['name']} 패턴에 대해 분석해줘."}]
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"AI 호출 오류: {e}")