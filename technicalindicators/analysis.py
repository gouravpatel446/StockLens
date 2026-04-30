"""
Technical Analysis Dashboard Module
Displays comprehensive technical analysis with charts and indicators
"""

import streamlit as st  # For building interactive web app
import yfinance as yf  # For fetching stock data
import plotly.graph_objects as go  # For interactive charts
from plotly.subplots import make_subplots  # For multi-subplot charts
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations

# Import custom indicator functions
from .indicators import calculate_rsi, calculate_macd, calculate_moving_averages
# Import scoring module for fundamental analysis
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scoring import fundamental_score, get_score_color, get_score_emoji


def calculate_technical_rating(hist, info):
    """
    Calculate a comprehensive technical rating out of 10 based on multiple indicators

    Returns:
    --------
    tuple: (rating, zone, details, recommendations)
    """
    rating = 0
    max_score = 10
    details = []
    recommendations = []

    current_price = hist['Close'].iloc[-1] if 'Close' in hist.columns and len(hist) > 0 else None

    # ── 1. Trend Analysis (2 points) ──
    if len(hist) >= 200:
        ma50 = hist['MA50'].iloc[-1]
        ma200 = hist['MA200'].iloc[-1]

        # Price vs MAs
        if current_price is not None and current_price > ma50 and ma50 > ma200:
            rating += 2  # Strong uptrend
            details.append("✅ Strong uptrend: Price > MA50 > MA200")
        elif current_price > ma50:
            rating += 1.5  # Moderate uptrend
            details.append("⚠️ Moderate uptrend: Price > MA50")
        elif current_price < ma200:
            rating += 0  # Downtrend
            details.append("❌ Downtrend: Price < MA200")
            recommendations.append("Consider waiting for trend reversal")
        else:
            rating += 1  # Sideways
            details.append("➡️ Sideways trend")
    else:
        details.append("⚠️ Insufficient data for trend analysis")

    # ── 2. RSI Analysis (1.5 points) ──
    rsi = hist['RSI'].iloc[-1]
    if 30 <= rsi <= 70:
        rating += 1.5  # Neutral zone
        details.append(f"✅ RSI: {rsi:.1f} (Neutral zone)")
    elif rsi < 30:
        rating += 1  # Oversold but potential
        details.append(f"⚠️ RSI: {rsi:.1f} (Oversold - potential buying opportunity)")
        recommendations.append("Oversold condition - monitor for bounce")
    else:  # > 70
        rating += 0.5  # Overbought
        details.append(f"⚠️ RSI: {rsi:.1f} (Overbought - potential selling pressure)")
        recommendations.append("Overbought condition - consider taking profits")

    # ── 3. MACD Analysis (1.5 points) ──
    macd = hist['MACD'].iloc[-1]
    signal = hist['MACD_Signal'].iloc[-1]
    histogram = hist['MACD_Histogram'].iloc[-1]

    if macd > signal and histogram > 0:
        rating += 1.5  # Bullish momentum
        details.append("✅ MACD: Bullish momentum")
    elif macd > signal:
        rating += 1  # Weak bullish
        details.append("⚠️ MACD: Weak bullish signal")
    elif macd < signal and histogram < 0:
        rating += 0  # Bearish momentum
        details.append("❌ MACD: Bearish momentum")
        recommendations.append("Bearish momentum - exercise caution")
    else:
        rating += 0.5  # Mixed signals
        details.append("➡️ MACD: Mixed signals")

    # ── 4. Volatility Analysis (1 point) ──
    if 'High' in hist.columns and 'Low' in hist.columns:
        # Calculate ATR-like volatility
        tr = pd.concat([
            hist['High'] - hist['Low'],
            (hist['High'] - hist['Close'].shift()).abs(),
            (hist['Low'] - hist['Close'].shift()).abs(),
        ], axis=1).max(axis=1)
        volatility = tr.rolling(14).mean().iloc[-1] / current_price * 100

        if volatility < 2:
            rating += 1  # Low volatility - stable
            details.append(f"✅ Low volatility: {volatility:.1f}% (Stable)")
        elif volatility < 5:
            rating += 0.7  # Moderate volatility
            details.append(f"⚠️ Moderate volatility: {volatility:.1f}%")
        else:
            rating += 0.3  # High volatility - risky
            details.append(f"⚠️ High volatility: {volatility:.1f}% (Risky)")
            recommendations.append("High volatility - use stop losses")

    # ── 5. Volume Analysis (1 point) ──
    if 'Volume' in hist.columns:
        avg_volume = hist['Volume'].tail(20).mean()
        recent_volume = hist['Volume'].tail(5).mean()

        if recent_volume > avg_volume * 1.5:
            rating += 1  # High volume - strong interest
            details.append("✅ High volume: Strong interest")
        elif recent_volume > avg_volume * 0.8:
            rating += 0.7  # Normal volume
            details.append("➡️ Normal volume")
        else:
            rating += 0.3  # Low volume - weak interest
            details.append("⚠️ Low volume: Weak interest")
            recommendations.append("Low volume may indicate lack of conviction")

    # ── 6. Support/Resistance Analysis (1 point) ──
    if len(hist) >= 50:
        recent_high = hist['High'].tail(20).max()
        recent_low = hist['Low'].tail(20).min()

        if current_price > recent_high * 0.98:
            rating += 0.5  # Near resistance
            details.append("⚠️ Near recent resistance")
            recommendations.append("Approaching resistance - monitor closely")
        elif current_price < recent_low * 1.02:
            rating += 0.8  # Near support
            details.append("✅ Near recent support")
            recommendations.append("Near support level - potential buying zone")
        else:
            rating += 1  # Clear of major levels
            details.append("✅ Clear of major support/resistance")

    # ── 7. Momentum Analysis (1 point) ──
    if len(hist) >= 10:
        # Price momentum over last 10 days
        momentum = (current_price - hist['Close'].iloc[-10]) / hist['Close'].iloc[-10] * 100

        if momentum > 5:
            rating += 1  # Strong positive momentum
            details.append(f"✅ Strong momentum: +{momentum:.1f}% (10-day)")
        elif momentum > 0:
            rating += 0.7  # Mild positive momentum
            details.append(f"⚠️ Mild momentum: +{momentum:.1f}% (10-day)")
        elif momentum > -5:
            rating += 0.3  # Mild negative momentum
            details.append(f"⚠️ Weak momentum: {momentum:.1f}% (10-day)")
        else:
            rating += 0  # Strong negative momentum
            details.append(f"❌ Weak momentum: {momentum:.1f}% (10-day)")
            recommendations.append("Strong downward momentum - avoid")

    # Determine zone and final rating
    final_rating = min(max(round(rating), 0), 10)

    if final_rating >= 7:
        zone = "GOOD"
        zone_color = "🟢"
        zone_desc = "Strong Buy - Excellent technical setup"
    elif final_rating >= 5:
        zone = "NEUTRAL"
        zone_color = "🟡"
        zone_desc = "Hold - Mixed signals, wait for clarity"
    else:
        zone = "BAD"
        zone_color = "🔴"
        zone_desc = "Avoid - Weak technical setup"

    return final_rating, zone, zone_color, zone_desc, details, recommendations


def calculate_bollinger_bands(data, window=20, num_std=2):
    df = data.copy()
    df['BB_MA'] = df['Close'].rolling(window).mean()
    df['BB_STD'] = df['Close'].rolling(window).std()
    df['BB_upper'] = df['BB_MA'] + num_std * df['BB_STD']
    df['BB_lower'] = df['BB_MA'] - num_std * df['BB_STD']
    return df


def calculate_weekly_data(data):
    weekly = data.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    if not weekly.empty:
        weekly = calculate_moving_averages(weekly)
    return weekly


def fetch_index_data(period, index_symbol='^NSEI'):
    try:
        index_hist = yf.Ticker(index_symbol).history(period=period)
        if index_hist is None or getattr(index_hist, 'empty', True):
            return None
        return index_hist
    except Exception:
        return None


def calculate_relative_strength(hist, index_hist):
    rs = (hist['Close'] / index_hist['Close']).dropna() * 100
    return rs


def display_analysis_dashboard(theme='dark', preset_symbol=None):
    """
    Display the main technical analysis dashboard with charts and indicators

    Args:
        theme (str): 'dark' or 'light' theme
        preset_symbol (str): Pre-loaded stock symbol from Overview page
    """

    # ── Use preset symbol or default ──
    symbol = preset_symbol if preset_symbol else 'AAPL'
    period = '1y'  # Default time period

    # ── Fetch Data and Calculate Indicators ──
    with st.spinner('Loading data and calculating indicators...'):
        try:
            # Create Ticker object for the symbol
            stock = yf.Ticker(symbol.upper())
            
            # Fetch historical price data for the selected period
            hist = stock.history(period=period)
            
            # Check if data is available
            if hist is None or getattr(hist, 'empty', True):
                st.error('Stock not found or returned no data. Please check the symbol.')
                return

            # Fetch fundamental company information
            info = stock.info or {}
            hist = calculate_moving_averages(hist)
            
            # Calculate RSI (Relative Strength Index)
            hist['RSI'] = calculate_rsi(hist, period=14)
            
            # Calculate MACD, Signal Line, and Histogram
            hist['MACD'], hist['MACD_Signal'], hist['MACD_Histogram'] = calculate_macd(hist)
            
            # === Additional line-by-line analysis charts ===
            hist = calculate_bollinger_bands(hist)
            weekly_hist = calculate_weekly_data(hist)
            index_hist = fetch_index_data(period)
            rs_series = None
            if index_hist is not None and not index_hist.empty:
                merged_index = hist[['Close']].join(index_hist[['Close']].rename(columns={'Close': 'Index_Close'}), how='inner')
                if not merged_index.empty:
                    rs_series = merged_index['Close'] / merged_index['Index_Close'] * 100

            current_price = hist['Close'].iloc[-1]
            support1 = hist['Low'].tail(20).min()
            resistance1 = hist['High'].tail(20).max()
            stop_price = support1
            target_price = resistance1
            rr_ratio = None
            rr_label = 'N/A'
            if stop_price and current_price > stop_price:
                rr_ratio = (target_price - current_price) / (current_price - stop_price) if current_price != stop_price else None
                if rr_ratio is not None:
                    rr_label = f'{rr_ratio:.2f}:1'

            base_range = hist['Close'].tail(20).max() - hist['Close'].tail(20).min()
            breakout_text = 'Clean consolidation base detected' if base_range < current_price * 0.05 else 'No clean breakout base yet'
            weekly_trend_text = 'Uptrend' if not weekly_hist.empty and weekly_hist['Close'].iloc[-1] > weekly_hist['MA50'].iloc[-1] > weekly_hist['MA200'].iloc[-1] else 'Not in clear uptrend'
            ma_trend_text = 'Price above both MA50 and MA200' if current_price > hist['MA50'].iloc[-1] and current_price > hist['MA200'].iloc[-1] else 'Price not above both moving averages'
            golden_cross = 'Yes' if hist['MA50'].iloc[-2] < hist['MA200'].iloc[-2] and hist['MA50'].iloc[-1] > hist['MA200'].iloc[-1] else 'No'
            bb_status = 'Bounce from lower band' if hist['Close'].iloc[-1] <= hist['BB_lower'].iloc[-1] else 'No lower-band bounce'
            squeeze_status = 'Squeeze detected' if (hist['BB_upper'].iloc[-1] - hist['BB_lower'].iloc[-1]) < hist['Close'].rolling(20).std().iloc[-1] * 1.2 else 'No squeeze'
            macd_signal_text = 'Histogram green and/or MACD crossing up' if hist['MACD_Histogram'].iloc[-1] > 0 or hist['MACD'].iloc[-1] > hist['MACD_Signal'].iloc[-1] else 'MACD not bullish yet'

            if not weekly_hist.empty:
                st.markdown('### Weekly Trend Check')
                fig_weekly = go.Figure()
                fig_weekly.add_trace(go.Scatter(x=weekly_hist.index, y=weekly_hist['Close'], name='Weekly Close', line=dict(color='#1F4E79', width=2)))
                if 'MA50' in weekly_hist.columns:
                    fig_weekly.add_trace(go.Scatter(x=weekly_hist.index, y=weekly_hist['MA50'], name='Weekly MA50', line=dict(color='orange', dash='dash')))
                if 'MA200' in weekly_hist.columns:
                    fig_weekly.add_trace(go.Scatter(x=weekly_hist.index, y=weekly_hist['MA200'], name='Weekly MA200', line=dict(color='red', dash='dot')))
                fig_weekly.update_layout(title='Weekly Price Trend', xaxis_title='Date', yaxis_title='Price (INR)', template='plotly_white', height=420)
                st.plotly_chart(fig_weekly, width='stretch')
                st.info(f'Weekly Trend: {weekly_trend_text} | {ma_trend_text} | Golden Cross: {golden_cross}')
                st.divider()

            st.markdown('### 50 & 200 MA Trend')
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close', line=dict(color='#1F4E79', width=2)))
            fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='MA50', line=dict(color='orange', dash='dash')))
            fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], name='MA200', line=dict(color='red', dash='dot')))
            fig_ma.update_layout(title='Daily 50 / 200 Moving Averages', xaxis_title='Date', yaxis_title='Price (INR)', template='plotly_white', height=420)
            st.plotly_chart(fig_ma, width='stretch')
            st.info(f'{ma_trend_text} | Golden Cross: {golden_cross}')
            st.divider()

            entry_price = support1 * 1.01 if current_price > support1 else current_price
            if entry_price >= resistance1:
                entry_price = min(current_price, resistance1 * 0.98)

            entry_rr = None
            if entry_price > stop_price and target_price > entry_price:
                entry_rr = (target_price - entry_price) / (entry_price - stop_price)

            st.markdown('### Support & Resistance')
            fig_sr = go.Figure()
            fig_sr.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close', line=dict(color='#1F4E79', width=2)))
            fig_sr.add_trace(go.Scatter(x=hist.index, y=[support1] * len(hist), name='Support / Stop Loss', line=dict(color='green', dash='dash')))
            fig_sr.add_trace(go.Scatter(x=hist.index, y=[resistance1] * len(hist), name='Target / Resistance', line=dict(color='red', dash='dash')))
            fig_sr.add_trace(go.Scatter(x=hist.index, y=[entry_price] * len(hist), name='Potential Entry', line=dict(color='blue', dash='dot')))
            fig_sr.update_layout(title='Support and Resistance Levels', xaxis_title='Date', yaxis_title='Price (INR)', template='plotly_white', height=420)
            st.plotly_chart(fig_sr, width='stretch')
            st.info(f'Buying near support? Support: ₹{support1:.2f} | Resistance: ₹{resistance1:.2f}')

            st.markdown('### Potential Trade Setup')
            setup_col1, setup_col2, setup_col3 = st.columns(3)
            setup_col1.metric('Entry Zone', f'₹{entry_price:.2f}')
            setup_col2.metric('Stop Loss', f'₹{stop_price:.2f}')
            setup_col3.metric('Target', f'₹{target_price:.2f}', delta=f'{entry_rr:.2f}:1' if entry_rr is not None else 'N/A')
            if entry_rr is not None:
                st.write(f'Potential R:R based on entry zone: **{entry_rr:.2f}:1**')
            else:
                st.write('No valid risk/reward setup found with current support/target levels.')
            st.divider()

            st.markdown('### RSI Momentum')
            st.write('')
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='purple', width=2)))
            fig_rsi.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought', annotation_position='top right')
            fig_rsi.add_hline(y=50, line_dash='dot', line_color='gray', annotation_text='Cooling Zone', annotation_position='top right')
            fig_rsi.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold', annotation_position='bottom right')
            fig_rsi.update_layout(title='RSI Momentum', xaxis_title='Date', yaxis_title='RSI', template='plotly_white', height=360)
            st.plotly_chart(fig_rsi, width='stretch')
            st.info(f'RSI Status: {hist["RSI"].iloc[-1]:.1f} | {"Cooling" if 30 <= hist["RSI"].iloc[-1] <= 50 else "Overbought" if hist["RSI"].iloc[-1] > 70 else "Neutral"}')
            st.divider()

            st.markdown('### Volume Confirmation')
            st.write('')
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color='steelblue'))
            fig_vol.update_layout(title='Volume Confirmation', xaxis_title='Date', yaxis_title='Volume', template='plotly_white', height=360)
            st.plotly_chart(fig_vol, width='stretch')
            vol_status = 'Volume confirms move' if hist['Volume'].iloc[-1] >= hist['Volume'].tail(20).mean() else 'Volume weak - no conviction'
            st.info(vol_status)
            st.divider()

            if 'Volume' in hist.columns:
                hist['OBV'] = ((np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0)).cumsum()
                st.markdown('### On-Balance Volume (OBV)')
                st.write('')
                fig_obv = go.Figure()
                fig_obv.add_trace(go.Scatter(x=hist.index, y=hist['OBV'], name='OBV', line=dict(color='teal', width=2)))
                fig_obv.update_layout(title='On-Balance Volume', xaxis_title='Date', yaxis_title='OBV', template='plotly_white', height=360)
                st.plotly_chart(fig_obv, width='stretch')
                st.info('OBV shows whether volume is supporting the current price move.')
                st.divider()

            if 'Close' in hist.columns:
                hist['Daily_Return'] = hist['Close'].pct_change() * 100
                st.markdown('### Daily Returns Distribution')
                st.write('')
                fig_return = go.Figure()
                fig_return.add_trace(go.Histogram(x=hist['Daily_Return'].dropna(), nbinsx=30, marker_color='slateblue'))
                fig_return.update_layout(title='Daily Returns Distribution', xaxis_title='Daily Return (%)', yaxis_title='Frequency', template='plotly_white', height=360)
                st.plotly_chart(fig_return, width='stretch')
                st.info('Return distribution helps assess recent volatility and risk.')
                st.divider()

            st.markdown('### MACD + Histogram')
            st.write('')
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line=dict(color='blue', width=2)))
            fig_macd.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], name='Signal', line=dict(color='red', width=2)))
            fig_macd.add_trace(go.Bar(x=hist.index, y=hist['MACD_Histogram'], name='Histogram', marker_color=['green' if val >= 0 else 'red' for val in hist['MACD_Histogram']], opacity=0.4))
            fig_macd.update_layout(title='MACD + Histogram', xaxis_title='Date', yaxis_title='MACD', template='plotly_white', height=360)
            st.plotly_chart(fig_macd, width='stretch')
            st.info(macd_signal_text)
            st.divider()

            st.markdown('### Bollinger Bands')
            st.write('')
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close', line=dict(color='#1F4E79', width=2)))
            fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], name='Upper Band', line=dict(color='gray', dash='dash')))
            fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['BB_MA'], name='Middle Band', line=dict(color='orange', dash='dot')))
            fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], name='Lower Band', line=dict(color='gray', dash='dash')))
            fig_bb.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price (INR)', template='plotly_white', height=360)
            st.plotly_chart(fig_bb, width='stretch')
            st.info(f'{bb_status} | {squeeze_status}')
            st.divider()

            if {'High', 'Low', 'Close'}.issubset(hist.columns):
                tr = pd.concat([
                    hist['High'] - hist['Low'],
                    (hist['High'] - hist['Close'].shift()).abs(),
                    (hist['Low'] - hist['Close'].shift()).abs(),
                ], axis=1).max(axis=1)
                hist['ATR14'] = tr.rolling(14).mean()
                hist['BB_Width'] = hist['BB_upper'] - hist['BB_lower']

                st.markdown('### ATR / Volatility')
                st.write('')
                fig_atr = go.Figure()
                fig_atr.add_trace(go.Scatter(x=hist.index, y=hist['ATR14'], name='ATR (14)', line=dict(color='#1F4E79', width=2)))
                fig_atr.add_trace(go.Scatter(x=hist.index, y=hist['BB_Width'], name='Bollinger Width', line=dict(color='purple', dash='dash')))
                fig_atr.update_layout(title='ATR and Bollinger Band Width', xaxis_title='Date', yaxis_title='Volatility', template='plotly_white', height=360)
                st.plotly_chart(fig_atr, width='stretch')
                st.info('ATR and Bollinger width highlight volatility and squeeze/breakout risk.')
                st.divider()

            if rs_series is not None and not rs_series.empty:
                st.markdown('### Relative Strength')
                st.write('')
                fig_rs = go.Figure()
                fig_rs.add_trace(go.Scatter(x=rs_series.index, y=rs_series, name='Relative Strength', line=dict(color='teal', width=2)))
                fig_rs.update_layout(title='Relative Strength vs Nifty/Sensex', xaxis_title='Date', yaxis_title='RS Ratio (100 = parity)', template='plotly_white', height=360)
                st.plotly_chart(fig_rs, width='stretch')
                st.info('Relative strength shows whether the stock is outperforming the benchmark.')
                st.divider()

            st.markdown('### Risk-Reward & Stop Loss')
            rr_text = f'R:R = {rr_label}' if rr_label != 'N/A' else 'R:R undefined'
            st.info(f'{rr_text} | Stop Loss at ₹{stop_price:.2f} | Target near ₹{target_price:.2f}')
            st.markdown(f'**Breakout Pattern:** {breakout_text}')
            st.divider()

            # ── Comprehensive Technical Rating ──
            st.subheader('Technical Rating & Analysis')
            
            # Calculate comprehensive rating
            rating, zone, zone_color, zone_desc, details, recommendations = calculate_technical_rating(hist, info)
            
            # Display rating with clean, professional styling
            rating_col1, rating_col2, rating_col3 = st.columns([1.2, 2, 2])
            
            with rating_col1:
                # Determine colors based on zone
                if zone == "GOOD":
                    bg_gradient = "linear-gradient(135deg, #0f766e 0%, #14b8a6 100%)"
                    text_color = "#ecfdf5"
                elif zone == "NEUTRAL":
                    bg_gradient = "linear-gradient(135deg, #78350f 0%, #b45309 100%)"
                    text_color = "#fffbeb"
                else:  # BAD
                    bg_gradient = "linear-gradient(135deg, #7c2d12 0%, #dc2626 100%)"
                    text_color = "#fef2f2"
                
                st.markdown(f"""
                <div style='text-align: center; padding: 28px 16px; border-radius: 12px; background: {bg_gradient}; color: {text_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                    <div style='font-size: 48px; font-weight: 700; letter-spacing: -1px; margin: 0 0 8px 0;'>{rating}/10</div>
                    <div style='font-size: 16px; font-weight: 600; margin: 0 0 6px 0;'>{zone}</div>
                    <div style='font-size: 12px; opacity: 0.9; margin: 0;'>{zone_desc}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with rating_col2:
                st.markdown("**Key Signals:**")
                # Display only the most important details
                important_details = []
                for detail in details:
                    # Clean up emoji and extract key info
                    clean_detail = detail.replace("✅", "•").replace("⚠️", "⚡").replace("❌", "✗").replace("➡️", "•")
                    important_details.append(clean_detail)
                
                # Show only top 4-5 signals
                for detail in important_details[:5]:
                    st.write(detail)
            
            with rating_col3:
                if recommendations:
                    st.markdown("**Action Items:**")
                    for i, rec in enumerate(recommendations[:3], 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.markdown("**Status:**")
                    st.write("✓ Technical setup is favorable")
                    st.write("✓ No major red flags detected")
                    st.write("✓ Monitor for confirmation signals")
            
            # ── Detailed Indicator Analysis ──
            st.divider()
            st.subheader('Indicator Details')
            
            # Create expandable sections for detailed analysis
            with st.expander("Trend Analysis", expanded=True):
                col_trend1, col_trend2 = st.columns(2)
                
                with col_trend1:
                    st.markdown("**Moving Averages**")
                    current_price = hist['Close'].iloc[-1]
                    ma50 = hist['MA50'].iloc[-1] if 'MA50' in hist.columns else None
                    ma200 = hist['MA200'].iloc[-1] if 'MA200' in hist.columns else None
                    
                    if ma50 and ma200:
                        st.metric("Current Price", f"₹{current_price:.2f}")
                        st.metric("MA50 (Short-term)", f"₹{ma50:.2f}", 
                                delta=f"{((current_price/ma50-1)*100):+.1f}%")
                        st.metric("MA200 (Long-term)", f"₹{ma200:.2f}",
                                delta=f"{((current_price/ma200-1)*100):+.1f}%")
                        
                        if current_price > ma50 > ma200:
                            st.success("Status: Uptrend - Price above both moving averages")
                        elif current_price < ma200:
                            st.error("Status: Downtrend - Price below 200-day MA")
                        else:
                            st.info("Status: Mixed - Price in transition zone")
                
                with col_trend2:
                    st.markdown("**Momentum Indicators**")
                    rsi_val = hist['RSI'].iloc[-1]
                    macd_val = hist['MACD'].iloc[-1]
                    signal_val = hist['MACD_Signal'].iloc[-1]
                    
                    # RSI Status
                    if rsi_val > 70:
                        rsi_status = "Overbought"
                    elif rsi_val < 30:
                        rsi_status = "Oversold"
                    else:
                        rsi_status = "Neutral"
                    
                    st.metric("RSI (14)", f"{rsi_val:.1f}", rsi_status)
                    
                    # MACD Status
                    macd_status = "Bullish" if macd_val > signal_val else "Bearish"
                    st.metric("MACD Status", macd_status)
                    st.metric("MACD Line", f"{macd_val:.4f}")
            
            with st.expander("Volume & Volatility"):
                col_vol1, col_vol2 = st.columns(2)
                
                with col_vol1:
                    st.markdown("**Volume**")
                    if 'Volume' in hist.columns:
                        avg_vol = hist['Volume'].tail(20).mean()
                        recent_vol = hist['Volume'].tail(5).mean()
                        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
                        
                        vol_status = "High - Strong interest" if vol_ratio > 1.5 else "Normal" if vol_ratio > 0.8 else "Low - Weak interest"
                        st.metric("Recent Volume", f"{recent_vol/1000000:.1f}M")
                        st.metric("20-Day Average", f"{avg_vol/1000000:.1f}M")
                        st.metric("Volume Ratio", f"{vol_ratio:.2f}x", vol_status)
                    else:
                        st.warning("Volume data not available")
                
                with col_vol2:
                    st.markdown("**Volatility**")
                    if 'High' in hist.columns and 'Low' in hist.columns:
                        # Calculate True Range
                        tr = pd.concat([
                            hist['High'] - hist['Low'],
                            (hist['High'] - hist['Close'].shift()).abs(),
                            (hist['Low'] - hist['Close'].shift()).abs(),
                        ], axis=1).max(axis=1)
                        
                        # ATR (14-day)
                        atr = tr.rolling(14).mean().iloc[-1]
                        volatility_pct = (atr / current_price) * 100
                        
                        vol_level = "Low - Stable" if volatility_pct < 2 else "Medium" if volatility_pct < 5 else "High - Risky"
                        st.metric("ATR (14)", f"₹{atr:.2f}")
                        st.metric("Volatility", f"{volatility_pct:.1f}%", vol_level)
                        
                        # Price range
                        price_range = hist['High'].iloc[-1] - hist['Low'].iloc[-1]
                        range_pct = (price_range / hist['Close'].iloc[-1]) * 100
                        st.metric("Daily Range", f"₹{price_range:.2f} ({range_pct:.1f}%)")
            
            with st.expander("Support & Resistance"):
                col_sr1, col_sr2 = st.columns(2)
                
                with col_sr1:
                    st.markdown("**Support Levels**")
                    # Calculate support levels from recent lows
                    recent_lows = hist['Low'].tail(20)
                    support1 = recent_lows.min()
                    support2 = recent_lows.nlargest(2).iloc[-1] if len(recent_lows) >= 2 else support1
                    
                    current_price = hist['Close'].iloc[-1]
                    st.metric("Support 1", f"₹{support1:.2f}", 
                            f"{((current_price/support1-1)*100):+.1f}% away")
                    st.metric("Support 2", f"₹{support2:.2f}",
                            f"{((current_price/support2-1)*100):+.1f}% away")
                
                with col_sr2:
                    st.markdown("**Resistance Levels**")
                    # Calculate resistance levels from recent highs
                    recent_highs = hist['High'].tail(20)
                    resistance1 = recent_highs.max()
                    resistance2 = recent_highs.nsmallest(2).iloc[-1] if len(recent_highs) >= 2 else resistance1
                    
                    st.metric("Resistance 1", f"₹{resistance1:.2f}",
                            f"{((current_price/resistance1-1)*100):+.1f}% away")
                    st.metric("Resistance 2", f"₹{resistance2:.2f}",
                            f"{((current_price/resistance2-1)*100):+.1f}% away")
            
            # ── Display Key Information ──
            
            st.divider()
            
            # Create columns for key metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                # Display company name
                st.metric(
                    'Company',
                    info.get('longName', symbol.upper())
                )
            
            with metric_col2:
                # Display current price
                st.metric(
                    'Current Price',
                    f"₹{info.get('currentPrice', 'N/A')}"
                )
            
            with metric_col3:
                # Display market cap
                market_cap = info.get('marketCap', 'N/A')
                if isinstance(market_cap, (int, float)):
                    st.metric('Market Cap', f"₹{market_cap/1e9:.2f}B")
                else:
                    st.metric('Market Cap', market_cap)
            
            with metric_col4:
                # Display P/E ratio
                pe_ratio = info.get('trailingPE', 'N/A')
                if isinstance(pe_ratio, (int, float)):
                    st.metric('P/E Ratio', f"{pe_ratio:.1f}x")
                else:
                    st.metric('P/E Ratio', pe_ratio)
            
            # ── Quick Indicator Summary ──
            st.divider()
            st.subheader('Quick Summary')
            
            quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
            
            with quick_col1:
                rsi_val = hist['RSI'].iloc[-1]
                if rsi_val > 70:
                    st.error(f"RSI: {rsi_val:.1f}\nOverbought")
                elif rsi_val < 30:
                    st.success(f"RSI: {rsi_val:.1f}\nOversold")
                else:
                    st.info(f"RSI: {rsi_val:.1f}\nNeutral")
            
            with quick_col2:
                ma50 = hist['MA50'].iloc[-1] if 'MA50' in hist.columns else None
                ma200 = hist['MA200'].iloc[-1] if 'MA200' in hist.columns else None
                current = hist['Close'].iloc[-1]
                
                if ma50 and ma200 and current:
                    if current > ma50 > ma200:
                        st.success("Trend\nBullish")
                    elif current < ma200:
                        st.error("Trend\nBearish")
                    else:
                        st.warning("Trend\nMixed")
                else:
                    st.info("Trend\nN/A")
            
            with quick_col3:
                macd_val = hist['MACD'].iloc[-1]
                signal_val = hist['MACD_Signal'].iloc[-1]
                
                if macd_val > signal_val:
                    st.success("MACD\nBullish")
                else:
                    st.error("MACD\nBearish")
            
            with quick_col4:
                if 'Volume' in hist.columns:
                    avg_vol = hist['Volume'].tail(20).mean()
                    recent_vol = hist['Volume'].tail(5).mean()
                    
                    if recent_vol > avg_vol * 1.2:
                        st.success("Volume\nHigh")
                    elif recent_vol > avg_vol * 0.8:
                        st.info("Volume\nNormal")
                    else:
                        st.warning("Volume\nLow")
                else:
                    st.info("Volume\nN/A")
            
            # ── Performance Metrics ──
            st.divider()
            st.subheader('Performance Metrics')
            
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            
            with perf_col1:
                # 1-day change
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[-2]
                    current = hist['Close'].iloc[-1]
                    day_change = ((current - prev_close) / prev_close) * 100
                    color = "inverse" if day_change >= 0 else "normal"
                    st.metric("1-Day Change", f"{day_change:+.2f}%", 
                            delta=f"₹{current-prev_close:+.2f}", delta_color=color)
            
            with perf_col2:
                # 1-week change
                if len(hist) >= 5:
                    week_ago = hist['Close'].iloc[-5]
                    current = hist['Close'].iloc[-1]
                    week_change = ((current - week_ago) / week_ago) * 100
                    color = "inverse" if week_change >= 0 else "normal"
                    st.metric("1-Week Change", f"{week_change:+.2f}%", 
                            delta=f"₹{current-week_ago:+.2f}", delta_color=color)
            
            with perf_col3:
                # 1-month change
                if len(hist) >= 20:
                    month_ago = hist['Close'].iloc[-20]
                    current = hist['Close'].iloc[-1]
                    month_change = ((current - month_ago) / month_ago) * 100
                    color = "inverse" if month_change >= 0 else "normal"
                    st.metric("1-Month Change", f"{month_change:+.2f}%", 
                            delta=f"₹{current-month_ago:+.2f}", delta_color=color)
            
            with perf_col4:
                # Volatility
                if len(hist) >= 20:
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                    st.metric("Annual Volatility", f"{volatility:.1f}%")
            
            # ── Stock Rating & Zone Analysis ──
            st.divider()
            st.subheader('Detailed Rating Analysis')
            
            # Calculate comprehensive rating (0-10 scale)
            rating_score = 0
            max_score = 10
            rating_factors = []
            
            # RSI factor (2 points)
            rsi_val = hist['RSI'].iloc[-1]
            if 40 <= rsi_val <= 60:
                rating_score += 2
                rating_factors.append("RSI in optimal range")
            elif 30 <= rsi_val <= 70:
                rating_score += 1.5
                rating_factors.append("RSI acceptable")
            else:
                rating_factors.append("RSI extreme")
            
            # Trend factor (2 points)
            if 'MA50' in hist.columns and 'MA200' in hist.columns:
                ma50 = hist['MA50'].iloc[-1]
                ma200 = hist['MA200'].iloc[-1]
                current = hist['Close'].iloc[-1]
                
                if current > ma50 > ma200:
                    rating_score += 2
                    rating_factors.append("Strong uptrend")
                elif current > ma50:
                    rating_score += 1.5
                    rating_factors.append("Moderate uptrend")
                elif current > ma200:
                    rating_score += 1
                    rating_factors.append("Weak uptrend")
                else:
                    rating_factors.append("Downtrend")
            
            # MACD factor (2 points)
            if 'MACD' in hist.columns and 'MACD_Signal' in hist.columns:
                macd = hist['MACD'].iloc[-1]
                signal = hist['MACD_Signal'].iloc[-1]
                
                if macd > signal:
                    rating_score += 2
                    rating_factors.append("MACD bullish")
                else:
                    rating_factors.append("MACD bearish")
            
            # Volume factor (1 point)
            if 'Volume' in hist.columns and len(hist) >= 20:
                avg_vol = hist['Volume'].tail(20).mean()
                recent_vol = hist['Volume'].tail(5).mean()
                
                if recent_vol > avg_vol * 1.2:
                    rating_score += 1
                    rating_factors.append("High volume interest")
                elif recent_vol > avg_vol * 0.8:
                    rating_score += 0.5
                    rating_factors.append("Normal volume")
            
            # Momentum factor (1 point) - based on recent performance
            if len(hist) >= 10:
                recent_returns = hist['Close'].pct_change().tail(10).mean() * 100
                if recent_returns > 2:
                    rating_score += 1
                    rating_factors.append("Strong momentum")
                elif recent_returns > 0:
                    rating_score += 0.5
                    rating_factors.append("Positive momentum")
            
            # Volatility factor (2 points) - penalize excessive volatility
            if len(hist) >= 20:
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                
                if volatility < 20:
                    rating_score += 2
                    rating_factors.append("Low volatility")
                elif volatility < 40:
                    rating_score += 1
                    rating_factors.append("Moderate volatility")
                else:
                    rating_factors.append("High volatility")
            
            # Round to 1 decimal place
            rating_score = round(rating_score, 1)
            
            # Determine zone and color
            if rating_score >= 7:
                zone = "🟢 GOOD ZONE"
                zone_color = "green"
                zone_desc = "Strong technical signals suggest positive momentum"
            elif rating_score >= 4:
                zone = "🟡 NEUTRAL ZONE"
                zone_color = "orange"
                zone_desc = "Mixed signals, exercise caution"
            else:
                zone = "🔴 BAD ZONE"
                zone_color = "red"
                zone_desc = "Weak technical signals indicate potential risks"
            
            # Display rating and zone
            rating_col1, rating_col2 = st.columns([1, 2])
            
            with rating_col1:
                st.metric("Stock Rating", f"{rating_score}/10", 
                         delta=zone, delta_color=zone_color)
                
                # Rating gauge
                st.progress(rating_score/10)
            
            with rating_col2:
                st.markdown(f"**Zone Status:** {zone}")
                st.markdown(f"*{zone_desc}*")
                
                with st.expander("📊 Rating Factors"):
                    for factor in rating_factors:
                        st.markdown(f"• {factor}")
            
            # ── Additional Analysis Details ──
            st.divider()
            st.subheader('🔍 Additional Analysis Details')
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("**Technical Strength:**")
                
                # Calculate technical strength percentage
                strength_indicators = []
                
                # RSI strength
                if 45 <= rsi_val <= 55:
                    strength_indicators.append(1)
                elif 35 <= rsi_val <= 65:
                    strength_indicators.append(0.7)
                else:
                    strength_indicators.append(0.3)
                
                # Trend strength
                if 'MA50' in hist.columns and 'MA200' in hist.columns:
                    if current > ma50 and ma50 > ma200:
                        strength_indicators.append(1)
                    elif current > ma50 or ma50 > ma200:
                        strength_indicators.append(0.7)
                    else:
                        strength_indicators.append(0.3)
                
                # MACD strength
                if macd > signal:
                    strength_indicators.append(1)
                else:
                    strength_indicators.append(0.3)
                
                tech_strength = np.mean(strength_indicators) * 100 if strength_indicators else 0
                
                st.metric("Technical Strength", f"{tech_strength:.1f}%")
                
                # Support/Resistance levels
                st.markdown("**Key Levels:**")
                if len(hist) >= 50:
                    recent_high = hist['High'].tail(50).max()
                    recent_low = hist['Low'].tail(50).min()
                    st.write(f"Resistance: ₹{recent_high:.2f}")
                    st.write(f"Support: ₹{recent_low:.2f}")
            
            with detail_col2:
                st.markdown("**Market Sentiment:**")
                
                # Calculate sentiment based on various factors
                sentiment_score = 0
                
                # RSI sentiment
                if rsi_val > 60:
                    sentiment_score += 0.3  # Bullish
                elif rsi_val < 40:
                    sentiment_score -= 0.3  # Bearish
                
                # MACD sentiment
                if macd > signal:
                    sentiment_score += 0.3
                else:
                    sentiment_score -= 0.3
                
                # Volume sentiment
                if 'Volume' in hist.columns:
                    if recent_vol > avg_vol:
                        sentiment_score += 0.2
                
                # Trend sentiment
                if current > ma50:
                    sentiment_score += 0.2
                
                if sentiment_score > 0.3:
                    sentiment = "🐂 Bullish"
                    sent_color = "green"
                elif sentiment_score < -0.3:
                    sentiment = "🐻 Bearish"
                    sent_color = "red"
                else:
                    sentiment = "😐 Neutral"
                    sent_color = "gray"
                
                st.markdown(f"**{sentiment}**")
                
                # Risk assessment
                st.markdown("**Risk Assessment:**")
                if volatility > 50:
                    st.error("High Risk - Volatile")
                elif volatility > 30:
                    st.warning("Medium Risk")
                else:
                    st.success("Low Risk - Stable")
            
            # ── Fundamental Score ──
            st.divider()
            st.subheader('Fundamental Score')
            
            # Calculate fundamental score
            score, label, reasons = fundamental_score(info)
            
            # Determine category and colors
            if score >= 3:
                category = "STRONG"
                bg_color = "linear-gradient(135deg, #0f766e 0%, #14b8a6 100%)"
                text_color = "#ecfdf5"
                message = "Strong Fundamentals - Good potential"
            elif score >= 2:
                category = "NEUTRAL"
                bg_color = "linear-gradient(135deg, #78350f 0%, #b45309 100%)"
                text_color = "#fffbeb"
                message = "Average Fundamentals - Mixed signals"
            else:
                category = "WEAK"
                bg_color = "linear-gradient(135deg, #7c2d12 0%, #dc2626 100%)"
                text_color = "#fef2f2"
                message = "Weak Fundamentals - Watch carefully"
            
            # Display balanced score card
            col_score1, col_score2 = st.columns([1.2, 1.8])
            
            with col_score1:
                st.markdown(f"""
                <div style='text-align: center; padding: 24px 16px; border-radius: 12px; background: {bg_color}; color: {text_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
                    <div style='font-size: 28px; font-weight: 700; margin: 0 0 8px 0;'>{label}</div>
                    <div style='font-size: 14px; font-weight: 600; margin: 0;'>{category}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_score2:
                st.markdown(f"**Assessment:** {message}")
                st.markdown("**Key Metrics to Watch:**")
                for reason in reasons[:3]:
                    st.write(f"• {reason}")
            
            # ── Recent Data Table ──
            st.divider()
            st.subheader('📋 Recent Technical Data')
            
            # Show last 10 rows with key indicators
            display_data = hist[[
                'Close',
                'MA50',
                'MA200', 
                'RSI',
                'MACD',
                'MACD_Signal',
                'MACD_Histogram'
            ]].tail(10)
            
            # Format the data nicely
            formatted_data = display_data.copy()
            formatted_data = formatted_data.round({
                'Close': 2,
                'MA50': 2,
                'MA200': 2,
                'RSI': 1,
                'MACD': 3,
                'MACD_Signal': 3,
                'MACD_Histogram': 3
            })
            
            st.dataframe(
                formatted_data,
                width='stretch',
                column_config={
                    "Close": st.column_config.NumberColumn("Close (₹)", format="₹%.2f"),
                    "MA50": st.column_config.NumberColumn("MA50 (₹)", format="₹%.2f"),
                    "MA200": st.column_config.NumberColumn("MA200 (₹)", format="₹%.2f"),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                    "MACD": st.column_config.NumberColumn("MACD", format="%.3f"),
                    "MACD_Signal": st.column_config.NumberColumn("Signal", format="%.3f"),
                    "MACD_Histogram": st.column_config.NumberColumn("Histogram", format="%.3f"),
                }
            )
            
        except Exception as e:
            # Display error message if something goes wrong
            st.error(f'Error: {str(e)}')
            st.info('Please check the stock symbol and try again.')


# Run the dashboard
if __name__ == '__main__':
    display_analysis_dashboard()
 