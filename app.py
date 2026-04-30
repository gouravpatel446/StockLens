import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Import modules with error handling
try:
    from technicalindicators.analysis import display_analysis_dashboard
except ImportError:
    display_analysis_dashboard = None

try:
    from mlmodels import predict_linear, get_prediction_summary, predict_lstm, get_lstm_prediction_summary
except ImportError:
    predict_linear = None
    predict_lstm = None
    get_prediction_summary = None
    get_lstm_prediction_summary = None

# Configure page settings
st.set_page_config(page_title='Stock Market Analyzer', page_icon='chart', layout='wide', initial_sidebar_state='expanded')

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style='padding: 12px 0; text-align: center;'>
            <div style='font-size: 28px; font-weight: 800; color: #0f172a;'>🔎 Stock Lens</div>
            <div style='font-size: 14px; color: #475569; margin-top: 4px;'>Market dashboard + technical scanner</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()
    st.title('Menu')
    page = st.radio(
        'Go to:',
        options=['Overview', 'Analysis', 'ML Prediction'],
        index=0
    )
    st.divider()


# Main Content
if page == 'Overview':
    st.markdown("""
    <style>
        .overview-card {background: #ffffff; border-radius: 20px; padding: 28px; color: #0f172a; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);}
        .overview-input {background: rgba(226, 232, 240, 0.95); border-radius: 18px; padding: 24px;}
        .overview-metric {background: rgba(14, 165, 233, 0.12); border-radius: 16px; padding: 18px;}
        .overview-chart-header {font-size: 22px; font-weight: 600; color: #0f172a; margin-bottom: 8px;}
        .overview-chart-sub {color: #475569; margin-top: 0; margin-bottom: 14px;}
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        name = st.text_input(
            'Analyst Name (Optional)',
            placeholder='Your name for reference',
            help='Optional entry for personalized analysis'
        )
    with col2:
        symbol = st.text_input(
            'Stock Symbol',
            value=st.session_state.get('last_analyzed_symbol', 'RELIANCE.NS'),
            placeholder='e.g., RELIANCE.NS',
            help='Enter the stock ticker symbol to analyze',
            key='overview_symbol'
        )
    with col3:
        period = st.selectbox(
            'Time Period',
            options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=3,
            help='Select the historical timeframe for the chart'
        )

    if st.button('Analyze Stock', type='primary', key='overview_analyze', width='stretch'):
        if not symbol.strip():
            st.warning('Please enter a stock symbol.')
        else:
            st.session_state['last_analyzed_symbol'] = symbol.upper()
            with st.spinner('Loading stock data...'):
                try:
                    stock = yf.Ticker(symbol.upper())
                    hist = stock.history(period=period)
                    info = stock.info or {}
                except Exception:
                    st.error('Unable to fetch data from Yahoo Finance. Check your network connection and try again.')
                    st.info('If the problem continues, verify your internet access or use a different stock symbol.')
                    st.stop()

                if hist is None or getattr(hist, 'empty', True):
                    st.error('No historical data was returned for this ticker. Please check the symbol or try a different timeframe.')
                    st.stop()

                title = f'Analysis Complete for {symbol.upper()}'
                if name.strip():
                    title = f'Analysis Complete for {symbol.upper()}, {name}'

                st.success(title)
                st.markdown(f"**Period:** {period.upper()} • **Data points:** {len(hist)}")

                price = info.get('currentPrice') if isinstance(info, dict) else None
                if price is None:
                    price = hist['Close'].iloc[-1]

                change = hist['Close'].pct_change().iloc[-1] * 100 if len(hist) > 1 else 0
                high = info.get('dayHigh', hist['High'].max() if 'High' in hist.columns else price)
                low = info.get('dayLow', hist['Low'].min() if 'Low' in hist.columns else price)

                metric_cols = st.columns(4)
                metric_cols[0].metric('Current Price', f'₹{price:.2f}')
                metric_cols[1].metric('Today Change', f'{change:+.2f}%')
                metric_cols[2].metric('Day High', f'₹{high:.2f}')
                metric_cols[3].metric('Day Low', f'₹{low:.2f}')

                # Company profile and technical headline summary
                company_name = info.get('longName') if isinstance(info, dict) else None
                company_sector = info.get('sector') if isinstance(info, dict) else None
                company_industry = info.get('industry') if isinstance(info, dict) else None
                market_cap = info.get('marketCap') if isinstance(info, dict) else None
                beta = info.get('beta') if isinstance(info, dict) else None
                trailing_pe = info.get('trailingPE') if isinstance(info, dict) else None
                forward_pe = info.get('forwardPE') if isinstance(info, dict) else None
                eps = info.get('trailingEps') if isinstance(info, dict) else None
                volume = info.get('volume') if isinstance(info, dict) else (hist['Volume'].iloc[-1] if 'Volume' in hist.columns else None)
                avg_volume = info.get('averageDailyVolume3Month') if isinstance(info, dict) else None
                fifty_two_week_low = info.get('fiftyTwoWeekLow') if isinstance(info, dict) else None
                fifty_two_week_high = info.get('fiftyTwoWeekHigh') if isinstance(info, dict) else None
                dividend_yield = info.get('dividendYield') if isinstance(info, dict) else None
                revenue_ttm = info.get('totalRevenue') or info.get('revenueTTM') or info.get('revenue') if isinstance(info, dict) else None
                profit_margin = info.get('profitMargins') if isinstance(info, dict) else None
                return_on_equity = info.get('returnOnEquity') if isinstance(info, dict) else None
                debt_to_equity = info.get('debtToEquity') if isinstance(info, dict) else None
                company_description = info.get('longBusinessSummary') if isinstance(info, dict) else ''
                company_description = company_description.strip() if isinstance(company_description, str) else ''

                def fmt_currency(value):
                    if isinstance(value, (int, float)):
                        if abs(value) >= 1e12:
                            return f'₹{value / 1e12:.2f}T'
                        if abs(value) >= 1e9:
                            return f'₹{value / 1e9:.2f}B'
                        if abs(value) >= 1e6:
                            return f'₹{value / 1e6:.2f}M'
                        return f'₹{value:,.0f}'
                    return 'N/A'

                def fmt_percent(value, precision=2):
                    if isinstance(value, (int, float)):
                        return f'{value * 100:.{precision}f}%'
                    return 'N/A'

                def fmt_number(value):
                    if isinstance(value, (int, float)):
                        return f'{int(value):,}'
                    return 'N/A'

                beta_text = f'{beta:.2f}' if isinstance(beta, (int, float)) else 'N/A'
                trailing_pe_text = f'{trailing_pe:.2f}' if isinstance(trailing_pe, (int, float)) else 'N/A'
                forward_pe_text = f'{forward_pe:.2f}' if isinstance(forward_pe, (int, float)) else 'N/A'
                eps_text = f'{eps:.2f}' if isinstance(eps, (int, float)) else 'N/A'
                market_cap_text = fmt_currency(market_cap)
                range_text = f'₹{fifty_two_week_low:.2f} - ₹{fifty_two_week_high:.2f}' if isinstance(fifty_two_week_low, (int, float)) and isinstance(fifty_two_week_high, (int, float)) else 'N/A'
                liquidity_text = fmt_number(avg_volume)
                dividend_yield_text = fmt_percent(dividend_yield, 2) if dividend_yield is not None else 'N/A'
                revenue_text = fmt_currency(revenue_ttm)
                profit_margin_text = fmt_percent(profit_margin, 1) if profit_margin is not None else 'N/A'
                roe_text = fmt_percent(return_on_equity, 1) if return_on_equity is not None else 'N/A'
                debt_to_equity_text = f'{debt_to_equity:.2f}' if isinstance(debt_to_equity, (int, float)) else 'N/A'
                valuation_text = 'Growth-oriented' if forward_pe and trailing_pe and forward_pe > trailing_pe else 'Value-oriented' if trailing_pe else 'Neutral'
                momentum_text = 'Positive' if price > hist['Close'].rolling(20).mean().iloc[-1] else 'Negative'
                trend_text = 'Bullish' if change >= 0 else 'Bearish'
                volatility_text = 'High' if beta and beta > 1 else 'Moderate' if beta else 'Unknown'
                support_level = f'₹{hist["Close"].min():.2f}'
                resistance_level = f'₹{hist["Close"].max():.2f}'

                st.markdown('### Company & Technical Overview')
                overview_top_left, overview_top_right = st.columns([3, 1], gap='large')
                with overview_top_left:
                    st.markdown(f"**{company_name or symbol.upper()}** • {company_sector or 'Sector N/A'} / {company_industry or 'Industry N/A'}")
                    company_description_text = company_description[:280] + ('...' if len(company_description) > 280 else '')
                    st.markdown(company_description_text)
                with overview_top_right:
                    st.markdown('#### Market Snapshot')
                    st.metric('Today Change', f'{change:+.2f}%')
                    st.metric('Trend', trend_text)
                    st.metric('Support', support_level)
                    st.metric('Resistance', resistance_level)

                st.markdown('### Overview Metrics')
                row1 = st.columns(4)
                row1[0].metric('Market cap', market_cap_text)
                row1[1].metric('P/E ratio', trailing_pe_text if trailing_pe_text != 'N/A' else forward_pe_text)
                row1[2].metric('EPS (TTM)', eps_text)
                row1[3].metric('Avg volume', liquidity_text)

                row2 = st.columns(4)
                row2[0].metric('52w high', f'₹{fifty_two_week_high:.2f}' if isinstance(fifty_two_week_high, (int, float)) else 'N/A')
                row2[1].metric('52w low', f'₹{fifty_two_week_low:.2f}' if isinstance(fifty_two_week_low, (int, float)) else 'N/A')
                row2[2].metric('Dividend yield', dividend_yield_text)
                row2[3].metric('Beta', beta_text)

                row3 = st.columns(4)
                row3[0].metric('Revenue (TTM)', revenue_text)
                row3[1].metric('Profit margin', profit_margin_text)
                row3[2].metric('Return on equity', roe_text)
                row3[3].metric('Debt / equity', debt_to_equity_text)

                overview_cols = st.columns([2, 1])
                with overview_cols[0]:
                    st.markdown('**Fundamental Profile**')
                    st.markdown(f"""
                    - **Market Cap:** {market_cap_text}
                    - **52-Week Range:** {range_text}
                    - **Beta:** {beta_text}
                    - **Trailing P/E:** {trailing_pe_text}
                    - **EPS:** {eps_text}
                    - **Liquidity (3M avg):** {liquidity_text}
                    - **Dividend Yield:** {dividend_yield_text}
                    - **Revenue (TTM):** {revenue_text}
                    - **Profit Margin:** {profit_margin_text}
                    - **Return on Equity:** {roe_text}
                    - **Debt / Equity:** {debt_to_equity_text}
                    """)

                    st.markdown('**Technical Insight**')
                    st.markdown(f"""
                    - **Momentum:** {momentum_text}
                    - **Volatility:** {volatility_text}
                    - **Valuation Bias:** {valuation_text}
                    - **Price Action:** Close price relative to moving averages
                    - **Structure:** Support and resistance are derived from recent price range
                    """)

                with overview_cols[1]:
                    st.markdown('**Trading Signals**')
                    signal_text = 'Bullish continuation if price stays above MA20/MA50, watch for breakout volume.' if trend_text == 'Bullish' else 'Bearish pressure may persist until price reclaims MA20 support.'
                    st.markdown(signal_text)
                    st.markdown(f"**Relative Strength:** {trend_text} / **Liquidity:** {liquidity_text}")

                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                hist['MA50'] = hist['Close'].rolling(window=50).mean()

                # Create multi-panel chart like Analysis page
                fig = make_subplots(
                    rows=3,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=[
                        'Price & Moving Averages',
                        'Volume',
                        'Price Change %'
                    ],
                    row_heights=[0.5, 0.25, 0.25],
                    vertical_spacing=0.08
                )
                # ── Panel 1: Price and Moving Averages ──
                fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            name='Close Price',
                            line=dict(color='#1F4E79', width=2),
                            hovertemplate='<b>Close</b><br>%{x|%Y-%m-%d}<br>₹%{y:.2f}<extra></extra>'
                        ),
                        row=1,
                        col=1
                    )
                fig.add_trace(
                    go.Scatter(
                        x=hist.index,
                        y=hist['MA20'],
                        name='MA20',
                        line=dict(color='#38bdf8', width=2, dash='dot'),
                        hovertemplate='<b>20-day MA</b><br>%{x|%Y-%m-%d}<br>₹%{y:.2f}<extra></extra>'
                    ),
                    row=1,
                    col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=hist.index,
                        y=hist['MA50'],
                        name='MA50',
                        line=dict(color='#f97316', width=2, dash='dash'),
                        hovertemplate='<b>50-day MA</b><br>%{x|%Y-%m-%d}<br>₹%{y:.2f}<extra></extra>'
                    ),
                    row=1,
                    col=1
                )
                # ── Panel 2: Volume ──
                if 'Volume' in hist.columns and 'Open' in hist.columns:
                    colors = ['green' if hist['Close'].iloc[i] >= hist['Open'].iloc[i] else 'red' for i in range(len(hist))]
                    fig.add_trace(
                        go.Bar(
                            x=hist.index,
                            y=hist['Volume'],
                            name='Volume',
                            marker_color=colors,
                            opacity=0.7,
                            hovertemplate='<b>Volume</b><br>%{x|%Y-%m-%d}<br>%{y:,}<extra></extra>'
                        ),
                        row=2,
                        col=1
                    )
                elif 'Volume' in hist.columns:
                    fig.add_trace(
                        go.Bar(
                            x=hist.index,
                            y=hist['Volume'],
                            name='Volume',
                            opacity=0.7,
                            hovertemplate='<b>Volume</b><br>%{x|%Y-%m-%d}<br>%{y:,}<extra></extra>'
                        ),
                        row=2,
                        col=1
                    )
                price_change_pct = hist['Close'].pct_change() * 100
                colors_change = ['green' if val >= 0 else 'red' for val in price_change_pct]
                fig.add_trace(
                    go.Bar(
                        x=hist.index,
                        y=price_change_pct,
                        name='Daily Change %',
                        marker_color=colors_change,
                        opacity=0.6,
                        hovertemplate='<b>Daily Change</b><br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>'
                    ),
                    row=3,
                    col=1
                )

                # Update layout with light theme
                fig.update_layout(
                    height=800,
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#ffffff',
                    font=dict(color='#1e293b', family='Inter, sans-serif'),
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1,
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='rgba(0,0,0,0)',
                        font=dict(color='#1e293b')
                    ),
                    hoverlabel=dict(bgcolor='#f1f5f9', bordercolor='#cbd5e1', font=dict(color='#0f172a')),
                    xaxis=dict(showgrid=False, tickfont=dict(color='#475569'), linecolor='#cbd5e1', zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color='#475569'), linecolor='#cbd5e1'),
                    xaxis2=dict(showgrid=False, tickfont=dict(color='#475569'), linecolor='#cbd5e1'),
                    yaxis2=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color='#475569'), linecolor='#cbd5e1'),
                    xaxis3=dict(showgrid=False, tickfont=dict(color='#475569'), linecolor='#cbd5e1'),
                    yaxis3=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)', tickfont=dict(color='#475569'), linecolor='#cbd5e1')
                )

                st.plotly_chart(fig, width='stretch')

                # OHLC Data Presentation
                st.markdown('### Recent OHLC Data')
                ohlc_data = hist[['Open', 'High', 'Low', 'Close']].tail(10).copy()
                ohlc_data.index = ohlc_data.index.strftime('%Y-%m-%d')

                # Add change calculations
                ohlc_data['Change'] = ohlc_data['Close'].diff()
                ohlc_data['Change %'] = ohlc_data['Close'].pct_change() * 100

                # Format for display
                ohlc_display = pd.DataFrame({
                    'Date': ohlc_data.index,
                    'Open': ohlc_data['Open'].round(2),
                    'High': ohlc_data['High'].round(2),
                    'Low': ohlc_data['Low'].round(2),
                    'Close': ohlc_data['Close'].round(2),
                    'Change': ohlc_data['Change'].round(2),
                    'Change %': ohlc_data['Change %'].round(2)
                })

                st.dataframe(ohlc_display.style.format({
                    'Open': '₹{:.2f}',
                    'High': '₹{:.2f}',
                    'Low': '₹{:.2f}',
                    'Close': '₹{:.2f}',
                    'Change': '₹{:+.2f}',
                    'Change %': '{:+.2f}%'
                }).apply(lambda x: ['background-color: #dcfce7' if v > 0 else 'background-color: #fecaca' for v in x] if x.name == 'Change' else ['' for _ in x], axis=0), width='stretch')

elif page == 'Analysis':
    # Display technical analysis dashboard
    st.title('Technical Analysis')
    st.markdown('Comprehensive indicator analysis, trend evaluation, and trading signals.')

    if display_analysis_dashboard is not None:
        try:
            preset_symbol = st.session_state.get('last_analyzed_symbol')
            display_analysis_dashboard(preset_symbol=preset_symbol)
        except Exception as e:
            st.error(f'Analysis module error: {str(e)}')
            st.info('Technical analysis dashboard is not fully configured yet. Please check the technicalindicators module.')
    else:
        st.warning('Technical analysis module not available. Please ensure technicalindicators package is installed.')

elif page == 'ML Prediction':
    # ML Prediction Page
    st.title('AI Price Prediction Engine')
    st.markdown('Advanced machine learning algorithms forecasting future stock prices for the next 90 days.')

    # ── Input Section ──
    col1, col2 = st.columns([2, 1])

    with col1:
        # Get stock symbol from user
        symbol = st.text_input(
            'Stock Symbol',
            value='AAPL',
            placeholder='e.g., AAPL, GOOGL, RELIANCE.NS',
            key='prediction_symbol'
        )

    with col2:
        # Get time period for training data
        train_period = st.selectbox(
            'Training Data Period',
            options=['6mo', '1y', '2y', '5y'],
            index=1,
            help='Historical data used to train the ML model'
        )

    # Model Selection
    model_type = st.selectbox(
        'Choose AI Algorithm',
        options=['Linear Regression', 'LSTM Neural Network'],
        index=0,  # Default to Linear Regression
        help='Linear Regression: Statistical trend analysis | LSTM: Advanced deep learning with memory'
    )

    # Prediction Section
    if symbol:
        if predict_linear is None or predict_lstm is None:
            st.error('ML prediction modules not available. Please ensure mlmodels package is installed.')
            st.info('Install with: pip install -r requirements.txt')
        else:
            with st.spinner('Training ML model and generating predictions...'):
                try:
                    stock = yf.Ticker(symbol.upper())
                    hist = stock.history(period=train_period)
                    info = stock.info or {}
                except Exception:
                    st.error('Unable to fetch historical stock data. Check your network connection and try again.')
                    st.info('If the problem continues, verify your internet access or use a different stock symbol.')
                    st.stop()

                # Check if data is available
                if hist is None or getattr(hist, 'empty', True):
                    st.error('Stock not found or returned no data. Check the symbol.')
                    st.stop()

                st.subheader(f"{info.get('longName', symbol.upper())} - 90 Day Price Forecast")

                # Run ML prediction based on selected model
                if model_type == 'Linear Regression':
                    with st.spinner('Training Linear Regression model...'):
                        future_dates, predicted_prices, rmse = predict_linear(hist, forecast_days=90)

                    # Display model accuracy
                    st.metric(
                        'Linear Regression RMSE',
                        f"₹{rmse:.2f}",
                        help='Lower values indicate better prediction accuracy'
                    )

                    model_description = "**Linear Regression**: Simple trend-following model that finds the best straight line through your data."
                    accuracy_info = f"- **Model Accuracy**: ₹{rmse:.2f} RMSE (typical range: ₹1-50 for stocks)"
                    interpretation = """
                    **How to interpret:**
                    - Lower RMSE = More accurate predictions
                    - The dashed line shows the ML model's price trend prediction
                    - This is a trend-following model, not a crystal ball!"""

                else:  # LSTM Neural Network
                    with st.spinner('Training LSTM Neural Network (this may take longer)...'):
                        future_dates, predicted_prices, training_metrics = predict_lstm(hist, forecast_days=90)

                    # LSTM doesn't return RMSE in the same way, so we'll show a different metric
                    st.metric(
                        'AI Model',
                        'LSTM Neural Network',
                        help='Advanced AI that learns complex patterns in stock data'
                    )

                    model_description = "**LSTM Neural Network**: Advanced AI with 'memory' that learns complex patterns and relationships in stock data."
                    accuracy_info = "- **Model Type**: Advanced Neural Network with memory"
                    interpretation = """
                    **How to interpret:**
                    - LSTM learns complex patterns that simple models might miss
                    - The AI has "memory" of past 60 days of price action
                    - Neural networks can capture non-linear relationships
                    - Training takes longer but may find subtle patterns"""

                st.divider()

                # Prediction Chart
                st.subheader('Price Prediction Chart')

                # Create combined chart with historical and predicted data
                fig = go.Figure()

                # Historical data (last 30 days for context)
                hist_recent = hist.tail(30)
                fig.add_trace(go.Scatter(
                    x=hist_recent.index,
                    y=hist_recent['Close'],
                    name='Historical Prices',
                    line=dict(color='#1F4E79', width=2),
                    mode='lines'
                ))

                # Predicted data
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predicted_prices,
                    name='Predicted Prices',
                    line=dict(color='#FF6B35', width=3, dash='dash'),
                    mode='lines+markers',
                    marker=dict(size=4)
                ))

                # Update layout
                fig.update_layout(
                    title=f'{symbol.upper()} - Historical vs 90-Day {model_type} Prediction',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    template='plotly_white',
                    height=500,
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )

                # Display the chart
                st.plotly_chart(fig, width='stretch')

                st.divider()

                # Prediction Summary
                st.subheader('Prediction Summary')

                # Create summary table
                summary_data = {
                    'Date': [date.strftime('%Y-%m-%d') for date in future_dates],
                    'Predicted Price': [f"₹{price:.2f}" for price in predicted_prices]
                }

                # Show first 10 and last 10 predictions
                summary_df = pd.DataFrame(summary_data)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('**First 10 Days:**')
                    st.dataframe(summary_df.head(10), width='stretch')

                with col2:
                    st.markdown('**Last 10 Days:**')
                    st.dataframe(summary_df.tail(10), width='stretch')

                # Key insights
                st.subheader('Key Insights')

                current_price = None
                if isinstance(info, dict):
                    current_price = info.get('currentPrice')
                if current_price is None:
                    current_price = hist['Close'].iloc[-1]

                predicted_start = predicted_prices[0]
                predicted_end = predicted_prices[-1]

                price_change = ((predicted_end - predicted_start) / predicted_start) * 100
                trend = "Bullish" if price_change > 0 else "Bearish"

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        'Current Price',
                        f"₹{current_price:.2f}"
                    )

                with col2:
                    st.metric(
                        'Predicted Price (90 days)',
                        f"₹{predicted_end:.2f}",
                        f"{price_change:+.1f}%"
                    )

                with col3:
                    st.metric(
                        'Trend Direction',
                        trend,
                        help=f"Expected price movement over next 90 days"
                    )

                # Additional info
                st.info(f"""
                **{model_type} Details:**
                - **Algorithm**: {model_type}
                - **Training Data**: {len(hist)} days of historical prices
                - **Forecast Period**: 90 business days (3 months)
                {accuracy_info}

                {interpretation}
                """)


