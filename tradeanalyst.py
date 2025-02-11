import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ta
import requests
import os
import json
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from dataclasses import dataclass
from enum import Enum
import plotly.express as px
from streamlit_option_menu import option_menu
from dotenv import load_dotenv

class TradingAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    symbol: str
    action: TradingAction
    confidence: float
    timestamp: datetime
    rationale: str
    price_target: float
    stop_loss: float
    indicators: Dict[str, float]
    risk_level: str

class IndianMarketPatterns:
    """Enhanced patterns and strategies for Indian markets"""
    
    @staticmethod
    def identify_gap_up_opening(data: pd.DataFrame) -> bool:
        if len(data) < 2:
            return False
        return data['Open'].iloc[-1] > data['High'].iloc[-2]
    
    @staticmethod
    def volume_breakout(data: pd.DataFrame, threshold: float = 2.0) -> bool:
        avg_volume = data['Volume'].rolling(window=20).mean()
        return data['Volume'].iloc[-1] > threshold * avg_volume.iloc[-1]
    
    @staticmethod
    def nifty_correlation(data: pd.DataFrame) -> float:
        try:
            nifty = yf.download('^NSEI', start=data.index[0], end=data.index[-1])
            return data['Close'].pct_change().corr(nifty['Close'].pct_change())
        except Exception:
            return 0.0
    
    @staticmethod
    def identify_support_resistance(data: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Identify support and resistance levels"""
        rolling_min = data['Low'].rolling(window=window).min()
        rolling_max = data['High'].rolling(window=window).max()
        return rolling_min.iloc[-1], rolling_max.iloc[-1]

class AdvancedTechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators with error handling"""
        try:
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_13'] = ta.trend.ema_indicator(df['Close'], window=13)
            df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume']).money_flow_index()
            
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_middle'] = bb.bollinger_mavg()
            df['BB_lower'] = bb.bollinger_lband()
            
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            df['Daily_Range'] = df['High'] - df['Low']
            df['Gap_Up'] = df['Open'] > df['Close'].shift(1)
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            
            high = df['High'].max()
            low = df['Low'].min()
            diff = high - low
            df['Fib_23.6'] = high - (diff * 0.236)
            df['Fib_38.2'] = high - (diff * 0.382)
            df['Fib_50.0'] = high - (diff * 0.5)
            df['Fib_61.8'] = high - (diff * 0.618)
            
            df.dropna(inplace=True)
            return df
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return df

class EnhancedLLMAnalyzer:
    def __init__(self, api_key: str = None):
        from dotenv import load_dotenv
        load_dotenv()  # Load variables from .env

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        # Debugging line to check if the API key is loaded
        st.write(f"Loaded API Key: {self.api_key[:4]}...")  # Only display first few characters

        if not self.api_key:
            raise ValueError("API key is not set. Please add OPENROUTER_API_KEY to the .env file or set it as an environment variable.")

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def analyze_market_conditions(self, technical_data: pd.DataFrame, fundamental_data: Dict, market_patterns: Dict) -> Tuple[str, Dict]:
        try:
            # Send data to LLM
            prompt = self._create_enhanced_prompt(technical_data, fundamental_data, market_patterns)
            llm_response = self._get_llm_response(prompt)
            llm_analysis, llm_data = self._parse_llm_response(llm_response)

            # Extract action and confidence
            action = llm_data.get('action', 'HOLD')
            confidence = llm_data.get('confidence', 0.5)
            price_target = llm_data.get('price_target', technical_data['Close'].iloc[-1] * 1.05)
            stop_loss = llm_data.get('stop_loss', technical_data['Close'].iloc[-1] * 0.95)

            # Fallback explanations if missing
            simple_explanation = llm_data.get('simple_explanation')
            if not simple_explanation:
                if action == "BUY":
                    simple_explanation = "The stock is showing positive trends, suggesting potential for growth. It’s a good time to consider buying."
                elif action == "SELL":
                    simple_explanation = "The stock shows signs of potential decline. Selling now could help avoid losses."
                else:
                    simple_explanation = "The stock is stable without clear signs of rising or falling. Holding is the safest option for now."

            analysis_data = {
                "action": action,
                "confidence": confidence,
                "price_target": price_target,
                "stop_loss": stop_loss,
                "risk_level": llm_data.get('risk_level', 'MEDIUM'),
                "simple_explanation": simple_explanation
            }

            return "LLM-based analysis completed", analysis_data

        except Exception as e:
            st.error(f"Error in LLM-based analysis: {str(e)}")
            return "Error in analysis, using conservative estimates", {
                "action": "HOLD",
                "confidence": 0.3,
                "price_target": technical_data['Close'].iloc[-1] * 1.05,
                "stop_loss": technical_data['Close'].iloc[-1] * 0.95,
                "risk_level": "MEDIUM",
                "simple_explanation": "An error occurred. Defaulting to a conservative hold position."
            }


    def _create_enhanced_prompt(self, technical_data: pd.DataFrame, fundamental_data: Dict, market_patterns: Dict) -> str:
        latest_data = technical_data.iloc[-1]
       
        prompt = f"""
        You are an expert Indian stock market analyst with deep knowledge of technical analysis, market trends, and risk management. Based on the following stock data, provide a **detailed explanation of at least 4-5 sentences** on whether to BUY, SELL, or HOLD the stock.

        **Important Instructions for Your Response:**
        1. Clearly state the recommendation (BUY, SELL, HOLD).
        2. Reference specific technical indicators like RSI, MACD, Moving Averages, and Volume trends. Explain how each indicator supports the recommendation.
        3. Describe current market sentiment and trends (e.g., bullish momentum, bearish reversal).
        4. Highlight potential risks (like high volatility or overbought conditions) or opportunities (like strong bullish signals or support levels).
        5. **Do NOT summarize the explanation in a single line. Write at least 8-10 full sentences explaining the rationale.**

        The explanation should be **simple and clear**, even for someone with no stock market knowledge.

        ---

        **Stock Information:**
        - Current Price: ₹{latest_data['Close']:.2f}
        - Recent Price Movement: {'Increased' if latest_data['Close'] > latest_data['Open'] else 'Decreased'}
        - RSI (Relative Strength Index, measures overbought/oversold conditions): {latest_data['RSI']:.2f}
        - MACD (Momentum Indicator): {latest_data['MACD']:.2f}
        - Moving Averages: SMA 20 at ₹{latest_data['SMA_20']:.2f}, SMA 50 at ₹{latest_data['SMA_50']:.2f}

        **Market Patterns:**
        - Gap Up Opening: {'Yes' if market_patterns['gap_up'] else 'No'}
        - Volume Breakout (unusual trading activity): {'Yes' if market_patterns['volume_breakout'] else 'No'}
        - Correlation with Nifty (overall market trend): {market_patterns['nifty_correlation']:.2f}
        - Support Level: ₹{market_patterns['support']:.2f}
        - Resistance Level: ₹{market_patterns['resistance']:.2f}

        **Fundamental Data:**
        - Market Cap: {fundamental_data.get('info', {}).get('marketCap', 'N/A')}
        - P/E Ratio (valuation metric): {fundamental_data.get('info', {}).get('trailingPE', 'N/A')}
        - Dividend Yield: {fundamental_data.get('info', {}).get('dividendYield', 'N/A')}

        ---

        **Provide your response in this JSON format, and ensure the 'simple_explanation' field contains at least 4-5 detailed sentences:**

        {{
            "action": "SELL",
            "confidence": 0.75,
            "price_target": 1150.00,
            "stop_loss": 1300.00,
            "risk_level": "MEDIUM",
            "simple_explanation": "The stock shows a downward trend with the RSI indicating it is approaching oversold territory. The MACD is signaling a bearish crossover, suggesting weakening momentum. Additionally, the 20-day moving average has crossed below the 50-day, a sign of potential continued decline. Volume is decreasing, indicating less investor interest. Selling now could prevent further losses as technical indicators point toward a short-term decline."
        }}
        """
        return prompt

    def _get_llm_response(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": "deepseek/deepseek-r1-distill-llama-70b:free",
                    "messages": [
                        {"role": "system", "content": "You are an expert Indian stock market analyst. Provide analysis in JSON format."},
                        {"role": "user", "content": prompt}
                    ]
                }
                
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=15)
                response.raise_for_status()

                # Debug: Print the raw response
                raw_response = response.json()
                print(f"LLM Raw Response (Attempt {attempt + 1}):", raw_response)  # For debugging
                
                return raw_response['choices'][0]['message']['content']
            
            except requests.exceptions.RequestException as e:
                print(f"LLM Request Error (Attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
                continue

    def _parse_llm_response(self, response: str) -> Tuple[str, Dict]:
            try:
                data = json.loads(response)
                print("Parsed LLM Data:", data)  # Debugging parsed data
                return response, data
            except json.JSONDecodeError:
                print("Failed to parse LLM response. Attempting to clean and re-parse...")

                # Fallback: Try to extract JSON from the response
                cleaned_response = self._extract_json_from_response(response)
                if cleaned_response:
                    try:
                        data = json.loads(cleaned_response)
                        print("Successfully parsed cleaned LLM response:", data)
                        return response, data
                    except json.JSONDecodeError:
                        print("Failed to parse cleaned response.")
                
                return response, {}  # Return empty if parsing fails

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from response string with extra text.
        """
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return match.group(0)
        return ""


class EnhancedTradingBot:
    def __init__(self, api_key: str):
        self.technical_analyzer = AdvancedTechnicalAnalyzer()
        self.llm_analyzer = EnhancedLLMAnalyzer(api_key)
        self.signals_history: List[TradingSignal] = []

    def analyze_stock(self, symbol: str) -> Tuple[TradingSignal, pd.DataFrame, Dict]:
        try:
            # Step 1: Fetch technical data
            data = self._fetch_data(symbol)
            technical_data = self.technical_analyzer.calculate_indicators(data)

            # Step 2: Identify market patterns
            support, resistance = IndianMarketPatterns.identify_support_resistance(data)
            market_patterns = {
                'gap_up': IndianMarketPatterns.identify_gap_up_opening(data),
                'volume_breakout': IndianMarketPatterns.volume_breakout(data),
                'nifty_correlation': IndianMarketPatterns.nifty_correlation(data),
                'support': support,
                'resistance': resistance
            }

            # Step 3: Get fundamental data
            fundamental_data = self._get_fundamental_data(symbol)

            # Step 4: Let the LLM analyze all data and generate a simple explanation
            analysis_text, llm_data = self.llm_analyzer.analyze_market_conditions(
                technical_data, fundamental_data, market_patterns
            )

            # Step 5: Use LLM output for final decision
            signal = TradingSignal(
                symbol=symbol,
                action=TradingAction[llm_data.get('action', 'HOLD')],
                confidence=llm_data.get('confidence', 0.5),
                timestamp=datetime.now(),
                rationale=llm_data.get('simple_explanation', 'LLM provided no explanation.'),
                price_target=llm_data.get('price_target', technical_data['Close'].iloc[-1] * 1.1),
                stop_loss=llm_data.get('stop_loss', technical_data['Close'].iloc[-1] * 0.95),
                indicators={
                    'current_price': technical_data['Close'].iloc[-1],
                    'RSI': technical_data['RSI'].iloc[-1],
                    'MACD': technical_data['MACD'].iloc[-1],
                    'ADX': technical_data['ADX'].iloc[-1],
                    'MFI': technical_data['MFI'].iloc[-1],
                    'support': support,
                    'resistance': resistance
                },
                risk_level=llm_data.get('risk_level', 'MEDIUM')
            )

            self.signals_history.append(signal)
            return signal, technical_data, fundamental_data

        except Exception as e:
            st.error(f"Error in stock analysis: {str(e)}")
            raise

    def _fetch_data(self, symbol: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1y")
            if data.empty:
                raise ValueError(f"No data available for symbol {symbol}")
            return data
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    def _get_fundamental_data(self, symbol: str) -> Dict:
        try:
            stock = yf.Ticker(symbol)
            return {'info': stock.info}
        except Exception as e:
            st.warning(f"Error fetching fundamental data: {str(e)}")
            return {}

def create_technical_chart(data: pd.DataFrame, signal: TradingSignal) -> go.Figure:
    fig = make_subplots(
        rows=4, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("Price Action", "Volume", "RSI", "MACD")
    )

    # Price chart with candlesticks
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ),
        row=1, col=1
    )

    # Add moving averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_20'], name="SMA 20", line=dict(color='orange')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA_50'], name="SMA 50", line=dict(color='blue')),
        row=1, col=1
    )

    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(x=data.index, y=data['BB_upper'], name="BB Upper", line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['BB_lower'], name="BB Lower", line=dict(color='gray', dash='dash')),
        row=1, col=1
    )

    # Volume chart
    colors = ['red' if row['Close'] < row['Open'] else 'green' for index, row in data.iterrows()]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color=colors),
        row=2, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name="RSI", line=dict(color='purple')),
            row=3, col=1
        )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD'], name="MACD", line=dict(color='blue')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['MACD_signal'], name="Signal", line=dict(color='orange')),
        row=4, col=1
    )
    fig.add_trace(
        go.Bar(x=data.index, y=data['MACD_hist'], name="Histogram", marker_color='gray'),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=30, b=50)
    )
    return fig
# Define the main function
def main():
    # Page Configuration
    st.set_page_config(page_title="Personal Market Analyst", layout="wide")

    # Custom Styling for Enterprise Look
    st.markdown("""
        <style>
        .stApp {
            background-color: #f4f7fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            background-color: #0056b3;
            color: white;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #003f7f;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        st.image(r"C:\Users\Acer\stock.jpg", width=180)
        st.markdown("<h1 style='text-align:center;'>Market Navigator</h1>", unsafe_allow_html=True)
        menu_options = ["Analysis", "Watchlist", "Performance", "Settings"]
        selected = st.radio("Navigation", menu_options)

    # Analysis Section
    if selected == "Analysis":
        st.title("Advanced Stock Analysis")

        # Input Fields
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")
        with col2:
            period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        with col3:
            interval = st.selectbox("Interval", ["1d", "5d", "1wk", "1mo"], index=0)

        if st.button("Analyze Stock"):
            with st.spinner("Analyzing stock..."):
                try:
                    api_key = os.getenv("OPENROUTER_API_KEY")
                    trading_bot = EnhancedTradingBot(api_key)

                    # Run the real stock analysis
                    signal, technical_data, fundamental_data = trading_bot.analyze_stock(symbol)

                    st.success("Analysis Complete")

                    # Creating Sub-Tabs
                    tabs = st.tabs(["Overview", "Technical Analysis", "Price Targets", "Fundamental Analysis"])

                    # Overview Tab
                    with tabs[0]:
                        st.metric("Action", signal.action.value, delta="LLM Decision")
                        st.metric("Confidence", f"{signal.confidence * 100:.2f}%")
                        st.metric("Current Price", f"₹{signal.indicators['current_price']:.2f}")

                        st.subheader("Rationale")
                        st.markdown(f"""
                            <div class='metric-card' style='line-height: 1.6;'>{signal.rationale.replace('\\n', ' ')}</div>
                        """, unsafe_allow_html=True)

                    # Technical Analysis Tab
                    with tabs[1]:
                        st.subheader("Technical Chart")
                        st.plotly_chart(create_technical_chart(technical_data, signal), use_container_width=True)

                    # Price Targets Tab
                    with tabs[2]:
                        st.subheader("Price Targets")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Support", f"₹{signal.indicators['support']:.2f}")
                            st.metric("Target Price", f"₹{signal.price_target:.2f}")
                        with col2:
                            st.metric("Resistance", f"₹{signal.indicators['resistance']:.2f}")
                            st.metric("Stop Loss", f"₹{signal.stop_loss:.2f}")

                    # Fundamental Analysis Tab
                    with tabs[3]:
                        st.subheader("Fundamental Data")
                        info_data = fundamental_data.get('info', {})
                        key_info = {
                            "Website": info_data.get("website", "N/A"),
                            "Industry": info_data.get("industry", "N/A"),
                            "Sector": info_data.get("sector", "N/A"),
                            "Full-Time Employees": info_data.get("fullTimeEmployees", "N/A"),
                            "Market Cap": info_data.get("marketCap", "N/A"),
                            "PE Ratio": info_data.get("trailingPE", "N/A"),
                            "Dividend Yield": info_data.get("dividendYield", "N/A")
                        }

                        for key, value in key_info.items():
                            st.metric(key, value)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

    # Watchlist Section
    elif selected == "Watchlist":
        st.title("Watchlist")
        st.info("Watchlist feature coming soon!")

    # Performance Section
    elif selected == "Performance":
        st.title("Performance Analytics")
        st.info("Performance analytics feature coming soon!")

    # Settings Section
    elif selected == "Settings":
        st.title("Settings")
        st.info("Settings configuration coming soon!")


if __name__ == "__main__":
    main()