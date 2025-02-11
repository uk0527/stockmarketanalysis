# Market Navigator

**Market Navigator** is a comprehensive stock analysis tool tailored for the Indian stock market. This application leverages technical indicators, machine learning models, and advanced language models (LLMs) to provide actionable trading insights. Built using Python, Streamlit, and Plotly, it offers an interactive and visually rich experience for traders and investors.

## Features

1. **Technical Analysis:**
   - Calculates technical indicators such as SMA, EMA, RSI, MACD, Bollinger Bands, ADX, OBV, MFI, ATR, and Fibonacci retracement levels.
   - Visualizes price actions using candlestick charts with overlays for moving averages, Bollinger Bands, volume, RSI, and MACD.

2. **Market Patterns Specific to Indian Markets:**
   - Identifies gap-up openings, volume breakouts, support, and resistance levels.
   - Calculates correlation with the Nifty 50 index for better market sentiment analysis.

3. **LLM-Powered Analysis:**
   - Integrates with OpenRouter’s API for GPT-4-based market analysis.
   - Provides detailed explanations for BUY, SELL, or HOLD decisions, with confidence levels and risk assessments.

4. **Fundamental Analysis:**
   - Retrieves fundamental data like Market Cap, P/E Ratio, Dividend Yield, and more using Yahoo Finance.

5. **Interactive Dashboard:**
   - Built with Streamlit for an interactive web-based interface.
   - Allows users to navigate between different sections like Analysis, Watchlist, Performance, and Settings.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   - Create a `.env` file in the project root directory and add your API key:
     ```ini
     OPENROUTER_API_KEY=your_api_key_here
     ```

## Usage

1. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

2. **Navigate Through the Dashboard:**
   - **Analysis:** Enter the stock symbol (e.g., `RELIANCE.NS`) and run the analysis.
   - **Watchlist:** Manage your favorite stocks (feature coming soon).
   - **Performance:** Analyze your trading performance (feature coming soon).
   - **Settings:** Customize your app preferences (feature coming soon).

## Dependencies

- **Python Libraries:**
  - `yfinance` for fetching stock data.
  - `pandas`, `numpy` for data manipulation.
  - `ta` for technical analysis indicators.
  - `plotly`, `streamlit`, `streamlit_option_menu` for visualization and UI.
  - `requests`, `dotenv` for API interaction and environment management.

- **External APIs:**
  - OpenRouter’s GPT-4 API for LLM-based stock analysis.

## File Structure

```
.
├── app.py                  # Main application script
├── requirements.txt        # List of dependencies
├── .env                    # Environment variables (API keys)
├── README.md               # Project documentation
└── assets/
    └── stock.jpg           # Image used in the sidebar
```

## Customization

- **Styling:**
  The application uses custom CSS for a modern enterprise look. You can modify the styling inside the `st.markdown` section of the `main()` function.

- **API Key:**
  Ensure your API key is securely stored in the `.env` file. The application will not run without a valid key for LLM analysis.

## Troubleshooting

- **Common Issues:**
  - **API Key Errors:** Ensure your `.env` file is correctly formatted and the key is valid.
  - **Data Fetching Errors:** Verify the stock symbol is correct and Yahoo Finance has data available for it.
  - **LLM Response Issues:** If the LLM response is not parsed correctly, the fallback mechanisms will default to conservative "HOLD" decisions.

- **Debugging:**
  The application logs certain outputs (e.g., API key prefix, LLM raw responses) for debugging. Review these logs in the Streamlit app or console for troubleshooting.

## Contributing

Contributions are welcome! Feel free to fork the repository, make changes, and submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **OpenRouter API** for providing GPT-4-based LLM services.
- **Yahoo Finance (yfinance)** for stock data retrieval.
- **Streamlit** and **Plotly** for the interactive UI and visualizations.

---

Happy Trading! 📈📉

