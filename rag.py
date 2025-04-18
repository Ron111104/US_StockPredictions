import os
import yfinance as yf
import praw
import requests
from newsapi import NewsApiClient
from alpha_vantage.fundamentaldata import FundamentalData
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up global headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
}

class RAGStockAnalyzer:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-0125-preview"
        )
        self.vector_store = None

        # Initialize APIs with error handling
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent=HEADERS['User-Agent']
            )
        except Exception as e:
            print(f"Error initializing Reddit API: {str(e)}")
        
        try:
            self.newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
        except Exception as e:
            print(f"Error initializing NewsAPI: {str(e)}")

        try:
            self.alpha_vantage = FundamentalData(key=os.getenv("ALPHA_VANTAGE_KEY"))
        except Exception as e:
            print(f"Error initializing Alpha Vantage API: {str(e)}")

    def gather_reddit_posts(self, subreddits=['stocks', 'wallstreetbets', 'investing'], count=20) -> List[str]:
        """Fetch recent Reddit posts related to the stock."""
        posts = []
        try:
            for subreddit in subreddits:
                for post in self.reddit.subreddit(subreddit).search(
                    self.ticker, sort='new', time_filter='month', limit=count
                ):
                    posts.append(f"[Reddit {subreddit}] Title: {post.title}\nContent: {post.selftext}")
        except Exception as e:
            print(f"Error gathering Reddit posts: {str(e)}")
        return posts

    def gather_news(self) -> List[str]:
        """Fetch recent news articles using NewsAPI."""
        try:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            articles = self.newsapi.get_everything(
                q=self.ticker, language='en', sort_by='publishedAt', from_param=from_date, page_size=20
            )
            return [
                f"[News {article['publishedAt']}] Title: {article['title']}\nDescription: {article['description']}" 
                for article in articles['articles']
            ]
        except Exception as e:
            print(f"Error gathering news: {str(e)}")
            return []

    def gather_yahoo_data(self) -> List[str]:
        """Gather recent Yahoo Finance data."""
        try:
            info = self.stock.info
            print(self.stock)  # For debugging

            # Instead of using the deprecated quarterly_earnings,
            # extract Net Income from the income statement.
            income_stmt = self.stock.income_stmt
            net_income_str = ""
            if income_stmt is not None and isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty:
                if "Net Income" in income_stmt.index:
                    net_income = income_stmt.loc["Net Income"].iloc[0]
                    net_income_str = f"Net Income: {net_income}"
                else:
                    net_income_str = "Net Income data not found in income statement."
            else:
                net_income_str = "Income statement data is unavailable."

            # Analyst Recommendations (unchanged)
            recommendations = self.stock.recommendations
            recommendations_str = ""
            if recommendations is not None and isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                recommendations_str = "Recent Analyst Recommendations:\n" + recommendations.tail(5).to_string()

            return [
                f"Business Summary: {info.get('longBusinessSummary', '')}",
                f"Industry: {info.get('industry', '')}",
                f"Sector: {info.get('sector', '')}",
                net_income_str,
                recommendations_str
            ]
        except Exception as e:
            print(f"Error gathering Yahoo data: {str(e)}")
            return []

    def gather_alpha_vantage_data(self) -> Dict:
        """Fetch financial data from Alpha Vantage."""
        try:
            balance_sheet_df, _ = self.alpha_vantage.get_balance_sheet_annual(self.ticker)
            income_statement_df, _ = self.alpha_vantage.get_income_statement_annual(self.ticker)
            
            balance_sheet_dict = {}
            income_statement_dict = {}
            
            if balance_sheet_df is not None and not balance_sheet_df.empty:
                balance_sheet_dict = balance_sheet_df.head(1).to_dict('records')[0]
            if income_statement_df is not None and not income_statement_df.empty:
                income_statement_dict = income_statement_df.head(1).to_dict('records')[0]
            
            return {
                'balance_sheet': balance_sheet_dict,
                'income_statement': income_statement_dict
            }
        except Exception as e:
            print(f"Error gathering Alpha Vantage data: {str(e)}")
            return {}

    def create_knowledge_base(self):
        """Create vector store from gathered data."""
        try:
            sources = self.gather_news() + self.gather_reddit_posts() + self.gather_yahoo_data()
            sources = [s for s in sources if s]  # Remove empty entries

            if not sources:
                raise ValueError("No data gathered from any source")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            from langchain.schema import Document
            documents = [Document(page_content=text) for text in sources]
            splits = text_splitter.split_documents(documents)

            if not splits:
                raise ValueError("No documents created after splitting")

            print(f"Created {len(splits)} document chunks for analysis")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)

        except Exception as e:
            print(f"Error creating knowledge base: {str(e)}")
            raise

    def analyze_aspect(self, aspect: str) -> str:
        """Analyze specific aspects using RAG."""
        try:
            if not self.vector_store:
                self.create_knowledge_base()

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5})
            )

            query = f"Analyze the {aspect} of {self.ticker}. Provide details and evidence."
            # Use invoke instead of run to avoid deprecation warnings
            return qa_chain.invoke(query)
        except Exception as e:
            return f"Error analyzing {aspect}: {str(e)}"

    def get_full_analysis(self) -> Dict:
        """Generate a full stock analysis report."""
        try:
            info = self.stock.info
            return {
                "ticker": self.ticker,
                "company_name": info.get('longName', ''),
                "sector": info.get('sector', ''),
                "current_price": info.get('currentPrice', 0),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('forwardPE', 0),
                "investment_thesis": {
                    "business_model": self.analyze_aspect("business model"),
                    "financials": self.analyze_aspect("financial performance"),
                    "risks": self.analyze_aspect("risks and challenges"),
                    "growth": self.analyze_aspect("growth potential"),
                    "valuation": self.analyze_aspect("valuation and price targets")
                },
                "financial_data": self.gather_alpha_vantage_data()
            }
        except Exception as e:
            return {"error": f"Error generating full analysis: {str(e)}"}

if __name__ == "__main__":
    ticker = input("Enter a stock ticker: ").upper()
    analyzer = RAGStockAnalyzer(ticker)
    report = analyzer.get_full_analysis()
    print(report)
