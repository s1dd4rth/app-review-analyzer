import re
import csv
from datetime import datetime
from collections import Counter
import streamlit as st
import pandas as pd
import plotly.express as px
from app_store_scraper import AppStore
from google_play_scraper import reviews, Sort
from textblob import TextBlob
from tqdm import tqdm
import time
import requests
import logging
from typing import List, Dict, Any

# Configuration
MAX_REVIEWS = 5000
MIN_REVIEWS = 100
DEFAULT_REVIEWS = 1000
POLARITY_THRESHOLD = 0.1

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- URL Parsing and Validation Functions ---

def parse_app_store_url(url):
    """Parses App Store URL to extract country, name, and ID."""
    regex = r"https:\/\/apps\.apple\.com\/(?P<country>[a-z]{2})\/app\/(?P<name>[^\/]+)\/id(?P<id>\d+)"
    match = re.match(regex, url)
    if not match:
        raise ValueError("Invalid App Store URL.")
    return match.group('country'), match.group('name'), match.group('id')

def extract_play_store_package_name(url):
    """Extracts package name from Play Store URL."""
    match = re.search(r"id=([a-zA-Z0-9_.]+)", url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid Play Store URL. Could not find a package name.")

# --- Data Processing Functions ---

def filter_reviews_by_year(reviews, start_year, end_year):
    """Filters reviews based on the specified year range."""
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31, 23, 59, 59)
    
    def get_review_date(review):
        date = review['date']
        if isinstance(date, (int, float)):  # If it's a timestamp
            return datetime.fromtimestamp(date)
        elif isinstance(date, str):  # If it's a string
            return datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        return date  # If it's already a datetime object
    
    return [review for review in reviews if start_date <= get_review_date(review) <= end_date]

def analyze_sentiment(text, rating, polarity_threshold=POLARITY_THRESHOLD):
    """Analyzes sentiment of a review using TextBlob and rating."""
    analysis = TextBlob(text)
    # TODO: Consider using a more sophisticated sentiment analysis model
    # TextBlob might not be the most accurate for this task
    if analysis.sentiment.polarity > polarity_threshold or rating >= 4:
        return 'Positive'
    elif analysis.sentiment.polarity < -polarity_threshold or rating <= 2:
        return 'Negative'
    else:
        return 'Neutral'

# --- Data Fetching Functions ---

def fetch_app_store_reviews(app, how_many):
    """Fetches reviews from the App Store with progress bar and error handling."""
    with st.spinner("Fetching App Store reviews..."):
        progress_bar = st.progress(0)
        try:
            app.review(how_many=how_many)
            time.sleep(5)  # Give some time for the reviews to be fetched
            reviews = app.reviews
            if not reviews:
                st.warning("No reviews were found for this app. The app might not have any reviews yet.")
                return []
            for i, _ in enumerate(reviews):
                progress_bar.progress((i + 1) / how_many)
            return reviews
        except Exception as e:
            st.error(f"Error fetching App Store reviews: {e}")
            st.info("This could be due to rate limiting, network issues, or the app not being available.")
            return []
        finally:
            progress_bar.empty()

def fetch_play_store_reviews(package_name, how_many, start_year, end_year):
    """Fetches reviews from the Play Store with progress bar and error handling."""
    fetched_reviews = []
    continuation_token = None
    with st.spinner("Fetching Play Store reviews..."):
        progress_bar = st.progress(0)
        while len(fetched_reviews) < how_many:
            try:
                # Limit the number of reviews fetched in a single request
                batch_size = min(100, how_many - len(fetched_reviews))
                result, continuation_token = reviews(
                    package_name,
                    lang='en',
                    country='us',
                    sort=Sort.NEWEST,
                    count=batch_size,
                    continuation_token=continuation_token
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching Play Store reviews: {e}")
                return fetched_reviews 
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                return fetched_reviews

            # Filter reviews by year
            result = [r for r in result if start_year <= (r['at'].year if isinstance(r['at'], datetime) else datetime.fromtimestamp(r['at']).year) <= end_year]
            fetched_reviews.extend(result)
            progress_bar.progress(min(len(fetched_reviews) / how_many, 1.0))
            
            if continuation_token is None or not result:
                break
            time.sleep(1)  # Respect rate limits

    return fetched_reviews[:how_many]  # Ensure we don't return more than requested

# --- Streamlit App ---

def validate_years(start_year, end_year):
    """Validates the input years."""
    current_year = datetime.now().year
    if start_year < 2000 or end_year > current_year:
        raise ValueError(f"Years must be between 2000 and {current_year}")
    if start_year > end_year:
        raise ValueError("Start year must be less than or equal to end year")

def validate_app_url(store_type, url):
    """Validates the app URL based on the store type."""
    if store_type == "App Store":
        if not re.match(r"https:\/\/apps\.apple\.com\/[a-z]{2}\/app\/[^\/]+\/id\d+", url):
            raise ValueError("Invalid App Store URL format.")
    else:  # Play Store
        if not re.match(r"https:\/\/play\.google\.com\/store\/apps\/details\?id=[a-zA-Z0-9_.]+", url):
            raise ValueError("Invalid Play Store URL format.")

def create_sentiment_chart(df):
    """Creates a pie chart for sentiment distribution with specific colors."""
    sentiment_counts = Counter(df['Sentiment'])
    colors = {'Positive': '#00FF00', 'Neutral': '#FFA500', 'Negative': '#FF0000'}
    
    fig = px.pie(
        values=sentiment_counts.values(),
        names=sentiment_counts.keys(),
        title="Sentiment Distribution",
        color=sentiment_counts.keys(),
        color_discrete_map=colors
    )
    return fig

def create_rating_chart(df):
    """Creates a bar chart for rating distribution."""
    fig = px.bar(df['Rating'].value_counts().sort_index(), title="Rating Distribution")
    fig.update_xaxes(title="Rating")
    fig.update_yaxes(title="Count")
    return fig

def create_trend_chart(df):
    """Creates a line chart for review trends over time."""
    df['date'] = pd.to_datetime(df['date'])
    df_grouped = df.groupby([df['date'].dt.to_period('M'), 'sentiment']).size().unstack(fill_value=0)
    fig = px.line(df_grouped, x=df_grouped.index.astype(str), y=df_grouped.columns)
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Number of Reviews")
    return fig

def fetch_and_analyze_reviews(store_type: str, app_url: str, start_year: int, end_year: int, max_reviews: int) -> List[Dict[str, Any]]:
    try:
        if store_type == "App Store":
            country, app_name, app_id = parse_app_store_url(app_url)
            st.info(f"Fetching reviews for {app_name} (ID: {app_id}) from {country} App Store")
            app = AppStore(country=country, app_name=app_name, app_id=app_id)
            reviews = fetch_app_store_reviews(app, max_reviews)
        else:  # Play Store
            package_name = extract_play_store_package_name(app_url)
            st.info(f"Fetching reviews for app with package name: {package_name}")
            reviews = fetch_play_store_reviews(package_name, max_reviews, start_year, end_year)

        if not reviews:
            st.warning("No reviews were fetched. This could be due to the app having no reviews, or issues with accessing the app store.")
            return []

        review_data = []
        for review in reviews:
            if store_type == "App Store":
                date = review.get('date')
                rating = review.get('rating')
                content = review.get('review', '')
            else:  # Play Store
                date = review.get('at')
                if isinstance(date, (int, float)):
                    date = datetime.fromtimestamp(date)
                rating = review.get('score')
                content = review.get('content', '')
            
            if date and rating is not None:
                sentiment = analyze_sentiment(content, rating)
                review_data.append({
                    'Date': date.strftime('%Y-%m-%d') if isinstance(date, datetime) else str(date),
                    'Rating': rating,
                    'Sentiment': sentiment,
                    'Review': content[:100] + '...' if len(content) > 100 else content  # Truncate long reviews
                })
        
        return review_data
    except Exception as e:
        st.error(f"An error occurred while fetching and analyzing reviews: {str(e)}")
        logger.error(f"Error details: {e}", exc_info=True)
        st.info("Please check the app URL and ensure it's correct. If the problem persists, the app store might be experiencing issues.")
        return []

def main():
    """Main function for the Streamlit app."""
    st.title("App Review Analyzer")

    store_type = st.radio("Select the app store:", ("App Store", "Play Store"))
    app_url = st.text_input("Paste the app URL:")
    start_year = st.number_input("Start Year:", min_value=2000, max_value=datetime.now().year, value=datetime.now().year - 1)
    end_year = st.number_input("End Year:", min_value=2000, max_value=datetime.now().year, value=datetime.now().year)
    max_reviews = st.number_input("Maximum number of reviews to fetch:", min_value=MIN_REVIEWS, max_value=MAX_REVIEWS, value=DEFAULT_REVIEWS, step=100)

    if st.button("Analyze Reviews"):
        try:
            st.info("Starting review analysis...")
            review_data = fetch_and_analyze_reviews(store_type, app_url, start_year, end_year, max_reviews)

            if review_data:
                st.success(f"Successfully analyzed {len(review_data)} reviews.")
                # Create DataFrame
                df = pd.DataFrame(review_data)

                # Display table of reviews
                st.subheader("Review Data")
                st.dataframe(df, height=400)

                # Display visualizations
                st.subheader("Sentiment Distribution")
                fig_sentiment = create_sentiment_chart(df)
                st.plotly_chart(fig_sentiment)

                st.subheader("Rating Distribution")
                fig_rating = create_rating_chart(df)
                st.plotly_chart(fig_rating)

                # Provide option to download data
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name="app_reviews_analysis.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No reviews found for the specified criteria.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
