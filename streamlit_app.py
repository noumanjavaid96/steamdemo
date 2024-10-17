import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode
from openai import OpenAI
import altair as alt
import random

# Set the app title and layout
st.set_page_config(
    page_title="Steam App Reviews Dashboard",
    layout="wide",
)

# User input for OpenAI API key
api_key_input = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    help="Your API key will be used to access the OpenAI API for sentiment analysis."
)

# Function to initialize OpenAI client
@st.cache_resource
def get_openai_client(api_key_input):
    if api_key_input:
        return OpenAI()
    return None

# Initialize OpenAI client
client = OpenAI(api_key=api_key_input)

if not api_key_input:
    st.sidebar.warning("Please enter your OpenAI API Key to proceed.")

st.title("🎮 Steam App Reviews Dashboard")

# Sidebar inputs for user interaction
st.sidebar.header("User Input Parameters")

# User input for App ID
appid = st.sidebar.text_input("Enter the Steam App ID:", value="271590")

# Date input for selecting a week
st.sidebar.write("Select the date range for reviews:")
start_date = st.sidebar.date_input(
    "Start Date", value=datetime.today() - timedelta(days=7)
)
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# Check if the end date is after the start date
if start_date > end_date:
    st.error("Error: End date must fall after start date.")
elif not api_key_input:
    st.info("Please input your OpenAI API Key to proceed.")
else:
    # Fetch reviews button
    if st.sidebar.button("Fetch and Analyze Reviews"):
        st.write("Fetching reviews...")

        # Convert dates to timestamps
        start_timestamp = int(time.mktime(start_date.timetuple()))
        end_timestamp = int(
            time.mktime((end_date + timedelta(days=1)).timetuple())
        ) - 1  # Include the entire end date

        # Define the base API URL
        base_url = f"https://store.steampowered.com/appreviews/{appid}?json=1"

        # Define initial API parameters
        params = {
            "filter": "recent",
            "language": "all",
            "day_range": "365",  # Maximum allowed
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": "100",
            "cursor": "*",
            "filter_offtopic_activity": 0,
        }

        # Function to fetch reviews
        def fetch_reviews():
            """
            Fetches Steam reviews for the specified app within the given date range.
            The function uses pagination to retrieve reviews and applies a limit to avoid infinite loops.

            Returns:
                list: A list of reviews within the specified date range.
            """
            reviews_list = []
            request_count = 0
            max_requests = 50  # Limit the number of requests to avoid infinite loops
            while True:
                # URL encode the cursor parameter
                params_encoded = params.copy()
                params_encoded["cursor"] = params["cursor"].replace("+", "%2B")

                # Construct the full URL with parameters
                url = base_url + "&" + urlencode(params_encoded)

                # Make the API request
                response = requests.get(url)
                data = response.json()

                # Check if the request was successful
                if data["success"] != 1:
                    st.error("Failed to fetch reviews.")
                    return None

                # Append reviews to the list
                reviews = data.get("reviews", [])
                for review in reviews:
                    # Filter reviews based on timestamp
                    if start_timestamp <= review["timestamp_created"] <= end_timestamp:
                        reviews_list.append(review)
                    elif review["timestamp_created"] < start_timestamp:
                        # Since reviews are ordered by most recent, we can break early
                        return reviews_list

                # Update the cursor for the next batch
                new_cursor = data.get("cursor")
                if params["cursor"] == new_cursor:
                    break  # Exit if the cursor hasn't changed to avoid infinite loop
                params["cursor"] = new_cursor

                # Check if there are no more reviews
                if not reviews:
                    break

                # Increment request count and check limit
                request_count += 1
                if request_count >= max_requests:
                    st.warning("Reached maximum number of requests. Some reviews may not be fetched.")
                    break

                # Optional: To avoid hitting the rate limit
                time.sleep(0.2)

            return reviews_list

        # Fetch the reviews
        reviews_data = fetch_reviews()

        # Check if reviews were fetched
        if reviews_data:
            st.success(f"Fetched {len(reviews_data)} reviews from App ID {appid}.")

            # Create a DataFrame from the review data
            df = pd.DataFrame(
                [
                    {
                        "Review ID": review["recommendationid"],
                        "Author SteamID": review["author"]["steamid"],
                        "Language": review["language"],
                        "Review": review["review"],
                        "Posted On": datetime.fromtimestamp(
                            review["timestamp_created"]
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                        "Recommended": "Yes" if review["voted_up"] else "No",
                        "Helpful Votes": review["votes_up"],
                        "Playtime at Review (hours)": round(
                            review["author"]["playtime_at_review"] / 60, 2
                        ),
                    }
                    for review in reviews_data
                ]
            )

            # Perform sentiment analysis using OpenAI GPT-4
            st.write("### Sentiment Analysis")

            @st.cache_data
            def analyze_sentiments(reviews):
                """
                Analyzes the sentiment of the given reviews using OpenAI's GPT-4 model.

                Args:
                    reviews (list): List of review texts to analyze.

                Returns:
                    list: A list of sentiment labels ('Positive', 'Negative', 'Neutral') for each review.
                """
                sentiments = []
                for review_text in reviews:
                    # Prepare the prompt for GPT-4
                    messages = [{"role": "user", "content": f"Please analyze the sentiment of the following Steam game review and classify it as 'Positive: Where the user is happy'\n, 'Negative: Where the user expressed disatisfaction',\n or 'Neutral'.\n\nReview: \"{review_text}\"\n\nSentiment: "}]
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": "analyze_sentiment",
                                "description": "Analyze the sentiment of the given text and the positive should be positive",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "The text to analyze",
                                        }
                                    },
                                    "required": ["text"],
                                },
                            }
                        }
                    ]

                    try:
                        # Call OpenAI API
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            temperature=0
                        )
                        sentiment = response.choices[0]["message"]["content"].strip() if "message" in response.choices[0] and "content" in response.choices[0]["message"] else 'Neutral'
                        # Ensure the sentiment is one of the expected values
                        if sentiment not in ["Positive", "Negative", "Neutral"]:
                            sentiment = "Neutral"
                        sentiments.append(sentiment)
                    except Exception as e:
                        st.error(f"OpenAI API Error: {e}")
                        sentiments.append("Neutral")  # Default to Neutral on error

                return sentiments

            # Analyze sentiments
            df["Sentiment"] = analyze_sentiments(df["Review"].tolist())

            # Generate and display interactive word cloud using Vega
            st.write("### Interactive Word Cloud of Reviews")
            all_reviews_text = " ".join(df["Review"])
            words = all_reviews_text.split()
            word_freq = pd.DataFrame([{"word": word, "count": words.count(word)} for word in set(words)])

            wordcloud_chart = alt.Chart(word_freq).mark_text().encode(
                text="word",
                size=alt.Size("count:Q", scale=alt.Scale(range=[10, 100])),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["word", "count"]
            ).configure_mark(
                align="center",
                baseline="top"
            ).properties(
                width=800,
                height=400
            )

            st.altair_chart(wordcloud_chart)

            # Display the DataFrame
            st.write("### Reviews Data")
            st.dataframe(df)

            # Display Top 5 Positive and Top 5 Negative Reviews
            st.write("### Top 5 Positive Reviews")
            top_5_positive = df[df["Sentiment"] == "Positive"].sort_values(by="Helpful Votes", ascending=False).head(5)
            st.dataframe(top_5_positive)

            st.write("### Top 5 Negative Reviews")
            top_5_negative = df[df["Sentiment"] == "Negative"].sort_values(by="Helpful Votes", ascending=False).head(5)
            st.dataframe(top_5_negative)

            # Calculate and display NPS Score
            st.write("### Net Promoter Score (NPS)")

            # Define NPS categories based on sentiment labels
            def categorize_nps(sentiment):
                """
                Categorizes the sentiment into NPS categories.

                Args:
                    sentiment (str): The sentiment label ('Positive', 'Negative', 'Neutral').

                Returns:
                    str: The NPS category ('Promoter', 'Passive', 'Detractor').
                """
                if sentiment.lower() == "positive":
                    return "Promoter"
                elif sentiment.lower() == "neutral":
                    return "Passive"
                else:
                    return "Detractor"

            df["NPS Category"] = df["Sentiment"].apply(categorize_nps)
            nps_counts = df["NPS Category"].value_counts()
            total_responses = nps_counts.sum()

            promoters = nps_counts.get("Promoter", 0)
            detractors = nps_counts.get("Detractor", 0)

            nps_score = ((promoters - detractors) / total_responses) * 100

            st.write(f"**Net Promoter Score (NPS):** {nps_score:.2f}")

            # Interactive Bar Chart for NPS Categories
            st.write("### NPS Category Distribution")
            nps_df = pd.DataFrame({
                "Category": ["Promoters", "Passives", "Detractors"],
                "Count": [
                    nps_counts.get("Promoter", 0),
                    nps_counts.get("Passive", 0),
                    nps_counts.get("Detractor", 0),
                ],
            })

            nps_chart = alt.Chart(nps_df).mark_bar().encode(
                x=alt.X("Category", sort=None),
                y="Count",
                color=alt.Color("Category", scale=alt.Scale(domain=["Promoters", "Passives", "Detractors"], range=["#00CC00", "#CCCC00", "#CC0000"])),
                tooltip=["Category", "Count"]
            ).properties(
                width=600,
                height=400
            )

            st.altair_chart(nps_chart)
