import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode
from openai import OpenAI
from io import StringIO
import json

# Set up the application title and layout
st.set_page_config(
    page_title="Steam App Reviews - Themes Analysis",
)

st.title("🎮 Steam App Reviews - Themes Analysis")

# Sidebar inputs for user interaction
st.sidebar.header("User Input Parameters")

# User input for OpenAI API key
api_key_input = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    help="Your API key will be used to access the OpenAI API for theme extraction.",
)

# Initialize OpenAI client
client = None
if api_key_input:
    client = OpenAI(api_key=api_key_input)
else:
    st.sidebar.warning("Please enter your OpenAI API Key to proceed.")

# User input for App ID
appid = st.sidebar.text_input("Enter the Steam App ID:", value="1782120")

# Date input for selecting a week
st.sidebar.write("Select the date range for reviews:")
start_date = st.sidebar.date_input(
    "Start Date", value=datetime.today() - timedelta(days=7)
)
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# Validate dates
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
        def fetch_reviews(max_reviews=200):
            """
            Fetches Steam reviews for the specified app within the given date range.
            The function uses pagination to retrieve reviews and applies a limit to avoid infinite loops.

            Args:
                max_reviews (int): Maximum number of reviews to fetch.

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
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Steam API Error: {e}")
                    return None

                # Check if the request was successful
                if data.get("success") != 1:
                    st.error("Failed to fetch reviews from Steam API.")
                    return None

                # Append reviews to the list
                reviews = data.get("reviews", [])
                if not reviews:
                    # No more reviews
                    break

                for review in reviews:
                    # Filter reviews based on timestamp
                    if start_timestamp <= review.get("timestamp_created", 0) <= end_timestamp:
                        reviews_list.append(review)
                        if len(reviews_list) >= max_reviews:
                            break
                    elif review.get("timestamp_created", 0) < start_timestamp:
                        # Since reviews are ordered by most recent, we can break early
                        break

                # Update the cursor for the next batch
                new_cursor = data.get("cursor")
                if params["cursor"] == new_cursor:
                    # Exit if the cursor hasn't changed to avoid infinite loop
                    break
                params["cursor"] = new_cursor

                # Check if maximum number of reviews fetched
                if len(reviews_list) >= max_reviews:
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
        reviews_data = fetch_reviews(max_reviews=200)

        # Check if reviews were fetched
        if reviews_data:
            st.success(f"Fetched {len(reviews_data)} reviews from App ID {appid}.")

            # Create a DataFrame from the review data
            df = pd.DataFrame(
                [
                    {
                        "Review ID": str(review.get("recommendationid")),
                        "Author SteamID": review.get("author", {}).get("steamid"),
                        "Language": review.get("language"),
                        "Review": review.get("review"),
                        "Posted On": datetime.fromtimestamp(
                            review.get("timestamp_created", 0)
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    for review in reviews_data
                ]
            )

            # Function to extract themes using OpenAI GPT-4o
            def extract_themes(df):
                """
                Uses OpenAI's GPT-4o model to identify the most common themes,
                provide descriptions, and list review IDs where each theme is mentioned.

                Args:
                    df (DataFrame): DataFrame containing the reviews.

                Returns:
                    DataFrame: A DataFrame containing themes, descriptions, and review references.
                """
                # Combine reviews into a single string with IDs
                reviews_text = "\n".join(
                    [
                        f"Review ID: {row['Review ID']}\nReview Text: {row['Review']}"
                        for _, row in df.iterrows()
                    ]
                )

                # Prepare the prompt
                prompt = f"""
Analyze the following user reviews and identify the most common themes or topics being discussed.
For each theme, provide a brief description and list the Review IDs where the theme is mentioned.

Provide the output as a JSON array matching the following structure:
[
    {{"Theme": "<theme_name>", "Description": "<description>", "Review IDs": ["<review_id1>", "<review_id2>", ...]}},
    ...
]

Ensure the output is valid JSON.

Reviews:
{reviews_text}
"""
                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "analyze_game_reviews",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "themes": {
                                                "type": "array",
                                                "description": "List of themes identified in the game reviews",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "Theme": {
                                                            "type": "string",
                                                            "description": "The main theme derived from the game reviews.",
                                                        },
                                                        "Description": {
                                                            "type": "string",
                                                            "description": "A summary of the issues or sentiments related to the theme.",
                                                        },
                                                        "Review IDs": {
                                                            "type": "array",
                                                            "description": "Array of IDs for reviews that relate to this theme.",
                                                            "items": {
                                                                "type": "string",
                                                                "description": "Unique identifier for a review.",
                                                            },
                                                        },
                                                    },
                                                    "required": [
                                                        "Theme",
                                                        "Description",
                                                        "Review IDs",
                                                    ],
                                                    "additionalProperties": False,
                                                },
                                            }
                                        },
                                        "required": ["themes"],
                                        "additionalProperties": False,
                                    }
                                },
                                "required": ["parameters"],
                                "additionalProperties": False,
                            },
                        },
                    },
                )

                # Get the response content
                response_text = response.choices[0].message.content  # Extract the text

                # Parse the JSON output
                try:
                    response_data = json.loads(response_text)
                    themes_list = (
                        response_data.get("parameters", {}).get("themes", [])
                    )  # Extract the 'themes' list
                    themes_df = pd.DataFrame(themes_list)  # Create DataFrame from the list
                    return themes_df
                except ValueError as ve:
                    st.error(f"Error parsing JSON response: {ve}")
                    st.error(f"Raw API Response: {response_text}")
                    return None

            # Extract themes
            themes_df = extract_themes(df)

            if themes_df is not None:
                st.write("### Most Common Themes")
                st.dataframe(themes_df)

                # Display detailed information for each theme
                for index, row in themes_df.iterrows():
                    theme = row["Theme"]
                    description = row["Description"]
                    review_ids = row["Review IDs"]

                    st.write(f"#### Theme: {theme}")
                    st.write(f"**Description:** {description}")
                    st.write(f"**Mentioned in {len(review_ids)} reviews.**")

                    # Get the reviews that mention the theme
                    try:
                        reviews_with_theme = df[
                            df["Review ID"].isin(review_ids)
                        ][["Review ID", "Review"]]
                        st.write("Reviews mentioning this theme:")
                        st.dataframe(reviews_with_theme)
                    except ValueError as e:
                        st.error(
                            f"Error displaying reviews for theme '{theme}': {e}"
                        )
            else:
                st.warning("Failed to extract themes.")

        else:
            st.warning("No reviews to display.")
