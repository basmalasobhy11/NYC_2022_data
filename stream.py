import streamlit as st
import pandas as pd
import plotly.express as px

# PAGE CONFIGURATION

st.set_page_config(
    page_title="NYC 311 Complaint Intelligence",
    page_icon="🏙️",
    layout="wide"
)

# LOAD DATA
@st.cache_data
def load_data():

    df = pd.read_csv(
        "Final_File.csv"
    )

    # Convert created date to datetime

    df["created_date"] = pd.to_datetime(
        df["created_date"],
        errors="coerce"
    )

    # Coordinates

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # Time Features

    df["Month"] = df["created_date"].dt.month

    df["Month Name"] = (
        df["created_date"]
        .dt.strftime("%b")
    )

    df["Day"] = (
        df["created_date"]
        .dt.day_name()
    )

    df["Hour"] = (
        df["created_date"]
        .dt.hour
    )

    df["Date"] = (
        df["created_date"]
        .dt.date
    )

    return df


df = load_data()

# KEEP ONLY THE THREE REQUIRED CATEGORIES

allowed_categories = [
    "Social",
    "Political",
    "Economic"
]

df = df[
    df["Category"].isin(allowed_categories)
].copy()

# NAVIGATION

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "🏙️ Analysis",
        "🤖 Machine Learning"
    ]
)


# ANALYSIS PAGE


if page == "🏙️ Analysis":

    # CATEGORY SELECTION

    st.sidebar.divider()

    st.sidebar.title("📂 Category")

    selected_category = st.sidebar.selectbox(
        "Choose Category",
        allowed_categories
    )

    # TITLE

    st.title(
        f"🏙️ NYC 311 {selected_category} Complaints Dashboard"
    )

    st.markdown(
        f"### Analysis of {selected_category.lower()} complaints for 2022"
    )

    # SIDEBAR FILTERS

    st.sidebar.divider()

    st.sidebar.title("🔎 Analysis Filters")

    # AGENCY

    agency_values = sorted(
        df["agency"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_agencies = st.sidebar.multiselect(
        "Agency",
        agency_values,
        default=agency_values
    )

    # STATUS

    status_values = sorted(
        df["status"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_statuses = st.sidebar.multiselect(
        "Status",
        status_values,
        default=status_values
    )

    # SENTIMENT

    sentiment_values = sorted(
        df["Sentiment"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_sentiments = st.sidebar.multiselect(
        "Sentiment",
        sentiment_values,
        default=sentiment_values
    )

    # EMOTION
    emotion_values = sorted(
        df["Emotion"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_emotions = st.sidebar.multiselect(
        "Emotion",
        emotion_values,
        default=emotion_values
    )

    # COMPLAINT TYPE

    complaint_values = sorted(
        df["complaint_type"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_complaints = st.sidebar.multiselect(
        "Complaint Type",
        complaint_values,
        default=complaint_values
    )

    # MONTH

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec"
    ]

    selected_months = st.sidebar.multiselect(
        "Month",
        month_names,
        default=month_names
    )


    # HOUR

    selected_hours = st.sidebar.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=(0, 23)
    )

    # FILTER DATA
    # Category is filtered HERE.Therefore every plot below uses ONLY the selected category.

    filtered_df = df[
        (df["Category"] == selected_category)
        &
        (df["agency"].isin(selected_agencies))
        &
        (df["status"].isin(selected_statuses))
        &
        (df["Sentiment"].isin(selected_sentiments))
        &
        (df["Emotion"].isin(selected_emotions))
        &
        (df["complaint_type"].isin(selected_complaints))
        &
        (df["Month Name"].isin(selected_months))
        &
        (
            df["Hour"].between(
                selected_hours[0],
                selected_hours[1]
            )
        )
    ].copy()


    # KPI SECTION

    st.subheader(
        f"📊 {selected_category} Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    # Total complaints

    with col1:

        st.metric(
            "Total Complaints",
            f"{len(filtered_df):,}"
        )

    # Complaint types

    with col2:

        st.metric(
            "Complaint Types",
            filtered_df[
                "complaint_type"
            ].nunique()
        )

    # Agencies

    with col3:

        st.metric(
            "Agencies",
            filtered_df[
                "agency"
            ].nunique()
        )

    # Negative percentage

    with col4:

        if len(filtered_df) > 0:

            negative_percentage = (
                (
                    filtered_df["Sentiment"]
                    == "Negative"
                ).mean()
                * 100
            )

        else:

            negative_percentage = 0

        st.metric(
            "Negative Sentiment",
            f"{negative_percentage:.1f}%"
        )


    st.divider()

    # ROW 1 : COMPLAINT + SENTIMENT
    col1, col2 = st.columns(2)

    # TOP COMPLAINT TYPES

    with col1:

        st.subheader(
            "📋 Top Complaint Types"
        )

        top_complaints = (
            filtered_df[
                "complaint_type"
            ]
            .value_counts()
            .head(10)
            .reset_index()
        )

        top_complaints.columns = [
            "Complaint Type",
            "Number of Complaints"
        ]

        fig_complaints = px.bar(
            top_complaints,
            x="Number of Complaints",
            y="Complaint Type",
            orientation="h",
            title=(
                f"Top 10 {selected_category} "
                "Complaint Types"
            )
        )

        fig_complaints.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            }
        )

        st.plotly_chart(
            fig_complaints,
            use_container_width=True
        )

    # SENTIMENT

    with col2:

        st.subheader(
            "💭 Sentiment Distribution"
        )

        sentiment_counts = (
            filtered_df[
                "Sentiment"
            ]
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = [
            "Sentiment",
            "Number of Complaints"
        ]

        fig_sentiment = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Number of Complaints",
            color="Sentiment",
            title=(
                f"{selected_category} "
                "Complaints by Sentiment"
            ),
            color_discrete_map={
                "Negative": "red",
                "Neutral": "blue",
                "Positive": "green"
            }
        )

        st.plotly_chart(
            fig_sentiment,
            use_container_width=True
        )

    # EMOTION DISTRIBUTION

    st.subheader(
        "😊 Emotion Distribution"
    )

    emotion_counts = (
        filtered_df[
            "Emotion"
        ]
        .value_counts()
        .reset_index()
    )

    emotion_counts.columns = [
        "Emotion",
        "Number of Complaints"
    ]

    fig_emotion = px.bar(
        emotion_counts,
        x="Emotion",
        y="Number of Complaints",
        color="Emotion",
        title=(
            f"{selected_category} "
            "Complaints by Emotion"
        )
    )

    st.plotly_chart(
        fig_emotion,
        use_container_width=True
    )

    # AGENCY VS STATUS

    st.subheader(
        "🏢 Complaint Status by Agency"
    )

    agency_status = (
        filtered_df
        .groupby(
            [
                "agency",
                "status"
            ]
        )
        .size()
        .reset_index(
            name="Number of Complaints"
        )
    )

    fig_status = px.bar(
        agency_status,
        x="agency",
        y="Number of Complaints",
        color="status",
        barmode="group",
        title=(
            f"{selected_category} "
            "Complaint Status by Agency"
        )
    )

    st.plotly_chart(
        fig_status,
        use_container_width=True
    )

    # TOP 5 VS LOCATION

    st.subheader(
        "🔥 Top 5 Complaint Types vs Location Type"
    )

    top5_types = (
        filtered_df[
            "complaint_type"
        ]
        .value_counts()
        .head(5)
        .index
    )

    top5_df = filtered_df[
        filtered_df[
            "complaint_type"
        ].isin(top5_types)
    ]

    pivot = pd.crosstab(
        top5_df[
            "complaint_type"
        ],
        top5_df[
            "location_type"
        ]
    )

    fig_heatmap = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Location Type",
            "y": "Complaint Type",
            "color": "Number of Complaints"
        },
        title=(
            f"Top 5 {selected_category} "
            "Complaint Types vs Location Type"
        )
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )

    # COMPLAINT TYPE OVER TIME

    st.subheader(
        "📈 Complaint Type Over Time"
    )

    time_complaints = filtered_df.copy()

    top10_types = (
        time_complaints[
            "complaint_type"
        ]
        .value_counts()
        .head(10)
        .index
    )

    time_complaints = time_complaints[
        time_complaints[
            "complaint_type"
        ].isin(top10_types)
    ]

    complaint_time = (
        time_complaints
        .groupby(
            [
                "Month",
                "Month Name",
                "complaint_type"
            ]
        )
        .size()
        .reset_index(
            name="Number of Complaints"
        )
    )

    complaint_time[
        "Month Name"
    ] = pd.Categorical(
        complaint_time[
            "Month Name"
        ],
        categories=month_names,
        ordered=True
    )

    complaint_time = (
        complaint_time
        .sort_values("Month")
    )

    fig_complaint_time = px.line(
        complaint_time,
        x="Month Name",
        y="Number of Complaints",
        color="complaint_type",
        markers=True,
        title=(
            f"Top {selected_category} "
            "Complaint Types Across 2022"
        ),
        labels={
            "Month Name": "Month",
            "Number of Complaints":
                "Complaints",
            "complaint_type":
                "Complaint Type"
        }
    )

    fig_complaint_time.update_layout(
        xaxis={
            "categoryorder": "array",
            "categoryarray": month_names
        }
    )

    st.plotly_chart(
        fig_complaint_time,
        use_container_width=True
    )

    # COMPLAINT TYPE × MONTH

    st.subheader(
        "🔥 Complaint Type × Month"
    )

    month_heatmap = pd.crosstab(
        time_complaints[
            "complaint_type"
        ],
        time_complaints[
            "Month Name"
        ]
    )

    month_heatmap = month_heatmap.reindex(
        columns=month_names,
        fill_value=0
    )

    fig_month_heatmap = px.imshow(
        month_heatmap,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Month",
            "y": "Complaint Type",
            "color": "Complaints"
        },
        title=(
            f"{selected_category} "
            "Complaint Frequency by Month"
        )
    )

    st.plotly_chart(
        fig_month_heatmap,
        use_container_width=True
    )

    # COMPLAINT TYPE × HOUR

    st.subheader(
        "🕐 Complaint Type × Hour"
    )

    hour_data = filtered_df.copy()

    top10_hour_types = (
        hour_data[
            "complaint_type"
        ]
        .value_counts()
        .head(10)
        .index
    )

    hour_data = hour_data[
        hour_data[
            "complaint_type"
        ].isin(top10_hour_types)
    ]

    hour_table = pd.crosstab(
        hour_data[
            "complaint_type"
        ],
        hour_data[
            "Hour"
        ]
    )

    hour_table = hour_table.reindex(
        columns=range(24),
        fill_value=0
    )

    fig_hour_heatmap = px.imshow(
        hour_table,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Hour of Day",
            "y": "Complaint Type",
            "color": "Complaints"
        },
        title=(
            f"{selected_category} "
            "Complaint Types by Hour of Day"
        )
    )

    fig_hour_heatmap.update_layout(
        xaxis={
            "tickmode": "linear",
            "dtick": 1
        }
    )

    st.plotly_chart(
        fig_hour_heatmap,
        use_container_width=True
    )

    # TOTAL COMPLAINTS BY HOUR
    st.subheader(
        "⏰ Complaints by Hour"
    )

    hour_counts = (
        filtered_df
        .groupby("Hour")
        .size()
        .reindex(
            range(24),
            fill_value=0
        )
        .reset_index(
            name="Number of Complaints"
        )
    )

    fig_hour = px.line(
        hour_counts,
        x="Hour",
        y="Number of Complaints",
        markers=True,
        title=(
            f"Total {selected_category} "
            "Complaints by Hour of Day"
        )
    )

    fig_hour.update_layout(
        xaxis={
            "tickmode": "linear",
            "dtick": 1
        },
        xaxis_title="Hour of Day",
        yaxis_title="Number of Complaints"
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )

    # DAY OF WEEK

    st.subheader(
        "📅 Complaints by Day of Week"
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    day_counts = (
        filtered_df
        .groupby("Day")
        .size()
        .reindex(
            day_order,
            fill_value=0
        )
        .reset_index(
            name="Number of Complaints"
        )
    )

    fig_day = px.bar(
        day_counts,
        x="Day",
        y="Number of Complaints",
        title=(
            f"{selected_category} "
            "Complaints by Day of Week"
        )
    )

    st.plotly_chart(
        fig_day,
        use_container_width=True
    )

    # DAY × COMPLAINT TYPE
    st.subheader(
        "📅 Complaint Type × Day of Week"
    )

    day_type_table = pd.crosstab(
        filtered_df[
            "complaint_type"
        ],
        filtered_df[
            "Day"
        ]
    )

    day_type_table = day_type_table.reindex(
        columns=day_order,
        fill_value=0
    )

    top_day_types = (
        filtered_df[
            "complaint_type"
        ]
        .value_counts()
        .head(10)
        .index
    )

    day_type_table = (
        day_type_table.loc[
            day_type_table.index.intersection(
                top_day_types
            )
        ]
    )

    fig_day_heatmap = px.imshow(
        day_type_table,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Day",
            "y": "Complaint Type",
            "color": "Complaints"
        },
        title=(
            f"{selected_category} "
            "Complaint Types by Day of Week"
        )
    )

    st.plotly_chart(
        fig_day_heatmap,
        use_container_width=True
    )
    # TIME + LOCATION MAP

    st.subheader(
        "🗺️ Complaints by Time and Location"
    )

    map_df = filtered_df.dropna(
        subset=[
            "latitude",
            "longitude",
            "created_date"
        ]
    ).copy()

    map_df["Time"] = (
        map_df[
            "created_date"
        ].dt.strftime("%H:%M")
    )

    fig_map = px.scatter_map(
        map_df,
        lat="latitude",
        lon="longitude",
        color="complaint_type",
        hover_name="complaint_type",
        hover_data={
            "latitude": False,
            "longitude": False,
            "complaint_type": True,
            "agency": True,
            "status": True,
            "location_type": True,
            "Sentiment": True,
            "Emotion": True,
            "Time": True,
            "Day": True
        },
        center={
            "lat": 40.7128,
            "lon": -74.0060
        },
        zoom=9,
        height=700
    )

    fig_map.update_traces(
        marker={
            "size": 7,
            "opacity": 0.65
        }
    )

    fig_map.update_layout(
        title=(
            f"{selected_category} Complaints "
            "by Time and Location"
        )
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

    # CATEGORY × SENTIMENT MAPS
    # ONLY THE SELECTED CATEGORY IS SHOWN

    st.subheader(
        f"🗺️ {selected_category} Complaints by Sentiment"
    )

    # Keep only rows with valid coordinates
    sentiment_map_df = filtered_df.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).copy()

    # NYC map settings
    NYC_CENTER = {
        "lat": 40.7128,
        "lon": -74.0060
    }

    # Function to create sentiment map

    def create_sentiment_map(
        data,
        sentiment,
        category
    ):

        subset = data[
            data["Sentiment"] == sentiment
        ].copy()


        # Sentiment colors

        sentiment_colors = {
            "Positive": "green",
            "Negative": "red",
            "Neutral": "blue"
        }


        # If no data

        if subset.empty:

            st.info(
                f"No {sentiment} "
                f"{category} complaints found."
            )

            return


        # Create map

        fig = px.scatter_map(
            subset,

            lat="latitude",
            lon="longitude",

            color="Sentiment",

            color_discrete_map={
                sentiment:
                    sentiment_colors[sentiment]
            },

            hover_name="complaint_type",

            hover_data={
                "latitude": False,
                "longitude": False,

                "complaint_type": True,
                "agency": True,
                "status": True,
                "location_type": True,
                "Sentiment": True
            },

            center=NYC_CENTER,

            zoom=9.5,

            height=450,

            title=f"{category} - {sentiment}"
        )


        # Make points clearer

        fig.update_traces(
            marker={
                "size": 6,
                "opacity": 0.65
            }
        )


        fig.update_layout(
            margin={
                "r": 0,
                "t": 50,
                "l": 0,
                "b": 0
            },

            showlegend=False
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # THREE SENTIMENT MAPS

    col1, col2, col3 = st.columns(3)

    # POSITIVE MAP

    with col1:

        create_sentiment_map(
            sentiment_map_df,
            "Positive",
            selected_category
        )


    # NEGATIVE MAP

    with col2:

        create_sentiment_map(
            sentiment_map_df,
            "Negative",
            selected_category
        )

    # NEUTRAL MAP

    with col3:

        create_sentiment_map(
            sentiment_map_df,
            "Neutral",
            selected_category
        )

    # EMOTION MAP
    # COMMENTED OUT BECAUSE THE REQUIRED EMOTION
    # LIBRARY/MODEL IS NOT CURRENTLY INSTALLED.
    # ========================================================

    # st.subheader(
    #     f"😊 {selected_category} Complaints by Emotion"
    # )
    #
    #
    # emotion_map_df = filtered_df.dropna(
    #     subset=[
    #         "latitude",
    #         "longitude",
    #         "Emotion"
    #     ]
    # ).copy()
    #
    #
    # fig_emotion_map = px.scatter_map(
    #
    #     emotion_map_df,
    #
    #     lat="latitude",
    #     lon="longitude",
    #
    #     color="Emotion",
    #
    #     hover_name="complaint_type",
    #
    #     hover_data={
    #         "latitude": False,
    #         "longitude": False,
    #         "complaint_type": True,
    #         "agency": True,
    #         "status": True,
    #         "location_type": True,
    #         "Sentiment": True,
    #         "Emotion": True,
    #         "Category": True
    #     },
    #
    #     center={
    #         "lat": 40.7128,
    #         "lon": -74.0060
    #     },
    #
    #     zoom=9.5,
    #
    #     height=700,
    #
    #     title=(
    #         f"{selected_category} "
    #         "Complaints by Emotion"
    #     )
    # )
    #
    #
    # fig_emotion_map.update_traces(
    #     marker={
    #         "size": 7,
    #         "opacity": 0.7
    #     }
    # )
    #
    #
    # fig_emotion_map.update_layout(
    #     legend_title="Emotion"
    # )
    #
    #
    # st.plotly_chart(
    #     fig_emotion_map,
    #     use_container_width=True
    # )

    # TIME PERIOD


    st.subheader(
        "🌙 Complaint Distribution by Time Period"
    )

    period_df = filtered_df.copy()


    def get_period(hour):

        if 5 <= hour < 12:
            return "Morning"

        elif 12 <= hour < 17:
            return "Afternoon"

        elif 17 <= hour < 21:
            return "Evening"

        else:
            return "Night"


    period_df[
        "Time Period"
    ] = period_df[
        "Hour"
    ].apply(get_period)


    period_order = [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]


    period_counts = (
        period_df[
            "Time Period"
        ]
        .value_counts()
        .reindex(
            period_order,
            fill_value=0
        )
        .reset_index()
    )

    period_counts.columns = [
        "Time Period",
        "Number of Complaints"
    ]


    fig_period = px.bar(
        period_counts,
        x="Time Period",
        y="Number of Complaints",
        title=(
            f"{selected_category} "
            "Complaints by Time Period"
        )
    )

    st.plotly_chart(
        fig_period,
        use_container_width=True
    )

    # TIME PERIOD × COMPLAINT TYPE

    st.subheader(
        "🌙 Complaint Type × Time Period"
    )

    period_table = pd.crosstab(
        period_df[
            "complaint_type"
        ],
        period_df[
            "Time Period"
        ]
    )

    period_table = period_table.reindex(
        columns=period_order,
        fill_value=0
    )


    period_table = (
        period_table.loc[
            period_table.index.intersection(
                top10_types
            )
        ]
    )


    fig_period_heatmap = px.imshow(
        period_table,
        text_auto=False,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Time Period",
            "y": "Complaint Type",
            "color": "Complaints"
        },
        title=(
            f"{selected_category} "
            "Complaint Types by Time Period"
        )
    )

    st.plotly_chart(
        fig_period_heatmap,
        use_container_width=True
    )
    # SENTIMENT × EMOTION

    st.subheader(
        "💭 Sentiment × Emotion"
    )

    sentiment_emotion = pd.crosstab(
        filtered_df[
            "Sentiment"
        ],
        filtered_df[
            "Emotion"
        ]
    )

    fig_sentiment_emotion = px.imshow(
        sentiment_emotion,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Emotion",
            "y": "Sentiment",
            "color": "Complaints"
        },
        title=(
            f"{selected_category}: "
            "Relationship Between Sentiment and Emotion"
        )
    )

    st.plotly_chart(
        fig_sentiment_emotion,
        use_container_width=True
    )

    # FILTERED DATA

    st.divider()

    st.subheader(
        f"📄 Filtered {selected_category} Data"
    )

    st.write(
        f"Showing {len(filtered_df):,} "
        f"{selected_category.lower()} complaints"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )

# MACHINE LEARNING PAGE


elif page == "🤖 Machine Learning":

    st.title(
        "🤖 Complaint Intelligence"
    )

    st.markdown(
        """
        ### Analyze a New Complaint

        Enter a complaint below and the machine
        learning models will predict its:

        - 💭 **Sentiment**
        - 😊 **Emotion**
        """
    )

    st.divider()

    # LOAD MODELS
    @st.cache_resource
    def load_ml_models():

        import torch
        from transformers import pipeline

        device = (
            0
            if torch.cuda.is_available()
            else -1
        )

        # SENTIMENT MODEL

        sentiment_model = pipeline(
            "sentiment-analysis",
            model=(
                "distilbert-base-uncased-"
                "finetuned-sst-2-english"
            ),
            device=device
        )

        # EMOTION MODEL

        emotion_model = pipeline(
            "text-classification",
            model=(
                "j-hartmann/"
                "emotion-english-"
                "distilroberta-base"
            ),
            device=device
        )


        return (
            sentiment_model,
            emotion_model
        )

    # LOAD MODELS

    with st.spinner(
        "Loading machine learning models..."
    ):

        sentiment_model, emotion_model = (
            load_ml_models()
        )

    # USER INPUT

    st.subheader(
        "📝 Enter Complaint"
    )

    complaint_text = st.text_area(
        "Complaint",
        placeholder=(
            "Example: My neighbor is playing "
            "very loud music every night and "
            "I cannot sleep."
        ),
        height=180
    )

    # ANALYZE BUTTON

    analyze_button = st.button(
        "🔍 Analyze Complaint",
        type="primary",
        use_container_width=True
    )

    # ANALYSIS

    if analyze_button:

        if not complaint_text.strip():

            st.warning(
                "⚠️ Please enter a complaint first."
            )

        else:

            with st.spinner(
                "Analyzing complaint..."
            ):

                # SENTIMENT

                sentiment_result = (
                    sentiment_model(
                        complaint_text,
                        truncation=True
                    )[0]
                )

                sentiment_label = (
                    sentiment_result["label"]
                )

                sentiment_score = (
                    sentiment_result["score"]
                )

                # EMOTION
                emotion_result = (
                    emotion_model(
                        complaint_text,
                        truncation=True
                    )[0]
                )

                emotion_label = (
                    emotion_result["label"]
                    .capitalize()
                )

                emotion_score = (
                    emotion_result["score"]
                )

            # RESULTS

            st.divider()

            st.subheader(
                "📊 Analysis Result"
            )

            col1, col2 = st.columns(2)

            # SENTIMENT RESULT
            with col1:

                st.markdown(
                    "### 💭 Sentiment"
                )

                if sentiment_label == "POSITIVE":

                    st.success(
                        "Positive"
                    )

                else:

                    st.error(
                        "Negative"
                    )

                st.progress(
                    float(sentiment_score)
                )

                st.caption(
                    f"Confidence: "
                    f"{sentiment_score:.2%}"
                )

            # EMOTION RESULT
            with col2:

                st.markdown(
                    "### 😊 Emotion"
                )

                st.info(
                    emotion_label
                )

                st.progress(
                    float(emotion_score)
                )

                st.caption(
                    f"Confidence: "
                    f"{emotion_score:.2%}"
                )

            # COMPLAINT TEXT

            st.divider()

            st.subheader(
                "📝 Complaint Analyzed"
            )

            st.info(
                complaint_text
            )

            # RESULT TABLE
            result_df = pd.DataFrame(
                {
                    "Analysis": [
                        "Sentiment",
                        "Emotion"
                    ],

                    "Prediction": [
                        sentiment_label.capitalize(),
                        emotion_label
                    ],

                    "Confidence": [
                        f"{sentiment_score:.2%}",
                        f"{emotion_score:.2%}"
                    ]
                }
            )


            st.subheader(
                "📋 Prediction Summary"
            )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )


            # INTERPRETATION

            st.subheader(
                "💡 Interpretation"
            )

            if sentiment_label == "NEGATIVE":

                st.write(
                    "The complaint has a negative "
                    "overall sentiment."
                )

            else:

                st.write(
                    "The complaint has a positive "
                    "overall sentiment."
                )

            st.write(
                f"The detected emotional state "
                f"is **{emotion_label}**."
            )