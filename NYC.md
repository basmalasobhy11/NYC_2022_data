# NYC 311 Complaint Intelligence

## Project Overview

NYC 311 Complaint Intelligence is an interactive data analysis and machine learning application developed to analyze NYC 311 complaints from 2022.

The project combines data analytics, interactive visualization, geographic analysis, sentiment analysis, and emotion detection in one dashboard. It helps identify patterns in citizen complaints based on complaint type, time, location, agency, status, sentiment, and emotion.

The application is divided into two main sections: **Analysis** and **Machine Learning**.

---

## Analysis Dashboard

The Analysis section provides an interactive dashboard where users can explore the NYC 311 complaints using different filters.

Users can filter the data by:

* Agency
* Complaint Status
* Sentiment
* Emotion
* Complaint Type
* Month
* Hour of Day

All visualizations and statistics are updated based on the selected filters.

### Overview

The dashboard provides an overview of the filtered data, including:

* Total number of complaints
* Number of different complaint types
* Number of agencies
* Percentage of complaints with negative sentiment

### Complaint Analysis

The dashboard identifies the most common complaint types and shows how complaints are distributed across different categories.

It also analyzes the relationship between complaint types and location types, helping to understand where different types of complaints are commonly reported.

### Sentiment and Emotion Analysis

The project analyzes complaints based on their sentiment and detected emotion.

The dashboard includes:

* Sentiment distribution
* Emotion distribution
* Sentiment versus emotion analysis

This provides a better understanding of how citizens express their experiences through 311 complaints.

### Time Analysis

The project analyzes complaint patterns across different time dimensions:

* Monthly trends
* Hour of the day
* Day of the week
* Time periods

Complaints are grouped into four time periods:

* Morning
* Afternoon
* Evening
* Night

The dashboard also compares complaint types across months, hours, days, and time periods to identify recurring patterns.

### Geographic Analysis

The application uses the latitude and longitude of complaints to display them on interactive New York City maps.

The maps provide information such as:

* Complaint type
* Agency
* Status
* Location type
* Sentiment
* Emotion
* Time
* Day

The project also provides category-based geographic analysis. Complaints are divided into three categories:

* Political
* Economic
* Social

For each category, the dashboard provides maps showing complaints according to their sentiment and detected emotion.

---

## Machine Learning

The Machine Learning section allows users to analyze a completely new complaint.

The user enters a complaint, and the application processes the text using pre-trained Transformer-based Natural Language Processing models.

The system predicts two main aspects:

### Sentiment Prediction

The system determines whether the complaint has a:

* Positive sentiment
* Negative sentiment

It also displays the model's confidence in the prediction.

### Emotion Prediction

The system detects the main emotional state expressed in the complaint and displays the predicted emotion along with its confidence score.

The application uses a DistilBERT-based model for sentiment analysis and a DistilRoBERTa-based model for emotion classification.

The results are presented in an easy-to-understand format containing the predicted sentiment, emotion, confidence scores, and a short interpretation of the complaint.

---

## Dataset Processing

The application works with a processed NYC 311 dataset containing complaint information along with sentiment and emotion analysis.

The complaint date is processed to extract additional time-related information such as:

* Month
* Month Name
* Day
* Hour
* Date

Latitude and longitude values are also processed to support the geographic visualizations.

---

## Technologies Used

* Python
* Streamlit
* Pandas
* Plotly
* PyTorch
* Hugging Face Transformers
* VADER Sentiment

---

## Project Goals

The main goals of the project are to:

1. Explore and understand NYC 311 complaint patterns.
2. Identify the most common types of complaints.
3. Analyze complaints across time and location.
4. Understand the sentiment and emotions expressed in complaints.
5. Discover relationships between complaint characteristics.
6. Provide an interactive dashboard for data exploration.
7. Apply machine learning and NLP to analyze new complaints.

---

## Team

* Basmala Mohammed
* Noreen Ali


