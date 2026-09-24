# Zepto Data AI Platform

## Module 1 - Data Collection and SQL Pipeline

This module collects book data from Books to Scrape, cleans and transforms the data using Python and Pandas, stores the data in SQLite, and performs SQL analysis.

## Data Collection

The project collects book information from multiple categories:

- Travel
- Mystery
- Fiction
- Romance
- Science

A total of 87 books were collected.

## Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- SQL

## Data Cleaning and Transformation

The collected data was cleaned and converted into appropriate data types.

Main columns:

- title
- price_gbp
- price_inr
- rating
- in_stock
- category

Price values were converted to numeric values and INR prices were calculated from GBP prices.

Ratings were converted into numeric values and availability was converted into a Boolean field.

## SQLite Database

The cleaned data was stored in a SQLite database named:

`books.db`

The database contains two related tables:

### Categories Table

- category_id - Primary Key
- category_name

### Books Table

- book_id - Primary Key
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id - Foreign Key

The `category_id` creates a relationship between the Books and Categories tables.

## SQL Analysis

The following SQL queries were performed:

1. Total number of books
2. Number of books by category
3. Average book price
4. Books with rating 4 or higher
5. Most expensive books

A SQL JOIN was also performed between the Books and Categories tables.

## Pandas Integration

SQL results were read using:

`pandas.read_sql_query()`

The relationship between Books and Categories was also reproduced using:

`pandas.merge()`

## Module 1 Outcome

The pipeline successfully demonstrates:

- Web scraping
- Data cleaning
- Data type conversion
- Currency transformation
- SQLite database creation
- Primary Key and Foreign Key relationships
- SQL queries
- SQL JOIN
- Pandas SQL integration
- Pandas merge



# Module 2 - Analytics Pipeline

## Overview

Module 2 performs exploratory data analysis, data cleaning, machine learning classification, hyperparameter tuning, cross-validation, and regression using the Titanic dataset.

## Dataset

The Titanic dataset was loaded using Seaborn and saved locally as:

`titanic.csv`

After cleaning, the processed dataset was saved as:

`titanic_cleaned.csv`

## Data Cleaning

The following cleaning steps were performed:

- Missing Age values were filled using the median.
- Missing Embarked values were filled using the mode.
- The Deck column was removed because it contained a large number of missing values.
- Categorical variables were converted using one-hot encoding.

## Exploratory Data Analysis

The following visualizations were created:

- Age Distribution
- Survival Rate by Gender
- Survival Rate by Passenger Class
- Correlation Heatmap

## Classification Models

Three classification models were trained:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Model Evaluation

Confusion matrices and ROC curves were generated to evaluate classification performance.

## Hyperparameter Tuning

Hyperparameter tuning was performed using:

`GridSearchCV`

Five-fold cross-validation was used to evaluate model performance.

The best parameters for Decision Tree and Random Forest were identified.

## Regression

A Linear Regression model was trained to predict passenger fare.

Regression metrics used:

- MAE
- MSE
- RMSE
- R²

## Final Pipeline

The final end-to-end pipeline performs:

Dataset Loading → Feature Selection → Encoding → Train/Test Split → Scaling → Model Training → Prediction → Evaluation

The final pipeline is implemented in:

`08_final_pipeline.py`

## Technologies Used

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

# Module 3 - Support Assistant

## Overview

Module 3 implements a small GenAI Support Assistant for Zepto using a Retrieval-Augmented Generation (RAG) pipeline.

The system uses a local embedding model, ChromaDB vector storage, LangGraph for query routing, Pydantic for structured output validation, and FastAPI for serving the application locally.

The required graded baseline runs completely offline without an API key or network access to an LLM provider.

## Technologies Used

- Python
- ChromaDB
- Sentence Transformers
- all-MiniLM-L6-v2
- LangChain Text Splitters
- LangGraph
- Pydantic
- FastAPI
- Uvicorn

## Project Structure

```text
ZEPTO_DATA_AI_PLATFORM/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── chroma_db/
│
├── ingest.py
├── assistant.py
├── prompt_template.py
├── graph.py
├── schemas.py
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md