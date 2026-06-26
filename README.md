---
title: Movie Review Sentiment Analysis
emoji: 🎬
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
-------------

# 🎬 Movie Review Sentiment Analysis

A deep learning-based web application that classifies movie reviews as **Positive** or **Negative** using a fine-tuned Transformer model.

## Features

* Fine-tuned Transformer model for sentiment analysis
* Binary sentiment prediction (Positive / Negative)
* Interactive web interface
* Dockerized deployment
* Hosted on Hugging Face Spaces

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Streamlit
* Docker

## Project Structure

```text
.
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
└── movie_sentiment_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Deployment

The application is containerized using Docker and deployed on Hugging Face Spaces.

