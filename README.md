Movie Review Sentiment Analysis

This project is a deep learning–based sentiment analysis application that classifies movie reviews as Positive or Negative using a fine-tuned Transformer model.

The application provides an intuitive interface where users can enter a movie review and receive an instant sentiment prediction.

Features
Fine-tuned Transformer model for sentiment classification
Predicts Positive or Negative sentiment
Interactive and easy-to-use web interface
Dockerized for reproducible deployment
Hosted on Hugging Face Spaces
Tech Stack
Python
PyTorch
Hugging Face Transformers
Streamlit
Docker
Project Structure
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
Run Locally

Install the dependencies:

pip install -r requirements.txt

Start the application:

streamlit run app.py
Deployment

This application is containerized using Docker and deployed on Hugging Face Spaces.