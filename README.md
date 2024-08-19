# mednotes
Patient Notes Summarization with T5 and Streamlit
This project focuses on developing an AI-powered tool to generate concise summaries of patient notes using a fine-tuned T5 (Text-To-Text Transfer Transformer) model. The primary goal is to assist healthcare providers by quickly summarizing long patient histories into key points, enabling more efficient review and decision-making.

Key Features:
Natural Language Processing: Utilizes the T5 model for summarizing unstructured clinical text.
Interactive Streamlit App: Provides an easy-to-use interface for uploading patient notes, generating summaries, and downloading results.
Customizable Parameters: Allows users to tweak model settings such as input length and summary length for optimal performance.
Data Preprocessing: Includes steps to clean and normalize the text, remove duplicates, and handle missing values.
Exploratory Data Analysis: Visualizes text length distribution and identifies the most common words in the dataset.
Evaluation: Implements mechanisms for evaluating the quality of the generated summaries against reference data.
Technologies Used:
Python: Core language for data processing and model implementation.
Streamlit: Framework for building and deploying the web application.
PyTorch: Deep learning framework used for loading and fine-tuning the T5 model.
Transformers: Hugging Face library for implementing the T5 model.
NLTK: Natural Language Toolkit for preprocessing text data.
Matplotlib/Seaborn: Libraries for data visualization.
This project represents a step forward in leveraging AI to enhance healthcare workflows, particularly in the efficient management and summarization of patient records.
