import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import nltk
import pandas as pd
from tqdm import tqdm

nltk.download('stopwords')

# Setting up the Streamlit app
st.title("Patient Notes Summarization App")
st.write("This app generates concise summaries of patient notes using a T5 model.")

# Data Uploading
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file with patient notes", type=["csv"])

if uploaded_file is not None:
    try:
        patients_data = pd.read_csv(uploaded_file)
        if 'pn_history' not in patients_data.columns:
            st.error("The uploaded CSV does not contain a 'pn_history' column.")
        else:
            st.write("Uploaded Data:")
            st.dataframe(patients_data.head())

            # Data Preprocessing
            st.sidebar.header("Data Preprocessing")
            if st.sidebar.checkbox("Remove Duplicates"):
                patients_data.drop_duplicates(subset=['pn_history'], inplace=True)
                st.write("Duplicates removed.")

            if st.sidebar.checkbox("Handle Missing Values"):
                patients_data.dropna(subset=['pn_history'], inplace=True)
                st.write("Missing values handled.")

            patients_data['pn_history'] = patients_data['pn_history'].str.lower().str.replace('[^\w\s]', '', regex=True)
            st.write("Text normalization done.")

            # EDA Section
            st.sidebar.header("Exploratory Data Analysis")
            if st.sidebar.checkbox("Show Text Length Distribution"):
                patients_data['text_length'] = patients_data['pn_history'].apply(len)
                plt.figure(figsize=(10, 6))
                sns.histplot(patients_data['text_length'], bins=50, kde=True)
                plt.title('Distribution of Patient Note Lengths')
                plt.xlabel('Text Length')
                plt.ylabel('Frequency')
                st.pyplot(plt)

            if st.sidebar.checkbox("Show Most Common Words"):
                stop_words = set(nltk.corpus.stopwords.words('english'))
                patients_data['tokenized'] = patients_data['pn_history'].apply(lambda x: [word for word in x.split() if word not in stop_words])
                all_words = [word for tokens in patients_data['tokenized'] for word in tokens]
                word_freq = Counter(all_words)
                common_words = word_freq.most_common(20)
                words, counts = zip(*common_words)
                plt.figure(figsize=(10, 6))
                sns.barplot(x=list(counts), y=list(words))
                plt.title('Most Common Words in Patient Notes')
                plt.xlabel('Frequency')
                plt.ylabel('Words')
                st.pyplot(plt)

            # Model Section
            st.sidebar.header("Model Settings")
            model_name = st.sidebar.selectbox("Choose Model Version", ["t5-small", "t5-base", "t5-large"])
            max_input_length = st.sidebar.slider("Max Input Length", min_value=50, max_value=512, value=512)
            max_summary_length = st.sidebar.slider("Max Summary Length", min_value=50, max_value=150, value=150)

            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            # Summary Generation Section
            st.header("Generate Summaries")
            if st.button("Generate Summaries for Uploaded Data"):
                def generate_summary(text):
                    inputs = tokenizer(text, max_length=max_input_length, truncation=True, return_tensors="pt", padding="max_length")
                    input_ids = inputs.input_ids.to(device)
                    attention_mask = inputs.attention_mask.to(device)
                    summary_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=max_summary_length, num_beams=4, length_penalty=2.0, early_stopping=True)
                    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Progress Bar
                progress = st.progress(0)
                for i, _ in enumerate(tqdm(patients_data['pn_history'])):
                    progress.progress((i + 1) / len(patients_data['pn_history']))

                try:
                    patients_data['generated_summary'] = patients_data['pn_history'].apply(generate_summary)
                    st.write("Summarization complete. Here are some examples:")
                    st.dataframe(patients_data[['pn_history', 'generated_summary']].head())

                    # Download option
                    csv = patients_data[['pn_history', 'generated_summary']].to_csv(index=False)
                    st.download_button(label="Download Summaries", data=csv, file_name="summarized_patient_notes.csv", mime="text/csv")

                except Exception as e:
                    st.error(f"An error occurred during summarization: {str(e)}")

            # Single Text Summarization
            st.subheader("Or, Input a Single Patient Note to Summarize")
            input_text = st.text_area("Enter Patient Note", "")
            if st.button("Summarize"):
                if input_text:
                    try:
                        summary = generate_summary(input_text)
                        st.write("Generated Summary:")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {str(e)}")

            # Evaluation Section 
            st.header("Model Evaluation")
            if st.checkbox("Show Evaluation Metrics"):
                st.write("Add code here to display evaluation metrics like BLEU or ROUGE scores.")

            # User Feedback Section 
            st.header("User Feedback")
            feedback = st.text_input("Please provide your feedback or rate the summaries:")
            if st.button("Submit Feedback"):
                st.write("Thank you for your feedback!")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

