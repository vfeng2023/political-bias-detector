import streamlit as st
import openai
import pandas as pd

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"

# Prompt engineering function
def classify_bias(article_text):
    prompt = (
        "Classify the political bias of the following news article as Liberal, Conservative, or Neutral.\n\n"
        f"Article: {article_text}\n\n"
        "Classification:"
    )
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=10
        )
        classification = response.choices[0].text.strip()
        return classification
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app layout
st.title("Political Bias Detection System")
st.markdown("Classify the political bias of news articles as **Liberal**, **Conservative**, or **Neutral**.")

# Input: User-provided article text
article_text = st.text_area("Enter the news article text below:", height=200)

if st.button("Classify Bias"):
    if article_text.strip():
        with st.spinner("Classifying..."):
            classification = classify_bias(article_text)
        st.success(f"Political Bias: {classification}")
    else:
        st.warning("Please enter an article to classify.")

# Option to upload and classify a dataset
st.markdown("---")
st.header("Batch Analysis")
file_upload = st.file_uploader("Upload a CSV file with a column named 'Article':", type=["csv"])

if file_upload:
    df = pd.read_csv(file_upload)
    if "Article" in df.columns:
        with st.spinner("Classifying articles..."):
            df["Bias"] = df["Article"].apply(classify_bias)
        st.write("Classification Complete! Here are the results:")
        st.dataframe(df)

        # Download link
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_download = convert_df(df)
        st.download_button(
            label="Download Results as CSV",
            data=csv_download,
            file_name="bias_classification_results.csv",
            mime="text/csv"
        )
    else:
        st.error("The uploaded CSV must contain a column named 'Article'.")

st.markdown("---")
st.info("This tool uses OpenAI's GPT model for classification. Ensure you have an API key set.")
