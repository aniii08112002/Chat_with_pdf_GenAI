from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        df = pd.read_csv(csv_file)

        # Display the dataframe
        st.write("Data Preview:")
        st.write(df.head())

        agent = create_csv_agent(OpenAI(temperature=0), csv_file, verbose=True)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                result = agent.run(user_question)
                st.write(result)

        # Example of displaying a simple chart
        st.subheader("Chart")
        if st.button("Generate Chart"):
            # Assuming the CSV has a column named 'data_column' to plot
            plt.figure(figsize=(10, 5))
            plt.plot(df['data_column'], label='Data')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Data Column Chart')
            plt.legend()
            st.pyplot(plt)


if __name__ == "__main__":
    main()
