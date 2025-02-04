import streamlit as st
import pandas as pd
from analyse_lib import *

NITs_cutoff = pd.read_csv("./Data/2024/NITs_2024.csv")

# Title of the dashboard
st.title('NIT Admissions Dashboard')

# Sidebar inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    NIT_name = st.sidebar.selectbox('Select NIT', NITs_cutoff['Institute'].unique())
    Branch_name = st.sidebar.multiselect('Select Branch', NITs_cutoff[NITs_cutoff["Institute"] == NIT_name]['Academic Program Name'].unique())
    Gender = st.sidebar.selectbox('Select Gender', NITs_cutoff['Gender'].unique())
    Category = st.sidebar.selectbox('Select Category', NITs_cutoff['Seat Type'].unique())
    Quota = st.sidebar.selectbox('Select Quota', ['HS', 'OS'])

    data = {'NIT Name': [NIT_name] * len(Branch_name),
            'Branch Name': Branch_name,
            'Gender': [Gender] * len(Branch_name),
            'Category': [Category] * len(Branch_name),
            'Quota': [Quota] * len(Branch_name)}

    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

# Debugging statements
st.write("Input DataFrame:")
st.write(input_df)

mvr_2024 = Marks_vs_Rank_Analyser(2024)
mvr_2023 = Marks_vs_Rank_Analyser(2023)

# Check if input_df is empty
if input_df.empty:
    st.error("Input DataFrame is empty. Please select appropriate values in the sidebar.")
else:
    # Filtering the DataFrame based on user input
    filtered_db = NITs_cutoff[
        (NITs_cutoff['Institute'] == input_df['NIT Name'].iloc[0]) &
        (NITs_cutoff['Academic Program Name'].isin(input_df['Branch Name'])) &
        (NITs_cutoff['Gender'] == input_df['Gender'].iloc[0]) &
        (NITs_cutoff['Seat Type'] == input_df['Category'].iloc[0]) &
        (NITs_cutoff['Quota'] == input_df['Quota'].iloc[0])
        ]
    filtered_db.loc[:, "2024 Min Score"] = mvr_2024.predict_marks(filtered_db["Closing Rank"])[0]
    filtered_db.loc[:, "2024 Max Score"] = mvr_2024.predict_marks(filtered_db["Closing Rank"])[1]
    filtered_db.loc[:, "2024 Average Score"] = mvr_2024.predict_marks(filtered_db["Closing Rank"])[2]
    filtered_db.loc[:, "2024 Median Score"] = mvr_2024.predict_marks(filtered_db["Closing Rank"])[3]

    filtered_db.loc[:, "2023 Min Score"] = mvr_2023.predict_marks(filtered_db["Closing Rank"])[0]
    filtered_db.loc[:, "2023 Max Score"] = mvr_2023.predict_marks(filtered_db["Closing Rank"])[1]
    filtered_db.loc[:, "2023 Average Score"] = mvr_2023.predict_marks(filtered_db["Closing Rank"])[2]
    filtered_db.loc[:, "2023 Median Score"] = mvr_2023.predict_marks(filtered_db["Closing Rank"])[3]

    filtered_db.loc[:, "Predicted Score"] = ((2*filtered_db["2024 Average Score"] - filtered_db["2023 Average Score"])+(2*filtered_db["2024 Median Score"] - filtered_db["2023 Median Score"]))//2
    # Display filtered data
    st.subheader('Filtered Data')
    st.write(filtered_db[["Academic Program Name", "Closing Rank", "Predicted Score","2024 Min Score", "2024 Max Score", "2024 Average Score", "2024 Median Score", "2023 Min Score", "2023 Max Score", "2023 Average Score", "2023 Median Score"]])

