import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import skcriteria as skc
from skcriteria.pipeline import mkpipe
from skcriteria.madm.simple import WeightedProductModel, WeightedSumModel
from skcriteria.madm.similarity import TOPSIS

# Load data
df = pd.read_csv('diabetes4.csv', encoding='mac_roman')

# Create the Decision Matrix
dm = skc.mkdm(
    matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    criteria=["Criterion 1", "Criterion 2", "Criterion 3"],
    weights=[0.3, 0.3, 0.4],
    objectives=[max, max, max], # Define objectives here
    alternatives=["Alt 1", "Alt 2", "Alt 3"]
)

# Create empty lists to store results
wsum_results = []
wprod_results = []
topsis_results = []

# Create Streamlit app
st.title('Fuzzy TOPSIS Calculator')

# Print the initial decision matrix
st.subheader('Initial Decision Matrix')
st.write(dm)

# Function to append results when buttons are clicked
def append_results():
    ws_pipe = mkpipe(WeightedSumModel())
    wp_pipe = mkpipe(WeightedProductModel())
    tp_pipe = mkpipe(TOPSIS())

    wsum_result = ws_pipe.evaluate(dm)
    wprod_result = wp_pipe.evaluate(dm)
    tp_result = tp_pipe.evaluate(dm)
    
    wsum_results.append(wsum_result)
    wprod_results.append(wprod_result)
    topsis_results.append(tp_result)

# Button to append results
if st.button('Calculate Results'):
    append_results()

# Display results when the respective button is clicked
if wsum_results:
    st.subheader('Weighted Sum Model Result')
    st.write(wsum_results[-1])

if wprod_results:
    st.subheader('Weighted Product Model Result')
    st.write(wprod_results[-1])

if topsis_results:
    st.subheader('TOPSIS Result')
    st.write(topsis_results[-1])

# Calculate ranks and create comparison dataframes
if wsum_results:
    rank_wsum = wsum_results[-1].rank_
    st.subheader('Rank Comparison for Weighted Sum Model')
    st.write(rank_wsum)

if wprod_results:
    rank_wprod = wprod_results[-1].rank_
    st.subheader('Rank Comparison for Weighted Product Model')
    st.write(rank_wprod)

if topsis_results:
    rank_tp = topsis_results[-1].rank_
    st.subheader('Rank Comparison for TOPSIS')
    st.write(rank_tp)
