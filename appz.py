import streamlit as st
import pandas as pd
import skcriteria as skc
import matplotlib.pyplot as plt
import platform
import skcriteria
from skcriteria.madm import similarity  
from skcriteria.pipeline import mkpipe  
from skcriteria.preprocessing import invert_objectives, scalers
import numpy as np
from skcriteria.pipeline import mkpipe
from skcriteria.preprocessing.invert_objectives import InvertMinimize,NegateMinimize 
from skcriteria.preprocessing.filters import FilterNonDominated
from skcriteria.preprocessing.scalers import SumScaler, VectorScaler
from skcriteria.madm.simple import WeightedProductModel, WeightedSumModel
from skcriteria.madm.similarity import TOPSIS


def check_versions():
    python_version = platform.python_version()
    skcriteria_version = skcriteria.__version__
    return python_version, skcriteria_version

small_constant = 1e-6
replace_dict = {
    'Low': '1,3,5',
    'Medium': '3,5,7',
    'High': '5,7,9'
}

def replace_zeros_with_constant(df, constant):
    """
    Replace zero values with a small constant in the DataFrame.
    """
    return df.replace(0, constant)

def replace_values_with_dict(df, replace_dict):
    """
    Replace values in the DataFrame according to the specified dictionary.
    """
    return df.replace(replace_dict, inplace=True)

def find_inverse(value):
    """
    Find the inverse of a string of numbers separated by commas.
    """
    numbers = value.split(',')
    inverse_numbers = [format(1 / float(num), '.2f') for num in numbers]
    return ','.join(inverse_numbers)

def apply_inverse_to_min_columns(df):
    """
    Apply the find_inverse function to columns with "MIN" in their names.
    """
    min_columns = [col for col in df.columns if 'MIN' in col]
    for col in min_columns:
        df[col] = df[col].apply(lambda x: find_inverse(x) if isinstance(x, str) else x)
    return df

def normalize_max(column):
    """
    Normalize a column based on its maximum value.
    """
    values = column.split(',')
    max_value = max(map(int, values))
    return [round(int(val) / max_value, 3) for val in values]

def normalize_min(column):
    """
    Normalize a column based on its minimum value.
    """
    values = column.split(',')
    min_value = min(map(float, values))
    normalized_values = [round(min_value / float(val), 3) for val in values]
    return normalized_values

def apply_normalization(df):
    """
    Apply normalization based on column names containing "MAX" or "MIN".
    """
    for column_name in df.columns:
        if 'MAX' in column_name:
            df[column_name] = df[column_name].apply(normalize_max)
        elif 'MIN' in column_name:
            df[column_name] = df[column_name].apply(normalize_min)
    return df

def remove_square_brackets_from_numerical(df):
    """
    Remove square brackets from numerical columns.
    """
    numerical_columns = df.columns.difference(['index'])
    df[numerical_columns] = df[numerical_columns].applymap(lambda x: ', '.join(map(str, x)))
    return df

def convert_to_matrix_and_calculate_averages(df):
    """
    Convert the DataFrame to a matrix of floats and calculate the average for each innermost list.
    """
    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: [float(val) for val in x.split(', ')])
    matrix = df.iloc[:, 1:].values.tolist()
    averages = [[sum(values) / len(values) for values in row] for row in matrix]
    return averages

def create_decision_matrix(matrix, objectives, weights, alternatives):
    """
    Create the decision matrix using skcriteria.
    """
    dm = skc.mkdm(matrix, objectives, weights, criteria=alternatives)
    return dm

def main():
    st.title("Ranking Pharma & Non Pharma Features with Fuzzy TOPSIS")

    python_version, skcriteria_version = check_versions()
    st.write("Python version:", python_version)
    st.write("scikit-criteria version:", skcriteria_version)

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        st.write("File successfully uploaded!")

        try:
            # Load data
            df = pd.read_csv(uploaded_file)

            # Display data
            st.write(df)

            # Button to trigger all preprocessing steps
            if st.button("Run Fuzzy TOPSIS Calculations"):
                replace_values_with_dict(df, replace_dict)
                apply_inverse_to_min_columns(df)
                apply_normalization(df)
                remove_square_brackets_from_numerical(df)
                averages = convert_to_matrix_and_calculate_averages(df)

                
                weights = [0.3, 0.3, 0.3, 0.1, 0.1, 0.1]
                objectives = [max, max, max, min, min, min]

                st.write("Preprocessing completed.")
                st.write(df)

                matrix = averages

                dm = skc.mkdm(
                    matrix,
                    objectives,
                    weights,
                    criteria=["Patient1 (PMC)", "Patient2 (PMC)", "Patient3 (PMC)", "Patient4 (NPMC)",
                              "Patient5 (NPMC)", "Patient6 (NPMC)"]
                )

                alternatives=[
                    "Patient's height, weight and BMI",
                    "Patients daily physical activity level",
                    "Patient's diet intake",
                    "Expert's knowledge",
                    "Social support",
                    "Patient preferences of treatment options",
                    "Hemoglobin A1c (HbA1c) level",
                    "Hypoglycemia risk",
                    "Medication availability",
                    "Presence of comorbidity",
                    "Risk of infection",
                    "Risk of gastrointestinal problems",
                    "Patient’s mental health",
                    "Daily activities functioning",
                    "Blood glucose level",
                    "Cardiovascular (CVD) risk",
                    "Renal failure",
                    "Duration of diabetes",
                    "TOC (Target Organ Complication)",
                    "Adverse effect of drugs",
                    "Age (Life expectancy)",
                    "Patient’s blood pressure",
                    "Patient’s knowledge",
                    "Polypharmacy",
                    "Dyslipidaemia",
                    "Working condition"
                ]

                dm = dm.copy(alternatives=alternatives)

                # Create a figure for the criteria KDE plot
                fig1, axs1 = plt.subplots(figsize=(6, 5))

                # Plot the criteria KDE
                dm.plot.kde(ax=axs1)
                axs1.set_title("Criteria KDE")

                st.header("1. Criteria KDE for each patient")
                # Display the criteria KDE plot in Streamlit
                st.pyplot(fig1)

                # Create a figure for the weights as bars plot
                fig2, axs2 = plt.subplots(figsize=(6, 5))

                # Plot the weights as bars
                dm.plot.wbar(ax=axs2)
                axs2.set_title("Weights as Bars")

                # Display the weights as bars plot in Streamlit
                st.header("2. Bar plot on Patient weights ")
                st.write("(Assuming pharma patients 3x prioritized)")
                st.pyplot(fig2)
                
                from skcriteria.madm import similarity  
                from skcriteria.pipeline import mkpipe  
                from skcriteria.preprocessing import invert_objectives, scalers

                from skcriteria.pipeline import mkpipe
                from skcriteria.preprocessing.invert_objectives import InvertMinimize,NegateMinimize 
                from skcriteria.preprocessing.filters import FilterNonDominated
                from skcriteria.preprocessing.scalers import SumScaler, VectorScaler
                from skcriteria.madm.simple import WeightedProductModel, WeightedSumModel
                from skcriteria.madm.similarity import TOPSIS

                # TOPSIS Calculation
                pipe = mkpipe(
                    invert_objectives.NegateMinimize(),
                    scalers.VectorScaler(target="matrix"),  # this scaler transform the matrix
                    scalers.SumScaler(target="weights"),  # and this transform the weights
                    similarity.TOPSIS(),
                )

                rankTOPSIS = pipe.evaluate(dm)

                # st.write("TOPSIS Ranking:")
                # st.write(rankTOPSIS)

                # Create DataFrame for TOPSIS ranking
                rank_list = rankTOPSIS.rank_
                rank_alternatives = rankTOPSIS.alternatives

                rank_df_topsis = pd.DataFrame({
                    'Rank': rank_list,
                    'Feature': rank_alternatives
                })

                # Sort the DataFrame by rank in ascending order
                rank_df_topsis = rank_df_topsis.sort_values(by='Rank', ascending=True)
                rank_df_topsis = rank_df_topsis.reset_index(drop=True)

                # Display TOPSIS ranking DataFrame
                st.header("3. TOPSIS Ranking DataFrame:")
                st.write(rank_df_topsis)

                # Padding the arrays to make them the same length
                e_ = rankTOPSIS.e_

                # Find the maximum length among the arrays
                max_length = max(len(e_.ideal), len(e_.anti_ideal), len(e_.similarity))

                # Pad the arrays with NaN values to make them the same length
                e_.ideal = np.pad(e_.ideal, (0, max_length - len(e_.ideal)), 'constant', constant_values=(np.nan,))
                e_.anti_ideal = np.pad(e_.anti_ideal, (0, max_length - len(e_.anti_ideal)), 'constant', constant_values=(np.nan,))
                e_.similarity = np.pad(e_.similarity, (0, max_length - len(e_.similarity)), 'constant', constant_values=(np.nan,))

                data = {
                    'ideal': e_.ideal,
                    'anti_ideal': e_.anti_ideal,
                    'similarity': e_.similarity,
                    'similarity index': alternatives
                }

                # Create the DataFrame
                df = pd.DataFrame(data)

                # Print the DataFrame
                st.header("4. TOPSIS Metrics DataFrame:")
                st.write(df.head())

                pharma = True
                non_pharma = True


                pharma_features = [
                    "Hemoglobin A1c (HbA1c) level",
                    "Blood glucose level",
                    "Patient's height, weight and BMI",
                    "Patient’s blood pressure",
                    "Duration of diabetes",
                    "Risk of infection",
                    "Working condition",
                    "Polypharmacy",
                    "Dyslipidaemia"
                ]

                non_pharma_features = [
                    "Patient’s blood pressure",
                    "Age (Life expectancy)",
                    "Daily activities functioning",
                    "Patient’s mental health",
                    "Patient preferences of treatment options",
                    "Social support",
                    "Patient's diet intake"
                ]

                # Filter and print the ranks based on the specified features if pharma is True
                if pharma:
                    pharma_ranks = rank_df_topsis[rank_df_topsis['Feature'].isin(pharma_features)]
                    pharma_ranks = pharma_ranks.sort_values(by="Rank")  # Sort by Rank
                    pharma_ranks.reset_index(drop=True, inplace=True)  # Reset the index
                    pharma_ranks['Rank'] = range(1, len(pharma_ranks) + 1)  # Reset Rank
                    

                # Filter and print the ranks based on the specified non-pharma features if non_pharma is True
                if non_pharma:
                    non_pharma_ranks = rank_df_topsis[rank_df_topsis['Feature'].isin(non_pharma_features)]
                    non_pharma_ranks = non_pharma_ranks.sort_values(by="Rank", ascending=False)  # Sort by Rank in descending order
                    non_pharma_ranks.reset_index(drop=True, inplace=True)  # Reset the index
                    non_pharma_ranks['Rank'] = range(1, len(non_pharma_ranks) + 1)  # Reset Rank

                st.header("5. Pharma Ranks:")
                st.write(pharma_ranks)
                st.header("6. Non Pharma Ranks:")
                st.write(non_pharma_ranks)

                from skcriteria.preprocessing import invert_objectives, scalers

                # Invert minimize - 1/ criterion

                inverter = invert_objectives.InvertMinimize()
                dmt = inverter.transform(dm)

                # Sum Scaler - Normalize scale (0-1)
                scaler = scalers.SumScaler(target="both")
                dmt = scaler.transform(dmt)

                from skcriteria.pipeline import mkpipe
                from skcriteria.preprocessing.invert_objectives import InvertMinimize,NegateMinimize 
                from skcriteria.preprocessing.filters import FilterNonDominated
                from skcriteria.preprocessing.scalers import SumScaler, VectorScaler
                from skcriteria.madm.simple import WeightedProductModel, WeightedSumModel
                from skcriteria.madm.similarity import TOPSIS

                # InvertMinimize(),
                # FilterNonDominated(),

                ws_pipe = mkpipe(
                SumScaler(target="weights"),
                VectorScaler(target="matrix"),
                WeightedSumModel(),
                )

                wp_pipe = mkpipe(
                SumScaler(target="weights"),
                VectorScaler(target="matrix"),
                WeightedProductModel(),
                )

                tp_pipe = mkpipe(
                SumScaler(target="weights"),
                VectorScaler(target="matrix"),
                TOPSIS(),
                )

                wsum_result = ws_pipe.evaluate(dmt)
                wprod_result = wp_pipe.evaluate(dmt)
                tp_result = tp_pipe.evaluate(dmt)

                rank_wsum = wsum_result.rank_
                alt_wsum = wsum_result.alternatives

                rank_wprod =wprod_result.rank_
                alt_wprod = wprod_result.alternatives

                rank_tp = tp_result.rank_
                alt_tp = tp_result.alternatives

                rank_comparison = pd.DataFrame({
                    'Feature': alt_wsum,
                    'Weight Sum Rank': rank_wsum,
                    'Weight Product Rank' : rank_wprod,
                    'TOPSIS Rank' : rank_tp
                    
                })


                rank_comparison['Average Rank'] = rank_comparison[['Weight Sum Rank', 'Weight Product Rank', 'TOPSIS Rank']].mean(axis=1)
                rank_comparison = rank_comparison.sort_values(by='Average Rank').reset_index(drop=True)
                
                st.header("7. Rank Comparison:")
                st.write(rank_comparison)

                pharma = True
                non_pharma = True


        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
