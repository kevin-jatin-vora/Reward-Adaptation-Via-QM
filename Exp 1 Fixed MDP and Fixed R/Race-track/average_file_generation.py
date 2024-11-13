import os
import pandas as pd
import numpy as np

def calculate_average_csv_files(file_list):
    """Calculates the average of multiple CSV files.

    Args:
        file_list: A list of CSV file paths.

    Returns:
        A pandas DataFrame containing the average of the CSV files.
    """

    dataframes = []
    for file_path in file_list:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    average_df = pd.concat(dataframes)#.groupby(level=0).mean()

    return average_df
avg = 3
# Create lists of CSV file paths
ours_1 = [f'{i}//ours_1.csv' for i in range(avg)]
ours_3 = [f'{i}//ours_3.csv' for i in range(avg)]
ours_5 = [f'{i}//ours_5.csv' for i in range(avg)]
ours_7 = [f'{i}//ours_7.csv' for i in range(avg)]

ql_1 = [f'{i}//QL_1.csv' for i in range(avg)]
ql_3 = [f'{i}//QL_3.csv' for i in range(avg)]
ql_5 = [f'{i}//QL_5.csv' for i in range(avg)]
ql_7 = [f'{i}//QL_7.csv' for i in range(avg)]

sfql_1 = [f'{i}//sfql_1.csv' for i in range(avg)]
sfql_3 = [f'{i}//sfql_3.csv' for i in range(avg)]
sfql_5 = [f'{i}//sfql_5.csv' for i in range(avg)]
sfql_7 = [f'{i}//sfql_7.csv' for i in range(avg)]

# Calculate averages for each group of files
ours_1_avg = calculate_average_csv_files(ours_1)
ours_3_avg = calculate_average_csv_files(ours_3)
ours_5_avg = calculate_average_csv_files(ours_5)
ours_7_avg = calculate_average_csv_files(ours_7)

ql_1_avg = calculate_average_csv_files(ql_1)
ql_3_avg = calculate_average_csv_files(ql_3)
ql_5_avg = calculate_average_csv_files(ql_5)
ql_7_avg = calculate_average_csv_files(ql_7)

sfql_1_avg = calculate_average_csv_files(sfql_1)
sfql_3_avg = calculate_average_csv_files(sfql_3)
sfql_5_avg = calculate_average_csv_files(sfql_5)
sfql_7_avg = calculate_average_csv_files(sfql_7)

# Save the average DataFrames to CSV files (optional)
ours_1_avg.to_csv('ours_1.csv', index=False)
ours_3_avg.to_csv('ours_3.csv', index=False)
ours_5_avg.to_csv('ours_5.csv', index=False)
ours_7_avg.to_csv('ours_7.csv', index=False)

ql_1_avg.to_csv('QL_1.csv', index=False)
ql_3_avg.to_csv('QL_3.csv', index=False)
ql_5_avg.to_csv('QL_5.csv', index=False)
ql_7_avg.to_csv('QL_7.csv', index=False)

sfql_1_avg.to_csv('sfql_1.csv', index=False)
sfql_3_avg.to_csv('sfql_3.csv', index=False)
sfql_5_avg.to_csv('sfql_5.csv', index=False)
sfql_7_avg.to_csv('sfql_7.csv', index=False)