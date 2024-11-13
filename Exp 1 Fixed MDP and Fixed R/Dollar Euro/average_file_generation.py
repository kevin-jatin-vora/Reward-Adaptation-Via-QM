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
ours_2 = [f'{i}//ours_2.csv' for i in range(avg)]
ours_3 = [f'{i}//ours_3.csv' for i in range(avg)]
ours_4 = [f'{i}//ours_4.csv' for i in range(avg)]

ql_1 = [f'{i}//QL_1.csv' for i in range(avg)]
ql_2 = [f'{i}//QL_2.csv' for i in range(avg)]
ql_3 = [f'{i}//QL_3.csv' for i in range(avg)]
ql_4 = [f'{i}//QL_4.csv' for i in range(avg)]

sfql_1 = [f'{i}//sfql_1.csv' for i in range(avg)]
sfql_2 = [f'{i}//sfql_2.csv' for i in range(avg)]
sfql_3 = [f'{i}//sfql_3.csv' for i in range(avg)]
sfql_4 = [f'{i}//sfql_4.csv' for i in range(avg)]

# Calculate averages for each group of files
ours_1_avg = calculate_average_csv_files(ours_1)
ours_2_avg = calculate_average_csv_files(ours_2)
ours_3_avg = calculate_average_csv_files(ours_3)
ours_4_avg = calculate_average_csv_files(ours_4)

ql_1_avg = calculate_average_csv_files(ql_1)
ql_2_avg = calculate_average_csv_files(ql_2)
ql_3_avg = calculate_average_csv_files(ql_3)
ql_4_avg = calculate_average_csv_files(ql_4)

sfql_1_avg = calculate_average_csv_files(sfql_1)
sfql_2_avg = calculate_average_csv_files(sfql_2)
sfql_3_avg = calculate_average_csv_files(sfql_3)
sfql_4_avg = calculate_average_csv_files(sfql_4)

# Save the average DataFrames to CSV files (optional)
ours_1_avg.to_csv('ours_1.csv', index=False)
ours_2_avg.to_csv('ours_2.csv', index=False)
ours_3_avg.to_csv('ours_3.csv', index=False)
ours_4_avg.to_csv('ours_4.csv', index=False)

ql_1_avg.to_csv('QL_1.csv', index=False)
ql_2_avg.to_csv('QL_2.csv', index=False)
ql_3_avg.to_csv('QL_3.csv', index=False)
ql_4_avg.to_csv('QL_4.csv', index=False)

sfql_1_avg.to_csv('sfql_1.csv', index=False)
sfql_2_avg.to_csv('sfql_2.csv', index=False)
sfql_3_avg.to_csv('sfql_3.csv', index=False)
sfql_4_avg.to_csv('sfql_4.csv', index=False)