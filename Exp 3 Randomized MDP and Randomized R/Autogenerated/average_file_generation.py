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
avg = 20
# Create lists of CSV file paths
ours_0_1 = [f'{i}//ours_0_1.csv' for i in range(avg)]
ours_0_3 = [f'{i}//ours_0_3.csv' for i in range(avg)]
ours_0_5 = [f'{i}//ours_0_5.csv' for i in range(avg)]

ql_0_1 = [f'{i}//QL_0_1.csv' for i in range(avg)]
ql_0_3 = [f'{i}//QL_0_3.csv' for i in range(avg)]
ql_0_5 = [f'{i}//QL_0_5.csv' for i in range(avg)]

sfql_0_1 = [f'{i}//SFQL_0_1.csv' for i in range(avg)]
sfql_0_3 = [f'{i}//SFQL_0_3.csv' for i in range(avg)]
sfql_0_5 = [f'{i}//SFQL_0_5.csv' for i in range(avg)]

# Calculate averages for each group of files
ours_0_1_avg = calculate_average_csv_files(ours_0_1)
ours_0_3_avg = calculate_average_csv_files(ours_0_3)
ours_0_5_avg = calculate_average_csv_files(ours_0_5)

ql_0_1_avg = calculate_average_csv_files(ql_0_1)
ql_0_3_avg = calculate_average_csv_files(ql_0_3)
ql_0_5_avg = calculate_average_csv_files(ql_0_5)

sfql_0_1_avg = calculate_average_csv_files(sfql_0_1)
sfql_0_3_avg = calculate_average_csv_files(sfql_0_3)
sfql_0_5_avg = calculate_average_csv_files(sfql_0_5)

# Save the average DataFrames to CSV files
ours_0_1_avg.to_csv('ours_0_1.csv', index=False)
ours_0_3_avg.to_csv('ours_0_3.csv', index=False)
ours_0_5_avg.to_csv('ours_0_5.csv', index=False)

ql_0_1_avg.to_csv('QL_0_1.csv', index=False)
ql_0_3_avg.to_csv('QL_0_3.csv', index=False)
ql_0_5_avg.to_csv('QL_0_5.csv', index=False)

sfql_0_1_avg.to_csv('SFQL_0_1.csv', index=False)
sfql_0_3_avg.to_csv('SFQL_0_3.csv', index=False)
sfql_0_5_avg.to_csv('SFQL_0_5.csv', index=False)

# # Create lists of CSV file paths
# ours_0_1 = [f'{i}//ours_0_1.csv' for i in range(avg)]
# ours_0_4 = [f'{i}//ours_0_4.csv' for i in range(avg)]
# ours_0_7 = [f'{i}//ours_0_7.csv' for i in range(avg)]

# ql_0_1 = [f'{i}//QL_0_1.csv' for i in range(avg)]
# ql_0_4 = [f'{i}//QL_0_4.csv' for i in range(avg)]
# ql_0_7 = [f'{i}//QL_0_7.csv' for i in range(avg)]

# sfql_0_1 = [f'{i}//SFQL_0_1.csv' for i in range(avg)]
# sfql_0_4 = [f'{i}//SFQL_0_4.csv' for i in range(avg)]
# # sfql_0_3 = [f'{i}//sfql_0_5.csv' for i in range(avg)]
# sfql_0_7 = [f'{i}//SFQL_0_7.csv' for i in range(avg)]

# # Calculate averages for each group of files
# ours_0_1_avg = calculate_average_csv_files(ours_0_1)
# ours_0_4_avg = calculate_average_csv_files(ours_0_4)
# ours_0_7_avg = calculate_average_csv_files(ours_0_7)

# ql_0_1_avg = calculate_average_csv_files(ql_0_1)
# ql_0_4_avg = calculate_average_csv_files(ql_0_4)
# # ql_0_3_avg = calculate_average_csv_files(ql_0_3)
# ql_0_7_avg = calculate_average_csv_files(ql_0_7)

# sfql_0_1_avg = calculate_average_csv_files(sfql_0_1)
# sfql_0_4_avg = calculate_average_csv_files(sfql_0_4)
# sfql_0_7_avg = calculate_average_csv_files(sfql_0_7)

# # Save the average DataFrames to CSV files (optional)
# ours_0_1_avg.to_csv('ours_0_1.csv', index=False)
# ours_0_4_avg.to_csv('ours_0_4.csv', index=False)
# ours_0_7_avg.to_csv('ours_0_7.csv', index=False)

# ql_0_1_avg.to_csv('QL_0_1.csv', index=False)
# ql_0_4_avg.to_csv('QL_0_4.csv', index=False)
# ql_0_7_avg.to_csv('QL_0_7.csv', index=False)

# sfql_0_1_avg.to_csv('SFQL_0_1.csv', index=False)
# sfql_0_4_avg.to_csv('SFQL_0_4.csv', index=False)
# sfql_0_7_avg.to_csv('SFQL_0_7.csv', index=False)