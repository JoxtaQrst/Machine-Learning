import pandas as pd
import random
import numpy as np

# Read the data from the original CSV file
data = pd.read_csv('2023_LoL_esports_match_data_from_OraclesElixir.csv')

# Filter rows for the specified leagues (LCS, LEC, LCK, LPL)
leagues = ['LCS', 'LEC', 'LCK', 'LPL']
filtered_data = data[data['league'].isin(leagues)]

# Filter out rows with empty 'champion' or 'team' in the 'side' column
filtered_data = filtered_data.dropna(subset=['champion'])
filtered_data = filtered_data[filtered_data['side'] != 'team']

# Use the desired columns for champion performances
relevant_columns = filtered_data[['champion', 'league', 'result', 'kills', 'deaths', 'assists',
                                 'side', 'damagetochampions', 'position']]

# Merge with the champion attributes template
champion_attributes = pd.read_csv('fabian.csv')
relevant_columns = relevant_columns.merge(champion_attributes, left_on='champion', right_on='champion', how='left')

roles = ['top', 'mid', 'jng', 'sup', 'bot']

# Create a column for each role
for role in roles:
    relevant_columns[role] = ['yes' if role == str(val).lower() else 'no' for val in relevant_columns['position']]

# Skip lines with missing values in the 'position' column
relevant_columns = relevant_columns[relevant_columns['position'].notna()]

# Shuffle rows randomly within each league
shuffled_data = relevant_columns.groupby('league', group_keys=False).apply(lambda group: group.sample(frac=1))

# Limit each league to 75 rows
limited_data = shuffled_data.groupby('league').head(75)

# Remove the 'position'
limited_data = limited_data.drop(columns=['position'])

# Assuming 'firstblood' and 'firstbloodkill' columns exist in your original data CSV
limited_data['first_blood_kill'] = np.where((data.loc[limited_data.index, 'firstblood'] == 1) & (limited_data['kills'] >= 1), 'yes', 'no')

# Save the selected columns to a new CSV file
limited_data.to_csv('champions_info.csv', index=False, header=True)
print(filtered_data[['firstblood', 'firstbloodkill']])


# # Calculate average statistics for each champion
# average_stats = shuffled_data.groupby('champion').agg({
#     'result': ['sum'],
#     'kills': ['sum'],
#     'deaths': ['sum'],
#     'assists': ['sum'],
#     'damagetochampions': ['mean'],
#     'teamkills': ['mean'],
#     'teamdeaths': ['mean'],
#     'top': ['max'],
#     'mid': ['max'],
#     'jng': ['max'],
#     'sup': ['max'],
#     'bot': ['max'],
#     'Difficulty': ['max'],
#     'Items_For_Spike': ['max'],
#     'Attack_Type': ['max']
# }).reset_index()
#
# # Ensure that 'result' column is calculated as a count of wins
# average_stats['result', 'sum'] = round(average_stats['result', 'sum'])  # Assuming 75 games per champion
# average_stats['kills', 'sum'] = round(average_stats['kills', 'sum'])
# average_stats['deaths', 'sum'] = round(average_stats['deaths', 'sum'])
# average_stats['assists', 'sum'] = round(average_stats['assists', 'sum'])
# average_stats['damagetochampions', 'mean'] = round(average_stats['damagetochampions', 'mean'], 3)
# average_stats['teamkills', 'mean'] = round(average_stats['teamkills', 'mean'], 3)
# average_stats['teamdeaths', 'mean'] = round(average_stats['teamdeaths', 'mean'], 3)
#
#
# # Rename columns for clarity
# average_stats.columns = ['champion', 'number of wins', 'kills', 'deaths', 'assists', 'average damagetochampions', 'average teamkills', 'average teamdeaths', 'top', 'mid', 'jng', 'sup', 'bot', 'difficulty', 'items needed for spike', 'melee or ranged']
#
# # Save the calculated average statistics to a new CSV file
# average_stats.to_csv('champion_average_stats.csv', index=False, header=True)
