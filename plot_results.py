import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the CSV data into a DataFrame
csv_file_path = 'path_to_your_csv_file.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Convert 'Time' column to timedelta and 'Island' to a categorical type for plotting
df['Time'] = pd.to_timedelta(df['Time'].str.replace('h', 'hours').str.replace('m', 'minutes').str.replace('s', 'seconds'))
df['Island'] = df['Island'].astype('category')

# Calculate the average score at each timestamp
average_scores = df.groupby('Time')['Score'].mean().reset_index(name='AverageScore')

# Plotting
fig, ax = plt.subplots()
for island, group in df.groupby('Island'):
    ax.plot(group['Time'], group['Score'], label=f'Island {island}')

# Plot the average score
ax.plot(average_scores['Time'], average_scores['AverageScore'], label='Average', color='black', linewidth=2)

# Formatting the x-axis to show hours, minutes, and seconds
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

# Labels and title
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Island Score Over Time with Average')
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show grid
plt.grid(True)

# Show the plot
plt.show()
