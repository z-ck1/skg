# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 07:38:39 2025
@author: zck1
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Config / adjustable variables for decay & text/background
config = {
    'k_main': 0.15,         # Moderate decay for main prediction
    'k_fast': 0.2,          # Faster decay (below)
    'k_slow': 0.125,        # Slower decay (middle above)
    'k_very_slow': 0.1,     # Even slower decay (top above)
    'background_color': '#1d1d25',  # Dark blue background color
    'text_color': '#FFFFFF'  # Text color (white)
}

latest_entries = [892880, 1048205,1135896,1180006,1212725,1245543, 1272085] #July 4 onward

# Read the CSV file / check datetime
df = pd.read_csv('C:/temp/skg_csv.csv', delimiter=';')
df['time'] = pd.to_datetime(df['time'])

# start/end of the graph
start_date = datetime(2024, 7, 31)
end_date = datetime(2025, 7, 31)
df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

# Calculate daily signatures
df['daily_signatures'] = df['signatures'].diff().fillna(0)
df.iloc[0, df.columns.get_loc('daily_signatures')] = df.iloc[0, df.columns.get_loc('signatures')]  # Set first day to initial signature count

# Graph date breakpoints (middle part squeezed)
break1 = datetime(2024, 9, 1)  # Start of middle period
break2 = datetime(2025, 6, 20)   # Start of final

# Proportional segment
seg1 = 2/5*31/51    #segment 1 scaled to 31/51 days prop to break 2
seg2 = 3/5          #segment 2 end (third segment 2/5 of the graph in length)

# Convert dates to numerical values & squeeze the middle data
df['days'] = (df['time'] - start_date).dt.total_seconds() / (24 * 3600)
break1_days = (break1 - start_date).total_seconds() / (24 * 3600)
break2_days = (break2 - start_date).total_seconds() / (24 * 3600)
end_days = (end_date - start_date).total_seconds() / (24 * 3600)

# Define additional tick dates
tick_dates = [
    datetime(2024, 7, 31),  # Start
    datetime(2024, 8, 15),   # Additional tick
    datetime(2024, 9, 1),   # Additional tick
    datetime(2024, 11, 1),  # Segment boundary
    datetime(2025, 1, 1),   # Additional tick
    datetime(2025, 4, 1),   # Additional tick
    datetime(2025, 6, 20),   # Segment boundary
    datetime(2025, 7, 1),   # Additional tick
    datetime(2025, 7, 15),   # Additional tick
    datetime(2025, 7, 31)   # End
]
tick_days = [(d - start_date).total_seconds() / (24 * 3600) for d in tick_dates]
tick_labels = [
    '2024-07-31',           # 2024-07-31 initiative launch
    '2024-08-15',           # additional tick
    '2024-09-01',           # 2024-09-01 1 month
    '11-01',           # 2024-10-01 break
    '2025-01-01',           # 2025-01-01 new year
    '04-01',                # 2025-03-01 2 month
    '2025-06-20',           # 2025-06-01 break
    '2025-07-01',           # 2025-07-01 1 month
    '2025-07-15',           # 2025-07-01 1 month
    '2025-07-31'            # 2025-07-31 initiative end
]

# Transformation for x-axis (each segment gets 1/3 of the plot)
def transform_x(days):
    transformed = np.zeros_like(days)
    for i, d in enumerate(days):
        if d < break1_days:
            transformed[i] = (d / break1_days) * (seg1)
        elif d < break2_days:
            transformed[i] = (seg1) + ((d - break1_days) / (break2_days - break1_days)) * (seg2-seg1)
        else:
            transformed[i] = (seg2) + ((d - break2_days) / (end_days - break2_days)) * (1-seg2)
    return transformed

# Transform tick positions and data
tick_positions = transform_x(np.array(tick_days))
df['transformed_x'] = transform_x(df['days'])

# Get the latest cumulative signature and last daily signature
last_entry = df['time'].max()
start_signatures = df.loc[df['time'] == last_entry, 'signatures'].iloc[0]  # 1,048,205 on July 3
last_daily_signatures = df.loc[df.index[-1], 'daily_signatures']  # Calculated daily difference

# Create predicted dates and x-values dynamically from last_entry to end_date
future_dates = [last_entry + timedelta(days=i) for i in range(0, (end_date - last_entry).days + 1)]
future_days = [(d - start_date).total_seconds() / (24 * 3600) for d in future_dates]
future_x_transformed = transform_x(np.array(future_days))

# Predict daily signatures with exponential decay starting from last_entry
remaining_days = (end_date - last_entry).days  # Days from last_entry to end_date
t = np.linspace(0, remaining_days, len(future_dates))  # Points from last_entry to end_date
predicted_daily = np.zeros(len(future_dates))
predicted_daily[0] = last_daily_signatures  # Latest day's actual daily value as the starting point
predicted_daily[1:] = np.maximum(0, last_daily_signatures * np.exp(-config['k_main'] * t[1:]))  # Decay from next day

# Calculate cumulative signatures
future_predicted_signatures = np.concatenate([[start_signatures], np.cumsum(predicted_daily[1:]) + start_signatures])

# Calculate shaded regions with different decay rates
lower_fast_daily = np.zeros(len(future_dates))
lower_fast_daily[0] = last_daily_signatures
lower_fast_daily[1:] = np.maximum(0, last_daily_signatures * np.exp(-config['k_fast'] * t[1:]))
middle_slow_daily = np.zeros(len(future_dates))
middle_slow_daily[0] = last_daily_signatures
middle_slow_daily[1:] = np.maximum(0, last_daily_signatures * np.exp(-config['k_slow'] * t[1:]))
upper_very_slow_daily = np.zeros(len(future_dates))
upper_very_slow_daily[0] = last_daily_signatures
upper_very_slow_daily[1:] = np.maximum(0, last_daily_signatures * np.exp(-config['k_very_slow'] * t[1:]))
future_lower_fast = np.concatenate([[start_signatures], np.cumsum(lower_fast_daily[1:]) + start_signatures])
future_middle_slow = np.concatenate([[start_signatures], np.cumsum(middle_slow_daily[1:]) + start_signatures])
future_upper_very_slow = np.concatenate([[start_signatures], np.cumsum(upper_very_slow_daily[1:]) + start_signatures])

# Combine actual and predicted data
combined_daily = np.concatenate([df['daily_signatures'].values, predicted_daily])
combined_x_daily = transform_x(np.concatenate([df['days'].values, future_days]))
combined_signatures = np.concatenate([df['signatures'].values, future_predicted_signatures])

# Create figure / axes / colors
fig = plt.figure(figsize=(16, 9), facecolor=config['background_color'])
ax1 = fig.add_subplot(2, 1, 1, facecolor=config['background_color'])
ax2 = fig.add_subplot(2, 1, 2, facecolor=config['background_color'])
plt.rcParams['text.color'] = config['text_color']
plt.rcParams['axes.labelcolor'] = config['text_color']
plt.rcParams['xtick.color'] = config['text_color']
plt.rcParams['ytick.color'] = config['text_color']
ax1.yaxis.set_ticks_position('both')
ax1.yaxis.set_label_position('left')
ax2.yaxis.set_ticks_position('both')
ax2.yaxis.set_label_position('left')
for spine in ax1.spines.values():
    spine.set_edgecolor(config['text_color'])
for spine in ax2.spines.values():
    spine.set_edgecolor(config['text_color'])
"""
# extra lines
special_dates = [
    datetime(2024, 8, 8),   # piratesoftware line
    datetime(2025, 6, 23),  # ross line
    datetime(2025, 6, 24),  # moist line
    datetime(2025, 7, 1)    # july first line
]
special_days = [(d - start_date).total_seconds() / (24 * 3600) for d in special_dates]
special_positions = transform_x(np.array(special_days))
for ax in [ax1, ax2]:
    for pos in special_positions:
        ax.axvline(pos, color='#AAAAAA', linestyle='-', alpha=0.75, linewidth=0.5)
"""
# Filled areas between decays lines
ax1.fill_between(future_x_transformed, future_middle_slow, future_upper_very_slow, color='tab:green', alpha=0.3, label='Slow Decay, k=0.1')
ax1.fill_between(future_x_transformed, future_predicted_signatures, future_middle_slow, color='tab:orange', alpha=0.3, label='Moderate, Decay k=0.125')
ax1.fill_between(future_x_transformed, future_lower_fast, future_predicted_signatures, color='tab:red', alpha=0.3, label='Fast Decay, k=0.2')

# Upper plot / cumulative signatures and prediction
ax1.plot(future_x_transformed, future_predicted_signatures, color='pink', linestyle='--', label=f'Exp decay k={config["k_main"]}')
ax1.plot(df['transformed_x'], df['signatures'], color='lightblue', label='Cumulative Signatures')
ax1.plot(future_x_transformed[:len(latest_entries)],latest_entries, color='lightblue', linestyle='solid', marker='o') #new values
ax1.set_title('Cumulative Signatures Over Time')
ax1.set_ylabel('Signatures')
ax1.set_ylim(0, 2500000)
ax1.set_yticks([0, 250000, 500000, 750000, 1000000, 1500000, 2000000, 2250000, 2500000])  # Updated ticks
ax1.get_yaxis().set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
ax1.ticklabel_format(axis='y', style='plain')
ax1.grid(True, color='#555555')  # Light gray grid for dark mode
ax1.legend(loc='upper left', framealpha=0.5, facecolor=config['background_color'])
ax1.set_xticklabels([])  # Hide x-axis labels for upper plot
ax1.yaxis.set_label_coords(-0.06, 0.5)  # y label position relative to figure edge

# Filled areas for predicted daily ranges
ax2.fill_between(future_x_transformed, middle_slow_daily, upper_very_slow_daily, color='tab:green', alpha=0.3, label='Slow Decay')
ax2.fill_between(future_x_transformed, predicted_daily, middle_slow_daily, color='tab:orange', alpha=0.3, label='Moderate Decay')
ax2.fill_between(future_x_transformed, lower_fast_daily, predicted_daily, color='tab:red', alpha=0.3, label='Fast Decay')

# Lower plot / daily signatures and prediction
ax2.plot(combined_x_daily[-len(future_dates):], combined_daily[-len(future_dates):], color='pink', linestyle='--', label=f'Exp decay k={config["k_main"]}')
ax2.plot(combined_x_daily[:-len(future_dates)], combined_daily[:-len(future_dates)], color='lightblue', label='Daily Signatures')
ax2.plot(future_x_transformed[:len(latest_entries)][:len(latest_entries)],np.insert(np.diff(latest_entries), 0, combined_daily[-len(future_dates)]), color='lightblue', linestyle='solid', marker='o')
ax2.set_title('Daily Signatures')
ax2.set_ylabel('Daily Signatures')
ax2.set_yscale('log')  # Set logarithmic scale
ax2.grid(True, which="both", color='#555555')  # Light gray grid for dark mode
ax2.legend(loc='lower left', framealpha=0.5, facecolor=config['background_color'])  # Move legend to bottom left corner
ax2.yaxis.set_label_coords(-0.06, 0.5)  # Fix y-label position relative to figure edge

# Set x-axis limits and custom ticks
for ax in [ax1, ax2]:
    ax.set_xlim(0, 1)
    ax.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels)  # Only bottom plot has x-axis labels

# Add vertical lines to mark segment boundaries
for ax in [ax1, ax2]:
    ax.axvline(seg1, color=config['text_color'], linestyle='--', alpha=0.5)
    ax.axvline(seg2, color=config['text_color'], linestyle='--', alpha=0.5)

# Add ticks & labels
ax2.tick_params(axis='x', top=True, labeltop=False)
ax2.tick_params(axis='y', right=True, labelright=True)
ax1.tick_params(axis='y', right=True, labelright=True)

# add arrow lines for truncated axis
ax1.annotate(
    '',  # no text on this arrow
    xy=(seg1, 1250000),  #left arrow location
    xytext=((seg1+seg2)/2*0.85, 1250000),   #text begin location
    arrowprops=dict(facecolor=config['text_color'], edgecolor=config['text_color'], arrowstyle='->', linewidth=1)
    )
ax1.annotate(
    'Truncated September-June',  # arrow text
    xy=(seg2, 1250000),  #right arrow location
    xytext=((seg1+seg2)/2*0.85, 1230000),  #text begin location
    arrowprops=dict(facecolor=config['text_color'], edgecolor=config['text_color'], arrowstyle='->', linewidth=1)
    )

ax2.annotate(
    '',  # no text on this arrow
    xy=(seg1, 14000),   #left arrow location
    xytext=((seg1+seg2)/2*0.85, 14000), # end point + text start 
    arrowprops=dict(facecolor=config['text_color'], edgecolor=config['text_color'], arrowstyle='->', linewidth=1)
    )
ax2.annotate(
    'Truncated September-June',  # arrow text lower graph
    xy=(seg2, 14000),  # right arrow location
    xytext=((seg1+seg2)/2*0.85, 13000),  # end point / text start should be done other way around for simplicity
    arrowprops=dict(facecolor=config['text_color'], edgecolor=config['text_color'], arrowstyle='->', linewidth=1)
    )

plt.tight_layout()
plt.show()
