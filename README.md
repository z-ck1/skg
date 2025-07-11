# skg
data &amp; code snippets for plots

skg_csv.csv
includes combined data from the SKG tracker from JULY 31 2024 to JULY 3 2024
!NOTE:
  JUNE 2 - JUNE 17 IS INTERPOLATED, if you havev daily / total values please provide data&source
  JUNE 1 and JUNE 18 are real datapoints

skg_plot.py:
  plots cumulative and daily signatures while squeezing a period from october to may, as that period was extremely stagnant.
  "prediction" not really a prediction, more so just decay
  Not sure if I'm gonna be updating skg_csv.csv, so added a list for next values from 4 to 8 of July at the top of the code

skg_plot_adjusted.py:
  Tried fixing proportions and segments, adding annotations for truncated x axis, the line on lower plot is smoothed, took quite a bit out of me though. Coding is frustrating and I have a new thing I hate. Bezier curves.
