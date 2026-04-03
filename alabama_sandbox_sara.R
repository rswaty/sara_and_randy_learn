#load packages

library (tidyverse)
library(janitor)
library(scales)


#Raw data

data_raw <- read_csv('bps_evt_combine.csv')
view(data_raw)


#Randy's suggestion:
#i.	Select for BPS_MODEL, and Count.  
#ii.	Groupby BPS_MODEL then summarize something like “bps_total_count = sum(Count)”
#iii.	Join in attributes you want

bps_slice <- data_raw |>
  select(BPS_MODEL, Count) |>
  group_by(BPS_MODEL) |>
  summarize(bps_total_count = sum(Count)) |>
  #I'm assuming count is not in the right acreage format. 
  mutate(bps_acres = round(replace_na(bps_total_count, 0) * 0.2223945)) |>
  arrange(desc(bps_total_count))
(bps_slice)

# Understanding the data:
# How many evts are there and how do they stack up
evt_count <- data_raw |>
  select(EVT_NAME, Count) |>
  group_by(EVT_NAME) |>
  summarise(evt_total_count = sum(Count), .groups = "drop") |>
  mutate(evt_acres = round(replace_na(evt_total_count, 0) * 0.2223945),
         prcnt_of_total = (evt_acres/sum(evt_acres)*100)) |>
  arrange(desc(evt_total_count))
view(evt_count)

class(prcnt_of_total)

?percent

# Look at which biophysical setting has been turned into development the most by acreage and percentage
# what about is grouped by GROUPVEG





