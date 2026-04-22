#load packages

library (tidyverse)
library(janitor)
library(scales)
#library(ggExtra)
#library(reshape2)

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
  #I'm assuming count is not in the right acreage format as it is the count of pixels 
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
         prcnt_of_total = paste0(round(evt_acres/sum(evt_acres)*100, 1), "%")) |>
  arrange(desc(evt_total_count))
view(evt_count)

## Results ## Southeastern North American Temperate Forest Plantation is 23% of total acreage in Alabama

#total pixels
total_count <- sum(evt_count$evt_total_count, na.rm = TRUE)
(total_count)

#bring this down to only developed land
dvlp_evt_count <- data_raw |>
  select(EVT_NAME, , EVT_LF, Count) |>
  filter(EVT_LF %in% c("Developed", "Agriculture")) |>
                       #, "Barren", "Sparse")) |>
  group_by(EVT_LF, EVT_NAME) |>
  summarise(dvlp_evt_total_count = sum(Count), .groups = "drop") |>
  mutate(dvlp_evt_acres = round(replace_na(dvlp_evt_total_count, 0) * 0.2223945),
         prcnt_of_total = paste0(round(dvlp_evt_acres/sum(dvlp_evt_acres)*100, 1), "%")) |>
  arrange(desc(dvlp_evt_total_count))
view(dvlp_evt_count)


total_Dev <- sum(dvlp_evt_count$dvlp_evt_total_count[dvlp_evt_count$EVT_LF == "Developed"], na.rm = TRUE)
(total_Dev)

## Results ## Developed roads is 43% of all dev/ag land
# Easterns Warm and Cool Temp Row Crops are 33% of all dev/ag land
#Developed land comprises 65% of all


#breakdown of all evt_lf by total acreage
evt_lf <- data_raw |>
  select(EVT_NAME, , EVT_LF, Count) |>
  group_by(EVT_LF) |>
  summarise(evt_total_count = sum(Count), .groups = "drop") |>
  mutate(evt_acres = round(replace_na(evt_total_count, 0) * 0.2223945),
         prcnt_of_total = paste0(round(evt_acres/sum(evt_acres)*100, 1), "%")) |>
  arrange(desc(evt_total_count))
view(evt_lf)

#check to confirm this matches evt_lf table
(paste0(round(total_Dev/total_count*100, 1), "%"))

## Results ## Developed land is 6.4% of all land, Agriculture land is 3.4%

         
#what was that before?
southern_plantation <- data_raw |>
  select(BPS_NAME, EVT_NAME, Count) |>
  filter(EVT_NAME %in% c("Southeastern North American Temperate Forest Plantation")) |>
  group_by(BPS_NAME) |>
  summarise(plnt_total_count = sum(Count), .groups = "drop") |>
  mutate(plnt_acres = round(replace_na(plnt_total_count, 0) * 0.2223945),
         prcnt_of_total = paste0(round(plnt_acres/sum(plnt_acres)*100, 1), "%")) |>
  arrange(desc(plnt_total_count))
view(southern_plantation)

#check to confirm numbers match between evt_count and southern_plantation
total <- sum(southern_plantation[["plnt_total_count"]], na.rm = TRUE)
(total)

## Results ## Top 3 for what the Southeastern Plantations were before--
#1. East Gulf Coastal Plain Interior Upland Longleaf Pine Woodland 23%
#2. Southern Coastal Plain Dry Upland Hardwood Forest 21%
#3. Southern Coastal Plain Mesic Slope Forest 9%

#what was that before for developed and ag land?

#First, developed land summary
dev_bps <- data_raw |>
  select(BPS_NAME, EVT_NAME, EVT_LF, Count) |>
  filter(EVT_LF %in% c("Developed")) |>
  group_by(BPS_NAME) |>
  summarise(plnt_total_count = sum(Count), .groups = "drop") |>
  mutate(plnt_acres = round(replace_na(plnt_total_count, 0) * 0.2223945),
         prcnt_of_total = paste0(round(plnt_acres/sum(plnt_acres)*100, 1), "%")) |>
  arrange(desc(plnt_total_count))
view(dev_bps)

## Results ## Top 50% for what was Developed--
#1. East Gulf Coastal Plain Interior Upland Longleaf Pine Woodland 15%
#2. Allegheny-Cumberland Dry Oak Forest and Woodland 15%
#3. Central Interior and Appalachian Riparian Systems 10%
#4. Southern Coastal Plain Dry Upland Hardwood Forest 8%
#5. Southern Ridge and Valley/Cumberland Dry Calcareous Forest 7%
#6. Southern Interior Low Plateau Dry-Mesic Oak Forest 7%

#second, agriculture land summary
ag_bps <- data_raw |>
  select(BPS_NAME, EVT_NAME, EVT_LF, Count) |>
  filter(EVT_LF %in% c("Agriculture")) |>
  group_by(BPS_NAME) |>
  summarise(plnt_total_count = sum(Count), .groups = "drop") |>
  mutate(plnt_acres = round(replace_na(plnt_total_count, 0) * 0.2223945),
         prcnt_of_total = paste0(round(plnt_acres/sum(plnt_acres)*100, 1), "%")) |>
  arrange(desc(plnt_total_count))
view(ag_bps)

## Results ## Top 50% for what was turned into Ag--
#1. East Gulf Coastal Plain Interior Upland Longleaf Pine Woodland 21%
#2. Southern Interior Low Plateau Dry-Mesic Oak Forest 18%
#3. Central Interior and Appalachian Riparian Systems 17%

## Results show that in both, the East Gulf Coastal Plain Longleaf Woodland and the
# Central Interior and Appalachian Riparian System were among the top 50% in dev and ag.

#show this together graphically
#first, join the ag and dev tables

disturbed_summ <- merge(dev_bps, ag_bps, by = "BPS_NAME", suffixes = c(".dev", ".ag")) |>
  arrange(desc(plnt_total_count.dev)) |>
  top_n(10)
  # remove line below to get everything
  #select(BPS_NAME, plnt_acres.dev, plnt_acres.ag)
view(disturbed_summ)

#prepare the chart
p1 <- ggplot(disturbed_summ, aes(x = BPS_NAME, group = 1)) +
  geom_bar(aes(y = plnt_acres.dev), stat="identity", color = "purple", alpha = 0.4) +
  geom_line(aes(y = plnt_acres.ag), stat= "identity", color = "orange", size = 1) +
               
  scale_y_continuous(
    name = "Development Acreage",
    sec.axis = sec_axis(~., name = "Agriculture Acreage")) +
               
  theme_minimal() +
  labs(
    title = "Disturbed Acreage for top 10 BPS sites") +
  theme(legend.position = 'top',
        plot.title.position = "plot",
        plot.caption.position =  "plot",
        legend.title = element_text(size = 11),
        axis.title.y.right = element_text( angle = 90, color = "orange", size = 15),
        axis.title.y = element_text(color = "purple", size = 15),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) 
             
print(p1)


#Next steps:
#what about fire?
#group by colors in some way - find what are the prominent colors by BPS_NAME, EVT_LF, EVT_PHYS
#create a visual chart of what this could look like.

#Identifying risk areas, maybe places near already developed regions, or places that are medium or low development
#let's put this on a chart

############################################################
#other tables
#different count of various labels
count_of_groupings <- data_raw |>
  select(BPS_MODEL, BPS_NAME, GROUPVEG, EVT_NAME, EVT_PHYS, EVT_LF) |>
  summarise(across(everything(), n_distinct))
view(count_of_groupings)

#get unique values of EVT_LF
unique(data_raw$EVT_LF)

#Dictionary for Reference

####LANDFIRE Existing Vegetation Type Attribute Data Dictionar (pg 80)
#VALUE The LF assigned code identifying vegetation and land cover types.
#-9999 Fill - NoData
#4401 - 9994 The code identifies the vegetation and land cover types.
#EVT_NAME Class name in the LANDFIRE EVT legend.
#LFRDB Code stored in the LFRDB.
#4401 - 9994 The code identifies the EVT value stored in the LFRDB. Some LFRDB codes have been split 
#into more than one value, this field provides the codes lineage.
#EVT_FUEL Fuels EVT code. The code identifies the vegetation and land cover types used for fuels mapping.
#EVT_Fuel_N Fuels EVT class name.
#EVT_LF EVT Lifeform.
#EVT_GP EVT Group code.
#EVT_PHYS EVT Physiognomy.
#EVT_GP_N EVT Group name.
#SAF_SRM Crosswalk to Society of American Foresters and Society for Range Management cover 
#type.
#EVT_ORDER EVT Physiognomic Order from Federal Geographic Data Committee classification system.
#EVT_CLASS EVT Physiognomic Class from Federal Geographic Data Committee classification system.
#EVT_SBCLS EVT Physiognomic Subclass from Federal Geographic Data Committee classification system.

####LANDFIRE Biophysical Settings Attribute Data Dictionar (pg 80)
#VALUE LANDFIRE's (LF) Biophysical Settings (BPS) product represents the vegetation that may 
#have been dominant on the landscape prior to Euro-American settlement. BPS is based on 
#both the current biophysical environment and an approximation of the historical 
#disturbance regime. Map units are based on NatureServe's Ecological Systems 
#classification and represent the natural plant communities that may have been present 
#during the reference period.
#-9999 Fill-NoData
#-1111 Fill-Not Mapped
#11 Open Water
#12 Perennial Ice/Snow
#31 Barren-Rock/Sand/Clay
#4406 to 17220 The BPS value is a unique identifier for a unique combination of the BPS_Code and Zone.
#BPS_CODE 11 to 17220 Map units are based on NatureServe's Ecological Systems classification and represent the 
#natural plant communities that may have been present during the reference period.
#BPS_MODEL The BPS_CODE followed by the MAP ZONE.
#BPS_NAME BPS name.
#GROUPVEG Coarse categorization of BpS grouping.
#FRI_REPLAC Fire Return Interval (FRI) replacement fire.
#FRI_MIXED Fire Return Interval mixed fire.
#FRI_SURFAC Fire Return Interval surface fire.
#FRI_ALLFIR Fire Return Interval all fire. Quantifies the average period between fires under the 
#presumed historical fire regime. Previously Mean Fire Return Interval (MFRI).
#PRC_REPLAC Percent replacement fire. Previously Percent of Replacement-severity Fire (PRS). 
#Quantifies the amount of replacement-severity fires relative to low- and mixed-severity 
#fires under the presumed historical fire regime. Replacement severity is defined as greater 
#than 75 percent average top-kill within a typical fire perimeter for a given vegetation type.
#PRC_MIXED Percent mixed fire. Previously the Percent of Mixed-severity Fire (PMS). Quantifies the 
#amount of mixed severity fires relative to low- and replacement-severity fires under the 
#presumed historical fire regime. Mixed severity is defined as between 25 and 75 percent 
#average top-kill within a typical fire perimeter for a given vegetation type.
#PRC_SURFAC Percent of surface fire. Previously the Percent of Low-severity Fire (PLS). Quantifies the 
#amount of low severity fires relative to mixed- and replacement-severity fires under the 
#presumed historical fire regime. Low severity is defined as less than 25 percent average 
#top-kill within a typical fire perimeter for a given vegetation type.
#FRG_NEW Fire Regime Group.
#I-A Percent replacement fire less than 66.7%, fire return interval 0-5 years
#I-B Percent replacement fire less than 66.7%, fire return interval 6-15 years
#I-C Percent replacement fire less than 66.7%, fire return interval 16-35 years
#II-A Percent replacement fire greater than 66.7%, fire return interval 0-5 years
#II-B Percent replacement fire greater than 66.7%, fire return interval 6-15 years
#II-C Percent replacement fire greater than 66.7%, fire return interval 16-35 years
#III-A Percent replacement fire less than 80%, fire return interval 36-100 years
#III-B Percent replacement fire less than 66.7%, fire return interval 101-200 years
#IV-A Percent replacement fire greater than 80%, fire return interval 36-100 years
#IV-B Percent replacement fire greater than 66.7%, fire return interval 101-200 years
#V-A Any severity, fire return interval 201-500 years
#V-B Any severity, fire return interval 501 or more year


