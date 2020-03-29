
###################################################### Preparation ######################################################
# Remove everything
rm(list = ls())

# Set working directory
setwd("C:/Users/Micha/OneDrive/Desktop/Machine Learning using Matlab - Project/Data/Data Dang")





########################################## Import and Cleaning of ISIN Numbers ##########################################
# Import ISIN numbers from csv files
# install.packages("rio")
library(rio)
convert2010 <- import("convert2010.csv")
convert2010 <- convert2010$isin
convert2011 <- import("convert2011.csv")
convert2011 <- convert2011$isin
convert2012 <- import("convert2012.csv")
convert2012 <- convert2012$isin
convert2013 <- import("convert2013.csv")
convert2013 <- convert2013$isin
convert2014 <- import("convert2014.csv")
convert2014 <- convert2014$isin
convert2015 <- import("convert2015.csv")
convert2015 <- convert2015$isin
convert2016 <- import("convert2016.csv")
convert2016 <- convert2016$isin
convert2017 <- import("convert2017.csv")
convert2017 <- convert2017$isin
convert2018 <- import("convert2018.csv")
convert2018 <- convert2018$isin
convert2019 <- import("convert2019.csv")
convert2019 <- convert2019$isin


# Remove observation with missing ISIN
convert2010 <- convert2010[!is.na(convert2010)]
convert2011 <- convert2011[!is.na(convert2011)]
convert2012 <- convert2012[!is.na(convert2012)]
convert2013 <- convert2013[!is.na(convert2013)]
convert2014 <- convert2014[!is.na(convert2014)]
convert2015 <- convert2015[!is.na(convert2015)]
convert2016 <- convert2016[!is.na(convert2016)]
convert2017 <- convert2017[!is.na(convert2017)]
convert2018 <- convert2018[!is.na(convert2018)]
convert2019 <- convert2019[!is.na(convert2019)]


# Get unique ISIN numbers
ISIN <- unique(c(convert2010, convert2011, convert2012, convert2013, convert2014, convert2015, convert2016, convert2017, convert2018, convert2019))


# Remov not needed objects
rm(convert2010, convert2011, convert2012, convert2013, convert2014, convert2015, convert2016, convert2017, convert2018, convert2019)





######################################### Export ISIN Numbers to an Excel File #########################################
# Export ISIN Numbers to an Excel File in order to be able to download stock data for each compny
# Export ISIN numbers to Excel
library(rio)
export(ISIN, "C:/Users/Micha/OneDrive/Desktop/Machine Learning using Matlab - Project/Data/Data Dang/ISIN numbers/ISIN.xlsx")


# Export ISIN numbers in subfiles of 500 numbers to Excel
library(rio)
export(ISIN[1:500], "C:/Users/Micha/OneDrive/Desktop/Machine Learning using Matlab - Project/Data/Data Dang/ISIN numbers/ISIN1.xlsx")
export(ISIN[501:1000], "C:/Users/Micha/OneDrive/Desktop/Machine Learning using Matlab - Project/Data/Data Dang/ISIN numbers/ISIN2.xlsx")
export(ISIN[1001:1500], "C:/Users/Micha/OneDrive/Desktop/Machine Learning using Matlab - Project/Data/Data Dang/ISIN numbers/ISIN3.xlsx")
export(ISIN[1501:length(ISIN)], "C:/Users/Micha/OneDrive/Desktop/Machine Learning using Matlab - Project/Data/Data Dang/ISIN numbers/ISIN4.xlsx")



# Remove not required vectors
rm(ISIN)


