# set workdirectory
setwd("~/Documents/CDC")
# raw data
clients <- read.csv("7ParkData-raw.csv", sep = ",", header = TRUE)
# unique clients - 35
clientlist <- unique(clients$Client.Name) 

library(dplyr)
library(lubridate)
library(ggplot2)
# extract "month" from the date
#month(as.POSIXlt(clients$Date.of.Contact, format="%m/%d/%Y"))
new <- clients %>% mutate(Month = month(as.POSIXlt(clients$Date.of.Contact, format="%Y-%m-%d")),
                          Year = year(as.POSIXlt(clients$Date.of.Contact, format="%Y-%m-%d"))) %>% 
                   group_by(Year,Month) %>%  summarise(TotalUnique = length(unique(Client.Name)))

newdf <- as.data.frame(new)
newdf$Year <- as.character(newdf$Year)
gplot <-ggplot(data = newdf, aes(x=Month,y=TotalUnique),xlim=c(1,12))+geom_line(aes(colour=Year))+ 
  ggtitle("Number of Unique Clients Contacted by month")+
  scale_x_continuous(breaks = seq(1, 12, by = 1))

ggsave("7ParkData_R.png", gplot, height=4.5, width=4.5)  

