#### Test 1: check true, false capital_state of USA and capital_country. 
####  Data include 40 true fact USAcapital - USAstate along with 10 true fact capital - country. From each true fact, we random create 5 false fact by combining its USAcapital 
#### with 5 different USA states or country.


# ---- Cleanup everything before start ----
options(warn=-1)
rm(list = ls())
gc()

# ---- GBSERVER API ----
source("./Rscript/experimentAPI.R")

list.of.packages <- c("bear", "FSelector", "ggplot2")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos="http://cran.rstudio.com/")
library(FSelector)
#library(ggplot2)
# ---- INPUT and CONFIGURATIONS ----

EDGE_TYPE_FILE = "./data/infobox.edgetypes" # Example : "../data/lobbyist.edgetypes"
CLUSTER_SIZE = 5 # Number of workers in gbserver
FALSE_PER_TRUE = 5
DISCARD_REL = 191
ASSOCIATE_REL = c(404)
max_depth = 3

# ---- Load edge type file ----
mapfile <- read.csv(EDGE_TYPE_FILE, sep="\t", header=F)
mapfile$V1 <- as.numeric(mapfile$V1)
mapfile$V2 <- as.character(mapfile$V2)


# ---- Load input data ----
INPUT_FILE = "./data_id/state_capital.csv" # Example : "../facts/lobbyist/firm_payee.csv" col 1 and 2 are ids and 3 is label
dat_state_capital.true <- read.csv(INPUT_FILE)

if (ncol(dat_state_capital.true) < 3)
  dat_state_capital.true$label <- T

INPUT_FILE = "./data_id/country_captial.csv" 
# ---- Load input data ----
dat_country_capital.true <- read.csv(INPUT_FILE)

if (ncol(dat_country_capital.true) < 3)
  dat_country_capital.true$label <- T

# ---- Construct false labeled data -----
set.seed(233)

colnames(dat_state_capital.true) <- c("src","dst","label")
colnames(dat_country_capital.true) <- c("src","dst","label")
dat_country_state_capital.true <- rbind(dat_country_capital.true[163:172,], dat_state_capital.true[1:40,])
# TODO: reformat this so it is universal and file independent
dat_country_state_capital.false <- rbind.fill(apply(dat_country_state_capital.true, 1, function(x){
  candidates <- unique(dat_country_state_capital.true[which(dat_country_state_capital.true[,1] != x[1]), 2])
  candidates <- unlist(lapply(candidates, function(y){
    if(length(which(dat_country_state_capital.true[,1] == x[1] & dat_country_state_capital.true[,2] == y) != 0)) {
      return(NULL)
    }
    return(y)
  }))
  return(data.frame(src=x[1], 
                    dst=sample(candidates, FALSE_PER_TRUE),
                    label=F))
}))


dat_country_state_capital <- rbind(dat_country_state_capital.true, dat_country_state_capital.false)


# ---- Init workers ----
cl <- makeCluster(CLUSTER_SIZE) 
clusterExport(cl = cl, varlist=c("adamic_adar", "semantic_proximity", "ppagerank", "heter_path",  "max_depth",
                                 "preferential_attachment", "katz", "pcrw", "heter_full_path", "meta_path",
                                 "multidimensional_adamic_adar", "heterogeneous_adamic_adar",
                                 "connectedby", "rel_path", "truelabeled", "falselabeled", "str_split",
                                 "as.numeric", "request","DISCARD_REL"), envir = environment())

# Find discriminative paths
tmp.paths <- rbind.fill(parApply(cl, dat_country_state_capital, 1, function(x) {
  tmp_paths <- rel_path(as.numeric(x[1]), as.numeric(x[2]), max_depth = 3, F, DISCARD_REL)
  if(length(tmp_paths) == 0) {
    return(data.frame(label = as.logical(x[3])))
  }
  rtn <- as.data.frame(t(tmp_paths$Freq))
  colnames(rtn) <- tmp_paths$paths
  rtn <- cbind(label = as.logical(x[3]), rtn)
  return(rtn)
}))
tmp.paths[is.na(tmp.paths)] <- 0

write.csv(tmp.paths, "./Predicate_paths/capitals_2.csv")
