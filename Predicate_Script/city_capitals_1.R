#### Test 1: check true, false capital_state of USA. 
####  Data include 50 true fact USAcapital - USAstate. From each true fact, we random create 5 false fact by combining its USAcapital 
#### with 5 different USAstate


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
FALSE_PER_TRUE = 0
DISCARD_REL = 191
ASSOCIATE_REL = c(404)
max_depth = 3

# ---- Load edge type file ----
mapfile <- read.csv(EDGE_TYPE_FILE, sep="\t", header=F)
mapfile$V1 <- as.numeric(mapfile$V1)
mapfile$V2 <- as.character(mapfile$V2)


# ---- Load input data ----
INPUT_FILE = "./data_id/city_capital.csv" # Example : "../facts/lobbyist/firm_payee.csv" col 1 and 2 are ids and 3 is label
dat_city_capital <- read.csv(INPUT_FILE)



# ---- Construct false labeled data -----
set.seed(233)

# ---- Init workers ----
cl <- makeCluster(CLUSTER_SIZE) 
clusterExport(cl = cl, varlist=c("adamic_adar", "semantic_proximity", "ppagerank", "heter_path",  "max_depth",
                                 "preferential_attachment", "katz", "pcrw", "heter_full_path", "meta_path",
                                 "multidimensional_adamic_adar", "heterogeneous_adamic_adar",
                                 "connectedby", "rel_path", "truelabeled", "falselabeled", "str_split",
                                 "as.numeric", "request","DISCARD_REL"), envir = environment())

# Find discriminative paths
tmp.paths <- rbind.fill(parApply(cl, dat_city_capital, 1, function(x) {
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

write.csv(tmp.paths, "./Predicate_paths/city_capitals_1.csv")

res_capital_state <- list()
res_capital_state[["raw"]] <- tmp.paths
res_capital_state[["model"]] <- Logistic(label~.,res_capital_state[["raw"]])
res_capital_state[["eval"]] <- evaluate_Weka_classifier(res_capital_state[["model"]], numFolds = 10, complexity = T, class = T, seed = 233)
res_capital_state[["eval"]]


