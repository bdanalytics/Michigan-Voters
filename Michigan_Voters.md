# Michigan Voters: voting classification
bdanalytics  

**  **    
**Date: (Fri) Apr 24, 2015**    

# Introduction:  

Data: 
Source: 
    Training:   https://courses.edx.org/c4x/MITx/15.071x_2/asset/gerber.csv  
    New:        <newdt_url>  
Time period: 



# Synopsis:

Based on analysis utilizing <> techniques, <conclusion heading>:  

### ![](<filename>.png)

## Potential next steps include:
- Organization:
    - Categorize by chunk
    - Priority criteria:
        0. Ease of change
        1. Impacts report
        2. Cleans innards
        3. Bug report
        
- manage.missing.data chunk:
    - cleaner way to manage re-splitting of training vs. new entity

- fit.models chunk:
    - Prediction accuracy scatter graph:
    -   Add tiles (raw vs. PCA)
    -   Use shiny for drop-down of "important" features
    -   Use plot.ly for interactive plots ?
    
    - Change .fit suffix of model metrics to .mdl if it's data independent (e.g. AIC, Adj.R.Squared - is it truly data independent ?, etc.)
    - move model_type parameter to myfit_mdl before indep_vars_vctr (keep all model_* together)
    - create a custom model for rpart that has minbucket as a tuning parameter
    - varImp for randomForest crashes in caret version:6.0.41 -> submit bug report

- Probability handling for multinomials vs. desired binomial outcome
-   ROCR currently supports only evaluation of binary classification tasks (version 1.0.7)
-   extensions toward multiclass classification are scheduled for the next release

- Skip trControl.method="cv" for dummy classifier ?
- Add custom model to caret for a dummy (baseline) classifier (binomial & multinomial) that generates proba/outcomes which mimics the freq distribution of glb_rsp_var values; Right now glb_dmy_glm_mdl always generates most frequent outcome in training data
- glm_dmy_mdl should use the same method as glm_sel_mdl until custom dummy classifer is implemented

- Compare glb_sel_mdl vs. glb_fin_mdl:
    - varImp
    - Prediction differences (shd be minimal ?)

- Move glb_analytics_diag_plots to mydsutils.R: (+) Easier to debug (-) Too many glb vars used
- Add print(ggplot.petrinet(glb_analytics_pn) + coord_flip()) at the end of every major chunk
- Parameterize glb_analytics_pn
- Move glb_impute_missing_data to mydsutils.R: (-) Too many glb vars used; glb_<>_df reassigned
- Replicate myfit_mdl_classification features in myfit_mdl_regression
- Do non-glm methods handle interaction terms ?
- f-score computation for classifiers should be summation across outcomes (not just the desired one ?)
- Add accuracy computation to glb_dmy_mdl in predict.data.new chunk
- Why does splitting fit.data.training.all chunk into separate chunks add an overhead of ~30 secs ? It's not rbind b/c other chunks have lower elapsed time. Is it the number of plots ?
- Incorporate code chunks in print_sessionInfo
- Test against 
    - projects in github.com/bdanalytics
    - lectures in jhu-datascience track

# Analysis: 

```r
rm(list=ls())
set.seed(12345)
options(stringsAsFactors=FALSE)
source("~/Dropbox/datascience/R/mydsutils.R")
source("~/Dropbox/datascience/R/myplot.R")
source("~/Dropbox/datascience/R/mypetrinet.R")
# Gather all package requirements here
#suppressPackageStartupMessages(require())
#packageVersion("snow")

#require(sos); findFn("pinv", maxPages=2, sortby="MaxScore")

# Analysis control global variables
glb_trnng_url <- "https://courses.edx.org/c4x/MITx/15.071x_2/asset/gerber.csv"
glb_newdt_url <- "<newdt_url>"
glb_is_separate_newent_dataset <- FALSE    # or TRUE
glb_split_entity_newent_datasets <- TRUE   # or FALSE
glb_split_newdata_method <- "sample"          # "condition" or "sample"
glb_split_newdata_condition <- "<col_name> <condition_operator> <value>"    # or NULL
glb_split_newdata_size_ratio <- 0.4               # > 0 & < 1
glb_split_sample.seed <- 88               # or any integer
glb_max_obs <- 5000

glb_is_regression <- FALSE; glb_is_classification <- TRUE; glb_is_binomial <- TRUE

glb_rsp_var_raw <- "voting"

# for classification, the response variable has to be a factor
glb_rsp_var <- "voting.fctr"

# if the response factor is based on numbers e.g (0/1 vs. "A"/"B"), 
#   caret predict(..., type="prob") crashes
glb_map_rsp_raw_to_var <- function(raw) {
    relevel(factor(ifelse(raw == 1, "Y", "N")), as.factor(c("Y", "N")), ref="N")
    #as.factor(paste0("B", raw))
}
glb_map_rsp_raw_to_var(c(1, 0))
```

```
## [1] Y N
## Levels: N Y
```

```r
glb_map_rsp_var_to_raw <- function(var) {
    as.numeric(var) - 1
    #as.numeric(var)
}
glb_map_rsp_var_to_raw(glb_map_rsp_raw_to_var(c(1, 0, 0, 1)))
```

```
## [1] 1 0 0 1
```

```r
if ((glb_rsp_var != glb_rsp_var_raw) & is.null(glb_map_rsp_raw_to_var))
    stop("glb_map_rsp_raw_to_var function expected")

glb_rsp_var_out <- paste0(glb_rsp_var, ".predict.") # model_id is appended later
glb_id_vars <- NULL # or c("<id_var>")

# List transformed vars  
glb_exclude_vars_as_features <- c("voting.fctr")    
# List feats that shd be excluded due to known causation by prediction variable
if (glb_rsp_var_raw != glb_rsp_var)
    glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
                                            glb_rsp_var_raw)
glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
                                      c(NULL)) # or c("<col_name>")
# List output vars (useful during testing in console)          
# glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, 
#                         grep(glb_rsp_var_out, names(glb_trnent_df), value=TRUE)) 

glb_impute_na_data <- FALSE            # or TRUE
glb_mice_complete.seed <- 144               # or any integer

# Regression
#   rpart:  .rnorm messes with the models badly
#           caret creates dummy vars for factor feats which messes up the tuning
#               - better to feed as.numeric(<feat>.fctr) to caret 

#glb_models_method_vctr <- c("lm", "glm", "rpart", "rf")

# Classification
#   rpart:  .rnorm messes with the models badly
#           caret creates dummy vars for factor feats which messes up the tuning
#               - better to feed as.numeric(<feat>.fctr) to caret 

glb_models_method_vctr <- c("glm", "rpart", "rf")   # Binomials
#glb_models_method_vctr <- c("rpart", "rf")          # Multinomials

glb_models_lst <- list(); glb_models_df <- data.frame()
# Baseline prediction model feature(s)
glb_Baseline_mdl_var <- NULL # or c("<col_name>")

glb_model_metric_terms <- NULL # or matrix(c(
#                               0,1,2,3,4,
#                               2,0,1,2,3,
#                               4,2,0,1,2,
#                               6,4,2,0,1,
#                               8,6,4,2,0
#                           ), byrow=TRUE, nrow=5)
glb_model_metric <- NULL # or "<metric_name>"
glb_model_metric_maximize <- NULL # or FALSE (TRUE is not the default for both classification & regression) 
glb_model_metric_smmry <- NULL # or function(data, lev=NULL, model=NULL) {
#     confusion_mtrx <- t(as.matrix(confusionMatrix(data$pred, data$obs)))
#     #print(confusion_mtrx)
#     #print(confusion_mtrx * glb_model_metric_terms)
#     metric <- sum(confusion_mtrx * glb_model_metric_terms) / nrow(data)
#     names(metric) <- glb_model_metric
#     return(metric)
# }

glb_tune_models_df <- 
   rbind(
    #data.frame(parameter="cp", min=0.00005, max=0.00005, by=0.000005),
                            #seq(from=0.01,  to=0.01, by=0.01)
    #data.frame(parameter="mtry", min=2, max=4, by=1),
    data.frame(parameter="dummy", min=2, max=4, by=1)
        ) 
# or NULL
glb_n_cv_folds <- 3 # or NULL

glb_clf_proba_threshold <- NULL # 0.5

# Model selection criteria
if (glb_is_regression)
    glb_model_evl_criteria <- c("min.RMSE.OOB", "max.R.sq.OOB", "max.Adj.R.sq.fit")
if (glb_is_classification) {
    if (glb_is_binomial)
        glb_model_evl_criteria <- c("max.Accuracy.OOB", "max.Kappa.OOB", "min.aic.fit") else
        glb_model_evl_criteria <- c("max.Accuracy.OOB", "max.Kappa.OOB")
}

glb_sel_mdl_id <- NULL
glb_fin_mdl_id <- glb_sel_mdl_id # or "Final"

# Depict process
glb_analytics_pn <- petrinet(name="glb_analytics_pn",
                        trans_df=data.frame(id=1:6,
    name=c("data.training.all","data.new",
           "model.selected","model.final",
           "data.training.all.prediction","data.new.prediction"),
    x=c(   -5,-5,-15,-25,-25,-35),
    y=c(   -5, 5,  0,  0, -5,  5)
                        ),
                        places_df=data.frame(id=1:4,
    name=c("bgn","fit.data.training.all","predict.data.new","end"),
    x=c(   -0,   -20,                    -30,               -40),
    y=c(    0,     0,                      0,                 0),
    M0=c(   3,     0,                      0,                 0)
                        ),
                        arcs_df=data.frame(
    begin=c("bgn","bgn","bgn",        
            "data.training.all","model.selected","fit.data.training.all",
            "fit.data.training.all","model.final",    
            "data.new","predict.data.new",
            "data.training.all.prediction","data.new.prediction"),
    end  =c("data.training.all","data.new","model.selected",
            "fit.data.training.all","fit.data.training.all","model.final",
            "data.training.all.prediction","predict.data.new",
            "predict.data.new","data.new.prediction",
            "end","end")
                        ))
#print(ggplot.petrinet(glb_analytics_pn))
print(ggplot.petrinet(glb_analytics_pn) + coord_flip())
```

```
## Loading required package: grid
```

![](Michigan_Voters_files/figure-html/set_global_options-1.png) 

```r
glb_analytics_avl_objs <- NULL

glb_script_tm <- proc.time()
glb_script_df <- data.frame(chunk_label="import_data", 
                            chunk_step_major=1, chunk_step_minor=0,
                            elapsed=(proc.time() - glb_script_tm)["elapsed"])
print(tail(glb_script_df, 2))
```

```
##         chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed import_data                1                0   0.002
```

## Step `1`: import data

```r
glb_entity_df <- myimport_data(url=glb_trnng_url, 
    comment=ifelse(!glb_is_separate_newent_dataset, "glb_entity_df", "glb_trnent_df"), 
                                force_header=TRUE)
```

```
## [1] "Reading file ./data/gerber.csv..."
## [1] "dimensions of data in ./data/gerber.csv: 344,084 rows x 8 cols"
##   sex  yob voting hawthorne civicduty neighbors self control
## 1   0 1941      0         0         1         0    0       0
## 2   1 1947      0         0         1         0    0       0
## 3   1 1982      1         1         0         0    0       0
## 4   1 1950      1         1         0         0    0       0
## 5   0 1951      1         1         0         0    0       0
## 6   1 1959      1         0         0         0    0       1
##        sex  yob voting hawthorne civicduty neighbors self control
## 11883    1 1954      0         0         0         1    0       0
## 52429    1 1945      0         0         0         0    0       1
## 111861   0 1945      1         0         0         0    0       1
## 175216   1 1954      0         1         0         0    0       0
## 250391   1 1975      1         0         0         1    0       0
## 340550   1 1936      1         0         0         0    0       1
##        sex  yob voting hawthorne civicduty neighbors self control
## 344079   0 1943      1         0         0         0    0       1
## 344080   1 1944      1         0         0         0    0       1
## 344081   1 1958      0         0         0         0    0       1
## 344082   0 1955      0         0         0         0    0       1
## 344083   1 1949      1         0         0         0    0       1
## 344084   0 1937      1         0         0         0    0       1
## 'data.frame':	344084 obs. of  8 variables:
##  $ sex      : int  0 1 1 1 0 1 0 0 1 0 ...
##  $ yob      : int  1941 1947 1982 1950 1951 1959 1956 1981 1968 1967 ...
##  $ voting   : int  0 0 1 1 1 1 1 0 0 0 ...
##  $ hawthorne: int  0 0 1 1 1 0 0 0 0 0 ...
##  $ civicduty: int  1 1 0 0 0 0 0 0 0 0 ...
##  $ neighbors: int  0 0 0 0 0 0 0 0 0 0 ...
##  $ self     : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ control  : int  0 0 0 0 0 1 1 1 1 1 ...
##  - attr(*, "comment")= chr "glb_entity_df"
## NULL
```

```r
if (!glb_is_separate_newent_dataset) {
    glb_trnent_df <- glb_entity_df; comment(glb_trnent_df) <- "glb_trnent_df"
} # else glb_entity_df is maintained as is for chunk:inspectORexplore.data
    
if (glb_is_separate_newent_dataset) {
    glb_newent_df <- myimport_data(
        url=glb_newdt_url, 
        comment="glb_newent_df", force_header=TRUE)
    
    # To make plots / stats / checks easier in chunk:inspectORexplore.data
    glb_entity_df <- rbind(glb_trnent_df, glb_newent_df); comment(glb_entity_df) <- "glb_entity_df"
} else {
    if (!glb_split_entity_newent_datasets) {
        stop("Not implemented yet") 
        glb_newent_df <- glb_trnent_df[sample(1:nrow(glb_trnent_df),
                                          max(2, nrow(glb_trnent_df) / 1000)),]                    
    } else      if (glb_split_newdata_method == "condition") {
            glb_newent_df <- do.call("subset", 
                list(glb_trnent_df, parse(text=glb_split_newdata_condition)))
            glb_trnent_df <- do.call("subset", 
                list(glb_trnent_df, parse(text=paste0("!(", 
                                                      glb_split_newdata_condition,
                                                      ")"))))
        } else if (glb_split_newdata_method == "sample") {
                require(caTools)
                
                set.seed(glb_split_sample.seed)
                split <- sample.split(glb_trnent_df[, glb_rsp_var_raw], 
                                      SplitRatio=(1-glb_split_newdata_size_ratio))
                glb_newent_df <- glb_trnent_df[!split, ] 
                glb_trnent_df <- glb_trnent_df[split ,]
        } else stop("glb_split_newdata_method should be %in% c('condition', 'sample')")   

    comment(glb_newent_df) <- "glb_newent_df"
    myprint_df(glb_newent_df)
    str(glb_newent_df)

    if (glb_split_entity_newent_datasets) {
        myprint_df(glb_trnent_df)
        str(glb_trnent_df)        
    }
}         
```

```
## Loading required package: caTools
```

```
##    sex  yob voting hawthorne civicduty neighbors self control
## 8    0 1981      0         0         0         0    0       1
## 10   0 1967      0         0         0         0    0       1
## 12   0 1945      0         1         0         0    0       0
## 17   1 1957      0         0         0         0    1       0
## 18   0 1955      0         0         0         0    1       0
## 20   0 1939      1         0         0         1    0       0
##        sex  yob voting hawthorne civicduty neighbors self control
## 7619     0 1931      0         0         0         0    0       1
## 65786    1 1953      0         1         0         0    0       0
## 121142   1 1944      0         0         0         1    0       0
## 140646   1 1939      1         0         0         0    0       1
## 171025   1 1962      0         0         0         0    0       1
## 221799   1 1959      0         0         0         0    1       0
##        sex  yob voting hawthorne civicduty neighbors self control
## 344067   0 1915      1         0         0         0    0       1
## 344071   0 1915      0         1         0         0    0       0
## 344076   1 1947      0         0         0         0    1       0
## 344077   1 1951      0         0         0         1    0       0
## 344078   0 1942      0         0         0         1    0       0
## 344079   0 1943      1         0         0         0    0       1
## 'data.frame':	137633 obs. of  8 variables:
##  $ sex      : int  0 0 0 1 0 0 0 0 0 1 ...
##  $ yob      : int  1981 1967 1945 1957 1955 1939 1940 1968 1979 1983 ...
##  $ voting   : int  0 0 0 0 0 1 1 1 0 0 ...
##  $ hawthorne: int  0 0 1 0 0 0 0 0 0 0 ...
##  $ civicduty: int  0 0 0 0 0 0 0 0 0 0 ...
##  $ neighbors: int  0 0 0 0 0 1 0 0 0 0 ...
##  $ self     : int  0 0 0 1 1 0 1 0 0 0 ...
##  $ control  : int  1 1 0 0 0 0 0 1 1 1 ...
##  - attr(*, "comment")= chr "glb_newent_df"
##   sex  yob voting hawthorne civicduty neighbors self control
## 1   0 1941      0         0         1         0    0       0
## 2   1 1947      0         0         1         0    0       0
## 3   1 1982      1         1         0         0    0       0
## 4   1 1950      1         1         0         0    0       0
## 5   0 1951      1         1         0         0    0       0
## 6   1 1959      1         0         0         0    0       1
##        sex  yob voting hawthorne civicduty neighbors self control
## 11840    1 1938      1         0         0         0    1       0
## 118195   1 1954      0         0         0         0    0       1
## 130256   0 1929      1         0         1         0    0       0
## 135819   0 1940      1         0         0         0    0       1
## 140538   1 1964      1         0         0         0    0       1
## 287093   1 1949      1         1         0         0    0       0
##        sex  yob voting hawthorne civicduty neighbors self control
## 344075   1 1934      1         1         0         0    0       0
## 344080   1 1944      1         0         0         0    0       1
## 344081   1 1958      0         0         0         0    0       1
## 344082   0 1955      0         0         0         0    0       1
## 344083   1 1949      1         0         0         0    0       1
## 344084   0 1937      1         0         0         0    0       1
## 'data.frame':	206451 obs. of  8 variables:
##  $ sex      : int  0 1 1 1 0 1 0 1 0 1 ...
##  $ yob      : int  1941 1947 1982 1950 1951 1959 1956 1968 1941 1949 ...
##  $ voting   : int  0 0 1 1 1 1 1 0 1 1 ...
##  $ hawthorne: int  0 0 1 1 1 0 0 0 0 1 ...
##  $ civicduty: int  1 1 0 0 0 0 0 0 0 0 ...
##  $ neighbors: int  0 0 0 0 0 0 0 0 0 0 ...
##  $ self     : int  0 0 0 0 0 0 0 0 0 0 ...
##  $ control  : int  0 0 0 0 0 1 1 1 1 0 ...
##  - attr(*, "comment")= chr "glb_trnent_df"
```

```r
if (!is.null(glb_max_obs)) {
    if (nrow(glb_trnent_df) > glb_max_obs) {
        warning("glb_trnent_df restricted to glb_max_obs: ", format(glb_max_obs, big.mark=","))
        org_entity_df <- glb_trnent_df
        glb_trnent_df <- org_entity_df[split <- 
            sample.split(org_entity_df[, glb_rsp_var_raw], SplitRatio=glb_max_obs), ]
        org_entity_df <- NULL
    }
    if (nrow(glb_newent_df) > glb_max_obs) {
        warning("glb_newent_df restricted to glb_max_obs: ", format(glb_max_obs, big.mark=","))        
        org_newent_df <- glb_newent_df
        glb_newent_df <- org_newent_df[split <- 
            sample.split(org_newent_df[, glb_rsp_var_raw], SplitRatio=glb_max_obs), ]
        org_newent_df <- NULL
    }    
}
```

```
## Warning: glb_trnent_df restricted to glb_max_obs: 5,000
```

```
## Warning: glb_newent_df restricted to glb_max_obs: 5,000
```

```r
glb_script_df <- rbind(glb_script_df,
                   data.frame(chunk_label="cleanse_data", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##           chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed   import_data                1                0   0.002
## elapsed1 cleanse_data                2                0   5.886
```

## Step `2`: cleanse data

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="inspectORexplore.data", 
                              chunk_step_major=max(glb_script_df$chunk_step_major), 
                              chunk_step_minor=1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed1          cleanse_data                2                0   5.886
## elapsed2 inspectORexplore.data                2                1   5.918
```

### Step `2`.`1`: inspect/explore data

```r
#print(str(glb_trnent_df))
#View(glb_trnent_df)

# List info gathered for various columns
# <col_name>:   <description>; <notes>
# "Civic Duty" (variable civicduty) group members were sent a letter that simply said "DO YOUR CIVIC DUTY - VOTE!"
# "Hawthorne Effect" (variable hawthorne) group members were sent a letter that had the "Civic Duty" message plus the additional message "YOU ARE BEING STUDIED" and they were informed that their voting behavior would be examined by means of public records.
# "Self" (variable self) group members received the "Civic Duty" message as well as the recent voting record of everyone in that household and a message stating that another message would be sent after the election with updated records.
# "Neighbors" (variable neighbors) group members were given the same message as that for the "Self" group, except the message not only had the household voting records but also that of neighbors - maximizing social pressure.
# "Control" (variable control) group members were not sent anything, and represented the typical voting situation.

# Create new features that help diagnostics
#   Create factors of string variables
str_vars <- sapply(1:length(names(glb_trnent_df)), 
    function(col) ifelse(class(glb_trnent_df[, names(glb_trnent_df)[col]]) == "character",
                         names(glb_trnent_df)[col], ""))
if (length(str_vars <- setdiff(str_vars[str_vars != ""], 
                               glb_exclude_vars_as_features)) > 0) {
    warning("Creating factors of string variables:", paste0(str_vars, collapse=", "))
    glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, str_vars)
    for (var in str_vars) {
        glb_entity_df[, paste0(var, ".fctr")] <- factor(glb_entity_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
        glb_trnent_df[, paste0(var, ".fctr")] <- factor(glb_trnent_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
        glb_newent_df[, paste0(var, ".fctr")] <- factor(glb_newent_df[, var], 
                        as.factor(unique(glb_entity_df[, var])))
    }
}

#   Convert factors to dummy variables
#   Build splines   require(splines); bsBasis <- bs(training$age, df=3)

add_new_diag_feats <- function(obs_df, ref_df=glb_entity_df) {
    require(plyr)
    
    obs_df <- mutate(obs_df,
#         <col_name>.NA=is.na(<col_name>),

#         <col_name>.fctr=factor(<col_name>, 
#                     as.factor(union(obs_df$<col_name>, obs_twin_df$<col_name>))), 
#         <col_name>.fctr=relevel(factor(<col_name>, 
#                     as.factor(union(obs_df$<col_name>, obs_twin_df$<col_name>))),
#                                   "<ref_val>"), 
#         <col2_name>.fctr=relevel(factor(ifelse(<col1_name> == <val>, "<oth_val>", "<ref_val>")), 
#                               as.factor(c("R", "<ref_val>")),
#                               ref="<ref_val>"),

          # This doesn't work - use sapply instead
#         <col_name>.fctr_num=grep(<col_name>, levels(<col_name>.fctr)), 
#         
#         Date.my=as.Date(strptime(Date, "%m/%d/%y %H:%M")),
#         Year=year(Date.my),
#         Month=months(Date.my),
#         Weekday=weekdays(Date.my)

#         <col_name>.log=log(<col.name>),        
#         <col_name>=<table>[as.character(<col2_name>)],
#         <col_name>=as.numeric(<col2_name>),

        .rnorm=rnorm(n=nrow(obs_df))
                        )

    # If levels of a factor are different across obs_df & glb_newent_df; predict.glm fails  
    # Transformations not handled by mutate
#     obs_df$<col_name>.fctr.num <- sapply(1:nrow(obs_df), 
#         function(row_ix) grep(obs_df[row_ix, "<col_name>"],
#                               levels(obs_df[row_ix, "<col_name>.fctr"])))
    
    print(summary(obs_df))
    print(sapply(names(obs_df), function(col) sum(is.na(obs_df[, col]))))
    return(obs_df)
}

glb_entity_df <- add_new_diag_feats(glb_entity_df)
```

```
## Loading required package: plyr
```

```
##       sex              yob           voting         hawthorne    
##  Min.   :0.0000   Min.   :1900   Min.   :0.0000   Min.   :0.000  
##  1st Qu.:0.0000   1st Qu.:1947   1st Qu.:0.0000   1st Qu.:0.000  
##  Median :0.0000   Median :1956   Median :0.0000   Median :0.000  
##  Mean   :0.4993   Mean   :1956   Mean   :0.3159   Mean   :0.111  
##  3rd Qu.:1.0000   3rd Qu.:1965   3rd Qu.:1.0000   3rd Qu.:0.000  
##  Max.   :1.0000   Max.   :1986   Max.   :1.0000   Max.   :1.000  
##    civicduty        neighbors          self           control      
##  Min.   :0.0000   Min.   :0.000   Min.   :0.0000   Min.   :0.0000  
##  1st Qu.:0.0000   1st Qu.:0.000   1st Qu.:0.0000   1st Qu.:0.0000  
##  Median :0.0000   Median :0.000   Median :0.0000   Median :1.0000  
##  Mean   :0.1111   Mean   :0.111   Mean   :0.1111   Mean   :0.5558  
##  3rd Qu.:0.0000   3rd Qu.:0.000   3rd Qu.:0.0000   3rd Qu.:1.0000  
##  Max.   :1.0000   Max.   :1.000   Max.   :1.0000   Max.   :1.0000  
##      .rnorm         
##  Min.   :-4.804169  
##  1st Qu.:-0.673747  
##  Median :-0.001393  
##  Mean   : 0.000376  
##  3rd Qu.: 0.675565  
##  Max.   : 4.345014  
##       sex       yob    voting hawthorne civicduty neighbors      self 
##         0         0         0         0         0         0         0 
##   control    .rnorm 
##         0         0
```

```r
glb_trnent_df <- add_new_diag_feats(glb_trnent_df)
```

```
##       sex              yob           voting        hawthorne     
##  Min.   :0.0000   Min.   :1908   Min.   :0.000   Min.   :0.0000  
##  1st Qu.:0.0000   1st Qu.:1947   1st Qu.:0.000   1st Qu.:0.0000  
##  Median :0.0000   Median :1956   Median :0.000   Median :0.0000  
##  Mean   :0.4898   Mean   :1956   Mean   :0.316   Mean   :0.1064  
##  3rd Qu.:1.0000   3rd Qu.:1965   3rd Qu.:1.000   3rd Qu.:0.0000  
##  Max.   :1.0000   Max.   :1986   Max.   :1.000   Max.   :1.0000  
##    civicduty       neighbors           self           control     
##  Min.   :0.000   Min.   :0.0000   Min.   :0.0000   Min.   :0.000  
##  1st Qu.:0.000   1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:0.000  
##  Median :0.000   Median :0.0000   Median :0.0000   Median :1.000  
##  Mean   :0.109   Mean   :0.1098   Mean   :0.1128   Mean   :0.562  
##  3rd Qu.:0.000   3rd Qu.:0.0000   3rd Qu.:0.0000   3rd Qu.:1.000  
##  Max.   :1.000   Max.   :1.0000   Max.   :1.0000   Max.   :1.000  
##      .rnorm        
##  Min.   :-3.55701  
##  1st Qu.:-0.67654  
##  Median :-0.00468  
##  Mean   :-0.00575  
##  3rd Qu.: 0.67165  
##  Max.   : 3.43410  
##       sex       yob    voting hawthorne civicduty neighbors      self 
##         0         0         0         0         0         0         0 
##   control    .rnorm 
##         0         0
```

```r
glb_newent_df <- add_new_diag_feats(glb_newent_df)
```

```
##       sex              yob           voting         hawthorne     
##  Min.   :0.0000   Min.   :1912   Min.   :0.0000   Min.   :0.0000  
##  1st Qu.:0.0000   1st Qu.:1947   1st Qu.:0.0000   1st Qu.:0.0000  
##  Median :0.0000   Median :1956   Median :0.0000   Median :0.0000  
##  Mean   :0.4964   Mean   :1956   Mean   :0.3158   Mean   :0.1096  
##  3rd Qu.:1.0000   3rd Qu.:1965   3rd Qu.:1.0000   3rd Qu.:0.0000  
##  Max.   :1.0000   Max.   :1986   Max.   :1.0000   Max.   :1.0000  
##    civicduty        neighbors           self           control      
##  Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
##  1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:0.0000  
##  Median :0.0000   Median :0.0000   Median :0.0000   Median :1.0000  
##  Mean   :0.1026   Mean   :0.1066   Mean   :0.1144   Mean   :0.5668  
##  3rd Qu.:0.0000   3rd Qu.:0.0000   3rd Qu.:0.0000   3rd Qu.:1.0000  
##  Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   Max.   :1.0000  
##      .rnorm         
##  Min.   :-3.209265  
##  1st Qu.:-0.673492  
##  Median :-0.001377  
##  Mean   : 0.009209  
##  3rd Qu.: 0.694014  
##  Max.   : 4.096131  
##       sex       yob    voting hawthorne civicduty neighbors      self 
##         0         0         0         0         0         0         0 
##   control    .rnorm 
##         0         0
```

```r
# Histogram of predictor in glb_trnent_df & glb_newent_df
plot_df <- rbind(cbind(glb_trnent_df[, glb_rsp_var_raw, FALSE], data.frame(.data="Training")),
                 cbind(glb_trnent_df[, glb_rsp_var_raw, FALSE], data.frame(.data="New")))
print(myplot_histogram(plot_df, glb_rsp_var_raw) + facet_wrap(~ .data))
```

```
## stat_bin: binwidth defaulted to range/30. Use 'binwidth = x' to adjust this.
## stat_bin: binwidth defaulted to range/30. Use 'binwidth = x' to adjust this.
```

![](Michigan_Voters_files/figure-html/inspectORexplore_data-1.png) 

```r
if (glb_is_classification) {
    xtab_df <- mycreate_xtab(plot_df, c(".data", glb_rsp_var_raw))
    rownames(xtab_df) <- xtab_df$.data
    xtab_df <- subset(xtab_df, select=-.data)
    print(xtab_df / rowSums(xtab_df))    
}    
```

```
## Loading required package: reshape2
```

```
##          voting.0 voting.1
## New         0.684    0.316
## Training    0.684    0.316
```

```r
# Check for duplicates in glb_id_vars
if (length(glb_id_vars) > 0) {
    id_vars_dups_df <- subset(id_vars_df <- 
            mycreate_tbl_df(glb_entity_df[, glb_id_vars, FALSE], glb_id_vars),
                                .freq > 1)
    if (nrow(id_vars_dups_df) > 0) {
        warning("Duplicates found in glb_id_vars data:", nrow(id_vars_dups_df))
        myprint_df(id_vars_dups_df)
    } else {
        # glb_id_vars are unique across obs in both glb_<>_df
        glb_exclude_vars_as_features <- union(glb_exclude_vars_as_features, glb_id_vars)
    }
}

#pairs(subset(glb_trnent_df, select=-c(col_symbol)))
# Check for glb_newent_df & glb_trnent_df features range mismatches

# Other diagnostics:
# print(subset(glb_trnent_df, <col1_name> == max(glb_trnent_df$<col1_name>, na.rm=TRUE) & 
#                         <col2_name> <= mean(glb_trnent_df$<col1_name>, na.rm=TRUE)))

# print(glb_trnent_df[which.max(glb_trnent_df$<col_name>),])

# print(<col_name>_freq_glb_trnent_df <- mycreate_tbl_df(glb_trnent_df, "<col_name>"))
# print(which.min(table(glb_trnent_df$<col_name>)))
# print(which.max(table(glb_trnent_df$<col_name>)))
# print(which.max(table(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>)[, 2]))
# print(table(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>))
# print(table(is.na(glb_trnent_df$<col1_name>), glb_trnent_df$<col2_name>))
# print(table(sign(glb_trnent_df$<col1_name>), glb_trnent_df$<col2_name>))
# print(mycreate_xtab(glb_trnent_df, <col1_name>))
# print(mycreate_xtab(glb_trnent_df, c(<col1_name>, <col2_name>)))
# print(<col1_name>_<col2_name>_xtab_glb_trnent_df <- 
#   mycreate_xtab(glb_trnent_df, c("<col1_name>", "<col2_name>")))
# <col1_name>_<col2_name>_xtab_glb_trnent_df[is.na(<col1_name>_<col2_name>_xtab_glb_trnent_df)] <- 0
# print(<col1_name>_<col2_name>_xtab_glb_trnent_df <- 
#   mutate(<col1_name>_<col2_name>_xtab_glb_trnent_df, 
#             <col3_name>=(<col1_name> * 1.0) / (<col1_name> + <col2_name>))) 

# print(<col2_name>_min_entity_arr <- 
#    sort(tapply(glb_trnent_df$<col1_name>, glb_trnent_df$<col2_name>, min, na.rm=TRUE)))
# print(<col1_name>_na_by_<col2_name>_arr <- 
#    sort(tapply(glb_trnent_df$<col1_name>.NA, glb_trnent_df$<col2_name>, mean, na.rm=TRUE)))
for (grp in c("hawthorne", "civicduty", "neighbors", "self")) {
    print(sprintf("group: %s", grp))
    print(voting_by_grp_arr <- 
        sort(tapply(glb_entity_df$voting, glb_entity_df[, grp], mean, na.rm=TRUE)))    
}
```

```
## [1] "group: hawthorne"
##         0         1 
## 0.3150909 0.3223746 
## [1] "group: civicduty"
##         1         0 
## 0.3145377 0.3160698 
## [1] "group: neighbors"
##         0         1 
## 0.3081505 0.3779482 
## [1] "group: self"
##         0         1 
## 0.3122446 0.3451515
```

```r
# Other plots:
# print(myplot_box(df=glb_trnent_df, ycol_names="<col1_name>"))
# print(myplot_box(df=glb_trnent_df, ycol_names="<col1_name>", xcol_name="<col2_name>"))
# print(myplot_line(subset(glb_trnent_df, Symbol %in% c("KO", "PG")), 
#                   "Date.my", "StockPrice", facet_row_colnames="Symbol") + 
#     geom_vline(xintercept=as.numeric(as.Date("2003-03-01"))) +
#     geom_vline(xintercept=as.numeric(as.Date("1983-01-01")))        
#         )
# print(myplot_scatter(glb_trnent_df, "<col1_name>", "<col2_name>", smooth=TRUE))
# print(myplot_scatter(glb_entity_df, "<col1_name>", "<col2_name>", colorcol_name="<Pred.fctr>") + 
#         geom_point(data=subset(glb_entity_df, <condition>), 
#                     mapping=aes(x=<x_var>, y=<y_var>), color="red", shape=4, size=5))
# print(myplot_scatter(glb_entity_df, "yob", "voting") + 
#         geom_point(data=subset(glb_entity_df, control == 1), 
#                     mapping=aes(x=yob, y=voting), color="blue"))
print(myplot_scatter(glb_trnent_df, ".rownames", "yob", colorcol_name="voting"))
```

```
## Warning in myplot_scatter(glb_trnent_df, ".rownames", "yob", colorcol_name
## = "voting"): converting voting to class:factor
```

![](Michigan_Voters_files/figure-html/inspectORexplore_data-2.png) 

```r
glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="manage_missing_data", 
        chunk_step_major=max(glb_script_df$chunk_step_major), 
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed2 inspectORexplore.data                2                1   5.918
## elapsed3   manage_missing_data                2                2  19.040
```

### Step `2`.`2`: manage missing data

```r
# print(sapply(names(glb_trnent_df), function(col) sum(is.na(glb_trnent_df[, col]))))
# print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))
# glb_trnent_df <- na.omit(glb_trnent_df)
# glb_newent_df <- na.omit(glb_newent_df)
# df[is.na(df)] <- 0

# Not refactored into mydsutils.R since glb_*_df might be reassigned
glb_impute_missing_data <- function(entity_df, newent_df) {
    if (!glb_is_separate_newent_dataset) {
        # Combine entity & newent
        union_df <- rbind(mutate(entity_df, .src = "entity"),
                          mutate(newent_df, .src = "newent"))
        union_imputed_df <- union_df[, setdiff(setdiff(names(entity_df), 
                                                       glb_rsp_var), 
                                               glb_exclude_vars_as_features)]
        print(summary(union_imputed_df))
    
        require(mice)
        set.seed(glb_mice_complete.seed)
        union_imputed_df <- complete(mice(union_imputed_df))
        print(summary(union_imputed_df))
    
        union_df[, names(union_imputed_df)] <- union_imputed_df[, names(union_imputed_df)]
        print(summary(union_df))
#         union_df$.rownames <- rownames(union_df)
#         union_df <- orderBy(~.rownames, union_df)
#         
#         imp_entity_df <- myimport_data(
#             url="<imputed_trnng_url>", 
#             comment="imp_entity_df", force_header=TRUE, print_diagn=TRUE)
#         print(all.equal(subset(union_df, select=-c(.src, .rownames, .rnorm)), 
#                         imp_entity_df))
        
        # Partition again
        glb_trnent_df <<- subset(union_df, .src == "entity", select=-c(.src, .rownames))
        comment(glb_trnent_df) <- "entity_df"
        glb_newent_df <<- subset(union_df, .src == "newent", select=-c(.src, .rownames))
        comment(glb_newent_df) <- "newent_df"
        
        # Generate summaries
        print(summary(entity_df))
        print(sapply(names(entity_df), function(col) sum(is.na(entity_df[, col]))))
        print(summary(newent_df))
        print(sapply(names(newent_df), function(col) sum(is.na(newent_df[, col]))))
    
    } else stop("Not implemented yet")
}

if (glb_impute_na_data) {
    if ((sum(sapply(names(glb_trnent_df), 
                    function(col) sum(is.na(glb_trnent_df[, col])))) > 0) | 
        (sum(sapply(names(glb_newent_df), 
                    function(col) sum(is.na(glb_newent_df[, col])))) > 0))
        glb_impute_missing_data(glb_trnent_df, glb_newent_df)
}    

glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="encode_retype_data", 
        chunk_step_major=max(glb_script_df$chunk_step_major), 
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                  chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed3 manage_missing_data                2                2  19.040
## elapsed4  encode_retype_data                2                3  30.736
```

### Step `2`.`3`: encode/retype data

```r
# map_<col_name>_df <- myimport_data(
#     url="<map_url>", 
#     comment="map_<col_name>_df", print_diagn=TRUE)
# map_<col_name>_df <- read.csv(paste0(getwd(), "/data/<file_name>.csv"), strip.white=TRUE)

# glb_trnent_df <- mymap_codes(glb_trnent_df, "<from_col_name>", "<to_col_name>", 
#     map_<to_col_name>_df, map_join_col_name="<map_join_col_name>", 
#                           map_tgt_col_name="<to_col_name>")
# glb_newent_df <- mymap_codes(glb_newent_df, "<from_col_name>", "<to_col_name>", 
#     map_<to_col_name>_df, map_join_col_name="<map_join_col_name>", 
#                           map_tgt_col_name="<to_col_name>")
    					
# glb_trnent_df$<col_name>.fctr <- factor(glb_trnent_df$<col_name>, 
#                     as.factor(union(glb_trnent_df$<col_name>, glb_newent_df$<col_name>)))
# glb_newent_df$<col_name>.fctr <- factor(glb_newent_df$<col_name>, 
#                     as.factor(union(glb_trnent_df$<col_name>, glb_newent_df$<col_name>)))

if (!is.null(glb_map_rsp_raw_to_var)) {
    glb_entity_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_entity_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_entity_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)
        
    glb_trnent_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_trnent_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_trnent_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)
        
    glb_newent_df[, glb_rsp_var] <- 
        glb_map_rsp_raw_to_var(glb_newent_df[, glb_rsp_var_raw])
    mycheck_map_results(mapd_df=glb_newent_df, 
                        from_col_name=glb_rsp_var_raw, to_col_name=glb_rsp_var)    
}
```

```
## Loading required package: sqldf
## Loading required package: gsubfn
## Loading required package: proto
## Loading required package: RSQLite
## Loading required package: DBI
## Loading required package: tcltk
```

```
##   voting voting.fctr     .n
## 1      0           N 235388
## 2      1           Y 108696
```

![](Michigan_Voters_files/figure-html/encode_retype_data_1-1.png) 

```
##   voting voting.fctr   .n
## 1      0           N 3420
## 2      1           Y 1580
```

![](Michigan_Voters_files/figure-html/encode_retype_data_1-2.png) 

```
##   voting voting.fctr   .n
## 1      0           N 3421
## 2      1           Y 1579
```

![](Michigan_Voters_files/figure-html/encode_retype_data_1-3.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="extract_features", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                 chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed4 encode_retype_data                2                3  30.736
## elapsed5   extract_features                3                0  35.739
```

## Step `3`: extract features

```r
# Create new features that help prediction
# <col_name>.lag.2 <- lag(zoo(glb_trnent_df$<col_name>), -2, na.pad=TRUE)
# glb_trnent_df[, "<col_name>.lag.2"] <- coredata(<col_name>.lag.2)
# <col_name>.lag.2 <- lag(zoo(glb_newent_df$<col_name>), -2, na.pad=TRUE)
# glb_newent_df[, "<col_name>.lag.2"] <- coredata(<col_name>.lag.2)
# 
# glb_newent_df[1, "<col_name>.lag.2"] <- glb_trnent_df[nrow(glb_trnent_df) - 1, 
#                                                    "<col_name>"]
# glb_newent_df[2, "<col_name>.lag.2"] <- glb_trnent_df[nrow(glb_trnent_df), 
#                                                    "<col_name>"]
                                                   
# glb_trnent_df <- mutate(glb_trnent_df,
#     <new_col_name>=
#                     )

# glb_newent_df <- mutate(glb_newent_df,
#     <new_col_name>=
#                     )

# print(summary(glb_trnent_df))
# print(summary(glb_newent_df))

# print(sapply(names(glb_trnent_df), function(col) sum(is.na(glb_trnent_df[, col]))))
# print(sapply(names(glb_newent_df), function(col) sum(is.na(glb_newent_df[, col]))))

# print(myplot_scatter(glb_trnent_df, "<col1_name>", "<col2_name>", smooth=TRUE))

replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all","data.new")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0
```

![](Michigan_Voters_files/figure-html/extract_features-1.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="select_features", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##               chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed5 extract_features                3                0  35.739
## elapsed6  select_features                4                0  37.151
```

## Step `4`: select features

```r
print(glb_feats_df <- myselect_features(entity_df=glb_trnent_df, 
                       exclude_vars_as_features=glb_exclude_vars_as_features, 
                       rsp_var=glb_rsp_var))
```

```
##                  id        cor.y exclude.as.feat   cor.y.abs
## voting       voting  1.000000000               1 1.000000000
## yob             yob -0.114539250               0 0.114539250
## control     control -0.076267307               0 0.076267307
## neighbors neighbors  0.061253311               0 0.061253311
## self           self  0.033691805               0 0.033691805
## hawthorne hawthorne  0.020770759               0 0.020770759
## sex             sex -0.017971837               0 0.017971837
## .rnorm       .rnorm  0.007838379               0 0.007838379
## civicduty civicduty  0.005217928               0 0.005217928
```

```r
glb_script_df <- rbind(glb_script_df, 
    data.frame(chunk_label="remove_correlated_features", 
        chunk_step_major=max(glb_script_df$chunk_step_major),
        chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))        
print(tail(glb_script_df, 2))
```

```
##                         chunk_label chunk_step_major chunk_step_minor
## elapsed6            select_features                4                0
## elapsed7 remove_correlated_features                4                1
##          elapsed
## elapsed6  37.151
## elapsed7  37.388
```

### Step `4`.`1`: remove correlated features

```r
print(glb_feats_df <- orderBy(~-cor.y, 
          myfind_cor_features(feats_df=glb_feats_df, entity_df=glb_trnent_df, 
                                rsp_var=glb_rsp_var)))
```

```
##                     yob      control    neighbors          self
## yob        1.0000000000 -0.005944470 -0.002134753 -0.0001225673
## control   -0.0059444696  1.000000000 -0.397821708 -0.4039009603
## neighbors -0.0021347529 -0.397821708  1.000000000 -0.1252279175
## self      -0.0001225673 -0.403900960 -0.125227917  1.0000000000
## hawthorne -0.0003301621 -0.390868198 -0.121187160 -0.1230390636
## sex        0.0630941271 -0.020432311  0.012924632  0.0022167536
## .rnorm     0.0233532361 -0.007055089  0.027961828  0.0149431847
## civicduty  0.0120565205 -0.396191815 -0.122837727 -0.1247148533
##               hawthorne          sex       .rnorm    civicduty
## yob       -0.0003301621  0.063094127  0.023353236  0.012056520
## control   -0.3908681979 -0.020432311 -0.007055089 -0.396191815
## neighbors -0.1211871603  0.012924632  0.027961828 -0.122837727
## self      -0.1230390636  0.002216754  0.014943185 -0.124714853
## hawthorne  1.0000000000  0.016123277  0.006731367 -0.120690651
## sex        0.0161232769  1.000000000  0.007685916  0.001359547
## .rnorm     0.0067313673  0.007685916  1.000000000 -0.038648966
## civicduty -0.1206906513  0.001359547 -0.038648966  1.000000000
##                    yob     control   neighbors         self    hawthorne
## yob       0.0000000000 0.005944470 0.002134753 0.0001225673 0.0003301621
## control   0.0059444696 0.000000000 0.397821708 0.4039009603 0.3908681979
## neighbors 0.0021347529 0.397821708 0.000000000 0.1252279175 0.1211871603
## self      0.0001225673 0.403900960 0.125227917 0.0000000000 0.1230390636
## hawthorne 0.0003301621 0.390868198 0.121187160 0.1230390636 0.0000000000
## sex       0.0630941271 0.020432311 0.012924632 0.0022167536 0.0161232769
## .rnorm    0.0233532361 0.007055089 0.027961828 0.0149431847 0.0067313673
## civicduty 0.0120565205 0.396191815 0.122837727 0.1247148533 0.1206906513
##                   sex      .rnorm   civicduty
## yob       0.063094127 0.023353236 0.012056520
## control   0.020432311 0.007055089 0.396191815
## neighbors 0.012924632 0.027961828 0.122837727
## self      0.002216754 0.014943185 0.124714853
## hawthorne 0.016123277 0.006731367 0.120690651
## sex       0.000000000 0.007685916 0.001359547
## .rnorm    0.007685916 0.000000000 0.038648966
## civicduty 0.001359547 0.038648966 0.000000000
##                  id        cor.y exclude.as.feat   cor.y.abs cor.low
## voting       voting  1.000000000               1 1.000000000       0
## neighbors neighbors  0.061253311               0 0.061253311       1
## self           self  0.033691805               0 0.033691805       1
## hawthorne hawthorne  0.020770759               0 0.020770759       1
## .rnorm       .rnorm  0.007838379               0 0.007838379       1
## civicduty civicduty  0.005217928               0 0.005217928       1
## sex             sex -0.017971837               0 0.017971837       1
## control     control -0.076267307               0 0.076267307       1
## yob             yob -0.114539250               0 0.114539250       1
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.models", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                         chunk_label chunk_step_major chunk_step_minor
## elapsed7 remove_correlated_features                4                1
## elapsed8                 fit.models                5                0
##          elapsed
## elapsed7  37.388
## elapsed8  37.437
```

## Step `5`: fit models

```r
max_cor_y_x_var <- orderBy(~ -cor.y.abs, 
        subset(glb_feats_df, (exclude.as.feat == 0) & (cor.low == 1)))[1, "id"]
if (!is.null(glb_Baseline_mdl_var)) {
    if ((max_cor_y_x_var != glb_Baseline_mdl_var) & 
        (glb_feats_df[max_cor_y_x_var, "cor.y.abs"] > 
         glb_feats_df[glb_Baseline_mdl_var, "cor.y.abs"]))
        stop(max_cor_y_x_var, " has a lower correlation with ", glb_rsp_var, 
             " than the Baseline var: ", glb_Baseline_mdl_var)
}

glb_model_type <- ifelse(glb_is_regression, "regression", "classification")
    
# Any models that have tuning parameters has "better" results with cross-validation (except rf)
#   & "different" results for different outcome metrics

# # Problem 1.3
# ret_lst <- myfit_mdl(model_id="Groups.X.entity", model_method="glm",
#                      model_type=glb_model_type,
#                         indep_vars_vctr=c("hawthorne", "civicduty", "neighbors", "self"),
#                         rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
#                         fit_df=glb_entity_df, OOB_df=NULL)
# 
# # Problem 2.1
# ret_lst <- myfit_mdl(model_id="Groups.X.entity", model_method="rpart",
#                      model_type=glb_model_type,
#                         indep_vars_vctr=c("hawthorne", "civicduty", "neighbors", "self"),
#                         rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
#                         fit_df=glb_entity_df, OOB_df=NULL)
# CARTmodel2 = rpart(voting ~ civicduty + hawthorne + self + neighbors, data=glb_entity_df, cp=0.0)
# prp(CARTmodel2)
# 
# # Problem 2.4
# CARTsex = rpart(voting ~ civicduty + hawthorne + self + neighbors + sex, data=glb_entity_df, cp=0.0)
# prp(CARTsex)
# 
# # Problem 3.1
# CARTcontrol = rpart(voting ~ control, data=glb_entity_df, cp=0.0)
# prp(CARTcontrol, digits=6)
# print(abs(0.34 - 0.296638))
# 
# # Problem 3.2
# CARTcontrolsex = rpart(voting ~ control + sex, data=glb_entity_df, cp=0.0)
# prp(CARTcontrolsex, digits=6)
# print(abs(0.34 - 0.296638))
# 
# # Problem 3.3
# GLMcontrolsex = glm(voting ~ control + sex, data=glb_entity_df, family=binomial)
# print(summary(GLMcontrolsex))
# Possibilities = data.frame(sex=c(0,0,1,1),control=c(0,1,0,1))
# predict(GLMcontrolsex, newdata=Possibilities, type="response")
# 
# # Problem 3.5
# GLMcontrolsex2 = glm(voting ~ control + sex + sex:control, data=glb_entity_df, family=binomial)
# print(summary(GLMcontrolsex2))
# Possibilities = data.frame(sex=c(0,0,1,1),control=c(0,1,0,1))
# predict(GLMcontrolsex2, newdata=Possibilities, type="response")

# Baseline
if (!is.null(glb_Baseline_mdl_var)) {
#     lm_mdl <- lm(reformulate(glb_Baseline_mdl_var, 
#                             response="bucket2009"), data=glb_trnent_df)
#     print(summary(lm_mdl))
#     plot(lm_mdl, ask=FALSE)
#     ret_lst <- myfit_mdl_fn(model_id="Baseline", 
#                             model_method=ifelse(glb_is_regression, "lm", 
#                                         ifelse(glb_is_binomial, "glm", "rpart")),
#                             indep_vars_vctr=glb_Baseline_mdl_var,
#                             rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
#                             fit_df=glb_trnent_df, OOB_df=glb_newent_df,
#                             n_cv_folds=0, tune_models_df=NULL,
#                             model_loss_mtrx=glb_model_metric_terms,
#                             model_summaryFunction=glb_model_metric_smmry,
#                             model_metric=glb_model_metric,
#                             model_metric_maximize=glb_model_metric_maximize)
    ret_lst <- myfit_mdl_fn(model_id="Baseline", model_method="mybaseln_classfr",
                            indep_vars_vctr=glb_Baseline_mdl_var,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df)
}

# Most Frequent Outcome "MFO" model: mean(y) for regression
#   Not using caret's nullModel since model stats not avl
#   Cannot use rpart for multinomial classification since it predicts non-MFO
ret_lst <- myfit_mdl(model_id="MFO", 
                     model_method=ifelse(glb_is_regression, "lm", "myMFO_classfr"), 
                     model_type=glb_model_type,
                        indep_vars_vctr=".rnorm",
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## Loading required package: caret
## Loading required package: lattice
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:survival':
## 
##     cluster
```

```
## [1] "fitting model: MFO.myMFO_classfr"
## [1] "    indep_vars: .rnorm"
## Fitting parameter = none on full training set
## [1] "in MFO.Classifier$fit"
## [1] "unique.vals:"
## [1] N Y
## Levels: N Y
## [1] "unique.prob:"
## y
##     N     Y 
## 0.684 0.316 
## [1] "MFO.val:"
## [1] "N"
##             Length Class      Mode     
## unique.vals 2      factor     numeric  
## unique.prob 2      -none-     numeric  
## MFO.val     1      -none-     character
## x.names     1      -none-     character
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   2      -none-     character
```

```
## Loading required package: ROCR
## Loading required package: gplots
## 
## Attaching package: 'gplots'
## 
## The following object is masked from 'package:stats':
## 
##     lowess
```

```
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##       N     Y
## 1 0.684 0.316
## 2 0.684 0.316
## 3 0.684 0.316
## 4 0.684 0.316
## 5 0.684 0.316
## 6 0.684 0.316
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.MFO.myMFO_classfr.N
## 1           N                                    3420
## 2           Y                                    1580
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.MFO.myMFO_classfr.N
## 1           N                                    3420
## 2           Y                                    1580
##   voting.fctr.predict.MFO.myMFO_classfr.Y
## 1                                       0
## 2                                       0
##          Prediction
## Reference    N    Y
##         N 3420    0
##         Y 1580    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.6840000      0.0000000      0.6709089      0.6968760      0.6840000 
## AccuracyPValue  McnemarPValue 
##      0.5068114      0.0000000 
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##       N     Y
## 1 0.684 0.316
## 2 0.684 0.316
## 3 0.684 0.316
## 4 0.684 0.316
## 5 0.684 0.316
## 6 0.684 0.316
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.MFO.myMFO_classfr.N
## 1           N                                    3421
## 2           Y                                    1579
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.MFO.myMFO_classfr.N
## 1           N                                    3421
## 2           Y                                    1579
##   voting.fctr.predict.MFO.myMFO_classfr.Y
## 1                                       0
## 2                                       0
##          Prediction
## Reference    N    Y
##         N 3421    0
##         Y 1579    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.6842000      0.0000000      0.6711109      0.6970737      0.6842000 
## AccuracyPValue  McnemarPValue 
##      0.5068134      0.0000000 
##            model_id  model_method  feats max.nTuningRuns
## 1 MFO.myMFO_classfr myMFO_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.341                 0.003         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0            0.684
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.6709089              0.696876             0         0.5
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5               0           0.6842
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.6711109             0.6970737             0
```

```r
if (glb_is_classification)
    # "random" model - only for classification; none needed for regression since it is same as MFO
    ret_lst <- myfit_mdl(model_id="Random", model_method="myrandom_classfr",
                            model_type=glb_model_type,                         
                            indep_vars_vctr=".rnorm",
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## [1] "fitting model: Random.myrandom_classfr"
## [1] "    indep_vars: .rnorm"
## Fitting parameter = none on full training set
##             Length Class      Mode     
## unique.vals 2      factor     numeric  
## unique.prob 2      table      numeric  
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   2      -none-     character
## [1] "in Random.Classifier$prob"
```

![](Michigan_Voters_files/figure-html/fit.models-1.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                             0
## 2           Y                                             0
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          3420
## 2                                          1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                             0
## 2           Y                                             0
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          3420
## 2                                          1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                             0
## 2           Y                                             0
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          3420
## 2                                          1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                             0
## 2           Y                                             0
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          3420
## 2                                          1580
##           Reference
## Prediction    N    Y
##          N 2353 1061
##          Y 1067  519
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          2353
## 2           Y                                          1061
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          1067
## 2                                           519
##           Reference
## Prediction    N    Y
##          N 2353 1061
##          Y 1067  519
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          2353
## 2           Y                                          1061
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          1067
## 2                                           519
##           Reference
## Prediction    N    Y
##          N 2353 1061
##          Y 1067  519
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          2353
## 2           Y                                          1061
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          1067
## 2                                           519
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          3420
## 2           Y                                          1580
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                             0
## 2                                             0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          3420
## 2           Y                                          1580
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                             0
## 2                                             0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          3420
## 2           Y                                          1580
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                             0
## 2                                             0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          3420
## 2           Y                                          1580
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                             0
## 2                                             0
##    threshold   f.score
## 1        0.0 0.4802432
## 2        0.1 0.4802432
## 3        0.2 0.4802432
## 4        0.3 0.4802432
## 5        0.4 0.3278585
## 6        0.5 0.3278585
## 7        0.6 0.3278585
## 8        0.7 0.0000000
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Michigan_Voters_files/figure-html/fit.models-2.png) 

```
## [1] "Classifier Probability Threshold: 0.3000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.Y
## 1           N                                          3420
## 2           Y                                          1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                             0
## 2           Y                                             0
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          3420
## 2                                          1580
##          Prediction
## Reference    N    Y
##         N    0 3420
##         Y    0 1580
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3160000      0.0000000      0.3031240      0.3290911      0.6840000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000 
## [1] "in Random.Classifier$prob"
```

![](Michigan_Voters_files/figure-html/fit.models-3.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                             0
## 2           Y                                             0
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          3421
## 2                                          1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                             0
## 2           Y                                             0
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          3421
## 2                                          1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                             0
## 2           Y                                             0
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          3421
## 2                                          1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                             0
## 2           Y                                             0
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          3421
## 2                                          1579
##           Reference
## Prediction    N    Y
##          N 2334 1076
##          Y 1087  503
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          2334
## 2           Y                                          1076
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          1087
## 2                                           503
##           Reference
## Prediction    N    Y
##          N 2334 1076
##          Y 1087  503
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          2334
## 2           Y                                          1076
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          1087
## 2                                           503
##           Reference
## Prediction    N    Y
##          N 2334 1076
##          Y 1087  503
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          2334
## 2           Y                                          1076
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          1087
## 2                                           503
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          3421
## 2           Y                                          1579
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                             0
## 2                                             0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          3421
## 2           Y                                          1579
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                             0
## 2                                             0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          3421
## 2           Y                                          1579
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                             0
## 2                                             0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                          3421
## 2           Y                                          1579
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                             0
## 2                                             0
##    threshold   f.score
## 1        0.0 0.4800122
## 2        0.1 0.4800122
## 3        0.2 0.4800122
## 4        0.3 0.4800122
## 5        0.4 0.3174503
## 6        0.5 0.3174503
## 7        0.6 0.3174503
## 8        0.7 0.0000000
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Michigan_Voters_files/figure-html/fit.models-4.png) 

```
## [1] "Classifier Probability Threshold: 0.3000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.Y
## 1           N                                          3421
## 2           Y                                          1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Random.myrandom_classfr.N
## 1           N                                             0
## 2           Y                                             0
##   voting.fctr.predict.Random.myrandom_classfr.Y
## 1                                          3421
## 2                                          1579
##          Prediction
## Reference    N    Y
##         N    0 3421
##         Y    0 1579
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3158000      0.0000000      0.3029263      0.3288891      0.6842000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000 
##                  model_id     model_method  feats max.nTuningRuns
## 1 Random.myrandom_classfr myrandom_classfr .rnorm               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.231                 0.002   0.5082464
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.3       0.4802432            0.316
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.303124             0.3290911             0   0.5004063
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.3       0.4800122           0.3158
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.3029263             0.3288891             0
```

```r
# Max.cor.Y
#   Check impact of cv
#       rpart is not a good candidate since caret does not optimize cp (only tuning parameter of rpart) well
ret_lst <- myfit_mdl(model_id="Max.cor.Y.cv.0", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df)
```

```
## [1] "fitting model: Max.cor.Y.cv.0.rpart"
## [1] "    indep_vars: yob"
```

```
## Loading required package: rpart
```

```
## Fitting cp = 0.0019 on full training set
```

```
## Loading required package: rpart.plot
```

![](Michigan_Voters_files/figure-html/fit.models-5.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 5000 
## 
##            CP nsplit rel error
## 1 0.001898734      0         1
## 
## Node number 1: 5000 observations
##   predicted class=N  expected loss=0.316  P(node) =1
##     class counts:  3420  1580
##    probabilities: 0.684 0.316 
## 
## n= 5000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 5000 1580 N (0.6840000 0.3160000) *
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1           N                                       3420
## 2           Y                                       1580
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1           N                                       3420
## 2           Y                                       1580
##   voting.fctr.predict.Max.cor.Y.cv.0.rpart.Y
## 1                                          0
## 2                                          0
##          Prediction
## Reference    N    Y
##         N 3420    0
##         Y 1580    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.6840000      0.0000000      0.6709089      0.6968760      0.6840000 
## AccuracyPValue  McnemarPValue 
##      0.5068114      0.0000000 
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1           N                                       3421
## 2           Y                                       1579
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.rpart.N
## 1           N                                       3421
## 2           Y                                       1579
##   voting.fctr.predict.Max.cor.Y.cv.0.rpart.Y
## 1                                          0
## 2                                          0
##          Prediction
## Reference    N    Y
##         N 3421    0
##         Y 1579    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.6842000      0.0000000      0.6711109      0.6970737      0.6842000 
## AccuracyPValue  McnemarPValue 
##      0.5068134      0.0000000 
##               model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.cv.0.rpart        rpart   yob               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.655                 0.079         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0            0.684
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.6709089              0.696876             0         0.5
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5               0           0.6842
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.6711109             0.6970737             0
```

```r
ret_lst <- myfit_mdl(model_id="Max.cor.Y.cv.0.cp.0", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=0, tune_models_df=data.frame(parameter="cp", min=0.0, max=0.0, by=0.1))
```

```
## [1] "fitting model: Max.cor.Y.cv.0.cp.0.rpart"
## [1] "    indep_vars: yob"
## Fitting cp = 0 on full training set
```

![](Michigan_Voters_files/figure-html/fit.models-6.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 5000 
## 
##             CP nsplit rel error
## 1 0.0018987342      0 1.0000000
## 2 0.0009493671      5 0.9905063
## 3 0.0006329114      9 0.9867089
## 4 0.0000000000     10 0.9860759
## 
## Variable importance
## yob 
## 100 
## 
## Node number 1: 5000 observations,    complexity param=0.001898734
##   predicted class=N  expected loss=0.316  P(node) =1
##     class counts:  3420  1580
##    probabilities: 0.684 0.316 
##   left son=2 (3443 obs) right son=3 (1557 obs)
##   Primary splits:
##       yob < 1949.5 to the right, improve=27.30605, (0 missing)
## 
## Node number 2: 3443 observations
##   predicted class=N  expected loss=0.2808597  P(node) =0.6886
##     class counts:  2476   967
##    probabilities: 0.719 0.281 
## 
## Node number 3: 1557 observations,    complexity param=0.001898734
##   predicted class=N  expected loss=0.3937058  P(node) =0.3114
##     class counts:   944   613
##    probabilities: 0.606 0.394 
##   left son=6 (317 obs) right son=7 (1240 obs)
##   Primary splits:
##       yob < 1933.5 to the left,  improve=3.107252, (0 missing)
## 
## Node number 6: 317 observations,    complexity param=0.0009493671
##   predicted class=N  expected loss=0.3312303  P(node) =0.0634
##     class counts:   212   105
##    probabilities: 0.669 0.331 
##   left son=12 (29 obs) right son=13 (288 obs)
##   Primary splits:
##       yob < 1918.5 to the left,  improve=0.5153952, (0 missing)
## 
## Node number 7: 1240 observations,    complexity param=0.001898734
##   predicted class=N  expected loss=0.4096774  P(node) =0.248
##     class counts:   732   508
##    probabilities: 0.590 0.410 
##   left son=14 (575 obs) right son=15 (665 obs)
##   Primary splits:
##       yob < 1944.5 to the right, improve=6.058948, (0 missing)
## 
## Node number 12: 29 observations
##   predicted class=N  expected loss=0.2413793  P(node) =0.0058
##     class counts:    22     7
##    probabilities: 0.759 0.241 
## 
## Node number 13: 288 observations,    complexity param=0.0009493671
##   predicted class=N  expected loss=0.3402778  P(node) =0.0576
##     class counts:   190    98
##    probabilities: 0.660 0.340 
##   left son=26 (281 obs) right son=27 (7 obs)
##   Primary splits:
##       yob < 1919.5 to the right, improve=2.007132, (0 missing)
## 
## Node number 14: 575 observations
##   predicted class=N  expected loss=0.3565217  P(node) =0.115
##     class counts:   370   205
##    probabilities: 0.643 0.357 
## 
## Node number 15: 665 observations,    complexity param=0.001898734
##   predicted class=N  expected loss=0.4556391  P(node) =0.133
##     class counts:   362   303
##    probabilities: 0.544 0.456 
##   left son=30 (79 obs) right son=31 (586 obs)
##   Primary splits:
##       yob < 1935.5 to the left,  improve=0.7169401, (0 missing)
## 
## Node number 26: 281 observations
##   predicted class=N  expected loss=0.3309609  P(node) =0.0562
##     class counts:   188    93
##    probabilities: 0.669 0.331 
## 
## Node number 27: 7 observations
##   predicted class=Y  expected loss=0.2857143  P(node) =0.0014
##     class counts:     2     5
##    probabilities: 0.286 0.714 
## 
## Node number 30: 79 observations
##   predicted class=N  expected loss=0.3924051  P(node) =0.0158
##     class counts:    48    31
##    probabilities: 0.608 0.392 
## 
## Node number 31: 586 observations,    complexity param=0.001898734
##   predicted class=N  expected loss=0.4641638  P(node) =0.1172
##     class counts:   314   272
##    probabilities: 0.536 0.464 
##   left son=62 (531 obs) right son=63 (55 obs)
##   Primary splits:
##       yob < 1936.5 to the right, improve=3.599657, (0 missing)
## 
## Node number 62: 531 observations,    complexity param=0.0009493671
##   predicted class=N  expected loss=0.4463277  P(node) =0.1062
##     class counts:   294   237
##    probabilities: 0.554 0.446 
##   left son=124 (271 obs) right son=125 (260 obs)
##   Primary splits:
##       yob < 1941.5 to the right, improve=0.370028, (0 missing)
## 
## Node number 63: 55 observations
##   predicted class=Y  expected loss=0.3636364  P(node) =0.011
##     class counts:    20    35
##    probabilities: 0.364 0.636 
## 
## Node number 124: 271 observations
##   predicted class=N  expected loss=0.4280443  P(node) =0.0542
##     class counts:   155   116
##    probabilities: 0.572 0.428 
## 
## Node number 125: 260 observations,    complexity param=0.0009493671
##   predicted class=N  expected loss=0.4653846  P(node) =0.052
##     class counts:   139   121
##    probabilities: 0.535 0.465 
##   left son=250 (147 obs) right son=251 (113 obs)
##   Primary splits:
##       yob < 1939.5 to the left,  improve=0.9167461, (0 missing)
## 
## Node number 250: 147 observations
##   predicted class=N  expected loss=0.4285714  P(node) =0.0294
##     class counts:    84    63
##    probabilities: 0.571 0.429 
## 
## Node number 251: 113 observations,    complexity param=0.0006329114
##   predicted class=Y  expected loss=0.4867257  P(node) =0.0226
##     class counts:    55    58
##    probabilities: 0.487 0.513 
##   left son=502 (57 obs) right son=503 (56 obs)
##   Primary splits:
##       yob < 1940.5 to the right, improve=0.1118061, (0 missing)
## 
## Node number 502: 57 observations
##   predicted class=N  expected loss=0.4912281  P(node) =0.0114
##     class counts:    29    28
##    probabilities: 0.509 0.491 
## 
## Node number 503: 56 observations
##   predicted class=Y  expected loss=0.4642857  P(node) =0.0112
##     class counts:    26    30
##    probabilities: 0.464 0.536 
## 
## n= 5000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##   1) root 5000 1580 N (0.6840000 0.3160000)  
##     2) yob>=1949.5 3443  967 N (0.7191403 0.2808597) *
##     3) yob< 1949.5 1557  613 N (0.6062942 0.3937058)  
##       6) yob< 1933.5 317  105 N (0.6687697 0.3312303)  
##        12) yob< 1918.5 29    7 N (0.7586207 0.2413793) *
##        13) yob>=1918.5 288   98 N (0.6597222 0.3402778)  
##          26) yob>=1919.5 281   93 N (0.6690391 0.3309609) *
##          27) yob< 1919.5 7    2 Y (0.2857143 0.7142857) *
##       7) yob>=1933.5 1240  508 N (0.5903226 0.4096774)  
##        14) yob>=1944.5 575  205 N (0.6434783 0.3565217) *
##        15) yob< 1944.5 665  303 N (0.5443609 0.4556391)  
##          30) yob< 1935.5 79   31 N (0.6075949 0.3924051) *
##          31) yob>=1935.5 586  272 N (0.5358362 0.4641638)  
##            62) yob>=1936.5 531  237 N (0.5536723 0.4463277)  
##             124) yob>=1941.5 271  116 N (0.5719557 0.4280443) *
##             125) yob< 1941.5 260  121 N (0.5346154 0.4653846)  
##               250) yob< 1939.5 147   63 N (0.5714286 0.4285714) *
##               251) yob>=1939.5 113   55 Y (0.4867257 0.5132743)  
##                 502) yob>=1940.5 57   28 N (0.5087719 0.4912281) *
##                 503) yob< 1940.5 56   26 Y (0.4642857 0.5357143) *
##            63) yob< 1936.5 55   20 Y (0.3636364 0.6363636) *
```

![](Michigan_Voters_files/figure-html/fit.models-7.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                               0
## 2           Y                                               0
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                            3420
## 2                                            1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                               0
## 2           Y                                               0
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                            3420
## 2                                            1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                               0
## 2           Y                                               0
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                            3420
## 2                                            1580
##           Reference
## Prediction    N    Y
##          N 2498  974
##          Y  922  606
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            2498
## 2           Y                                             974
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                             922
## 2                                             606
##           Reference
## Prediction    N    Y
##          N 3104 1303
##          Y  316  277
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3104
## 2           Y                                            1303
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                             316
## 2                                             277
##           Reference
## Prediction    N    Y
##          N 3372 1510
##          Y   48   70
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3372
## 2           Y                                            1510
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              48
## 2                                              70
##           Reference
## Prediction    N    Y
##          N 3398 1540
##          Y   22   40
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3398
## 2           Y                                            1540
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              22
## 2                                              40
##           Reference
## Prediction    N    Y
##          N 3418 1575
##          Y    2    5
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3418
## 2           Y                                            1575
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               2
## 2                                               5
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3420
## 2           Y                                            1580
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               0
## 2                                               0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3420
## 2           Y                                            1580
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               0
## 2                                               0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3420
## 2           Y                                            1580
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               0
## 2                                               0
##    threshold     f.score
## 1        0.0 0.480243161
## 2        0.1 0.480243161
## 3        0.2 0.480243161
## 4        0.3 0.389961390
## 5        0.4 0.254947078
## 6        0.5 0.082449941
## 7        0.6 0.048721072
## 8        0.7 0.006301197
## 9        0.8 0.000000000
## 10       0.9 0.000000000
## 11       1.0 0.000000000
```

![](Michigan_Voters_files/figure-html/fit.models-8.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1           N                                            3420
## 2           Y                                            1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                               0
## 2           Y                                               0
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                            3420
## 2                                            1580
##          Prediction
## Reference    N    Y
##         N    0 3420
##         Y    0 1580
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3160000      0.0000000      0.3031240      0.3290911      0.6840000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000
```

![](Michigan_Voters_files/figure-html/fit.models-9.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                               0
## 2           Y                                               0
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                            3421
## 2                                            1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                               0
## 2           Y                                               0
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                            3421
## 2                                            1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                               0
## 2           Y                                               0
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                            3421
## 2                                            1579
##           Reference
## Prediction    N    Y
##          N 2481  995
##          Y  940  584
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            2481
## 2           Y                                             995
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                             940
## 2                                             584
##           Reference
## Prediction    N    Y
##          N 3072 1334
##          Y  349  245
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3072
## 2           Y                                            1334
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                             349
## 2                                             245
##           Reference
## Prediction    N    Y
##          N 3358 1516
##          Y   63   63
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3358
## 2           Y                                            1516
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              63
## 2                                              63
##           Reference
## Prediction    N    Y
##          N 3390 1555
##          Y   31   24
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3390
## 2           Y                                            1555
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                              31
## 2                                              24
##           Reference
## Prediction    N    Y
##          N 3414 1578
##          Y    7    1
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3414
## 2           Y                                            1578
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               7
## 2                                               1
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3421
## 2           Y                                            1579
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               0
## 2                                               0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3421
## 2           Y                                            1579
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               0
## 2                                               0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                            3421
## 2           Y                                            1579
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                               0
## 2                                               0
##    threshold     f.score
## 1        0.0 0.480012160
## 2        0.1 0.480012160
## 3        0.2 0.480012160
## 4        0.3 0.376409926
## 5        0.4 0.225494708
## 6        0.5 0.073900293
## 7        0.6 0.029375765
## 8        0.7 0.001260239
## 9        0.8 0.000000000
## 10       0.9 0.000000000
## 11       1.0 0.000000000
```

![](Michigan_Voters_files/figure-html/fit.models-10.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1           N                                            3421
## 2           Y                                            1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.N
## 1           N                                               0
## 2           Y                                               0
##   voting.fctr.predict.Max.cor.Y.cv.0.cp.0.rpart.Y
## 1                                            3421
## 2                                            1579
##          Prediction
## Reference    N    Y
##         N    0 3421
##         Y    0 1579
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3158000      0.0000000      0.3029263      0.3288891      0.6842000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000 
##                    model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.cv.0.cp.0.rpart        rpart   yob               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.517                 0.073   0.5649086
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.2       0.4802432            0.316
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.303124             0.3290911             0   0.5515162
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2       0.4800122           0.3158
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.3029263             0.3288891             0
```

```r
ret_lst <- myfit_mdl(model_id="Max.cor.Y", 
                        model_method="rpart",
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Max.cor.Y.rpart"
## [1] "    indep_vars: yob"
## + Fold1: cp=0.0006329 
## - Fold1: cp=0.0006329 
## + Fold2: cp=0.0006329 
## - Fold2: cp=0.0006329 
## + Fold3: cp=0.0006329 
## - Fold3: cp=0.0006329 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.0019 on full training set
```

```
## Warning in myfit_mdl(model_id = "Max.cor.Y", model_method = "rpart",
## model_type = glb_model_type, : model's bestTune found at an extreme of
## tuneGrid for parameter: cp
```

![](Michigan_Voters_files/figure-html/fit.models-11.png) ![](Michigan_Voters_files/figure-html/fit.models-12.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 5000 
## 
##            CP nsplit rel error
## 1 0.001898734      0         1
## 
## Node number 1: 5000 observations
##   predicted class=N  expected loss=0.316  P(node) =1
##     class counts:  3420  1580
##    probabilities: 0.684 0.316 
## 
## n= 5000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
## 1) root 5000 1580 N (0.6840000 0.3160000) *
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.Max.cor.Y.rpart.N
## 1           N                                  3420
## 2           Y                                  1580
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.rpart.N
## 1           N                                  3420
## 2           Y                                  1580
##   voting.fctr.predict.Max.cor.Y.rpart.Y
## 1                                     0
## 2                                     0
##          Prediction
## Reference    N    Y
##         N 3420    0
##         Y 1580    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.6840000      0.0000000      0.6709089      0.6968760      0.6840000 
## AccuracyPValue  McnemarPValue 
##      0.5068114      0.0000000 
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.Max.cor.Y.rpart.N
## 1           N                                  3421
## 2           Y                                  1579
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.rpart.N
## 1           N                                  3421
## 2           Y                                  1579
##   voting.fctr.predict.Max.cor.Y.rpart.Y
## 1                                     0
## 2                                     0
##          Prediction
## Reference    N    Y
##         N 3421    0
##         Y 1579    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.6842000      0.0000000      0.6711109      0.6970737      0.6842000 
## AccuracyPValue  McnemarPValue 
##      0.5068134      0.0000000 
##          model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.rpart        rpart   yob               3
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      1.409                 0.078         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0        0.6785971
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.6709089              0.696876    0.01341244         0.5
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.5               0           0.6842
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.6711109             0.6970737             0
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.01292947       0.0125422
```

```r
# Used to compare vs. Interactions.High.cor.Y 
ret_lst <- myfit_mdl(model_id="Max.cor.Y", 
                        model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                     model_type=glb_model_type,
                        indep_vars_vctr=max_cor_y_x_var,
                        rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Max.cor.Y.glm"
## [1] "    indep_vars: yob"
## + Fold1: parameter=none 
## - Fold1: parameter=none 
## + Fold2: parameter=none 
## - Fold2: parameter=none 
## + Fold3: parameter=none 
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Michigan_Voters_files/figure-html/fit.models-13.png) ![](Michigan_Voters_files/figure-html/fit.models-14.png) ![](Michigan_Voters_files/figure-html/fit.models-15.png) ![](Michigan_Voters_files/figure-html/fit.models-16.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.1972  -0.8870  -0.8132   1.4137   1.7552  
## 
## Coefficients:
##              Estimate Std. Error z value Pr(>|z|)    
## (Intercept) 32.963356   4.192276   7.863 3.75e-15 ***
## yob         -0.017252   0.002145  -8.045 8.65e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 6238.2  on 4999  degrees of freedom
## Residual deviance: 6172.2  on 4998  degrees of freedom
## AIC: 6176.2
## 
## Number of Fisher Scoring iterations: 4
```

![](Michigan_Voters_files/figure-html/fit.models-17.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                3420
## 2                                1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                3420
## 2                                1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                3420
## 2                                1580
##           Reference
## Prediction    N    Y
##          N 1450  539
##          Y 1970 1041
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                1450
## 2           Y                                 539
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                1970
## 2                                1041
##           Reference
## Prediction    N    Y
##          N 3183 1454
##          Y  237  126
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3183
## 2           Y                                1454
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                 237
## 2                                 126
##           Reference
## Prediction    N    Y
##          N 3419 1580
##          Y    1    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3419
## 2           Y                                1580
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   1
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##    threshold   f.score
## 1        0.0 0.4802432
## 2        0.1 0.4802432
## 3        0.2 0.4802432
## 4        0.3 0.4534960
## 5        0.4 0.1296963
## 6        0.5 0.0000000
## 7        0.6 0.0000000
## 8        0.7 0.0000000
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Michigan_Voters_files/figure-html/fit.models-18.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.Y
## 1           N                                3420
## 2           Y                                1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                3420
## 2                                1580
##          Prediction
## Reference    N    Y
##         N    0 3420
##         Y    0 1580
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3160000      0.0000000      0.3031240      0.3290911      0.6840000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000
```

![](Michigan_Voters_files/figure-html/fit.models-19.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                3421
## 2                                1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                3421
## 2                                1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                3421
## 2                                1579
##           Reference
## Prediction    N    Y
##          N 1499  485
##          Y 1922 1094
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                1499
## 2           Y                                 485
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                1922
## 2                                1094
##           Reference
## Prediction    N    Y
##          N 3184 1454
##          Y  237  125
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3184
## 2           Y                                1454
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                 237
## 2                                 125
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                   0
## 2                                   0
##    threshold   f.score
## 1        0.0 0.4800122
## 2        0.1 0.4800122
## 3        0.2 0.4800122
## 4        0.3 0.4761697
## 5        0.4 0.1287996
## 6        0.5 0.0000000
## 7        0.6 0.0000000
## 8        0.7 0.0000000
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Michigan_Voters_files/figure-html/fit.models-20.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.Y
## 1           N                                3421
## 2           Y                                1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Max.cor.Y.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Max.cor.Y.glm.Y
## 1                                3421
## 2                                1579
##          Prediction
## Reference    N    Y
##         N    0 3421
##         Y    0 1579
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3158000      0.0000000      0.3029263      0.3288891      0.6842000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000 
##        model_id model_method feats max.nTuningRuns
## 1 Max.cor.Y.glm          glm   yob               1
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      1.063                 0.088   0.5724227
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.2       0.4802432        0.6844003
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.303124             0.3290911   0.002658553   0.5845416
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2       0.4800122           0.3158
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.3029263             0.3288891             0    6176.245
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.00148051     0.005674799
```

```r
# Interactions.High.cor.Y
if (nrow(int_feats_df <- subset(glb_feats_df, (cor.low == 0) & 
                                              (exclude.as.feat == 0))) > 0) {
    # lm & glm handle interaction terms; rpart & rf do not
    #   This does not work - why ???
#     indep_vars_vctr <- ifelse(glb_is_binomial, 
#         c(max_cor_y_x_var, paste(max_cor_y_x_var, 
#                         subset(glb_feats_df, is.na(cor.low))[, "id"], sep=":")),
#         union(max_cor_y_x_var, subset(glb_feats_df, is.na(cor.low))[, "id"]))
    if (glb_is_regression || glb_is_binomial) {
        indep_vars_vctr <- 
            c(max_cor_y_x_var, paste(max_cor_y_x_var, int_feats_df[, "id"], sep=":"))       
    } else { indep_vars_vctr <- union(max_cor_y_x_var, int_feats_df[, "id"]) }
    
    ret_lst <- myfit_mdl(model_id="Interact.High.cor.y", 
                            model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                         model_type=glb_model_type,
                            indep_vars_vctr,
                            glb_rsp_var, glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                            n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)                        
}    

# Low.cor.X
ret_lst <- myfit_mdl(model_id="Low.cor.X", 
                        model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
                        indep_vars_vctr=subset(glb_feats_df, cor.low == 1)[, "id"],
                         model_type=glb_model_type,                     
                        glb_rsp_var, glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Low.cor.X.glm"
## [1] "    indep_vars: neighbors, self, hawthorne, .rnorm, civicduty, sex, control, yob"
## + Fold1: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Michigan_Voters_files/figure-html/fit.models-21.png) ![](Michigan_Voters_files/figure-html/fit.models-22.png) ![](Michigan_Voters_files/figure-html/fit.models-23.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.3037  -0.8876  -0.7861   1.3803   1.8429  
## 
## Coefficients: (1 not defined because of singularities)
##              Estimate Std. Error z value Pr(>|z|)    
## (Intercept) 32.941519   4.214821   7.816 5.47e-15 ***
## neighbors    0.510025   0.097554   5.228 1.71e-07 ***
## self         0.351723   0.097866   3.594 0.000326 ***
## hawthorne    0.282158   0.101096   2.791 0.005255 ** 
## .rnorm       0.018739   0.031075   0.603 0.546506    
## civicduty    0.195654   0.101455   1.928 0.053796 .  
## sex         -0.054481   0.061665  -0.884 0.376963    
## control            NA         NA      NA       NA    
## yob         -0.017305   0.002157  -8.022 1.04e-15 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 6238.2  on 4999  degrees of freedom
## Residual deviance: 6134.5  on 4992  degrees of freedom
## AIC: 6150.5
## 
## Number of Fisher Scoring iterations: 4
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Michigan_Voters_files/figure-html/fit.models-24.png) ![](Michigan_Voters_files/figure-html/fit.models-25.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                3420
## 2                                1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                3420
## 2                                1580
##           Reference
## Prediction    N    Y
##          N  141   18
##          Y 3279 1562
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                 141
## 2           Y                                  18
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                3279
## 2                                1562
##           Reference
## Prediction    N    Y
##          N 1676  572
##          Y 1744 1008
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                1676
## 2           Y                                 572
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                1744
## 2                                1008
##           Reference
## Prediction    N    Y
##          N 3083 1329
##          Y  337  251
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3083
## 2           Y                                1329
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                 337
## 2                                 251
##           Reference
## Prediction    N    Y
##          N 3394 1562
##          Y   26   18
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3394
## 2           Y                                1562
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                  26
## 2                                  18
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                   0
## 2                                   0
##    threshold    f.score
## 1        0.0 0.48024316
## 2        0.1 0.48024316
## 3        0.2 0.48652858
## 4        0.3 0.46537396
## 5        0.4 0.23154982
## 6        0.5 0.02216749
## 7        0.6 0.00000000
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                 141
## 2           Y                                  18
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                3279
## 2                                1562
##           Reference
## Prediction    N    Y
##          N  141   18
##          Y 3279 1562
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                 141
## 2           Y                                  18
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                3279
## 2                                1562
##          Prediction
## Reference    N    Y
##         N  141 3279
##         Y   18 1562
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##     0.34060000     0.01918436     0.32746032     0.35392603     0.68400000 
## AccuracyPValue  McnemarPValue 
##     1.00000000     0.00000000
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Michigan_Voters_files/figure-html/fit.models-26.png) ![](Michigan_Voters_files/figure-html/fit.models-27.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                3421
## 2                                1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                3421
## 2                                1579
##           Reference
## Prediction    N    Y
##          N  153   24
##          Y 3268 1555
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                 153
## 2           Y                                  24
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                3268
## 2                                1555
##           Reference
## Prediction    N    Y
##          N 1689  579
##          Y 1732 1000
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                1689
## 2           Y                                 579
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                1732
## 2                                1000
##           Reference
## Prediction    N    Y
##          N 3058 1369
##          Y  363  210
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3058
## 2           Y                                1369
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                 363
## 2                                 210
##           Reference
## Prediction    N    Y
##          N 3394 1563
##          Y   27   16
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3394
## 2           Y                                1563
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                  27
## 2                                  16
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                   0
## 2                                   0
##    threshold    f.score
## 1        0.0 0.48001216
## 2        0.1 0.48001216
## 3        0.2 0.48578569
## 4        0.3 0.46392948
## 5        0.4 0.19516729
## 6        0.5 0.01972873
## 7        0.6 0.00000000
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

![](Michigan_Voters_files/figure-html/fit.models-28.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                 153
## 2           Y                                  24
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                3268
## 2                                1555
##           Reference
## Prediction    N    Y
##          N  153   24
##          Y 3268 1555
##   voting.fctr voting.fctr.predict.Low.cor.X.glm.N
## 1           N                                 153
## 2           Y                                  24
##   voting.fctr.predict.Low.cor.X.glm.Y
## 1                                3268
## 2                                1555
##          Prediction
## Reference    N    Y
##         N  153 3268
##         Y   24 1555
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##     0.34160000     0.01900987     0.32845044     0.35493473     0.68420000 
## AccuracyPValue  McnemarPValue 
##     1.00000000     0.00000000 
##        model_id model_method
## 1 Low.cor.X.glm          glm
##                                                              feats
## 1 neighbors, self, hawthorne, .rnorm, civicduty, sex, control, yob
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      1.372                 0.175
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.5877905                    0.2       0.4865286        0.6830003
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.3274603              0.353926   0.007655516   0.5819886
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2       0.4857857           0.3416
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.3284504             0.3549347    0.01900987    6150.502
##   max.AccuracySD.fit max.KappaSD.fit
## 1        0.001142904      0.01027105
```

```r
# Groups.X
ret_lst <- myfit_mdl(model_id="Groups.X", 
                        model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
    indep_vars_vctr=c("hawthorne","civicduty","neighbors","self","control",".rnorm"),
                         model_type=glb_model_type,                     
                        glb_rsp_var, glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: Groups.X.glm"
## [1] "    indep_vars: hawthorne, civicduty, neighbors, self, control, .rnorm"
## + Fold1: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Michigan_Voters_files/figure-html/fit.models-29.png) ![](Michigan_Voters_files/figure-html/fit.models-30.png) ![](Michigan_Voters_files/figure-html/fit.models-31.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.0182  -0.8822  -0.8182   1.4308   1.6022  
## 
## Coefficients: (1 not defined because of singularities)
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -0.92116    0.04180 -22.035  < 2e-16 ***
## hawthorne    0.27539    0.10039   2.743 0.006085 ** 
## civicduty    0.18226    0.10075   1.809 0.070442 .  
## neighbors    0.50261    0.09676   5.194 2.05e-07 ***
## self         0.34502    0.09719   3.550 0.000385 ***
## control           NA         NA      NA       NA    
## .rnorm       0.01246    0.03084   0.404 0.686270    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 6238.2  on 4999  degrees of freedom
## Residual deviance: 6202.0  on 4994  degrees of freedom
## AIC: 6214
## 
## Number of Fisher Scoring iterations: 4
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Michigan_Voters_files/figure-html/fit.models-32.png) ![](Michigan_Voters_files/figure-html/fit.models-33.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                                  0
## 2           Y                                  0
##   voting.fctr.predict.Groups.X.glm.Y
## 1                               3420
## 2                               1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                                  0
## 2           Y                                  0
##   voting.fctr.predict.Groups.X.glm.Y
## 1                               3420
## 2                               1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                                  0
## 2           Y                                  0
##   voting.fctr.predict.Groups.X.glm.Y
## 1                               3420
## 2                               1580
##           Reference
## Prediction    N    Y
##          N 2010  800
##          Y 1410  780
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               2010
## 2           Y                                800
##   voting.fctr.predict.Groups.X.glm.Y
## 1                               1410
## 2                                780
##           Reference
## Prediction    N    Y
##          N 3372 1542
##          Y   48   38
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3372
## 2           Y                               1542
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                 48
## 2                                 38
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3420
## 2           Y                               1580
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3420
## 2           Y                               1580
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3420
## 2           Y                               1580
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3420
## 2           Y                               1580
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3420
## 2           Y                               1580
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3420
## 2           Y                               1580
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##    threshold    f.score
## 1        0.0 0.48024316
## 2        0.1 0.48024316
## 3        0.2 0.48024316
## 4        0.3 0.41379310
## 5        0.4 0.04561825
## 6        0.5 0.00000000
## 7        0.6 0.00000000
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.Groups.X.glm.Y
## 1           N                               3420
## 2           Y                               1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                                  0
## 2           Y                                  0
##   voting.fctr.predict.Groups.X.glm.Y
## 1                               3420
## 2                               1580
##          Prediction
## Reference    N    Y
##         N    0 3420
##         Y    0 1580
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3160000      0.0000000      0.3031240      0.3290911      0.6840000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Michigan_Voters_files/figure-html/fit.models-34.png) ![](Michigan_Voters_files/figure-html/fit.models-35.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                                  0
## 2           Y                                  0
##   voting.fctr.predict.Groups.X.glm.Y
## 1                               3421
## 2                               1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                                  0
## 2           Y                                  0
##   voting.fctr.predict.Groups.X.glm.Y
## 1                               3421
## 2                               1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                                  0
## 2           Y                                  0
##   voting.fctr.predict.Groups.X.glm.Y
## 1                               3421
## 2                               1579
##           Reference
## Prediction    N    Y
##          N 1976  858
##          Y 1445  721
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               1976
## 2           Y                                858
##   voting.fctr.predict.Groups.X.glm.Y
## 1                               1445
## 2                                721
##           Reference
## Prediction    N    Y
##          N 3372 1555
##          Y   49   24
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3372
## 2           Y                               1555
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                 49
## 2                                 24
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3421
## 2           Y                               1579
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3421
## 2           Y                               1579
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3421
## 2           Y                               1579
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3421
## 2           Y                               1579
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3421
## 2           Y                               1579
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                               3421
## 2           Y                               1579
##   voting.fctr.predict.Groups.X.glm.Y
## 1                                  0
## 2                                  0
##    threshold    f.score
## 1        0.0 0.48001216
## 2        0.1 0.48001216
## 3        0.2 0.48001216
## 4        0.3 0.38504673
## 5        0.4 0.02905569
## 6        0.5 0.00000000
## 7        0.6 0.00000000
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

![](Michigan_Voters_files/figure-html/fit.models-36.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.Groups.X.glm.Y
## 1           N                               3421
## 2           Y                               1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.Groups.X.glm.N
## 1           N                                  0
## 2           Y                                  0
##   voting.fctr.predict.Groups.X.glm.Y
## 1                               3421
## 2                               1579
##          Prediction
## Reference    N    Y
##         N    0 3421
##         Y    0 1579
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3158000      0.0000000      0.3029263      0.3288891      0.6842000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000 
##       model_id model_method
## 1 Groups.X.glm          glm
##                                                    feats max.nTuningRuns
## 1 hawthorne, civicduty, neighbors, self, control, .rnorm               1
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                       1.62                 0.147   0.5474284
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.2       0.4802432        0.6840001
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.303124             0.3290911             0   0.5239316
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2       0.4800122           0.3158
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.3029263             0.3288891             0    6214.036
##   max.AccuracySD.fit max.KappaSD.fit
## 1        0.000236992               0
```

```r
# GrpGndr.X
ret_lst <- myfit_mdl(model_id="GrpGndr.X", 
                        model_method=ifelse(glb_is_regression, "lm", 
                                        ifelse(glb_is_binomial, "glm", "rpart")),
    indep_vars_vctr=c("hawthorne","civicduty","neighbors","self","control",".rnorm", "sex"),
                         model_type=glb_model_type,                     
                        glb_rsp_var, glb_rsp_var_out,
                        fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                        n_cv_folds=glb_n_cv_folds, tune_models_df=NULL)
```

```
## [1] "fitting model: GrpGndr.X.glm"
## [1] "    indep_vars: hawthorne, civicduty, neighbors, self, control, .rnorm, sex"
## + Fold1: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Michigan_Voters_files/figure-html/fit.models-37.png) ![](Michigan_Voters_files/figure-html/fit.models-38.png) ![](Michigan_Voters_files/figure-html/fit.models-39.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.0358  -0.8723  -0.8101   1.4293   1.6211  
## 
## Coefficients: (1 not defined because of singularities)
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -0.88030    0.05079 -17.332  < 2e-16 ***
## hawthorne    0.27825    0.10043   2.770 0.005598 ** 
## civicduty    0.18328    0.10077   1.819 0.068937 .  
## neighbors    0.50514    0.09680   5.218 1.81e-07 ***
## self         0.34619    0.09721   3.561 0.000369 ***
## control           NA         NA      NA       NA    
## .rnorm       0.01275    0.03085   0.413 0.679331    
## sex         -0.08579    0.06114  -1.403 0.160530    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 6238.2  on 4999  degrees of freedom
## Residual deviance: 6200.1  on 4993  degrees of freedom
## AIC: 6214.1
## 
## Number of Fisher Scoring iterations: 4
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Michigan_Voters_files/figure-html/fit.models-40.png) ![](Michigan_Voters_files/figure-html/fit.models-41.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                3420
## 2                                1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                3420
## 2                                1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                3420
## 2                                1580
##           Reference
## Prediction    N    Y
##          N 2008  797
##          Y 1412  783
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                2008
## 2           Y                                 797
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                1412
## 2                                 783
##           Reference
## Prediction    N    Y
##          N 3252 1481
##          Y  168   99
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3252
## 2           Y                                1481
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                 168
## 2                                  99
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3420
## 2           Y                                1580
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##    threshold   f.score
## 1        0.0 0.4802432
## 2        0.1 0.4802432
## 3        0.2 0.4802432
## 4        0.3 0.4148344
## 5        0.4 0.1072009
## 6        0.5 0.0000000
## 7        0.6 0.0000000
## 8        0.7 0.0000000
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.Y
## 1           N                                3420
## 2           Y                                1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                3420
## 2                                1580
##          Prediction
## Reference    N    Y
##         N    0 3420
##         Y    0 1580
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3160000      0.0000000      0.3031240      0.3290911      0.6840000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Michigan_Voters_files/figure-html/fit.models-42.png) ![](Michigan_Voters_files/figure-html/fit.models-43.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                3421
## 2                                1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                3421
## 2                                1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                3421
## 2                                1579
##           Reference
## Prediction    N    Y
##          N 1969  856
##          Y 1452  723
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                1969
## 2           Y                                 856
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                1452
## 2                                 723
##           Reference
## Prediction    N    Y
##          N 3247 1476
##          Y  174  103
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3247
## 2           Y                                1476
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                 174
## 2                                 103
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                3421
## 2           Y                                1579
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                   0
## 2                                   0
##    threshold   f.score
## 1        0.0 0.4800122
## 2        0.1 0.4800122
## 3        0.2 0.4800122
## 4        0.3 0.3851891
## 5        0.4 0.1109914
## 6        0.5 0.0000000
## 7        0.6 0.0000000
## 8        0.7 0.0000000
## 9        0.8 0.0000000
## 10       0.9 0.0000000
## 11       1.0 0.0000000
```

![](Michigan_Voters_files/figure-html/fit.models-44.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.Y
## 1           N                                3421
## 2           Y                                1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.GrpGndr.X.glm.N
## 1           N                                   0
## 2           Y                                   0
##   voting.fctr.predict.GrpGndr.X.glm.Y
## 1                                3421
## 2                                1579
##          Prediction
## Reference    N    Y
##         N    0 3421
##         Y    0 1579
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3158000      0.0000000      0.3029263      0.3288891      0.6842000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000 
##        model_id model_method
## 1 GrpGndr.X.glm          glm
##                                                         feats
## 1 hawthorne, civicduty, neighbors, self, control, .rnorm, sex
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      1.408                 0.164
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.5519065                    0.2       0.4802432        0.6840001
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.303124             0.3290911             0   0.5237511
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2       0.4800122           0.3158
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.3029263             0.3288891             0    6214.066
##   max.AccuracySD.fit max.KappaSD.fit
## 1        0.000236992               0
```

```r
# User specified
for (method in glb_models_method_vctr) {
    print(sprintf("iterating over method:%s", method))

    # All X that is not user excluded
    indep_vars_vctr <- setdiff(names(glb_trnent_df), 
        union(glb_rsp_var, glb_exclude_vars_as_features))
    
    # easier to exclude features
#     indep_vars_vctr <- setdiff(names(glb_trnent_df), 
#         union(union(glb_rsp_var, glb_exclude_vars_as_features), 
#               c("<feat1_name>", "<feat2_name>")))
    
    # easier to include features
#     indep_vars_vctr <- c("<feat1_name>", "<feat2_name>")

    # User specified bivariate models
#     indep_vars_vctr_lst <- list()
#     for (feat in setdiff(names(glb_trnent_df), 
#                          union(glb_rsp_var, glb_exclude_vars_as_features)))
#         indep_vars_vctr_lst[["feat"]] <- feat

    # User specified combinatorial models
#     indep_vars_vctr_lst <- list()
#     combn_mtrx <- combn(c("<feat1_name>", "<feat2_name>", "<featn_name>"), 
#                           <num_feats_to_choose>)
#     for (combn_ix in 1:ncol(combn_mtrx))
#         #print(combn_mtrx[, combn_ix])
#         indep_vars_vctr_lst[[combn_ix]] <- combn_mtrx[, combn_ix]

#     glb_sel_mdl <- glb_sel_wlm_mdl <- ret_lst[["model"]]
#     rpart_sel_wlm_mdl <- rpart(reformulate(indep_vars_vctr, response=glb_rsp_var), 
#                                data=glb_trnent_df, method="class", 
#                                control=rpart.control(cp=glb_sel_wlm_mdl$bestTune$cp),
#                            parms=list(loss=glb_model_metric_terms))
#     print("rpart_sel_wlm_mdl"); prp(rpart_sel_wlm_mdl)
# 
    model_id_pfx <- "All.X";
    ret_lst <- myfit_mdl(model_id=paste0(model_id_pfx, ""), model_method=method,
                            indep_vars_vctr=indep_vars_vctr,
                            model_type=glb_model_type,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                            fit_df=glb_trnent_df, OOB_df=glb_newent_df,
                n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df)

    # Since caret does not optimize rpart well
    if (method == "rpart")
        ret_lst <- myfit_mdl(model_id=paste0(model_id_pfx, ".cp.0"), model_method=method,
                                indep_vars_vctr=indep_vars_vctr,
                                model_type=glb_model_type,
                                rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                                fit_df=glb_trnent_df, OOB_df=glb_newent_df,        
            n_cv_folds=0, tune_models_df=data.frame(parameter="cp", min=0.0, max=0.0, by=0.1))
    
    # rf is hard-coded in caret to recognize only Accuracy / Kappa evaluation metrics
    #   only for OOB in trainControl ?

#     ret_lst <- myfit_mdl_fn(model_id=paste0(model_id_pfx, ""), model_method=method,
#                             indep_vars_vctr=indep_vars_vctr,
#                             rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
#                             fit_df=glb_trnent_df, OOB_df=glb_newent_df,
#                             n_cv_folds=glb_n_cv_folds, tune_models_df=glb_tune_models_df,
#                             model_loss_mtrx=glb_model_metric_terms,
#                             model_summaryFunction=glb_model_metric_smmry,
#                             model_metric=glb_model_metric,
#                             model_metric_maximize=glb_model_metric_maximize)
}
```

```
## [1] "iterating over method:glm"
## [1] "fitting model: All.X.glm"
## [1] "    indep_vars: sex, yob, hawthorne, civicduty, neighbors, self, control, .rnorm"
## + Fold1: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold1: parameter=none 
## + Fold2: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold2: parameter=none 
## + Fold3: parameter=none
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
```

![](Michigan_Voters_files/figure-html/fit.models-45.png) ![](Michigan_Voters_files/figure-html/fit.models-46.png) ![](Michigan_Voters_files/figure-html/fit.models-47.png) 

```
## 
## Call:
## NULL
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.3037  -0.8876  -0.7861   1.3803   1.8429  
## 
## Coefficients: (1 not defined because of singularities)
##              Estimate Std. Error z value Pr(>|z|)    
## (Intercept) 32.941519   4.214821   7.816 5.47e-15 ***
## sex         -0.054481   0.061665  -0.884 0.376963    
## yob         -0.017305   0.002157  -8.022 1.04e-15 ***
## hawthorne    0.282158   0.101096   2.791 0.005255 ** 
## civicduty    0.195654   0.101455   1.928 0.053796 .  
## neighbors    0.510025   0.097554   5.228 1.71e-07 ***
## self         0.351723   0.097866   3.594 0.000326 ***
## control            NA         NA      NA       NA    
## .rnorm       0.018739   0.031075   0.603 0.546506    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 6238.2  on 4999  degrees of freedom
## Residual deviance: 6134.5  on 4992  degrees of freedom
## AIC: 6150.5
## 
## Number of Fisher Scoring iterations: 4
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Michigan_Voters_files/figure-html/fit.models-48.png) ![](Michigan_Voters_files/figure-html/fit.models-49.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                               0
## 2           Y                               0
##   voting.fctr.predict.All.X.glm.Y
## 1                            3420
## 2                            1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                               0
## 2           Y                               0
##   voting.fctr.predict.All.X.glm.Y
## 1                            3420
## 2                            1580
##           Reference
## Prediction    N    Y
##          N  141   18
##          Y 3279 1562
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                             141
## 2           Y                              18
##   voting.fctr.predict.All.X.glm.Y
## 1                            3279
## 2                            1562
##           Reference
## Prediction    N    Y
##          N 1676  572
##          Y 1744 1008
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            1676
## 2           Y                             572
##   voting.fctr.predict.All.X.glm.Y
## 1                            1744
## 2                            1008
##           Reference
## Prediction    N    Y
##          N 3083 1329
##          Y  337  251
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3083
## 2           Y                            1329
##   voting.fctr.predict.All.X.glm.Y
## 1                             337
## 2                             251
##           Reference
## Prediction    N    Y
##          N 3394 1562
##          Y   26   18
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3394
## 2           Y                            1562
##   voting.fctr.predict.All.X.glm.Y
## 1                              26
## 2                              18
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3420
## 2           Y                            1580
##   voting.fctr.predict.All.X.glm.Y
## 1                               0
## 2                               0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3420
## 2           Y                            1580
##   voting.fctr.predict.All.X.glm.Y
## 1                               0
## 2                               0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3420
## 2           Y                            1580
##   voting.fctr.predict.All.X.glm.Y
## 1                               0
## 2                               0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3420
## 2           Y                            1580
##   voting.fctr.predict.All.X.glm.Y
## 1                               0
## 2                               0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3420
## 2           Y                            1580
##   voting.fctr.predict.All.X.glm.Y
## 1                               0
## 2                               0
##    threshold    f.score
## 1        0.0 0.48024316
## 2        0.1 0.48024316
## 3        0.2 0.48652858
## 4        0.3 0.46537396
## 5        0.4 0.23154982
## 6        0.5 0.02216749
## 7        0.6 0.00000000
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                             141
## 2           Y                              18
##   voting.fctr.predict.All.X.glm.Y
## 1                            3279
## 2                            1562
##           Reference
## Prediction    N    Y
##          N  141   18
##          Y 3279 1562
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                             141
## 2           Y                              18
##   voting.fctr.predict.All.X.glm.Y
## 1                            3279
## 2                            1562
##          Prediction
## Reference    N    Y
##         N  141 3279
##         Y   18 1562
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##     0.34060000     0.01918436     0.32746032     0.35392603     0.68400000 
## AccuracyPValue  McnemarPValue 
##     1.00000000     0.00000000
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

![](Michigan_Voters_files/figure-html/fit.models-50.png) ![](Michigan_Voters_files/figure-html/fit.models-51.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                               0
## 2           Y                               0
##   voting.fctr.predict.All.X.glm.Y
## 1                            3421
## 2                            1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                               0
## 2           Y                               0
##   voting.fctr.predict.All.X.glm.Y
## 1                            3421
## 2                            1579
##           Reference
## Prediction    N    Y
##          N  153   24
##          Y 3268 1555
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                             153
## 2           Y                              24
##   voting.fctr.predict.All.X.glm.Y
## 1                            3268
## 2                            1555
##           Reference
## Prediction    N    Y
##          N 1689  579
##          Y 1732 1000
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            1689
## 2           Y                             579
##   voting.fctr.predict.All.X.glm.Y
## 1                            1732
## 2                            1000
##           Reference
## Prediction    N    Y
##          N 3058 1369
##          Y  363  210
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3058
## 2           Y                            1369
##   voting.fctr.predict.All.X.glm.Y
## 1                             363
## 2                             210
##           Reference
## Prediction    N    Y
##          N 3394 1563
##          Y   27   16
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3394
## 2           Y                            1563
##   voting.fctr.predict.All.X.glm.Y
## 1                              27
## 2                              16
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3421
## 2           Y                            1579
##   voting.fctr.predict.All.X.glm.Y
## 1                               0
## 2                               0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3421
## 2           Y                            1579
##   voting.fctr.predict.All.X.glm.Y
## 1                               0
## 2                               0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3421
## 2           Y                            1579
##   voting.fctr.predict.All.X.glm.Y
## 1                               0
## 2                               0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3421
## 2           Y                            1579
##   voting.fctr.predict.All.X.glm.Y
## 1                               0
## 2                               0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                            3421
## 2           Y                            1579
##   voting.fctr.predict.All.X.glm.Y
## 1                               0
## 2                               0
##    threshold    f.score
## 1        0.0 0.48001216
## 2        0.1 0.48001216
## 3        0.2 0.48578569
## 4        0.3 0.46392948
## 5        0.4 0.19516729
## 6        0.5 0.01972873
## 7        0.6 0.00000000
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

![](Michigan_Voters_files/figure-html/fit.models-52.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                             153
## 2           Y                              24
##   voting.fctr.predict.All.X.glm.Y
## 1                            3268
## 2                            1555
##           Reference
## Prediction    N    Y
##          N  153   24
##          Y 3268 1555
##   voting.fctr voting.fctr.predict.All.X.glm.N
## 1           N                             153
## 2           Y                              24
##   voting.fctr.predict.All.X.glm.Y
## 1                            3268
## 2                            1555
##          Prediction
## Reference    N    Y
##         N  153 3268
##         Y   24 1555
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##     0.34160000     0.01900987     0.32845044     0.35493473     0.68420000 
## AccuracyPValue  McnemarPValue 
##     1.00000000     0.00000000 
##    model_id model_method
## 1 All.X.glm          glm
##                                                              feats
## 1 sex, yob, hawthorne, civicduty, neighbors, self, control, .rnorm
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               1                      1.501                 0.184
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1   0.5877905                    0.2       0.4865286        0.6830003
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.3274603              0.353926   0.007655516   0.5819886
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2       0.4857857           0.3416
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB min.aic.fit
## 1             0.3284504             0.3549347    0.01900987    6150.502
##   max.AccuracySD.fit max.KappaSD.fit
## 1        0.001142904      0.01027105
## [1] "iterating over method:rpart"
## [1] "fitting model: All.X.rpart"
## [1] "    indep_vars: sex, yob, hawthorne, civicduty, neighbors, self, control"
## + Fold1: cp=0.001266 
## - Fold1: cp=0.001266 
## + Fold2: cp=0.001266 
## - Fold2: cp=0.001266 
## + Fold3: cp=0.001266 
## - Fold3: cp=0.001266 
## Aggregating results
## Selecting tuning parameters
## Fitting cp = 0.00127 on full training set
```

```
## Warning in myfit_mdl(model_id = paste0(model_id_pfx, ""), model_method =
## method, : model's bestTune found at an extreme of tuneGrid for parameter:
## cp
```

![](Michigan_Voters_files/figure-html/fit.models-53.png) ![](Michigan_Voters_files/figure-html/fit.models-54.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 5000 
## 
##            CP nsplit rel error
## 1 0.002278481      0 1.0000000
## 2 0.002215190     12 0.9696203
## 3 0.001265823     14 0.9651899
## 
## Variable importance
##       yob   control neighbors hawthorne      self civicduty       sex 
##        75         9         5         5         2         2         2 
## 
## Node number 1: 5000 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.316  P(node) =1
##     class counts:  3420  1580
##    probabilities: 0.684 0.316 
##   left son=2 (3443 obs) right son=3 (1557 obs)
##   Primary splits:
##       yob       < 1949.5 to the right, improve=27.306050, (0 missing)
##       control   < 0.5    to the right, improve=12.572450, (0 missing)
##       neighbors < 0.5    to the left,  improve= 8.109654, (0 missing)
##       self      < 0.5    to the left,  improve= 2.453532, (0 missing)
##       hawthorne < 0.5    to the left,  improve= 0.932498, (0 missing)
## 
## Node number 2: 3443 observations
##   predicted class=N  expected loss=0.2808597  P(node) =0.6886
##     class counts:  2476   967
##    probabilities: 0.719 0.281 
## 
## Node number 3: 1557 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.3937058  P(node) =0.3114
##     class counts:   944   613
##    probabilities: 0.606 0.394 
##   left son=6 (877 obs) right son=7 (680 obs)
##   Primary splits:
##       control   < 0.5    to the right, improve=6.136099, (0 missing)
##       yob       < 1933.5 to the left,  improve=3.107252, (0 missing)
##       neighbors < 0.5    to the left,  improve=2.777167, (0 missing)
##       civicduty < 0.5    to the left,  improve=1.726215, (0 missing)
##       self      < 0.5    to the left,  improve=0.783096, (0 missing)
##   Surrogate splits:
##       self      < 0.5    to the left,  agree=0.681, adj=0.269, (0 split)
##       hawthorne < 0.5    to the left,  agree=0.674, adj=0.254, (0 split)
##       neighbors < 0.5    to the left,  agree=0.672, adj=0.249, (0 split)
##       civicduty < 0.5    to the left,  agree=0.663, adj=0.228, (0 split)
## 
## Node number 6: 877 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.354618  P(node) =0.1754
##     class counts:   566   311
##    probabilities: 0.645 0.355 
##   left son=12 (323 obs) right son=13 (554 obs)
##   Primary splits:
##       yob < 1944.5 to the right, improve=2.682090, (0 missing)
##       sex < 0.5    to the right, improve=0.301366, (0 missing)
## 
## Node number 7: 680 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.4441176  P(node) =0.136
##     class counts:   378   302
##    probabilities: 0.556 0.444 
##   left son=14 (295 obs) right son=15 (385 obs)
##   Primary splits:
##       yob       < 1943.5 to the right, improve=1.45278700, (0 missing)
##       hawthorne < 0.5    to the right, improve=0.95119450, (0 missing)
##       neighbors < 0.5    to the left,  improve=0.55642430, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.16707330, (0 missing)
##       sex       < 0.5    to the right, improve=0.04815394, (0 missing)
##   Surrogate splits:
##       self < 0.5    to the right, agree=0.568, adj=0.003, (0 split)
## 
## Node number 12: 323 observations
##   predicted class=N  expected loss=0.3034056  P(node) =0.0646
##     class counts:   225    98
##    probabilities: 0.697 0.303 
## 
## Node number 13: 554 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.3844765  P(node) =0.1108
##     class counts:   341   213
##    probabilities: 0.616 0.384 
##   left son=26 (233 obs) right son=27 (321 obs)
##   Primary splits:
##       yob < 1935.5 to the left,  improve=6.2762110, (0 missing)
##       sex < 0.5    to the right, improve=0.9308898, (0 missing)
## 
## Node number 14: 295 observations,    complexity param=0.00221519
##   predicted class=N  expected loss=0.4067797  P(node) =0.059
##     class counts:   175   120
##    probabilities: 0.593 0.407 
##   left son=28 (224 obs) right son=29 (71 obs)
##   Primary splits:
##       neighbors < 0.5    to the left,  improve=1.87992400, (0 missing)
##       yob       < 1948.5 to the left,  improve=1.87724200, (0 missing)
##       self      < 0.5    to the right, improve=0.61822300, (0 missing)
##       hawthorne < 0.5    to the right, improve=0.25956460, (0 missing)
##       sex       < 0.5    to the left,  improve=0.05781034, (0 missing)
## 
## Node number 15: 385 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.4727273  P(node) =0.077
##     class counts:   203   182
##    probabilities: 0.527 0.473 
##   left son=30 (127 obs) right son=31 (258 obs)
##   Primary splits:
##       yob       < 1933.5 to the left,  improve=2.3671190, (0 missing)
##       hawthorne < 0.5    to the right, improve=0.6735522, (0 missing)
##       self      < 0.5    to the left,  improve=0.4563151, (0 missing)
##       sex       < 0.5    to the right, improve=0.1909091, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.1364486, (0 missing)
## 
## Node number 26: 233 observations
##   predicted class=N  expected loss=0.2961373  P(node) =0.0466
##     class counts:   164    69
##    probabilities: 0.704 0.296 
## 
## Node number 27: 321 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.4485981  P(node) =0.0642
##     class counts:   177   144
##    probabilities: 0.551 0.449 
##   left son=54 (280 obs) right son=55 (41 obs)
##   Primary splits:
##       yob < 1943.5 to the left,  improve=3.2364910, (0 missing)
##       sex < 0.5    to the right, improve=0.1013274, (0 missing)
## 
## Node number 28: 224 observations
##   predicted class=N  expected loss=0.375  P(node) =0.0448
##     class counts:   140    84
##    probabilities: 0.625 0.375 
## 
## Node number 29: 71 observations,    complexity param=0.00221519
##   predicted class=Y  expected loss=0.4929577  P(node) =0.0142
##     class counts:    35    36
##    probabilities: 0.493 0.507 
##   left son=58 (8 obs) right son=59 (63 obs)
##   Primary splits:
##       yob < 1944.5 to the left,  improve=2.631847000, (0 missing)
##       sex < 0.5    to the right, improve=0.005778259, (0 missing)
## 
## Node number 30: 127 observations
##   predicted class=N  expected loss=0.3937008  P(node) =0.0254
##     class counts:    77    50
##    probabilities: 0.606 0.394 
## 
## Node number 31: 258 observations,    complexity param=0.002278481
##   predicted class=Y  expected loss=0.4883721  P(node) =0.0516
##     class counts:   126   132
##    probabilities: 0.488 0.512 
##   left son=62 (68 obs) right son=63 (190 obs)
##   Primary splits:
##       hawthorne < 0.5    to the right, improve=1.8416880, (0 missing)
##       sex       < 0.5    to the right, improve=0.8871797, (0 missing)
##       self      < 0.5    to the left,  improve=0.7775005, (0 missing)
##       yob       < 1941.5 to the right, improve=0.4265493, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.1140375, (0 missing)
## 
## Node number 54: 280 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.4214286  P(node) =0.056
##     class counts:   162   118
##    probabilities: 0.579 0.421 
##   left son=108 (250 obs) right son=109 (30 obs)
##   Primary splits:
##       yob < 1936.5 to the right, improve=3.01752400, (0 missing)
##       sex < 0.5    to the right, improve=0.02630583, (0 missing)
## 
## Node number 55: 41 observations
##   predicted class=Y  expected loss=0.3658537  P(node) =0.0082
##     class counts:    15    26
##    probabilities: 0.366 0.634 
## 
## Node number 58: 8 observations
##   predicted class=N  expected loss=0.125  P(node) =0.0016
##     class counts:     7     1
##    probabilities: 0.875 0.125 
## 
## Node number 59: 63 observations
##   predicted class=Y  expected loss=0.4444444  P(node) =0.0126
##     class counts:    28    35
##    probabilities: 0.444 0.556 
## 
## Node number 62: 68 observations
##   predicted class=N  expected loss=0.4117647  P(node) =0.0136
##     class counts:    40    28
##    probabilities: 0.588 0.412 
## 
## Node number 63: 190 observations,    complexity param=0.002278481
##   predicted class=Y  expected loss=0.4526316  P(node) =0.038
##     class counts:    86   104
##    probabilities: 0.453 0.547 
##   left son=126 (79 obs) right son=127 (111 obs)
##   Primary splits:
##       sex       < 0.5    to the right, improve=1.19081700, (0 missing)
##       yob       < 1939.5 to the left,  improve=0.61245890, (0 missing)
##       self      < 0.5    to the left,  improve=0.20382000, (0 missing)
##       neighbors < 0.5    to the right, improve=0.08241823, (0 missing)
##       civicduty < 0.5    to the right, improve=0.02687495, (0 missing)
##   Surrogate splits:
##       yob       < 1935.5 to the left,  agree=0.605, adj=0.051, (0 split)
##       neighbors < 0.5    to the right, agree=0.589, adj=0.013, (0 split)
## 
## Node number 108: 250 observations
##   predicted class=N  expected loss=0.396  P(node) =0.05
##     class counts:   151    99
##    probabilities: 0.604 0.396 
## 
## Node number 109: 30 observations
##   predicted class=Y  expected loss=0.3666667  P(node) =0.006
##     class counts:    11    19
##    probabilities: 0.367 0.633 
## 
## Node number 126: 79 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.4810127  P(node) =0.0158
##     class counts:    41    38
##    probabilities: 0.519 0.481 
##   left son=252 (33 obs) right son=253 (46 obs)
##   Primary splits:
##       yob       < 1938.5 to the left,  improve=0.85937530, (0 missing)
##       self      < 0.5    to the left,  improve=0.45637130, (0 missing)
##       civicduty < 0.5    to the right, improve=0.28546220, (0 missing)
##       neighbors < 0.5    to the right, improve=0.01990872, (0 missing)
## 
## Node number 127: 111 observations
##   predicted class=Y  expected loss=0.4054054  P(node) =0.0222
##     class counts:    45    66
##    probabilities: 0.405 0.595 
## 
## Node number 252: 33 observations
##   predicted class=N  expected loss=0.3939394  P(node) =0.0066
##     class counts:    20    13
##    probabilities: 0.606 0.394 
## 
## Node number 253: 46 observations,    complexity param=0.002278481
##   predicted class=Y  expected loss=0.4565217  P(node) =0.0092
##     class counts:    21    25
##    probabilities: 0.457 0.543 
##   left son=506 (20 obs) right son=507 (26 obs)
##   Primary splits:
##       yob       < 1941.5 to the right, improve=1.456856000, (0 missing)
##       civicduty < 0.5    to the right, improve=0.061381070, (0 missing)
##       self      < 0.5    to the left,  improve=0.017753620, (0 missing)
##       neighbors < 0.5    to the left,  improve=0.008626639, (0 missing)
##   Surrogate splits:
##       civicduty < 0.5    to the right, agree=0.609, adj=0.1, (0 split)
## 
## Node number 506: 20 observations
##   predicted class=N  expected loss=0.4  P(node) =0.004
##     class counts:    12     8
##    probabilities: 0.600 0.400 
## 
## Node number 507: 26 observations
##   predicted class=Y  expected loss=0.3461538  P(node) =0.0052
##     class counts:     9    17
##    probabilities: 0.346 0.654 
## 
## n= 5000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##   1) root 5000 1580 N (0.6840000 0.3160000)  
##     2) yob>=1949.5 3443  967 N (0.7191403 0.2808597) *
##     3) yob< 1949.5 1557  613 N (0.6062942 0.3937058)  
##       6) control>=0.5 877  311 N (0.6453820 0.3546180)  
##        12) yob>=1944.5 323   98 N (0.6965944 0.3034056) *
##        13) yob< 1944.5 554  213 N (0.6155235 0.3844765)  
##          26) yob< 1935.5 233   69 N (0.7038627 0.2961373) *
##          27) yob>=1935.5 321  144 N (0.5514019 0.4485981)  
##            54) yob< 1943.5 280  118 N (0.5785714 0.4214286)  
##             108) yob>=1936.5 250   99 N (0.6040000 0.3960000) *
##             109) yob< 1936.5 30   11 Y (0.3666667 0.6333333) *
##            55) yob>=1943.5 41   15 Y (0.3658537 0.6341463) *
##       7) control< 0.5 680  302 N (0.5558824 0.4441176)  
##        14) yob>=1943.5 295  120 N (0.5932203 0.4067797)  
##          28) neighbors< 0.5 224   84 N (0.6250000 0.3750000) *
##          29) neighbors>=0.5 71   35 Y (0.4929577 0.5070423)  
##            58) yob< 1944.5 8    1 N (0.8750000 0.1250000) *
##            59) yob>=1944.5 63   28 Y (0.4444444 0.5555556) *
##        15) yob< 1943.5 385  182 N (0.5272727 0.4727273)  
##          30) yob< 1933.5 127   50 N (0.6062992 0.3937008) *
##          31) yob>=1933.5 258  126 Y (0.4883721 0.5116279)  
##            62) hawthorne>=0.5 68   28 N (0.5882353 0.4117647) *
##            63) hawthorne< 0.5 190   86 Y (0.4526316 0.5473684)  
##             126) sex>=0.5 79   38 N (0.5189873 0.4810127)  
##               252) yob< 1938.5 33   13 N (0.6060606 0.3939394) *
##               253) yob>=1938.5 46   21 Y (0.4565217 0.5434783)  
##                 506) yob>=1941.5 20    8 N (0.6000000 0.4000000) *
##                 507) yob< 1941.5 26    9 Y (0.3461538 0.6538462) *
##             127) sex< 0.5 111   45 Y (0.4054054 0.5945946) *
```

![](Michigan_Voters_files/figure-html/fit.models-55.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                                 0
## 2           Y                                 0
##   voting.fctr.predict.All.X.rpart.Y
## 1                              3420
## 2                              1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                                 0
## 2           Y                                 0
##   voting.fctr.predict.All.X.rpart.Y
## 1                              3420
## 2                              1580
##           Reference
## Prediction    N    Y
##          N    7    1
##          Y 3413 1579
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                                 7
## 2           Y                                 1
##   voting.fctr.predict.All.X.rpart.Y
## 1                              3413
## 2                              1579
##           Reference
## Prediction    N    Y
##          N 2647 1037
##          Y  773  543
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              2647
## 2           Y                              1037
##   voting.fctr.predict.All.X.rpart.Y
## 1                               773
## 2                               543
##           Reference
## Prediction    N    Y
##          N 3260 1381
##          Y  160  199
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3260
## 2           Y                              1381
##   voting.fctr.predict.All.X.rpart.Y
## 1                               160
## 2                               199
##           Reference
## Prediction    N    Y
##          N 3312 1417
##          Y  108  163
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3312
## 2           Y                              1417
##   voting.fctr.predict.All.X.rpart.Y
## 1                               108
## 2                               163
##           Reference
## Prediction    N    Y
##          N 3385 1518
##          Y   35   62
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3385
## 2           Y                              1518
##   voting.fctr.predict.All.X.rpart.Y
## 1                                35
## 2                                62
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3420
## 2           Y                              1580
##   voting.fctr.predict.All.X.rpart.Y
## 1                                 0
## 2                                 0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3420
## 2           Y                              1580
##   voting.fctr.predict.All.X.rpart.Y
## 1                                 0
## 2                                 0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3420
## 2           Y                              1580
##   voting.fctr.predict.All.X.rpart.Y
## 1                                 0
## 2                                 0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3420
## 2           Y                              1580
##   voting.fctr.predict.All.X.rpart.Y
## 1                                 0
## 2                                 0
##    threshold    f.score
## 1        0.0 0.48024316
## 2        0.1 0.48024316
## 3        0.2 0.48052343
## 4        0.3 0.37500000
## 5        0.4 0.20526044
## 6        0.5 0.17612102
## 7        0.6 0.07394156
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

![](Michigan_Voters_files/figure-html/fit.models-56.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                                 7
## 2           Y                                 1
##   voting.fctr.predict.All.X.rpart.Y
## 1                              3413
## 2                              1579
##           Reference
## Prediction    N    Y
##          N    7    1
##          Y 3413 1579
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                                 7
## 2           Y                                 1
##   voting.fctr.predict.All.X.rpart.Y
## 1                              3413
## 2                              1579
##          Prediction
## Reference    N    Y
##         N    7 3413
##         Y    1 1579
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   0.3172000000   0.0008943371   0.3043101853   0.3303035530   0.6840000000 
## AccuracyPValue  McnemarPValue 
##   1.0000000000   0.0000000000
```

![](Michigan_Voters_files/figure-html/fit.models-57.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                                 0
## 2           Y                                 0
##   voting.fctr.predict.All.X.rpart.Y
## 1                              3421
## 2                              1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                                 0
## 2           Y                                 0
##   voting.fctr.predict.All.X.rpart.Y
## 1                              3421
## 2                              1579
##           Reference
## Prediction    N    Y
##          N    3    2
##          Y 3418 1577
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                                 3
## 2           Y                                 2
##   voting.fctr.predict.All.X.rpart.Y
## 1                              3418
## 2                              1577
##           Reference
## Prediction    N    Y
##          N 2607 1066
##          Y  814  513
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              2607
## 2           Y                              1066
##   voting.fctr.predict.All.X.rpart.Y
## 1                               814
## 2                               513
##           Reference
## Prediction    N    Y
##          N 3211 1432
##          Y  210  147
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3211
## 2           Y                              1432
##   voting.fctr.predict.All.X.rpart.Y
## 1                               210
## 2                               147
##           Reference
## Prediction    N    Y
##          N 3262 1467
##          Y  159  112
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3262
## 2           Y                              1467
##   voting.fctr.predict.All.X.rpart.Y
## 1                               159
## 2                               112
##           Reference
## Prediction    N    Y
##          N 3360 1537
##          Y   61   42
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3360
## 2           Y                              1537
##   voting.fctr.predict.All.X.rpart.Y
## 1                                61
## 2                                42
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3421
## 2           Y                              1579
##   voting.fctr.predict.All.X.rpart.Y
## 1                                 0
## 2                                 0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3421
## 2           Y                              1579
##   voting.fctr.predict.All.X.rpart.Y
## 1                                 0
## 2                                 0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3421
## 2           Y                              1579
##   voting.fctr.predict.All.X.rpart.Y
## 1                                 0
## 2                                 0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                              3421
## 2           Y                              1579
##   voting.fctr.predict.All.X.rpart.Y
## 1                                 0
## 2                                 0
##    threshold    f.score
## 1        0.0 0.48001216
## 2        0.1 0.48001216
## 3        0.2 0.47976879
## 4        0.3 0.35306263
## 5        0.4 0.15185950
## 6        0.5 0.12108108
## 7        0.6 0.04994055
## 8        0.7 0.00000000
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

![](Michigan_Voters_files/figure-html/fit.models-58.png) 

```
## [1] "Classifier Probability Threshold: 0.1000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.All.X.rpart.Y
## 1           N                              3421
## 2           Y                              1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.All.X.rpart.N
## 1           N                                 0
## 2           Y                                 0
##   voting.fctr.predict.All.X.rpart.Y
## 1                              3421
## 2                              1579
##          Prediction
## Reference    N    Y
##         N    0 3421
##         Y    0 1579
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3158000      0.0000000      0.3029263      0.3288891      0.6842000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000 
##      model_id model_method
## 1 All.X.rpart        rpart
##                                                      feats max.nTuningRuns
## 1 sex, yob, hawthorne, civicduty, neighbors, self, control               3
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      1.669                 0.155   0.5689083
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.2       0.4805234        0.6799964
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.3043102             0.3303036    0.06844145   0.5498702
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.1       0.4800122           0.3158
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.3029263             0.3288891             0
##   max.AccuracySD.fit max.KappaSD.fit
## 1         0.01560146      0.01907617
## [1] "fitting model: All.X.cp.0.rpart"
## [1] "    indep_vars: sex, yob, hawthorne, civicduty, neighbors, self, control"
## Fitting cp = 0 on full training set
```

![](Michigan_Voters_files/figure-html/fit.models-59.png) 

```
## Call:
## rpart(formula = .outcome ~ ., control = list(minsplit = 20, minbucket = 7, 
##     cp = 0, maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, 
##     surrogatestyle = 0, maxdepth = 30, xval = 0))
##   n= 5000 
## 
##              CP nsplit rel error
## 1  0.0022784810      0 1.0000000
## 2  0.0022151899     12 0.9696203
## 3  0.0012658228     14 0.9651899
## 4  0.0009493671     16 0.9626582
## 5  0.0008438819     18 0.9607595
## 6  0.0007594937     21 0.9582278
## 7  0.0006329114     26 0.9544304
## 8  0.0005424955     33 0.9500000
## 9  0.0005274262     54 0.9373418
## 10 0.0003164557     60 0.9341772
## 11 0.0002373418     62 0.9335443
## 12 0.0002109705     70 0.9316456
## 13 0.0001582278     73 0.9310127
## 14 0.0001054852     85 0.9291139
## 15 0.0000000000     91 0.9284810
## 
## Variable importance
##       yob   control hawthorne       sex neighbors civicduty      self 
##        66         7         7         6         5         5         4 
## 
## Node number 1: 5000 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.316  P(node) =1
##     class counts:  3420  1580
##    probabilities: 0.684 0.316 
##   left son=2 (3443 obs) right son=3 (1557 obs)
##   Primary splits:
##       yob       < 1949.5 to the right, improve=27.306050, (0 missing)
##       control   < 0.5    to the right, improve=12.572450, (0 missing)
##       neighbors < 0.5    to the left,  improve= 8.109654, (0 missing)
##       self      < 0.5    to the left,  improve= 2.453532, (0 missing)
##       hawthorne < 0.5    to the left,  improve= 0.932498, (0 missing)
## 
## Node number 2: 3443 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.2808597  P(node) =0.6886
##     class counts:  2476   967
##    probabilities: 0.719 0.281 
##   left son=4 (394 obs) right son=5 (3049 obs)
##   Primary splits:
##       yob       < 1980.5 to the right, improve=14.135280, (0 missing)
##       control   < 0.5    to the right, improve= 6.854334, (0 missing)
##       neighbors < 0.5    to the left,  improve= 5.421958, (0 missing)
##       self      < 0.5    to the left,  improve= 1.509617, (0 missing)
##       hawthorne < 0.5    to the left,  improve= 1.078995, (0 missing)
## 
## Node number 3: 1557 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.3937058  P(node) =0.3114
##     class counts:   944   613
##    probabilities: 0.606 0.394 
##   left son=6 (877 obs) right son=7 (680 obs)
##   Primary splits:
##       control   < 0.5    to the right, improve=6.136099, (0 missing)
##       yob       < 1933.5 to the left,  improve=3.107252, (0 missing)
##       neighbors < 0.5    to the left,  improve=2.777167, (0 missing)
##       civicduty < 0.5    to the left,  improve=1.726215, (0 missing)
##       self      < 0.5    to the left,  improve=0.783096, (0 missing)
##   Surrogate splits:
##       self      < 0.5    to the left,  agree=0.681, adj=0.269, (0 split)
##       hawthorne < 0.5    to the left,  agree=0.674, adj=0.254, (0 split)
##       neighbors < 0.5    to the left,  agree=0.672, adj=0.249, (0 split)
##       civicduty < 0.5    to the left,  agree=0.663, adj=0.228, (0 split)
## 
## Node number 4: 394 observations
##   predicted class=N  expected loss=0.1548223  P(node) =0.0788
##     class counts:   333    61
##    probabilities: 0.845 0.155 
## 
## Node number 5: 3049 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.2971466  P(node) =0.6098
##     class counts:  2143   906
##    probabilities: 0.703 0.297 
##   left son=10 (1717 obs) right son=11 (1332 obs)
##   Primary splits:
##       control   < 0.5    to the right, improve=5.9403250, (0 missing)
##       neighbors < 0.5    to the left,  improve=5.4323020, (0 missing)
##       hawthorne < 0.5    to the left,  improve=1.2219550, (0 missing)
##       yob       < 1969.5 to the left,  improve=0.8249939, (0 missing)
##       self      < 0.5    to the left,  improve=0.5327204, (0 missing)
##   Surrogate splits:
##       civicduty < 0.5    to the left,  agree=0.675, adj=0.256, (0 split)
##       self      < 0.5    to the left,  agree=0.675, adj=0.255, (0 split)
##       neighbors < 0.5    to the left,  agree=0.672, adj=0.249, (0 split)
##       hawthorne < 0.5    to the left,  agree=0.668, adj=0.239, (0 split)
## 
## Node number 6: 877 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.354618  P(node) =0.1754
##     class counts:   566   311
##    probabilities: 0.645 0.355 
##   left son=12 (323 obs) right son=13 (554 obs)
##   Primary splits:
##       yob < 1944.5 to the right, improve=2.682090, (0 missing)
##       sex < 0.5    to the right, improve=0.301366, (0 missing)
## 
## Node number 7: 680 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.4441176  P(node) =0.136
##     class counts:   378   302
##    probabilities: 0.556 0.444 
##   left son=14 (295 obs) right son=15 (385 obs)
##   Primary splits:
##       yob       < 1943.5 to the right, improve=1.45278700, (0 missing)
##       hawthorne < 0.5    to the right, improve=0.95119450, (0 missing)
##       neighbors < 0.5    to the left,  improve=0.55642430, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.16707330, (0 missing)
##       sex       < 0.5    to the right, improve=0.04815394, (0 missing)
##   Surrogate splits:
##       self < 0.5    to the right, agree=0.568, adj=0.003, (0 split)
## 
## Node number 10: 1717 observations,    complexity param=0.0005274262
##   predicted class=N  expected loss=0.2696564  P(node) =0.3434
##     class counts:  1254   463
##    probabilities: 0.730 0.270 
##   left son=20 (1497 obs) right son=21 (220 obs)
##   Primary splits:
##       yob < 1969.5 to the left,  improve=0.61429950, (0 missing)
##       sex < 0.5    to the right, improve=0.03152786, (0 missing)
## 
## Node number 11: 1332 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.3325826  P(node) =0.2664
##     class counts:   889   443
##    probabilities: 0.667 0.333 
##   left son=22 (1000 obs) right son=23 (332 obs)
##   Primary splits:
##       neighbors < 0.5    to the left,  improve=2.20648200, (0 missing)
##       civicduty < 0.5    to the right, improve=1.87218500, (0 missing)
##       yob       < 1954.5 to the right, improve=1.72631200, (0 missing)
##       sex       < 0.5    to the right, improve=0.20300450, (0 missing)
##       self      < 0.5    to the right, improve=0.07483468, (0 missing)
## 
## Node number 12: 323 observations
##   predicted class=N  expected loss=0.3034056  P(node) =0.0646
##     class counts:   225    98
##    probabilities: 0.697 0.303 
## 
## Node number 13: 554 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.3844765  P(node) =0.1108
##     class counts:   341   213
##    probabilities: 0.616 0.384 
##   left son=26 (233 obs) right son=27 (321 obs)
##   Primary splits:
##       yob < 1935.5 to the left,  improve=6.2762110, (0 missing)
##       sex < 0.5    to the right, improve=0.9308898, (0 missing)
## 
## Node number 14: 295 observations,    complexity param=0.00221519
##   predicted class=N  expected loss=0.4067797  P(node) =0.059
##     class counts:   175   120
##    probabilities: 0.593 0.407 
##   left son=28 (224 obs) right son=29 (71 obs)
##   Primary splits:
##       neighbors < 0.5    to the left,  improve=1.87992400, (0 missing)
##       yob       < 1948.5 to the left,  improve=1.87724200, (0 missing)
##       self      < 0.5    to the right, improve=0.61822300, (0 missing)
##       hawthorne < 0.5    to the right, improve=0.25956460, (0 missing)
##       sex       < 0.5    to the left,  improve=0.05781034, (0 missing)
## 
## Node number 15: 385 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.4727273  P(node) =0.077
##     class counts:   203   182
##    probabilities: 0.527 0.473 
##   left son=30 (127 obs) right son=31 (258 obs)
##   Primary splits:
##       yob       < 1933.5 to the left,  improve=2.3671190, (0 missing)
##       hawthorne < 0.5    to the right, improve=0.6735522, (0 missing)
##       self      < 0.5    to the left,  improve=0.4563151, (0 missing)
##       sex       < 0.5    to the right, improve=0.1909091, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.1364486, (0 missing)
## 
## Node number 20: 1497 observations
##   predicted class=N  expected loss=0.2645291  P(node) =0.2994
##     class counts:  1101   396
##    probabilities: 0.735 0.265 
## 
## Node number 21: 220 observations,    complexity param=0.0005274262
##   predicted class=N  expected loss=0.3045455  P(node) =0.044
##     class counts:   153    67
##    probabilities: 0.695 0.305 
##   left son=42 (91 obs) right son=43 (129 obs)
##   Primary splits:
##       yob < 1974.5 to the right, improve=1.6894180, (0 missing)
##       sex < 0.5    to the right, improve=0.2528956, (0 missing)
## 
## Node number 22: 1000 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.316  P(node) =0.2
##     class counts:   684   316
##    probabilities: 0.684 0.316 
##   left son=44 (763 obs) right son=45 (237 obs)
##   Primary splits:
##       yob       < 1954.5 to the right, improve=1.62144400, (0 missing)
##       sex       < 0.5    to the right, improve=1.19708200, (0 missing)
##       civicduty < 0.5    to the right, improve=0.84709830, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.47673220, (0 missing)
##       self      < 0.5    to the left,  improve=0.05840998, (0 missing)
## 
## Node number 23: 332 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.3825301  P(node) =0.0664
##     class counts:   205   127
##    probabilities: 0.617 0.383 
##   left son=46 (213 obs) right son=47 (119 obs)
##   Primary splits:
##       yob < 1962.5 to the left,  improve=1.4652740, (0 missing)
##       sex < 0.5    to the left,  improve=0.9162968, (0 missing)
## 
## Node number 26: 233 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.2961373  P(node) =0.0466
##     class counts:   164    69
##    probabilities: 0.704 0.296 
##   left son=52 (32 obs) right son=53 (201 obs)
##   Primary splits:
##       yob < 1921.5 to the left,  improve=1.4517660, (0 missing)
##       sex < 0.5    to the right, improve=0.9932914, (0 missing)
## 
## Node number 27: 321 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.4485981  P(node) =0.0642
##     class counts:   177   144
##    probabilities: 0.551 0.449 
##   left son=54 (280 obs) right son=55 (41 obs)
##   Primary splits:
##       yob < 1943.5 to the left,  improve=3.2364910, (0 missing)
##       sex < 0.5    to the right, improve=0.1013274, (0 missing)
## 
## Node number 28: 224 observations,    complexity param=0.0007594937
##   predicted class=N  expected loss=0.375  P(node) =0.0448
##     class counts:   140    84
##    probabilities: 0.625 0.375 
##   left son=56 (177 obs) right son=57 (47 obs)
##   Primary splits:
##       yob       < 1948.5 to the left,  improve=1.030773000, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.149455300, (0 missing)
##       sex       < 0.5    to the left,  improve=0.125808800, (0 missing)
##       self      < 0.5    to the right, improve=0.083003950, (0 missing)
##       hawthorne < 0.5    to the right, improve=0.002458728, (0 missing)
## 
## Node number 29: 71 observations,    complexity param=0.00221519
##   predicted class=Y  expected loss=0.4929577  P(node) =0.0142
##     class counts:    35    36
##    probabilities: 0.493 0.507 
##   left son=58 (8 obs) right son=59 (63 obs)
##   Primary splits:
##       yob < 1944.5 to the left,  improve=2.631847000, (0 missing)
##       sex < 0.5    to the right, improve=0.005778259, (0 missing)
## 
## Node number 30: 127 observations,    complexity param=0.001265823
##   predicted class=N  expected loss=0.3937008  P(node) =0.0254
##     class counts:    77    50
##    probabilities: 0.606 0.394 
##   left son=60 (113 obs) right son=61 (14 obs)
##   Primary splits:
##       yob       < 1919.5 to the right, improve=0.99401730, (0 missing)
##       sex       < 0.5    to the left,  improve=0.29488810, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.17658790, (0 missing)
##       neighbors < 0.5    to the right, improve=0.13574880, (0 missing)
##       self      < 0.5    to the right, improve=0.01556517, (0 missing)
## 
## Node number 31: 258 observations,    complexity param=0.002278481
##   predicted class=Y  expected loss=0.4883721  P(node) =0.0516
##     class counts:   126   132
##    probabilities: 0.488 0.512 
##   left son=62 (68 obs) right son=63 (190 obs)
##   Primary splits:
##       hawthorne < 0.5    to the right, improve=1.8416880, (0 missing)
##       sex       < 0.5    to the right, improve=0.8871797, (0 missing)
##       self      < 0.5    to the left,  improve=0.7775005, (0 missing)
##       yob       < 1941.5 to the right, improve=0.4265493, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.1140375, (0 missing)
## 
## Node number 42: 91 observations
##   predicted class=N  expected loss=0.2307692  P(node) =0.0182
##     class counts:    70    21
##    probabilities: 0.769 0.231 
## 
## Node number 43: 129 observations,    complexity param=0.0005274262
##   predicted class=N  expected loss=0.3565891  P(node) =0.0258
##     class counts:    83    46
##    probabilities: 0.643 0.357 
##   left son=86 (70 obs) right son=87 (59 obs)
##   Primary splits:
##       sex < 0.5    to the right, improve=0.5477936, (0 missing)
##       yob < 1971.5 to the left,  improve=0.4326191, (0 missing)
##   Surrogate splits:
##       yob < 1972.5 to the left,  agree=0.566, adj=0.051, (0 split)
## 
## Node number 44: 763 observations,    complexity param=0.0002373418
##   predicted class=N  expected loss=0.3001311  P(node) =0.1526
##     class counts:   534   229
##    probabilities: 0.700 0.300 
##   left son=88 (308 obs) right son=89 (455 obs)
##   Primary splits:
##       yob       < 1960.5 to the left,  improve=1.4251890, (0 missing)
##       sex       < 0.5    to the right, improve=0.9415617, (0 missing)
##       civicduty < 0.5    to the right, improve=0.7625635, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.1998662, (0 missing)
##       self      < 0.5    to the left,  improve=0.1956043, (0 missing)
## 
## Node number 45: 237 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.3670886  P(node) =0.0474
##     class counts:   150    87
##    probabilities: 0.633 0.367 
##   left son=90 (174 obs) right son=91 (63 obs)
##   Primary splits:
##       yob       < 1953.5 to the left,  improve=1.49166200, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.37257770, (0 missing)
##       sex       < 0.5    to the right, improve=0.18113290, (0 missing)
##       self      < 0.5    to the right, improve=0.13623750, (0 missing)
##       civicduty < 0.5    to the right, improve=0.05329427, (0 missing)
## 
## Node number 46: 213 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.3474178  P(node) =0.0426
##     class counts:   139    74
##    probabilities: 0.653 0.347 
##   left son=92 (124 obs) right son=93 (89 obs)
##   Primary splits:
##       yob < 1954.5 to the right, improve=0.9960777, (0 missing)
##       sex < 0.5    to the left,  improve=0.5278233, (0 missing)
## 
## Node number 47: 119 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.4453782  P(node) =0.0238
##     class counts:    66    53
##    probabilities: 0.555 0.445 
##   left son=94 (52 obs) right son=95 (67 obs)
##   Primary splits:
##       sex < 0.5    to the left,  improve=0.3186186, (0 missing)
##       yob < 1967.5 to the right, improve=0.3162020, (0 missing)
##   Surrogate splits:
##       yob < 1976.5 to the right, agree=0.605, adj=0.096, (0 split)
## 
## Node number 52: 32 observations
##   predicted class=N  expected loss=0.15625  P(node) =0.0064
##     class counts:    27     5
##    probabilities: 0.844 0.156 
## 
## Node number 53: 201 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.318408  P(node) =0.0402
##     class counts:   137    64
##    probabilities: 0.682 0.318 
##   left son=106 (16 obs) right son=107 (185 obs)
##   Primary splits:
##       yob < 1934.5 to the right, improve=0.5958081, (0 missing)
##       sex < 0.5    to the right, improve=0.5008059, (0 missing)
## 
## Node number 54: 280 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.4214286  P(node) =0.056
##     class counts:   162   118
##    probabilities: 0.579 0.421 
##   left son=108 (250 obs) right son=109 (30 obs)
##   Primary splits:
##       yob < 1936.5 to the right, improve=3.01752400, (0 missing)
##       sex < 0.5    to the right, improve=0.02630583, (0 missing)
## 
## Node number 55: 41 observations
##   predicted class=Y  expected loss=0.3658537  P(node) =0.0082
##     class counts:    15    26
##    probabilities: 0.366 0.634 
## 
## Node number 56: 177 observations,    complexity param=0.0007594937
##   predicted class=N  expected loss=0.3502825  P(node) =0.0354
##     class counts:   115    62
##    probabilities: 0.650 0.350 
##   left son=112 (89 obs) right son=113 (88 obs)
##   Primary splits:
##       sex       < 0.5    to the left,  improve=0.787903300, (0 missing)
##       self      < 0.5    to the right, improve=0.230844800, (0 missing)
##       yob       < 1947.5 to the left,  improve=0.191249700, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.184019400, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.005183392, (0 missing)
##   Surrogate splits:
##       yob       < 1946.5 to the right, agree=0.576, adj=0.148, (0 split)
##       hawthorne < 0.5    to the left,  agree=0.508, adj=0.011, (0 split)
##       civicduty < 0.5    to the right, agree=0.508, adj=0.011, (0 split)
## 
## Node number 57: 47 observations,    complexity param=0.0007594937
##   predicted class=N  expected loss=0.4680851  P(node) =0.0094
##     class counts:    25    22
##    probabilities: 0.532 0.468 
##   left son=114 (15 obs) right son=115 (32 obs)
##   Primary splits:
##       hawthorne < 0.5    to the right, improve=0.80008870, (0 missing)
##       sex       < 0.5    to the right, improve=0.63357860, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.42806480, (0 missing)
##       self      < 0.5    to the left,  improve=0.07092199, (0 missing)
##   Surrogate splits:
##       self < 0.5    to the left,  agree=0.745, adj=0.2, (0 split)
## 
## Node number 58: 8 observations
##   predicted class=N  expected loss=0.125  P(node) =0.0016
##     class counts:     7     1
##    probabilities: 0.875 0.125 
## 
## Node number 59: 63 observations,    complexity param=0.001265823
##   predicted class=Y  expected loss=0.4444444  P(node) =0.0126
##     class counts:    28    35
##    probabilities: 0.444 0.556 
##   left son=118 (10 obs) right son=119 (53 obs)
##   Primary splits:
##       yob < 1945.5 to the left,  improve=0.57526210, (0 missing)
##       sex < 0.5    to the right, improve=0.05656566, (0 missing)
## 
## Node number 60: 113 observations,    complexity param=0.0006329114
##   predicted class=N  expected loss=0.3716814  P(node) =0.0226
##     class counts:    71    42
##    probabilities: 0.628 0.372 
##   left son=120 (13 obs) right son=121 (100 obs)
##   Primary splits:
##       yob       < 1922.5 to the left,  improve=0.58337640, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.29967020, (0 missing)
##       neighbors < 0.5    to the right, improve=0.06965612, (0 missing)
##       self      < 0.5    to the right, improve=0.04401305, (0 missing)
##       sex       < 0.5    to the left,  improve=0.02054056, (0 missing)
## 
## Node number 61: 14 observations
##   predicted class=Y  expected loss=0.4285714  P(node) =0.0028
##     class counts:     6     8
##    probabilities: 0.429 0.571 
## 
## Node number 62: 68 observations,    complexity param=0.0009493671
##   predicted class=N  expected loss=0.4117647  P(node) =0.0136
##     class counts:    40    28
##    probabilities: 0.588 0.412 
##   left son=124 (29 obs) right son=125 (39 obs)
##   Primary splits:
##       yob < 1940.5 to the right, improve=1.04020400, (0 missing)
##       sex < 0.5    to the left,  improve=0.01996435, (0 missing)
## 
## Node number 63: 190 observations,    complexity param=0.002278481
##   predicted class=Y  expected loss=0.4526316  P(node) =0.038
##     class counts:    86   104
##    probabilities: 0.453 0.547 
##   left son=126 (79 obs) right son=127 (111 obs)
##   Primary splits:
##       sex       < 0.5    to the right, improve=1.19081700, (0 missing)
##       yob       < 1939.5 to the left,  improve=0.61245890, (0 missing)
##       self      < 0.5    to the left,  improve=0.20382000, (0 missing)
##       neighbors < 0.5    to the right, improve=0.08241823, (0 missing)
##       civicduty < 0.5    to the right, improve=0.02687495, (0 missing)
##   Surrogate splits:
##       yob       < 1935.5 to the left,  agree=0.605, adj=0.051, (0 split)
##       neighbors < 0.5    to the right, agree=0.589, adj=0.013, (0 split)
## 
## Node number 86: 70 observations,    complexity param=0.0005274262
##   predicted class=N  expected loss=0.3142857  P(node) =0.014
##     class counts:    48    22
##    probabilities: 0.686 0.314 
##   left son=172 (19 obs) right son=173 (51 obs)
##   Primary splits:
##       yob < 1972.5 to the right, improve=2.278756, (0 missing)
## 
## Node number 87: 59 observations,    complexity param=0.0005274262
##   predicted class=N  expected loss=0.4067797  P(node) =0.0118
##     class counts:    35    24
##    probabilities: 0.593 0.407 
##   left son=174 (49 obs) right son=175 (10 obs)
##   Primary splits:
##       yob < 1973.5 to the left,  improve=0.8990661, (0 missing)
## 
## Node number 88: 308 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.262987  P(node) =0.0616
##     class counts:   227    81
##    probabilities: 0.737 0.263 
##   left son=176 (151 obs) right son=177 (157 obs)
##   Primary splits:
##       sex       < 0.5    to the right, improve=1.1702630, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.6356138, (0 missing)
##       civicduty < 0.5    to the right, improve=0.3916738, (0 missing)
##       yob       < 1957.5 to the left,  improve=0.2905138, (0 missing)
##       self      < 0.5    to the right, improve=0.0178726, (0 missing)
##   Surrogate splits:
##       yob       < 1956.5 to the right, agree=0.575, adj=0.132, (0 split)
##       self      < 0.5    to the right, agree=0.536, adj=0.053, (0 split)
##       hawthorne < 0.5    to the left,  agree=0.513, adj=0.007, (0 split)
##       civicduty < 0.5    to the left,  agree=0.513, adj=0.007, (0 split)
## 
## Node number 89: 455 observations,    complexity param=0.0002373418
##   predicted class=N  expected loss=0.3252747  P(node) =0.091
##     class counts:   307   148
##    probabilities: 0.675 0.325 
##   left son=178 (300 obs) right son=179 (155 obs)
##   Primary splits:
##       self      < 0.5    to the left,  improve=0.41093940, (0 missing)
##       civicduty < 0.5    to the right, improve=0.28589440, (0 missing)
##       sex       < 0.5    to the right, improve=0.23119950, (0 missing)
##       yob       < 1978.5 to the right, improve=0.20096800, (0 missing)
##       hawthorne < 0.5    to the right, improve=0.01245181, (0 missing)
##   Surrogate splits:
##       hawthorne < 0.5    to the right, agree=0.67, adj=0.032, (0 split)
##       civicduty < 0.5    to the right, agree=0.67, adj=0.032, (0 split)
## 
## Node number 90: 174 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.3333333  P(node) =0.0348
##     class counts:   116    58
##    probabilities: 0.667 0.333 
##   left son=180 (99 obs) right son=181 (75 obs)
##   Primary splits:
##       yob       < 1951.5 to the right, improve=0.74989900, (0 missing)
##       self      < 0.5    to the right, improve=0.53935930, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.37810030, (0 missing)
##       sex       < 0.5    to the right, improve=0.25035240, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.02437999, (0 missing)
##   Surrogate splits:
##       civicduty < 0.5    to the left,  agree=0.592, adj=0.053, (0 split)
## 
## Node number 91: 63 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.4603175  P(node) =0.0126
##     class counts:    34    29
##    probabilities: 0.540 0.460 
##   left son=182 (22 obs) right son=183 (41 obs)
##   Primary splits:
##       civicduty < 0.5    to the right, improve=0.631964200, (0 missing)
##       self      < 0.5    to the left,  improve=0.471354700, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.015873020, (0 missing)
##       sex       < 0.5    to the left,  improve=0.004617605, (0 missing)
##   Surrogate splits:
##       hawthorne < 0.5    to the left,  agree=0.683, adj=0.091, (0 split)
##       self      < 0.5    to the left,  agree=0.667, adj=0.045, (0 split)
## 
## Node number 92: 124 observations
##   predicted class=N  expected loss=0.3064516  P(node) =0.0248
##     class counts:    86    38
##    probabilities: 0.694 0.306 
## 
## Node number 93: 89 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.4044944  P(node) =0.0178
##     class counts:    53    36
##    probabilities: 0.596 0.404 
##   left son=186 (34 obs) right son=187 (55 obs)
##   Primary splits:
##       yob < 1951.5 to the left,  improve=0.29244730, (0 missing)
##       sex < 0.5    to the left,  improve=0.05721258, (0 missing)
## 
## Node number 94: 52 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.4038462  P(node) =0.0104
##     class counts:    31    21
##    probabilities: 0.596 0.404 
##   left son=188 (44 obs) right son=189 (8 obs)
##   Primary splits:
##       yob < 1964.5 to the right, improve=0.9248252, (0 missing)
## 
## Node number 95: 67 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.4776119  P(node) =0.0134
##     class counts:    35    32
##    probabilities: 0.522 0.478 
##   left son=190 (57 obs) right son=191 (10 obs)
##   Primary splits:
##       yob < 1975.5 to the left,  improve=1.16266, (0 missing)
## 
## Node number 106: 16 observations
##   predicted class=N  expected loss=0.1875  P(node) =0.0032
##     class counts:    13     3
##    probabilities: 0.812 0.187 
## 
## Node number 107: 185 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.3297297  P(node) =0.037
##     class counts:   124    61
##    probabilities: 0.670 0.330 
##   left son=214 (73 obs) right son=215 (112 obs)
##   Primary splits:
##       sex < 0.5    to the right, improve=0.7497342, (0 missing)
##       yob < 1933.5 to the left,  improve=0.3815056, (0 missing)
##   Surrogate splits:
##       yob < 1931.5 to the right, agree=0.622, adj=0.041, (0 split)
## 
## Node number 108: 250 observations
##   predicted class=N  expected loss=0.396  P(node) =0.05
##     class counts:   151    99
##    probabilities: 0.604 0.396 
## 
## Node number 109: 30 observations
##   predicted class=Y  expected loss=0.3666667  P(node) =0.006
##     class counts:    11    19
##    probabilities: 0.367 0.633 
## 
## Node number 112: 89 observations
##   predicted class=N  expected loss=0.3033708  P(node) =0.0178
##     class counts:    62    27
##    probabilities: 0.697 0.303 
## 
## Node number 113: 88 observations,    complexity param=0.0007594937
##   predicted class=N  expected loss=0.3977273  P(node) =0.0176
##     class counts:    53    35
##    probabilities: 0.602 0.398 
##   left son=226 (54 obs) right son=227 (34 obs)
##   Primary splits:
##       yob       < 1945.5 to the right, improve=0.58828480, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.15909090, (0 missing)
##       civicduty < 0.5    to the right, improve=0.11791440, (0 missing)
##       self      < 0.5    to the right, improve=0.00951826, (0 missing)
## 
## Node number 114: 15 observations
##   predicted class=N  expected loss=0.3333333  P(node) =0.003
##     class counts:    10     5
##    probabilities: 0.667 0.333 
## 
## Node number 115: 32 observations,    complexity param=0.0006329114
##   predicted class=Y  expected loss=0.46875  P(node) =0.0064
##     class counts:    15    17
##    probabilities: 0.469 0.531 
##   left son=230 (13 obs) right son=231 (19 obs)
##   Primary splits:
##       sex       < 0.5    to the right, improve=0.2128036, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.1041667, (0 missing)
##       self      < 0.5    to the right, improve=0.1041667, (0 missing)
## 
## Node number 118: 10 observations
##   predicted class=N  expected loss=0.4  P(node) =0.002
##     class counts:     6     4
##    probabilities: 0.600 0.400 
## 
## Node number 119: 53 observations,    complexity param=0.0008438819
##   predicted class=Y  expected loss=0.4150943  P(node) =0.0106
##     class counts:    22    31
##    probabilities: 0.415 0.585 
##   left son=238 (41 obs) right son=239 (12 obs)
##   Primary splits:
##       yob < 1946.5 to the right, improve=0.84560520, (0 missing)
##       sex < 0.5    to the left,  improve=0.08108715, (0 missing)
## 
## Node number 120: 13 observations
##   predicted class=N  expected loss=0.2307692  P(node) =0.0026
##     class counts:    10     3
##    probabilities: 0.769 0.231 
## 
## Node number 121: 100 observations,    complexity param=0.0006329114
##   predicted class=N  expected loss=0.39  P(node) =0.02
##     class counts:    61    39
##    probabilities: 0.610 0.390 
##   left son=242 (84 obs) right son=243 (16 obs)
##   Primary splits:
##       yob       < 1925.5 to the right, improve=1.13357100, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.29491230, (0 missing)
##       self      < 0.5    to the right, improve=0.17071730, (0 missing)
##       neighbors < 0.5    to the right, improve=0.04666667, (0 missing)
##       sex       < 0.5    to the left,  improve=0.01636364, (0 missing)
## 
## Node number 124: 29 observations
##   predicted class=N  expected loss=0.3103448  P(node) =0.0058
##     class counts:    20     9
##    probabilities: 0.690 0.310 
## 
## Node number 125: 39 observations,    complexity param=0.0009493671
##   predicted class=N  expected loss=0.4871795  P(node) =0.0078
##     class counts:    20    19
##    probabilities: 0.513 0.487 
##   left son=250 (8 obs) right son=251 (31 obs)
##   Primary splits:
##       yob < 1935.5 to the left,  improve=1.13234100, (0 missing)
##       sex < 0.5    to the right, improve=0.01349528, (0 missing)
## 
## Node number 126: 79 observations,    complexity param=0.002278481
##   predicted class=N  expected loss=0.4810127  P(node) =0.0158
##     class counts:    41    38
##    probabilities: 0.519 0.481 
##   left son=252 (33 obs) right son=253 (46 obs)
##   Primary splits:
##       yob       < 1938.5 to the left,  improve=0.85937530, (0 missing)
##       self      < 0.5    to the left,  improve=0.45637130, (0 missing)
##       civicduty < 0.5    to the right, improve=0.28546220, (0 missing)
##       neighbors < 0.5    to the right, improve=0.01990872, (0 missing)
## 
## Node number 127: 111 observations,    complexity param=0.0006329114
##   predicted class=Y  expected loss=0.4054054  P(node) =0.0222
##     class counts:    45    66
##    probabilities: 0.405 0.595 
##   left son=254 (90 obs) right son=255 (21 obs)
##   Primary splits:
##       yob       < 1936.5 to the right, improve=0.742084900, (0 missing)
##       neighbors < 0.5    to the right, improve=0.005523606, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.004422604, (0 missing)
## 
## Node number 172: 19 observations
##   predicted class=N  expected loss=0.1052632  P(node) =0.0038
##     class counts:    17     2
##    probabilities: 0.895 0.105 
## 
## Node number 173: 51 observations,    complexity param=0.0005274262
##   predicted class=N  expected loss=0.3921569  P(node) =0.0102
##     class counts:    31    20
##    probabilities: 0.608 0.392 
##   left son=346 (32 obs) right son=347 (19 obs)
##   Primary splits:
##       yob < 1971.5 to the left,  improve=2.113068, (0 missing)
## 
## Node number 174: 49 observations
##   predicted class=N  expected loss=0.3673469  P(node) =0.0098
##     class counts:    31    18
##    probabilities: 0.633 0.367 
## 
## Node number 175: 10 observations
##   predicted class=Y  expected loss=0.4  P(node) =0.002
##     class counts:     4     6
##    probabilities: 0.400 0.600 
## 
## Node number 176: 151 observations
##   predicted class=N  expected loss=0.218543  P(node) =0.0302
##     class counts:   118    33
##    probabilities: 0.781 0.219 
## 
## Node number 177: 157 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.3057325  P(node) =0.0314
##     class counts:   109    48
##    probabilities: 0.694 0.306 
##   left son=354 (93 obs) right son=355 (64 obs)
##   Primary splits:
##       yob       < 1957.5 to the left,  improve=0.621791700, (0 missing)
##       civicduty < 0.5    to the right, improve=0.048662850, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.029868440, (0 missing)
##       self      < 0.5    to the left,  improve=0.003649783, (0 missing)
##   Surrogate splits:
##       self < 0.5    to the left,  agree=0.599, adj=0.016, (0 split)
## 
## Node number 178: 300 observations,    complexity param=0.0002373418
##   predicted class=N  expected loss=0.31  P(node) =0.06
##     class counts:   207    93
##    probabilities: 0.690 0.310 
##   left son=356 (92 obs) right son=357 (208 obs)
##   Primary splits:
##       yob       < 1963.5 to the left,  improve=0.3884950, (0 missing)
##       sex       < 0.5    to the right, improve=0.2365517, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.0600000, (0 missing)
##       civicduty < 0.5    to the right, improve=0.0600000, (0 missing)
## 
## Node number 179: 155 observations,    complexity param=0.0001054852
##   predicted class=N  expected loss=0.3548387  P(node) =0.031
##     class counts:   100    55
##    probabilities: 0.645 0.355 
##   left son=358 (108 obs) right son=359 (47 obs)
##   Primary splits:
##       yob < 1963.5 to the right, improve=0.67420370, (0 missing)
##       sex < 0.5    to the right, improve=0.01057995, (0 missing)
## 
## Node number 180: 99 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.2929293  P(node) =0.0198
##     class counts:    70    29
##    probabilities: 0.707 0.293 
##   left son=360 (43 obs) right son=361 (56 obs)
##   Primary splits:
##       self      < 0.5    to the right, improve=0.5541209, (0 missing)
##       sex       < 0.5    to the left,  improve=0.2693603, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.2442428, (0 missing)
##       yob       < 1952.5 to the left,  improve=0.2190806, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.1034343, (0 missing)
##   Surrogate splits:
##       hawthorne < 0.5    to the left,  agree=0.758, adj=0.442, (0 split)
##       civicduty < 0.5    to the left,  agree=0.677, adj=0.256, (0 split)
## 
## Node number 181: 75 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.3866667  P(node) =0.015
##     class counts:    46    29
##    probabilities: 0.613 0.387 
##   left son=362 (43 obs) right son=363 (32 obs)
##   Primary splits:
##       sex       < 0.5    to the right, improve=2.333508000, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.153600900, (0 missing)
##       civicduty < 0.5    to the right, improve=0.077892600, (0 missing)
##       self      < 0.5    to the right, improve=0.009607843, (0 missing)
##       yob       < 1950.5 to the right, improve=0.006233766, (0 missing)
##   Surrogate splits:
##       hawthorne < 0.5    to the left,  agree=0.64, adj=0.156, (0 split)
## 
## Node number 182: 22 observations
##   predicted class=N  expected loss=0.3636364  P(node) =0.0044
##     class counts:    14     8
##    probabilities: 0.636 0.364 
## 
## Node number 183: 41 observations,    complexity param=0.0005424955
##   predicted class=Y  expected loss=0.4878049  P(node) =0.0082
##     class counts:    20    21
##    probabilities: 0.488 0.512 
##   left son=366 (21 obs) right son=367 (20 obs)
##   Primary splits:
##       hawthorne < 0.5    to the right, improve=0.111614400, (0 missing)
##       self      < 0.5    to the left,  improve=0.111614400, (0 missing)
##       sex       < 0.5    to the right, improve=0.009544008, (0 missing)
##   Surrogate splits:
##       self < 0.5    to the left,  agree=1.000, adj=1.00, (0 split)
##       sex  < 0.5    to the right, agree=0.585, adj=0.15, (0 split)
## 
## Node number 186: 34 observations
##   predicted class=N  expected loss=0.3529412  P(node) =0.0068
##     class counts:    22    12
##    probabilities: 0.647 0.353 
## 
## Node number 187: 55 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.4363636  P(node) =0.011
##     class counts:    31    24
##    probabilities: 0.564 0.436 
##   left son=374 (27 obs) right son=375 (28 obs)
##   Primary splits:
##       sex < 0.5    to the left,  improve=0.08893699, (0 missing)
##       yob < 1952.5 to the left,  improve=0.05454545, (0 missing)
## 
## Node number 188: 44 observations
##   predicted class=N  expected loss=0.3636364  P(node) =0.0088
##     class counts:    28    16
##    probabilities: 0.636 0.364 
## 
## Node number 189: 8 observations
##   predicted class=Y  expected loss=0.375  P(node) =0.0016
##     class counts:     3     5
##    probabilities: 0.375 0.625 
## 
## Node number 190: 57 observations,    complexity param=0.0002109705
##   predicted class=N  expected loss=0.4385965  P(node) =0.0114
##     class counts:    32    25
##    probabilities: 0.561 0.439 
##   left son=380 (8 obs) right son=381 (49 obs)
##   Primary splits:
##       yob < 1970.5 to the right, improve=0.6620122, (0 missing)
## 
## Node number 191: 10 observations
##   predicted class=Y  expected loss=0.3  P(node) =0.002
##     class counts:     3     7
##    probabilities: 0.300 0.700 
## 
## Node number 214: 73 observations
##   predicted class=N  expected loss=0.2739726  P(node) =0.0146
##     class counts:    53    20
##    probabilities: 0.726 0.274 
## 
## Node number 215: 112 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.3660714  P(node) =0.0224
##     class counts:    71    41
##    probabilities: 0.634 0.366 
##   left son=430 (87 obs) right son=431 (25 obs)
##   Primary splits:
##       yob < 1932.5 to the left,  improve=1.525131, (0 missing)
## 
## Node number 226: 54 observations
##   predicted class=N  expected loss=0.3518519  P(node) =0.0108
##     class counts:    35    19
##    probabilities: 0.648 0.352 
## 
## Node number 227: 34 observations,    complexity param=0.0007594937
##   predicted class=N  expected loss=0.4705882  P(node) =0.0068
##     class counts:    18    16
##    probabilities: 0.529 0.471 
##   left son=454 (20 obs) right son=455 (14 obs)
##   Primary splits:
##       self      < 0.5    to the left,  improve=1.41260500, (0 missing)
##       hawthorne < 0.5    to the right, improve=0.61260500, (0 missing)
##       yob       < 1944.5 to the right, improve=0.05228758, (0 missing)
##   Surrogate splits:
##       hawthorne < 0.5    to the right, agree=0.824, adj=0.571, (0 split)
## 
## Node number 230: 13 observations
##   predicted class=N  expected loss=0.4615385  P(node) =0.0026
##     class counts:     7     6
##    probabilities: 0.538 0.462 
## 
## Node number 231: 19 observations
##   predicted class=Y  expected loss=0.4210526  P(node) =0.0038
##     class counts:     8    11
##    probabilities: 0.421 0.579 
## 
## Node number 238: 41 observations,    complexity param=0.0008438819
##   predicted class=Y  expected loss=0.4634146  P(node) =0.0082
##     class counts:    19    22
##    probabilities: 0.463 0.537 
##   left son=476 (26 obs) right son=477 (15 obs)
##   Primary splits:
##       yob < 1948.5 to the left,  improve=0.8005003, (0 missing)
##       sex < 0.5    to the right, improve=0.0702439, (0 missing)
## 
## Node number 239: 12 observations
##   predicted class=Y  expected loss=0.25  P(node) =0.0024
##     class counts:     3     9
##    probabilities: 0.250 0.750 
## 
## Node number 242: 84 observations
##   predicted class=N  expected loss=0.3571429  P(node) =0.0168
##     class counts:    54    30
##    probabilities: 0.643 0.357 
## 
## Node number 243: 16 observations
##   predicted class=Y  expected loss=0.4375  P(node) =0.0032
##     class counts:     7     9
##    probabilities: 0.438 0.562 
## 
## Node number 250: 8 observations
##   predicted class=N  expected loss=0.25  P(node) =0.0016
##     class counts:     6     2
##    probabilities: 0.750 0.250 
## 
## Node number 251: 31 observations,    complexity param=0.0003164557
##   predicted class=Y  expected loss=0.4516129  P(node) =0.0062
##     class counts:    14    17
##    probabilities: 0.452 0.548 
##   left son=502 (20 obs) right son=503 (11 obs)
##   Primary splits:
##       yob < 1936.5 to the right, improve=0.2639296, (0 missing)
##       sex < 0.5    to the left,  improve=0.2009926, (0 missing)
## 
## Node number 252: 33 observations
##   predicted class=N  expected loss=0.3939394  P(node) =0.0066
##     class counts:    20    13
##    probabilities: 0.606 0.394 
## 
## Node number 253: 46 observations,    complexity param=0.002278481
##   predicted class=Y  expected loss=0.4565217  P(node) =0.0092
##     class counts:    21    25
##    probabilities: 0.457 0.543 
##   left son=506 (20 obs) right son=507 (26 obs)
##   Primary splits:
##       yob       < 1941.5 to the right, improve=1.456856000, (0 missing)
##       civicduty < 0.5    to the right, improve=0.061381070, (0 missing)
##       self      < 0.5    to the left,  improve=0.017753620, (0 missing)
##       neighbors < 0.5    to the left,  improve=0.008626639, (0 missing)
##   Surrogate splits:
##       civicduty < 0.5    to the right, agree=0.609, adj=0.1, (0 split)
## 
## Node number 254: 90 observations,    complexity param=0.0006329114
##   predicted class=Y  expected loss=0.4333333  P(node) =0.018
##     class counts:    39    51
##    probabilities: 0.433 0.567 
##   left son=508 (32 obs) right son=509 (58 obs)
##   Primary splits:
##       yob       < 1939.5 to the left,  improve=0.952155200, (0 missing)
##       neighbors < 0.5    to the right, improve=0.228571400, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.195951400, (0 missing)
##       self      < 0.5    to the right, improve=0.001724138, (0 missing)
## 
## Node number 255: 21 observations
##   predicted class=Y  expected loss=0.2857143  P(node) =0.0042
##     class counts:     6    15
##    probabilities: 0.286 0.714 
## 
## Node number 346: 32 observations
##   predicted class=N  expected loss=0.28125  P(node) =0.0064
##     class counts:    23     9
##    probabilities: 0.719 0.281 
## 
## Node number 347: 19 observations
##   predicted class=Y  expected loss=0.4210526  P(node) =0.0038
##     class counts:     8    11
##    probabilities: 0.421 0.579 
## 
## Node number 354: 93 observations
##   predicted class=N  expected loss=0.2688172  P(node) =0.0186
##     class counts:    68    25
##    probabilities: 0.731 0.269 
## 
## Node number 355: 64 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.359375  P(node) =0.0128
##     class counts:    41    23
##    probabilities: 0.641 0.359 
##   left son=710 (43 obs) right son=711 (21 obs)
##   Primary splits:
##       civicduty < 0.5    to the left,  improve=0.85302460, (0 missing)
##       self      < 0.5    to the right, improve=0.21742440, (0 missing)
##       hawthorne < 0.5    to the right, improve=0.20511360, (0 missing)
##       yob       < 1959.5 to the right, improve=0.06334459, (0 missing)
##   Surrogate splits:
##       self < 0.5    to the right, agree=0.688, adj=0.048, (0 split)
## 
## Node number 356: 92 observations
##   predicted class=N  expected loss=0.2717391  P(node) =0.0184
##     class counts:    67    25
##    probabilities: 0.728 0.272 
## 
## Node number 357: 208 observations,    complexity param=0.0002373418
##   predicted class=N  expected loss=0.3269231  P(node) =0.0416
##     class counts:   140    68
##    probabilities: 0.673 0.327 
##   left son=714 (110 obs) right son=715 (98 obs)
##   Primary splits:
##       sex       < 0.5    to the right, improve=0.3384615, (0 missing)
##       yob       < 1968.5 to the right, improve=0.1510275, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.0697231, (0 missing)
##       civicduty < 0.5    to the right, improve=0.0697231, (0 missing)
##   Surrogate splits:
##       yob < 1976.5 to the left,  agree=0.572, adj=0.092, (0 split)
## 
## Node number 358: 108 observations,    complexity param=0.0001054852
##   predicted class=N  expected loss=0.3240741  P(node) =0.0216
##     class counts:    73    35
##    probabilities: 0.676 0.324 
##   left son=716 (12 obs) right son=717 (96 obs)
##   Primary splits:
##       yob < 1964.5 to the left,  improve=0.668981500, (0 missing)
##       sex < 0.5    to the left,  improve=0.003090677, (0 missing)
## 
## Node number 359: 47 observations
##   predicted class=N  expected loss=0.4255319  P(node) =0.0094
##     class counts:    27    20
##    probabilities: 0.574 0.426 
## 
## Node number 360: 43 observations
##   predicted class=N  expected loss=0.2325581  P(node) =0.0086
##     class counts:    33    10
##    probabilities: 0.767 0.233 
## 
## Node number 361: 56 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.3392857  P(node) =0.0112
##     class counts:    37    19
##    probabilities: 0.661 0.339 
##   left son=722 (28 obs) right son=723 (28 obs)
##   Primary splits:
##       yob       < 1952.5 to the left,  improve=0.89285710, (0 missing)
##       sex       < 0.5    to the left,  improve=0.50297620, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.00297619, (0 missing)
##       civicduty < 0.5    to the right, improve=0.00297619, (0 missing)
##   Surrogate splits:
##       hawthorne < 0.5    to the left,  agree=0.679, adj=0.357, (0 split)
##       civicduty < 0.5    to the right, agree=0.679, adj=0.357, (0 split)
##       sex       < 0.5    to the right, agree=0.607, adj=0.214, (0 split)
## 
## Node number 362: 43 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.2790698  P(node) =0.0086
##     class counts:    31    12
##    probabilities: 0.721 0.279 
##   left son=724 (34 obs) right son=725 (9 obs)
##   Primary splits:
##       hawthorne < 0.5    to the left,  improve=1.740234000, (0 missing)
##       civicduty < 0.5    to the right, improve=0.999694000, (0 missing)
##       yob       < 1950.5 to the right, improve=0.032760360, (0 missing)
##       self      < 0.5    to the right, improve=0.007087486, (0 missing)
## 
## Node number 363: 32 observations,    complexity param=0.0005424955
##   predicted class=Y  expected loss=0.46875  P(node) =0.0064
##     class counts:    15    17
##    probabilities: 0.469 0.531 
##   left son=726 (23 obs) right son=727 (9 obs)
##   Primary splits:
##       civicduty < 0.5    to the left,  improve=1.522041000, (0 missing)
##       hawthorne < 0.5    to the right, improve=1.508929000, (0 missing)
##       self      < 0.5    to the left,  improve=0.014794690, (0 missing)
##       yob       < 1950.5 to the right, improve=0.002277328, (0 missing)
## 
## Node number 366: 21 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.4761905  P(node) =0.0042
##     class counts:    11    10
##    probabilities: 0.524 0.476 
##   left son=732 (10 obs) right son=733 (11 obs)
##   Primary splits:
##       sex < 0.5    to the left,  improve=0.221645, (0 missing)
## 
## Node number 367: 20 observations,    complexity param=0.0005424955
##   predicted class=Y  expected loss=0.45  P(node) =0.004
##     class counts:     9    11
##    probabilities: 0.450 0.550 
##   left son=734 (7 obs) right son=735 (13 obs)
##   Primary splits:
##       sex < 0.5    to the right, improve=0.3175824, (0 missing)
## 
## Node number 374: 27 observations
##   predicted class=N  expected loss=0.4074074  P(node) =0.0054
##     class counts:    16    11
##    probabilities: 0.593 0.407 
## 
## Node number 375: 28 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.4642857  P(node) =0.0056
##     class counts:    15    13
##    probabilities: 0.536 0.464 
##   left son=750 (11 obs) right son=751 (17 obs)
##   Primary splits:
##       yob < 1952.5 to the left,  improve=0.3670741, (0 missing)
## 
## Node number 380: 8 observations
##   predicted class=N  expected loss=0.25  P(node) =0.0016
##     class counts:     6     2
##    probabilities: 0.750 0.250 
## 
## Node number 381: 49 observations,    complexity param=0.0002109705
##   predicted class=N  expected loss=0.4693878  P(node) =0.0098
##     class counts:    26    23
##    probabilities: 0.531 0.469 
##   left son=762 (12 obs) right son=763 (37 obs)
##   Primary splits:
##       yob < 1964.5 to the left,  improve=0.08834345, (0 missing)
## 
## Node number 430: 87 observations
##   predicted class=N  expected loss=0.3218391  P(node) =0.0174
##     class counts:    59    28
##    probabilities: 0.678 0.322 
## 
## Node number 431: 25 observations
##   predicted class=Y  expected loss=0.48  P(node) =0.005
##     class counts:    12    13
##    probabilities: 0.480 0.520 
## 
## Node number 454: 20 observations
##   predicted class=N  expected loss=0.35  P(node) =0.004
##     class counts:    13     7
##    probabilities: 0.650 0.350 
## 
## Node number 455: 14 observations
##   predicted class=Y  expected loss=0.3571429  P(node) =0.0028
##     class counts:     5     9
##    probabilities: 0.357 0.643 
## 
## Node number 476: 26 observations,    complexity param=0.0008438819
##   predicted class=N  expected loss=0.4615385  P(node) =0.0052
##     class counts:    14    12
##    probabilities: 0.538 0.462 
##   left son=952 (14 obs) right son=953 (12 obs)
##   Primary splits:
##       yob < 1947.5 to the right, improve=0.6611722, (0 missing)
## 
## Node number 477: 15 observations
##   predicted class=Y  expected loss=0.3333333  P(node) =0.003
##     class counts:     5    10
##    probabilities: 0.333 0.667 
## 
## Node number 502: 20 observations,    complexity param=0.0003164557
##   predicted class=N  expected loss=0.5  P(node) =0.004
##     class counts:    10    10
##    probabilities: 0.500 0.500 
##   left son=1004 (13 obs) right son=1005 (7 obs)
##   Primary splits:
##       yob < 1938.5 to the left,  improve=0.1098901, (0 missing)
##       sex < 0.5    to the left,  improve=0.1010101, (0 missing)
##   Surrogate splits:
##       sex < 0.5    to the right, agree=0.7, adj=0.143, (0 split)
## 
## Node number 503: 11 observations
##   predicted class=Y  expected loss=0.3636364  P(node) =0.0022
##     class counts:     4     7
##    probabilities: 0.364 0.636 
## 
## Node number 506: 20 observations
##   predicted class=N  expected loss=0.4  P(node) =0.004
##     class counts:    12     8
##    probabilities: 0.600 0.400 
## 
## Node number 507: 26 observations
##   predicted class=Y  expected loss=0.3461538  P(node) =0.0052
##     class counts:     9    17
##    probabilities: 0.346 0.654 
## 
## Node number 508: 32 observations,    complexity param=0.0006329114
##   predicted class=N  expected loss=0.46875  P(node) =0.0064
##     class counts:    17    15
##    probabilities: 0.531 0.469 
##   left son=1016 (7 obs) right son=1017 (25 obs)
##   Primary splits:
##       neighbors < 0.5    to the right, improve=0.60035710, (0 missing)
##       yob       < 1937.5 to the right, improve=0.52083330, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.23553920, (0 missing)
##       self      < 0.5    to the left,  improve=0.02840909, (0 missing)
## 
## Node number 509: 58 observations
##   predicted class=Y  expected loss=0.3793103  P(node) =0.0116
##     class counts:    22    36
##    probabilities: 0.379 0.621 
## 
## Node number 710: 43 observations
##   predicted class=N  expected loss=0.3023256  P(node) =0.0086
##     class counts:    30    13
##    probabilities: 0.698 0.302 
## 
## Node number 711: 21 observations,    complexity param=0.0001582278
##   predicted class=N  expected loss=0.4761905  P(node) =0.0042
##     class counts:    11    10
##    probabilities: 0.524 0.476 
##   left son=1422 (14 obs) right son=1423 (7 obs)
##   Primary splits:
##       yob < 1959.5 to the left,  improve=0.1904762, (0 missing)
## 
## Node number 714: 110 observations
##   predicted class=N  expected loss=0.3  P(node) =0.022
##     class counts:    77    33
##    probabilities: 0.700 0.300 
## 
## Node number 715: 98 observations,    complexity param=0.0002373418
##   predicted class=N  expected loss=0.3571429  P(node) =0.0196
##     class counts:    63    35
##    probabilities: 0.643 0.357 
##   left son=1430 (11 obs) right son=1431 (87 obs)
##   Primary splits:
##       yob       < 1978.5 to the right, improve=0.7617555, (0 missing)
##       hawthorne < 0.5    to the left,  improve=0.3750000, (0 missing)
##       civicduty < 0.5    to the right, improve=0.3750000, (0 missing)
## 
## Node number 716: 12 observations
##   predicted class=N  expected loss=0.1666667  P(node) =0.0024
##     class counts:    10     2
##    probabilities: 0.833 0.167 
## 
## Node number 717: 96 observations,    complexity param=0.0001054852
##   predicted class=N  expected loss=0.34375  P(node) =0.0192
##     class counts:    63    33
##    probabilities: 0.656 0.344 
##   left son=1434 (13 obs) right son=1435 (83 obs)
##   Primary splits:
##       yob < 1977.5 to the right, improve=0.383862400, (0 missing)
##       sex < 0.5    to the left,  improve=0.001311189, (0 missing)
## 
## Node number 722: 28 observations
##   predicted class=N  expected loss=0.25  P(node) =0.0056
##     class counts:    21     7
##    probabilities: 0.750 0.250 
## 
## Node number 723: 28 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.4285714  P(node) =0.0056
##     class counts:    16    12
##    probabilities: 0.571 0.429 
##   left son=1446 (19 obs) right son=1447 (9 obs)
##   Primary splits:
##       sex < 0.5    to the left,  improve=1.503759, (0 missing)
## 
## Node number 724: 34 observations
##   predicted class=N  expected loss=0.2058824  P(node) =0.0068
##     class counts:    27     7
##    probabilities: 0.794 0.206 
## 
## Node number 725: 9 observations
##   predicted class=Y  expected loss=0.4444444  P(node) =0.0018
##     class counts:     4     5
##    probabilities: 0.444 0.556 
## 
## Node number 726: 23 observations,    complexity param=0.0005424955
##   predicted class=N  expected loss=0.4347826  P(node) =0.0046
##     class counts:    13    10
##    probabilities: 0.565 0.435 
##   left son=1452 (14 obs) right son=1453 (9 obs)
##   Primary splits:
##       hawthorne < 0.5    to the right, improve=0.43133200, (0 missing)
##       self      < 0.5    to the left,  improve=0.43133200, (0 missing)
##       yob       < 1950.5 to the left,  improve=0.04280936, (0 missing)
##   Surrogate splits:
##       self < 0.5    to the left,  agree=1, adj=1, (0 split)
## 
## Node number 727: 9 observations
##   predicted class=Y  expected loss=0.2222222  P(node) =0.0018
##     class counts:     2     7
##    probabilities: 0.222 0.778 
## 
## Node number 732: 10 observations
##   predicted class=N  expected loss=0.4  P(node) =0.002
##     class counts:     6     4
##    probabilities: 0.600 0.400 
## 
## Node number 733: 11 observations
##   predicted class=Y  expected loss=0.4545455  P(node) =0.0022
##     class counts:     5     6
##    probabilities: 0.455 0.545 
## 
## Node number 734: 7 observations
##   predicted class=N  expected loss=0.4285714  P(node) =0.0014
##     class counts:     4     3
##    probabilities: 0.571 0.429 
## 
## Node number 735: 13 observations
##   predicted class=Y  expected loss=0.3846154  P(node) =0.0026
##     class counts:     5     8
##    probabilities: 0.385 0.615 
## 
## Node number 750: 11 observations
##   predicted class=N  expected loss=0.3636364  P(node) =0.0022
##     class counts:     7     4
##    probabilities: 0.636 0.364 
## 
## Node number 751: 17 observations
##   predicted class=Y  expected loss=0.4705882  P(node) =0.0034
##     class counts:     8     9
##    probabilities: 0.471 0.529 
## 
## Node number 762: 12 observations
##   predicted class=N  expected loss=0.4166667  P(node) =0.0024
##     class counts:     7     5
##    probabilities: 0.583 0.417 
## 
## Node number 763: 37 observations,    complexity param=0.0002109705
##   predicted class=N  expected loss=0.4864865  P(node) =0.0074
##     class counts:    19    18
##    probabilities: 0.514 0.486 
##   left son=1526 (14 obs) right son=1527 (23 obs)
##   Primary splits:
##       yob < 1967.5 to the right, improve=0.1510828, (0 missing)
## 
## Node number 952: 14 observations
##   predicted class=N  expected loss=0.3571429  P(node) =0.0028
##     class counts:     9     5
##    probabilities: 0.643 0.357 
## 
## Node number 953: 12 observations
##   predicted class=Y  expected loss=0.4166667  P(node) =0.0024
##     class counts:     5     7
##    probabilities: 0.417 0.583 
## 
## Node number 1004: 13 observations
##   predicted class=N  expected loss=0.4615385  P(node) =0.0026
##     class counts:     7     6
##    probabilities: 0.538 0.462 
## 
## Node number 1005: 7 observations
##   predicted class=Y  expected loss=0.4285714  P(node) =0.0014
##     class counts:     3     4
##    probabilities: 0.429 0.571 
## 
## Node number 1016: 7 observations
##   predicted class=N  expected loss=0.2857143  P(node) =0.0014
##     class counts:     5     2
##    probabilities: 0.714 0.286 
## 
## Node number 1017: 25 observations,    complexity param=0.0006329114
##   predicted class=Y  expected loss=0.48  P(node) =0.005
##     class counts:    12    13
##    probabilities: 0.480 0.520 
##   left son=2034 (7 obs) right son=2035 (18 obs)
##   Primary splits:
##       yob       < 1938.5 to the right, improve=0.16253970, (0 missing)
##       self      < 0.5    to the right, improve=0.01333333, (0 missing)
##       civicduty < 0.5    to the left,  improve=0.01333333, (0 missing)
## 
## Node number 1422: 14 observations
##   predicted class=N  expected loss=0.4285714  P(node) =0.0028
##     class counts:     8     6
##    probabilities: 0.571 0.429 
## 
## Node number 1423: 7 observations
##   predicted class=Y  expected loss=0.4285714  P(node) =0.0014
##     class counts:     3     4
##    probabilities: 0.429 0.571 
## 
## Node number 1430: 11 observations
##   predicted class=N  expected loss=0.1818182  P(node) =0.0022
##     class counts:     9     2
##    probabilities: 0.818 0.182 
## 
## Node number 1431: 87 observations,    complexity param=0.0002373418
##   predicted class=N  expected loss=0.3793103  P(node) =0.0174
##     class counts:    54    33
##    probabilities: 0.621 0.379 
##   left son=2862 (43 obs) right son=2863 (44 obs)
##   Primary splits:
##       hawthorne < 0.5    to the left,  improve=1.0078010, (0 missing)
##       civicduty < 0.5    to the right, improve=1.0078010, (0 missing)
##       yob       < 1975.5 to the left,  improve=0.5619458, (0 missing)
##   Surrogate splits:
##       civicduty < 0.5    to the right, agree=1.000, adj=1.000, (0 split)
##       yob       < 1972.5 to the right, agree=0.552, adj=0.093, (0 split)
## 
## Node number 1434: 13 observations
##   predicted class=N  expected loss=0.2307692  P(node) =0.0026
##     class counts:    10     3
##    probabilities: 0.769 0.231 
## 
## Node number 1435: 83 observations,    complexity param=0.0001054852
##   predicted class=N  expected loss=0.3614458  P(node) =0.0166
##     class counts:    53    30
##    probabilities: 0.639 0.361 
##   left son=2870 (24 obs) right son=2871 (59 obs)
##   Primary splits:
##       yob < 1966.5 to the left,  improve=3.287897e-01, (0 missing)
##       sex < 0.5    to the left,  improve=1.424136e-05, (0 missing)
## 
## Node number 1446: 19 observations
##   predicted class=N  expected loss=0.3157895  P(node) =0.0038
##     class counts:    13     6
##    probabilities: 0.684 0.316 
## 
## Node number 1447: 9 observations
##   predicted class=Y  expected loss=0.3333333  P(node) =0.0018
##     class counts:     3     6
##    probabilities: 0.333 0.667 
## 
## Node number 1452: 14 observations
##   predicted class=N  expected loss=0.3571429  P(node) =0.0028
##     class counts:     9     5
##    probabilities: 0.643 0.357 
## 
## Node number 1453: 9 observations
##   predicted class=Y  expected loss=0.4444444  P(node) =0.0018
##     class counts:     4     5
##    probabilities: 0.444 0.556 
## 
## Node number 1526: 14 observations
##   predicted class=N  expected loss=0.4285714  P(node) =0.0028
##     class counts:     8     6
##    probabilities: 0.571 0.429 
## 
## Node number 1527: 23 observations
##   predicted class=Y  expected loss=0.4782609  P(node) =0.0046
##     class counts:    11    12
##    probabilities: 0.478 0.522 
## 
## Node number 2034: 7 observations
##   predicted class=N  expected loss=0.4285714  P(node) =0.0014
##     class counts:     4     3
##    probabilities: 0.571 0.429 
## 
## Node number 2035: 18 observations
##   predicted class=Y  expected loss=0.4444444  P(node) =0.0036
##     class counts:     8    10
##    probabilities: 0.444 0.556 
## 
## Node number 2862: 43 observations
##   predicted class=N  expected loss=0.3023256  P(node) =0.0086
##     class counts:    30    13
##    probabilities: 0.698 0.302 
## 
## Node number 2863: 44 observations,    complexity param=0.0002373418
##   predicted class=N  expected loss=0.4545455  P(node) =0.0088
##     class counts:    24    20
##    probabilities: 0.545 0.455 
##   left son=5726 (18 obs) right son=5727 (26 obs)
##   Primary splits:
##       yob < 1968.5 to the right, improve=0.2626263, (0 missing)
## 
## Node number 2870: 24 observations
##   predicted class=N  expected loss=0.2916667  P(node) =0.0048
##     class counts:    17     7
##    probabilities: 0.708 0.292 
## 
## Node number 2871: 59 observations,    complexity param=0.0001054852
##   predicted class=N  expected loss=0.3898305  P(node) =0.0118
##     class counts:    36    23
##    probabilities: 0.610 0.390 
##   left son=5742 (26 obs) right son=5743 (33 obs)
##   Primary splits:
##       sex < 0.5    to the left,  improve=0.1773537, (0 missing)
##       yob < 1971.5 to the right, improve=0.1330140, (0 missing)
##   Surrogate splits:
##       yob < 1971.5 to the right, agree=0.61, adj=0.115, (0 split)
## 
## Node number 5726: 18 observations
##   predicted class=N  expected loss=0.3888889  P(node) =0.0036
##     class counts:    11     7
##    probabilities: 0.611 0.389 
## 
## Node number 5727: 26 observations,    complexity param=0.0002373418
##   predicted class=N  expected loss=0.5  P(node) =0.0052
##     class counts:    13    13
##    probabilities: 0.500 0.500 
##   left son=11454 (11 obs) right son=11455 (15 obs)
##   Primary splits:
##       yob < 1965.5 to the left,  improve=0.7090909, (0 missing)
## 
## Node number 5742: 26 observations
##   predicted class=N  expected loss=0.3461538  P(node) =0.0052
##     class counts:    17     9
##    probabilities: 0.654 0.346 
## 
## Node number 5743: 33 observations,    complexity param=0.0001054852
##   predicted class=N  expected loss=0.4242424  P(node) =0.0066
##     class counts:    19    14
##    probabilities: 0.576 0.424 
##   left son=11486 (10 obs) right son=11487 (23 obs)
##   Primary splits:
##       yob < 1971.5 to the right, improve=1.442951, (0 missing)
## 
## Node number 11454: 11 observations
##   predicted class=N  expected loss=0.3636364  P(node) =0.0022
##     class counts:     7     4
##    probabilities: 0.636 0.364 
## 
## Node number 11455: 15 observations
##   predicted class=Y  expected loss=0.4  P(node) =0.003
##     class counts:     6     9
##    probabilities: 0.400 0.600 
## 
## Node number 11486: 10 observations
##   predicted class=N  expected loss=0.2  P(node) =0.002
##     class counts:     8     2
##    probabilities: 0.800 0.200 
## 
## Node number 11487: 23 observations
##   predicted class=Y  expected loss=0.4782609  P(node) =0.0046
##     class counts:    11    12
##    probabilities: 0.478 0.522 
## 
## n= 5000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##     1) root 5000 1580 N (0.6840000 0.3160000)  
##       2) yob>=1949.5 3443  967 N (0.7191403 0.2808597)  
##         4) yob>=1980.5 394   61 N (0.8451777 0.1548223) *
##         5) yob< 1980.5 3049  906 N (0.7028534 0.2971466)  
##          10) control>=0.5 1717  463 N (0.7303436 0.2696564)  
##            20) yob< 1969.5 1497  396 N (0.7354709 0.2645291) *
##            21) yob>=1969.5 220   67 N (0.6954545 0.3045455)  
##              42) yob>=1974.5 91   21 N (0.7692308 0.2307692) *
##              43) yob< 1974.5 129   46 N (0.6434109 0.3565891)  
##                86) sex>=0.5 70   22 N (0.6857143 0.3142857)  
##                 172) yob>=1972.5 19    2 N (0.8947368 0.1052632) *
##                 173) yob< 1972.5 51   20 N (0.6078431 0.3921569)  
##                   346) yob< 1971.5 32    9 N (0.7187500 0.2812500) *
##                   347) yob>=1971.5 19    8 Y (0.4210526 0.5789474) *
##                87) sex< 0.5 59   24 N (0.5932203 0.4067797)  
##                 174) yob< 1973.5 49   18 N (0.6326531 0.3673469) *
##                 175) yob>=1973.5 10    4 Y (0.4000000 0.6000000) *
##          11) control< 0.5 1332  443 N (0.6674174 0.3325826)  
##            22) neighbors< 0.5 1000  316 N (0.6840000 0.3160000)  
##              44) yob>=1954.5 763  229 N (0.6998689 0.3001311)  
##                88) yob< 1960.5 308   81 N (0.7370130 0.2629870)  
##                 176) sex>=0.5 151   33 N (0.7814570 0.2185430) *
##                 177) sex< 0.5 157   48 N (0.6942675 0.3057325)  
##                   354) yob< 1957.5 93   25 N (0.7311828 0.2688172) *
##                   355) yob>=1957.5 64   23 N (0.6406250 0.3593750)  
##                     710) civicduty< 0.5 43   13 N (0.6976744 0.3023256) *
##                     711) civicduty>=0.5 21   10 N (0.5238095 0.4761905)  
##                      1422) yob< 1959.5 14    6 N (0.5714286 0.4285714) *
##                      1423) yob>=1959.5 7    3 Y (0.4285714 0.5714286) *
##                89) yob>=1960.5 455  148 N (0.6747253 0.3252747)  
##                 178) self< 0.5 300   93 N (0.6900000 0.3100000)  
##                   356) yob< 1963.5 92   25 N (0.7282609 0.2717391) *
##                   357) yob>=1963.5 208   68 N (0.6730769 0.3269231)  
##                     714) sex>=0.5 110   33 N (0.7000000 0.3000000) *
##                     715) sex< 0.5 98   35 N (0.6428571 0.3571429)  
##                      1430) yob>=1978.5 11    2 N (0.8181818 0.1818182) *
##                      1431) yob< 1978.5 87   33 N (0.6206897 0.3793103)  
##                        2862) hawthorne< 0.5 43   13 N (0.6976744 0.3023256) *
##                        2863) hawthorne>=0.5 44   20 N (0.5454545 0.4545455)  
##                          5726) yob>=1968.5 18    7 N (0.6111111 0.3888889) *
##                          5727) yob< 1968.5 26   13 N (0.5000000 0.5000000)  
##                           11454) yob< 1965.5 11    4 N (0.6363636 0.3636364) *
##                           11455) yob>=1965.5 15    6 Y (0.4000000 0.6000000) *
##                 179) self>=0.5 155   55 N (0.6451613 0.3548387)  
##                   358) yob>=1963.5 108   35 N (0.6759259 0.3240741)  
##                     716) yob< 1964.5 12    2 N (0.8333333 0.1666667) *
##                     717) yob>=1964.5 96   33 N (0.6562500 0.3437500)  
##                      1434) yob>=1977.5 13    3 N (0.7692308 0.2307692) *
##                      1435) yob< 1977.5 83   30 N (0.6385542 0.3614458)  
##                        2870) yob< 1966.5 24    7 N (0.7083333 0.2916667) *
##                        2871) yob>=1966.5 59   23 N (0.6101695 0.3898305)  
##                          5742) sex< 0.5 26    9 N (0.6538462 0.3461538) *
##                          5743) sex>=0.5 33   14 N (0.5757576 0.4242424)  
##                           11486) yob>=1971.5 10    2 N (0.8000000 0.2000000) *
##                           11487) yob< 1971.5 23   11 Y (0.4782609 0.5217391) *
##                   359) yob< 1963.5 47   20 N (0.5744681 0.4255319) *
##              45) yob< 1954.5 237   87 N (0.6329114 0.3670886)  
##                90) yob< 1953.5 174   58 N (0.6666667 0.3333333)  
##                 180) yob>=1951.5 99   29 N (0.7070707 0.2929293)  
##                   360) self>=0.5 43   10 N (0.7674419 0.2325581) *
##                   361) self< 0.5 56   19 N (0.6607143 0.3392857)  
##                     722) yob< 1952.5 28    7 N (0.7500000 0.2500000) *
##                     723) yob>=1952.5 28   12 N (0.5714286 0.4285714)  
##                      1446) sex< 0.5 19    6 N (0.6842105 0.3157895) *
##                      1447) sex>=0.5 9    3 Y (0.3333333 0.6666667) *
##                 181) yob< 1951.5 75   29 N (0.6133333 0.3866667)  
##                   362) sex>=0.5 43   12 N (0.7209302 0.2790698)  
##                     724) hawthorne< 0.5 34    7 N (0.7941176 0.2058824) *
##                     725) hawthorne>=0.5 9    4 Y (0.4444444 0.5555556) *
##                   363) sex< 0.5 32   15 Y (0.4687500 0.5312500)  
##                     726) civicduty< 0.5 23   10 N (0.5652174 0.4347826)  
##                      1452) hawthorne>=0.5 14    5 N (0.6428571 0.3571429) *
##                      1453) hawthorne< 0.5 9    4 Y (0.4444444 0.5555556) *
##                     727) civicduty>=0.5 9    2 Y (0.2222222 0.7777778) *
##                91) yob>=1953.5 63   29 N (0.5396825 0.4603175)  
##                 182) civicduty>=0.5 22    8 N (0.6363636 0.3636364) *
##                 183) civicduty< 0.5 41   20 Y (0.4878049 0.5121951)  
##                   366) hawthorne>=0.5 21   10 N (0.5238095 0.4761905)  
##                     732) sex< 0.5 10    4 N (0.6000000 0.4000000) *
##                     733) sex>=0.5 11    5 Y (0.4545455 0.5454545) *
##                   367) hawthorne< 0.5 20    9 Y (0.4500000 0.5500000)  
##                     734) sex>=0.5 7    3 N (0.5714286 0.4285714) *
##                     735) sex< 0.5 13    5 Y (0.3846154 0.6153846) *
##            23) neighbors>=0.5 332  127 N (0.6174699 0.3825301)  
##              46) yob< 1962.5 213   74 N (0.6525822 0.3474178)  
##                92) yob>=1954.5 124   38 N (0.6935484 0.3064516) *
##                93) yob< 1954.5 89   36 N (0.5955056 0.4044944)  
##                 186) yob< 1951.5 34   12 N (0.6470588 0.3529412) *
##                 187) yob>=1951.5 55   24 N (0.5636364 0.4363636)  
##                   374) sex< 0.5 27   11 N (0.5925926 0.4074074) *
##                   375) sex>=0.5 28   13 N (0.5357143 0.4642857)  
##                     750) yob< 1952.5 11    4 N (0.6363636 0.3636364) *
##                     751) yob>=1952.5 17    8 Y (0.4705882 0.5294118) *
##              47) yob>=1962.5 119   53 N (0.5546218 0.4453782)  
##                94) sex< 0.5 52   21 N (0.5961538 0.4038462)  
##                 188) yob>=1964.5 44   16 N (0.6363636 0.3636364) *
##                 189) yob< 1964.5 8    3 Y (0.3750000 0.6250000) *
##                95) sex>=0.5 67   32 N (0.5223881 0.4776119)  
##                 190) yob< 1975.5 57   25 N (0.5614035 0.4385965)  
##                   380) yob>=1970.5 8    2 N (0.7500000 0.2500000) *
##                   381) yob< 1970.5 49   23 N (0.5306122 0.4693878)  
##                     762) yob< 1964.5 12    5 N (0.5833333 0.4166667) *
##                     763) yob>=1964.5 37   18 N (0.5135135 0.4864865)  
##                      1526) yob>=1967.5 14    6 N (0.5714286 0.4285714) *
##                      1527) yob< 1967.5 23   11 Y (0.4782609 0.5217391) *
##                 191) yob>=1975.5 10    3 Y (0.3000000 0.7000000) *
##       3) yob< 1949.5 1557  613 N (0.6062942 0.3937058)  
##         6) control>=0.5 877  311 N (0.6453820 0.3546180)  
##          12) yob>=1944.5 323   98 N (0.6965944 0.3034056) *
##          13) yob< 1944.5 554  213 N (0.6155235 0.3844765)  
##            26) yob< 1935.5 233   69 N (0.7038627 0.2961373)  
##              52) yob< 1921.5 32    5 N (0.8437500 0.1562500) *
##              53) yob>=1921.5 201   64 N (0.6815920 0.3184080)  
##               106) yob>=1934.5 16    3 N (0.8125000 0.1875000) *
##               107) yob< 1934.5 185   61 N (0.6702703 0.3297297)  
##                 214) sex>=0.5 73   20 N (0.7260274 0.2739726) *
##                 215) sex< 0.5 112   41 N (0.6339286 0.3660714)  
##                   430) yob< 1932.5 87   28 N (0.6781609 0.3218391) *
##                   431) yob>=1932.5 25   12 Y (0.4800000 0.5200000) *
##            27) yob>=1935.5 321  144 N (0.5514019 0.4485981)  
##              54) yob< 1943.5 280  118 N (0.5785714 0.4214286)  
##               108) yob>=1936.5 250   99 N (0.6040000 0.3960000) *
##               109) yob< 1936.5 30   11 Y (0.3666667 0.6333333) *
##              55) yob>=1943.5 41   15 Y (0.3658537 0.6341463) *
##         7) control< 0.5 680  302 N (0.5558824 0.4441176)  
##          14) yob>=1943.5 295  120 N (0.5932203 0.4067797)  
##            28) neighbors< 0.5 224   84 N (0.6250000 0.3750000)  
##              56) yob< 1948.5 177   62 N (0.6497175 0.3502825)  
##               112) sex< 0.5 89   27 N (0.6966292 0.3033708) *
##               113) sex>=0.5 88   35 N (0.6022727 0.3977273)  
##                 226) yob>=1945.5 54   19 N (0.6481481 0.3518519) *
##                 227) yob< 1945.5 34   16 N (0.5294118 0.4705882)  
##                   454) self< 0.5 20    7 N (0.6500000 0.3500000) *
##                   455) self>=0.5 14    5 Y (0.3571429 0.6428571) *
##              57) yob>=1948.5 47   22 N (0.5319149 0.4680851)  
##               114) hawthorne>=0.5 15    5 N (0.6666667 0.3333333) *
##               115) hawthorne< 0.5 32   15 Y (0.4687500 0.5312500)  
##                 230) sex>=0.5 13    6 N (0.5384615 0.4615385) *
##                 231) sex< 0.5 19    8 Y (0.4210526 0.5789474) *
##            29) neighbors>=0.5 71   35 Y (0.4929577 0.5070423)  
##              58) yob< 1944.5 8    1 N (0.8750000 0.1250000) *
##              59) yob>=1944.5 63   28 Y (0.4444444 0.5555556)  
##               118) yob< 1945.5 10    4 N (0.6000000 0.4000000) *
##               119) yob>=1945.5 53   22 Y (0.4150943 0.5849057)  
##                 238) yob>=1946.5 41   19 Y (0.4634146 0.5365854)  
##                   476) yob< 1948.5 26   12 N (0.5384615 0.4615385)  
##                     952) yob>=1947.5 14    5 N (0.6428571 0.3571429) *
##                     953) yob< 1947.5 12    5 Y (0.4166667 0.5833333) *
##                   477) yob>=1948.5 15    5 Y (0.3333333 0.6666667) *
##                 239) yob< 1946.5 12    3 Y (0.2500000 0.7500000) *
##          15) yob< 1943.5 385  182 N (0.5272727 0.4727273)  
##            30) yob< 1933.5 127   50 N (0.6062992 0.3937008)  
##              60) yob>=1919.5 113   42 N (0.6283186 0.3716814)  
##               120) yob< 1922.5 13    3 N (0.7692308 0.2307692) *
##               121) yob>=1922.5 100   39 N (0.6100000 0.3900000)  
##                 242) yob>=1925.5 84   30 N (0.6428571 0.3571429) *
##                 243) yob< 1925.5 16    7 Y (0.4375000 0.5625000) *
##              61) yob< 1919.5 14    6 Y (0.4285714 0.5714286) *
##            31) yob>=1933.5 258  126 Y (0.4883721 0.5116279)  
##              62) hawthorne>=0.5 68   28 N (0.5882353 0.4117647)  
##               124) yob>=1940.5 29    9 N (0.6896552 0.3103448) *
##               125) yob< 1940.5 39   19 N (0.5128205 0.4871795)  
##                 250) yob< 1935.5 8    2 N (0.7500000 0.2500000) *
##                 251) yob>=1935.5 31   14 Y (0.4516129 0.5483871)  
##                   502) yob>=1936.5 20   10 N (0.5000000 0.5000000)  
##                    1004) yob< 1938.5 13    6 N (0.5384615 0.4615385) *
##                    1005) yob>=1938.5 7    3 Y (0.4285714 0.5714286) *
##                   503) yob< 1936.5 11    4 Y (0.3636364 0.6363636) *
##              63) hawthorne< 0.5 190   86 Y (0.4526316 0.5473684)  
##               126) sex>=0.5 79   38 N (0.5189873 0.4810127)  
##                 252) yob< 1938.5 33   13 N (0.6060606 0.3939394) *
##                 253) yob>=1938.5 46   21 Y (0.4565217 0.5434783)  
##                   506) yob>=1941.5 20    8 N (0.6000000 0.4000000) *
##                   507) yob< 1941.5 26    9 Y (0.3461538 0.6538462) *
##               127) sex< 0.5 111   45 Y (0.4054054 0.5945946)  
##                 254) yob>=1936.5 90   39 Y (0.4333333 0.5666667)  
##                   508) yob< 1939.5 32   15 N (0.5312500 0.4687500)  
##                    1016) neighbors>=0.5 7    2 N (0.7142857 0.2857143) *
##                    1017) neighbors< 0.5 25   12 Y (0.4800000 0.5200000)  
##                      2034) yob>=1938.5 7    3 N (0.5714286 0.4285714) *
##                      2035) yob< 1938.5 18    8 Y (0.4444444 0.5555556) *
##                   509) yob>=1939.5 58   22 Y (0.3793103 0.6206897) *
##                 255) yob< 1936.5 21    6 Y (0.2857143 0.7142857) *
```

![](Michigan_Voters_files/figure-html/fit.models-60.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                      0
## 2           Y                                      0
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   3420
## 2                                   1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                      0
## 2           Y                                      0
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   3420
## 2                                   1580
##           Reference
## Prediction    N    Y
##          N  416   76
##          Y 3004 1504
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                    416
## 2           Y                                     76
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   3004
## 2                                   1504
##           Reference
## Prediction    N    Y
##          N 2136  683
##          Y 1284  897
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   2136
## 2           Y                                    683
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   1284
## 2                                    897
##           Reference
## Prediction    N    Y
##          N 3099 1176
##          Y  321  404
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3099
## 2           Y                                   1176
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                    321
## 2                                    404
##           Reference
## Prediction    N    Y
##          N 3211 1258
##          Y  209  322
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3211
## 2           Y                                   1258
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                    209
## 2                                    322
##           Reference
## Prediction    N    Y
##          N 3314 1384
##          Y  106  196
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3314
## 2           Y                                   1384
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                    106
## 2                                    196
##           Reference
## Prediction    N    Y
##          N 3409 1549
##          Y   11   31
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3409
## 2           Y                                   1549
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                     11
## 2                                     31
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3420
## 2           Y                                   1580
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      0
## 2                                      0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3420
## 2           Y                                   1580
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      0
## 2                                      0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3420
## 2           Y                                   1580
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      0
## 2                                      0
##    threshold    f.score
## 1        0.0 0.48024316
## 2        0.1 0.48024316
## 3        0.2 0.49408673
## 4        0.3 0.47700080
## 5        0.4 0.35054230
## 6        0.5 0.30506869
## 7        0.6 0.20828905
## 8        0.7 0.03822441
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

![](Michigan_Voters_files/figure-html/fit.models-61.png) 

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                    416
## 2           Y                                     76
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   3004
## 2                                   1504
##           Reference
## Prediction    N    Y
##          N  416   76
##          Y 3004 1504
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                    416
## 2           Y                                     76
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   3004
## 2                                   1504
##          Prediction
## Reference    N    Y
##         N  416 3004
##         Y   76 1504
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##     0.38400000     0.04907278     0.37049010     0.39764547     0.68400000 
## AccuracyPValue  McnemarPValue 
##     1.00000000     0.00000000
```

![](Michigan_Voters_files/figure-html/fit.models-62.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                      0
## 2           Y                                      0
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   3421
## 2                                   1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                      0
## 2           Y                                      0
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   3421
## 2                                   1579
##           Reference
## Prediction    N    Y
##          N  414   87
##          Y 3007 1492
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                    414
## 2           Y                                     87
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   3007
## 2                                   1492
##           Reference
## Prediction    N    Y
##          N 2044  765
##          Y 1377  814
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   2044
## 2           Y                                    765
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   1377
## 2                                    814
##           Reference
## Prediction    N    Y
##          N 2982 1311
##          Y  439  268
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   2982
## 2           Y                                   1311
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                    439
## 2                                    268
##           Reference
## Prediction    N    Y
##          N 3100 1391
##          Y  321  188
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3100
## 2           Y                                   1391
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                    321
## 2                                    188
##           Reference
## Prediction    N    Y
##          N 3238 1454
##          Y  183  125
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3238
## 2           Y                                   1454
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                    183
## 2                                    125
##           Reference
## Prediction    N    Y
##          N 3395 1552
##          Y   26   27
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3395
## 2           Y                                   1552
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                     26
## 2                                     27
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3421
## 2           Y                                   1579
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      0
## 2                                      0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3421
## 2           Y                                   1579
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      0
## 2                                      0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                   3421
## 2           Y                                   1579
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                      0
## 2                                      0
##    threshold    f.score
## 1        0.0 0.48001216
## 2        0.1 0.48001216
## 3        0.2 0.49095097
## 4        0.3 0.43183024
## 5        0.4 0.23447069
## 6        0.5 0.18007663
## 7        0.6 0.13248543
## 8        0.7 0.03308824
## 9        0.8 0.00000000
## 10       0.9 0.00000000
## 11       1.0 0.00000000
```

```
## [1] "Classifier Probability Threshold: 0.2000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                    414
## 2           Y                                     87
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   3007
## 2                                   1492
##           Reference
## Prediction    N    Y
##          N  414   87
##          Y 3007 1492
##   voting.fctr voting.fctr.predict.All.X.cp.0.rpart.N
## 1           N                                    414
## 2           Y                                     87
##   voting.fctr.predict.All.X.cp.0.rpart.Y
## 1                                   3007
## 2                                   1492
##          Prediction
## Reference    N    Y
##         N  414 3007
##         Y   87 1492
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##     0.38120000     0.04400884     0.36771048     0.39482837     0.68420000 
## AccuracyPValue  McnemarPValue 
##     1.00000000     0.00000000 
##           model_id model_method
## 1 All.X.cp.0.rpart        rpart
##                                                      feats max.nTuningRuns
## 1 sex, yob, hawthorne, civicduty, neighbors, self, control               0
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.788                 0.145   0.6396408
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.2       0.4940867            0.384
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1             0.3704901             0.3976455    0.04907278   0.5711956
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                    0.2        0.490951           0.3812
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.3677105             0.3948284    0.04400884
## [1] "iterating over method:rf"
## [1] "fitting model: All.X.rf"
## [1] "    indep_vars: sex, yob, hawthorne, civicduty, neighbors, self, control, .rnorm"
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

![](Michigan_Voters_files/figure-html/fit.models-63.png) 

```
## + : mtry=2 
## - : mtry=2 
## + : mtry=5 
## - : mtry=5 
## + : mtry=8 
## - : mtry=8 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 2 on full training set
```

```
## Warning in myfit_mdl(model_id = paste0(model_id_pfx, ""), model_method =
## method, : model's bestTune found at an extreme of tuneGrid for parameter:
## mtry
```

![](Michigan_Voters_files/figure-html/fit.models-64.png) ![](Michigan_Voters_files/figure-html/fit.models-65.png) 

```
##                 Length Class      Mode     
## call                4  -none-     call     
## type                1  -none-     character
## predicted        5000  factor     numeric  
## err.rate         1500  -none-     numeric  
## confusion           6  -none-     numeric  
## votes           10000  matrix     numeric  
## oob.times        5000  -none-     numeric  
## classes             2  -none-     character
## importance          8  -none-     numeric  
## importanceSD        0  -none-     NULL     
## localImportance     0  -none-     NULL     
## proximity           0  -none-     NULL     
## ntree               1  -none-     numeric  
## mtry                1  -none-     numeric  
## forest             14  -none-     list     
## y                5000  factor     numeric  
## test                0  -none-     NULL     
## inbag               0  -none-     NULL     
## xNames              8  -none-     character
## problemType         1  -none-     character
## tuneValue           1  data.frame list     
## obsLevels           2  -none-     character
```

![](Michigan_Voters_files/figure-html/fit.models-66.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                              0
## 2           Y                              0
##   voting.fctr.predict.All.X.rf.Y
## 1                           3420
## 2                           1580
##           Reference
## Prediction    N    Y
##          N 3061 1066
##          Y  359  514
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3061
## 2           Y                           1066
##   voting.fctr.predict.All.X.rf.Y
## 1                            359
## 2                            514
##           Reference
## Prediction    N    Y
##          N 3308 1334
##          Y  112  246
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3308
## 2           Y                           1334
##   voting.fctr.predict.All.X.rf.Y
## 1                            112
## 2                            246
##           Reference
## Prediction    N    Y
##          N 3405 1473
##          Y   15  107
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3405
## 2           Y                           1473
##   voting.fctr.predict.All.X.rf.Y
## 1                             15
## 2                            107
##           Reference
## Prediction    N    Y
##          N 3417 1555
##          Y    3   25
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3417
## 2           Y                           1555
##   voting.fctr.predict.All.X.rf.Y
## 1                              3
## 2                             25
##           Reference
## Prediction    N    Y
##          N 3420 1577
##          Y    0    3
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3420
## 2           Y                           1577
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              3
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3420
## 2           Y                           1580
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3420
## 2           Y                           1580
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3420
## 2           Y                           1580
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3420
## 2           Y                           1580
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              0
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3420
## 2           Y                           1580
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              0
##    threshold     f.score
## 1        0.0 0.480243161
## 2        0.1 0.419078679
## 3        0.2 0.253869969
## 4        0.3 0.125734430
## 5        0.4 0.031094527
## 6        0.5 0.003790272
## 7        0.6 0.000000000
## 8        0.7 0.000000000
## 9        0.8 0.000000000
## 10       0.9 0.000000000
## 11       1.0 0.000000000
```

![](Michigan_Voters_files/figure-html/fit.models-67.png) 

```
## [1] "Classifier Probability Threshold: 0.0000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.All.X.rf.Y
## 1           N                           3420
## 2           Y                           1580
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3420 1580
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                              0
## 2           Y                              0
##   voting.fctr.predict.All.X.rf.Y
## 1                           3420
## 2                           1580
##          Prediction
## Reference    N    Y
##         N    0 3420
##         Y    0 1580
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3160000      0.0000000      0.3031240      0.3290911      0.6840000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000
```

![](Michigan_Voters_files/figure-html/fit.models-68.png) 

```
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                              0
## 2           Y                              0
##   voting.fctr.predict.All.X.rf.Y
## 1                           3421
## 2                           1579
##           Reference
## Prediction    N    Y
##          N 2861 1245
##          Y  560  334
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           2861
## 2           Y                           1245
##   voting.fctr.predict.All.X.rf.Y
## 1                            560
## 2                            334
##           Reference
## Prediction    N    Y
##          N 3201 1445
##          Y  220  134
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3201
## 2           Y                           1445
##   voting.fctr.predict.All.X.rf.Y
## 1                            220
## 2                            134
##           Reference
## Prediction    N    Y
##          N 3353 1532
##          Y   68   47
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3353
## 2           Y                           1532
##   voting.fctr.predict.All.X.rf.Y
## 1                             68
## 2                             47
##           Reference
## Prediction    N    Y
##          N 3410 1576
##          Y   11    3
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3410
## 2           Y                           1576
##   voting.fctr.predict.All.X.rf.Y
## 1                             11
## 2                              3
##           Reference
## Prediction    N    Y
##          N 3420 1579
##          Y    1    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3420
## 2           Y                           1579
##   voting.fctr.predict.All.X.rf.Y
## 1                              1
## 2                              0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3421
## 2           Y                           1579
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3421
## 2           Y                           1579
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3421
## 2           Y                           1579
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3421
## 2           Y                           1579
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              0
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                           3421
## 2           Y                           1579
##   voting.fctr.predict.All.X.rf.Y
## 1                              0
## 2                              0
##    threshold     f.score
## 1        0.0 0.480012160
## 2        0.1 0.270117266
## 3        0.2 0.138644594
## 4        0.3 0.055489965
## 5        0.4 0.003766478
## 6        0.5 0.000000000
## 7        0.6 0.000000000
## 8        0.7 0.000000000
## 9        0.8 0.000000000
## 10       0.9 0.000000000
## 11       1.0 0.000000000
```

![](Michigan_Voters_files/figure-html/fit.models-69.png) 

```
## [1] "Classifier Probability Threshold: 0.0000 to maximize f.score.OOB"
##   voting.fctr voting.fctr.predict.All.X.rf.Y
## 1           N                           3421
## 2           Y                           1579
##           Reference
## Prediction    N    Y
##          N    0    0
##          Y 3421 1579
##   voting.fctr voting.fctr.predict.All.X.rf.N
## 1           N                              0
## 2           Y                              0
##   voting.fctr.predict.All.X.rf.Y
## 1                           3421
## 2                           1579
##          Prediction
## Reference    N    Y
##         N    0 3421
##         Y    0 1579
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.3158000      0.0000000      0.3029263      0.3288891      0.6842000 
## AccuracyPValue  McnemarPValue 
##      1.0000000      0.0000000 
##   model_id model_method
## 1 All.X.rf           rf
##                                                              feats
## 1 sex, yob, hawthorne, civicduty, neighbors, self, control, .rnorm
##   max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1               3                     22.652                 3.236
##   max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1    0.738763                      0       0.4802432           0.6834
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.303124             0.3290911  -0.001199162   0.5474452
##   opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                      0       0.4800122           0.3158
##   max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1             0.3029263             0.3288891             0
```

```r
# Simplify a model
# fit_df <- glb_trnent_df; glb_mdl <- step(<complex>_mdl)

print(glb_models_df)
```

```
##                     model_id     model_method
## 1          MFO.myMFO_classfr    myMFO_classfr
## 2    Random.myrandom_classfr myrandom_classfr
## 3       Max.cor.Y.cv.0.rpart            rpart
## 4  Max.cor.Y.cv.0.cp.0.rpart            rpart
## 5            Max.cor.Y.rpart            rpart
## 6              Max.cor.Y.glm              glm
## 7              Low.cor.X.glm              glm
## 8               Groups.X.glm              glm
## 9              GrpGndr.X.glm              glm
## 10                 All.X.glm              glm
## 11               All.X.rpart            rpart
## 12          All.X.cp.0.rpart            rpart
## 13                  All.X.rf               rf
##                                                               feats
## 1                                                            .rnorm
## 2                                                            .rnorm
## 3                                                               yob
## 4                                                               yob
## 5                                                               yob
## 6                                                               yob
## 7  neighbors, self, hawthorne, .rnorm, civicduty, sex, control, yob
## 8            hawthorne, civicduty, neighbors, self, control, .rnorm
## 9       hawthorne, civicduty, neighbors, self, control, .rnorm, sex
## 10 sex, yob, hawthorne, civicduty, neighbors, self, control, .rnorm
## 11         sex, yob, hawthorne, civicduty, neighbors, self, control
## 12         sex, yob, hawthorne, civicduty, neighbors, self, control
## 13 sex, yob, hawthorne, civicduty, neighbors, self, control, .rnorm
##    max.nTuningRuns min.elapsedtime.everything min.elapsedtime.final
## 1                0                      0.341                 0.003
## 2                0                      0.231                 0.002
## 3                0                      0.655                 0.079
## 4                0                      0.517                 0.073
## 5                3                      1.409                 0.078
## 6                1                      1.063                 0.088
## 7                1                      1.372                 0.175
## 8                1                      1.620                 0.147
## 9                1                      1.408                 0.164
## 10               1                      1.501                 0.184
## 11               3                      1.669                 0.155
## 12               0                      0.788                 0.145
## 13               3                     22.652                 3.236
##    max.auc.fit opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1    0.5000000                    0.5       0.0000000        0.6840000
## 2    0.5082464                    0.3       0.4802432        0.3160000
## 3    0.5000000                    0.5       0.0000000        0.6840000
## 4    0.5649086                    0.2       0.4802432        0.3160000
## 5    0.5000000                    0.5       0.0000000        0.6785971
## 6    0.5724227                    0.2       0.4802432        0.6844003
## 7    0.5877905                    0.2       0.4865286        0.6830003
## 8    0.5474284                    0.2       0.4802432        0.6840001
## 9    0.5519065                    0.2       0.4802432        0.6840001
## 10   0.5877905                    0.2       0.4865286        0.6830003
## 11   0.5689083                    0.2       0.4805234        0.6799964
## 12   0.6396408                    0.2       0.4940867        0.3840000
## 13   0.7387630                    0.0       0.4802432        0.6834000
##    max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit max.auc.OOB
## 1              0.6709089             0.6968760   0.000000000   0.5000000
## 2              0.3031240             0.3290911   0.000000000   0.5004063
## 3              0.6709089             0.6968760   0.000000000   0.5000000
## 4              0.3031240             0.3290911   0.000000000   0.5515162
## 5              0.6709089             0.6968760   0.013412438   0.5000000
## 6              0.3031240             0.3290911   0.002658553   0.5845416
## 7              0.3274603             0.3539260   0.007655516   0.5819886
## 8              0.3031240             0.3290911   0.000000000   0.5239316
## 9              0.3031240             0.3290911   0.000000000   0.5237511
## 10             0.3274603             0.3539260   0.007655516   0.5819886
## 11             0.3043102             0.3303036   0.068441453   0.5498702
## 12             0.3704901             0.3976455   0.049072784   0.5711956
## 13             0.3031240             0.3290911  -0.001199162   0.5474452
##    opt.prob.threshold.OOB max.f.score.OOB max.Accuracy.OOB
## 1                     0.5       0.0000000           0.6842
## 2                     0.3       0.4800122           0.3158
## 3                     0.5       0.0000000           0.6842
## 4                     0.2       0.4800122           0.3158
## 5                     0.5       0.0000000           0.6842
## 6                     0.2       0.4800122           0.3158
## 7                     0.2       0.4857857           0.3416
## 8                     0.2       0.4800122           0.3158
## 9                     0.2       0.4800122           0.3158
## 10                    0.2       0.4857857           0.3416
## 11                    0.1       0.4800122           0.3158
## 12                    0.2       0.4909510           0.3812
## 13                    0.0       0.4800122           0.3158
##    max.AccuracyLower.OOB max.AccuracyUpper.OOB max.Kappa.OOB
## 1              0.6711109             0.6970737    0.00000000
## 2              0.3029263             0.3288891    0.00000000
## 3              0.6711109             0.6970737    0.00000000
## 4              0.3029263             0.3288891    0.00000000
## 5              0.6711109             0.6970737    0.00000000
## 6              0.3029263             0.3288891    0.00000000
## 7              0.3284504             0.3549347    0.01900987
## 8              0.3029263             0.3288891    0.00000000
## 9              0.3029263             0.3288891    0.00000000
## 10             0.3284504             0.3549347    0.01900987
## 11             0.3029263             0.3288891    0.00000000
## 12             0.3677105             0.3948284    0.04400884
## 13             0.3029263             0.3288891    0.00000000
##    max.AccuracySD.fit max.KappaSD.fit min.aic.fit
## 1                  NA              NA          NA
## 2                  NA              NA          NA
## 3                  NA              NA          NA
## 4                  NA              NA          NA
## 5         0.012929471     0.012542196          NA
## 6         0.001480510     0.005674799    6176.245
## 7         0.001142904     0.010271048    6150.502
## 8         0.000236992     0.000000000    6214.036
## 9         0.000236992     0.000000000    6214.066
## 10        0.001142904     0.010271048    6150.502
## 11        0.015601458     0.019076173          NA
## 12                 NA              NA          NA
## 13                 NA              NA          NA
```

```r
if (!is.null(glb_model_metric_smmry)) {
    stats_df <- glb_models_df[, "model_id", FALSE]

    stats_mdl_df <- data.frame()
    for (model_id in stats_df$model_id) {
        stats_mdl_df <- rbind(stats_mdl_df, 
            mypredict_mdl(glb_models_lst[[model_id]], glb_trnent_df, glb_rsp_var, 
                          glb_rsp_var_out, model_id, "fit",
        						glb_model_metric_smmry, glb_model_metric, 
        						glb_model_metric_maximize, ret_type="stats"))
    }
    stats_df <- merge(stats_df, stats_mdl_df, all.x=TRUE)
    
    stats_mdl_df <- data.frame()
    for (model_id in stats_df$model_id) {
        stats_mdl_df <- rbind(stats_mdl_df, 
            mypredict_mdl(glb_models_lst[[model_id]], glb_newent_df, glb_rsp_var, 
                          glb_rsp_var_out, model_id, "OOB",
            					glb_model_metric_smmry, glb_model_metric, 
        						glb_model_metric_maximize, ret_type="stats"))
    }
    stats_df <- merge(stats_df, stats_mdl_df, all.x=TRUE)
    
#     tmp_models_df <- orderBy(~model_id, glb_models_df)
#     rownames(tmp_models_df) <- seq(1, nrow(tmp_models_df))
#     all.equal(subset(tmp_models_df[, names(stats_df)], model_id != "Random.myrandom_classfr"),
#               subset(stats_df, model_id != "Random.myrandom_classfr"))
#     print(subset(tmp_models_df[, names(stats_df)], model_id != "Random.myrandom_classfr")[, c("model_id", "max.Accuracy.fit")])
#     print(subset(stats_df, model_id != "Random.myrandom_classfr")[, c("model_id", "max.Accuracy.fit")])

    print("Merging following data into glb_models_df:")
    print(stats_mrg_df <- stats_df[, c(1, grep(glb_model_metric, names(stats_df)))])
    print(tmp_models_df <- orderBy(~model_id, glb_models_df[, c("model_id", grep(glb_model_metric, names(stats_df), value=TRUE))]))

    tmp2_models_df <- glb_models_df[, c("model_id", setdiff(names(glb_models_df), grep(glb_model_metric, names(stats_df), value=TRUE)))]
    tmp3_models_df <- merge(tmp2_models_df, stats_mrg_df, all.x=TRUE, sort=FALSE)
    print(tmp3_models_df)
    print(names(tmp3_models_df))
    print(glb_models_df <- subset(tmp3_models_df, select=-model_id.1))
}

plt_models_df <- glb_models_df[, -grep("SD|Upper|Lower", names(glb_models_df))]
for (var in grep("^min.", names(plt_models_df), value=TRUE)) {
    plt_models_df[, sub("min.", "inv.", var)] <- 
        #ifelse(all(is.na(tmp <- plt_models_df[, var])), NA, 1.0 / tmp)
        1.0 / plt_models_df[, var]
    plt_models_df <- plt_models_df[ , -grep(var, names(plt_models_df))]
}
print(plt_models_df)
```

```
##                     model_id     model_method
## 1          MFO.myMFO_classfr    myMFO_classfr
## 2    Random.myrandom_classfr myrandom_classfr
## 3       Max.cor.Y.cv.0.rpart            rpart
## 4  Max.cor.Y.cv.0.cp.0.rpart            rpart
## 5            Max.cor.Y.rpart            rpart
## 6              Max.cor.Y.glm              glm
## 7              Low.cor.X.glm              glm
## 8               Groups.X.glm              glm
## 9              GrpGndr.X.glm              glm
## 10                 All.X.glm              glm
## 11               All.X.rpart            rpart
## 12          All.X.cp.0.rpart            rpart
## 13                  All.X.rf               rf
##                                                               feats
## 1                                                            .rnorm
## 2                                                            .rnorm
## 3                                                               yob
## 4                                                               yob
## 5                                                               yob
## 6                                                               yob
## 7  neighbors, self, hawthorne, .rnorm, civicduty, sex, control, yob
## 8            hawthorne, civicduty, neighbors, self, control, .rnorm
## 9       hawthorne, civicduty, neighbors, self, control, .rnorm, sex
## 10 sex, yob, hawthorne, civicduty, neighbors, self, control, .rnorm
## 11         sex, yob, hawthorne, civicduty, neighbors, self, control
## 12         sex, yob, hawthorne, civicduty, neighbors, self, control
## 13 sex, yob, hawthorne, civicduty, neighbors, self, control, .rnorm
##    max.nTuningRuns max.auc.fit opt.prob.threshold.fit max.f.score.fit
## 1                0   0.5000000                    0.5       0.0000000
## 2                0   0.5082464                    0.3       0.4802432
## 3                0   0.5000000                    0.5       0.0000000
## 4                0   0.5649086                    0.2       0.4802432
## 5                3   0.5000000                    0.5       0.0000000
## 6                1   0.5724227                    0.2       0.4802432
## 7                1   0.5877905                    0.2       0.4865286
## 8                1   0.5474284                    0.2       0.4802432
## 9                1   0.5519065                    0.2       0.4802432
## 10               1   0.5877905                    0.2       0.4865286
## 11               3   0.5689083                    0.2       0.4805234
## 12               0   0.6396408                    0.2       0.4940867
## 13               3   0.7387630                    0.0       0.4802432
##    max.Accuracy.fit max.Kappa.fit max.auc.OOB opt.prob.threshold.OOB
## 1         0.6840000   0.000000000   0.5000000                    0.5
## 2         0.3160000   0.000000000   0.5004063                    0.3
## 3         0.6840000   0.000000000   0.5000000                    0.5
## 4         0.3160000   0.000000000   0.5515162                    0.2
## 5         0.6785971   0.013412438   0.5000000                    0.5
## 6         0.6844003   0.002658553   0.5845416                    0.2
## 7         0.6830003   0.007655516   0.5819886                    0.2
## 8         0.6840001   0.000000000   0.5239316                    0.2
## 9         0.6840001   0.000000000   0.5237511                    0.2
## 10        0.6830003   0.007655516   0.5819886                    0.2
## 11        0.6799964   0.068441453   0.5498702                    0.1
## 12        0.3840000   0.049072784   0.5711956                    0.2
## 13        0.6834000  -0.001199162   0.5474452                    0.0
##    max.f.score.OOB max.Accuracy.OOB max.Kappa.OOB
## 1        0.0000000           0.6842    0.00000000
## 2        0.4800122           0.3158    0.00000000
## 3        0.0000000           0.6842    0.00000000
## 4        0.4800122           0.3158    0.00000000
## 5        0.0000000           0.6842    0.00000000
## 6        0.4800122           0.3158    0.00000000
## 7        0.4857857           0.3416    0.01900987
## 8        0.4800122           0.3158    0.00000000
## 9        0.4800122           0.3158    0.00000000
## 10       0.4857857           0.3416    0.01900987
## 11       0.4800122           0.3158    0.00000000
## 12       0.4909510           0.3812    0.04400884
## 13       0.4800122           0.3158    0.00000000
##    inv.elapsedtime.everything inv.elapsedtime.final  inv.aic.fit
## 1                  2.93255132           333.3333333           NA
## 2                  4.32900433           500.0000000           NA
## 3                  1.52671756            12.6582278           NA
## 4                  1.93423598            13.6986301           NA
## 5                  0.70972321            12.8205128           NA
## 6                  0.94073377            11.3636364 0.0001619107
## 7                  0.72886297             5.7142857 0.0001625884
## 8                  0.61728395             6.8027211 0.0001609260
## 9                  0.71022727             6.0975610 0.0001609252
## 10                 0.66622252             5.4347826 0.0001625884
## 11                 0.59916117             6.4516129           NA
## 12                 1.26903553             6.8965517           NA
## 13                 0.04414621             0.3090235           NA
```

```r
print(myplot_radar(radar_inp_df=plt_models_df))
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 13. Consider specifying shapes manually. if you must have them.
```

```
## Warning: Removed 5 rows containing missing values (geom_path).
```

```
## Warning: Removed 103 rows containing missing values (geom_point).
```

```
## Warning: Removed 8 rows containing missing values (geom_text).
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 13. Consider specifying shapes manually. if you must have them.
```

![](Michigan_Voters_files/figure-html/fit.models-70.png) 

```r
# print(myplot_radar(radar_inp_df=subset(plt_models_df, 
#         !(model_id %in% grep("random|MFO", plt_models_df$model_id, value=TRUE)))))

# Compute CI for <metric>SD
glb_models_df <- mutate(glb_models_df, 
                max.df = ifelse(max.nTuningRuns > 1, max.nTuningRuns - 1, NA),
                min.sd2ci.scaler = ifelse(is.na(max.df), NA, qt(0.975, max.df)))
for (var in grep("SD", names(glb_models_df), value=TRUE)) {
    # Does CI alredy exist ?
    var_components <- unlist(strsplit(var, "SD"))
    varActul <- paste0(var_components[1],          var_components[2])
    varUpper <- paste0(var_components[1], "Upper", var_components[2])
    varLower <- paste0(var_components[1], "Lower", var_components[2])
    if (varUpper %in% names(glb_models_df)) {
        warning(varUpper, " already exists in glb_models_df")
        # Assuming Lower also exists
        next
    }    
    print(sprintf("var:%s", var))
    # CI is dependent on sample size in t distribution; df=n-1
    glb_models_df[, varUpper] <- glb_models_df[, varActul] + 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
    glb_models_df[, varLower] <- glb_models_df[, varActul] - 
        glb_models_df[, "min.sd2ci.scaler"] * glb_models_df[, var]
}
```

```
## Warning: max.AccuracyUpper.fit already exists in glb_models_df
```

```
## [1] "var:max.KappaSD.fit"
```

```r
# Plot metrics with CI
plt_models_df <- glb_models_df[, "model_id", FALSE]
pltCI_models_df <- glb_models_df[, "model_id", FALSE]
for (var in grep("Upper", names(glb_models_df), value=TRUE)) {
    var_components <- unlist(strsplit(var, "Upper"))
    col_name <- unlist(paste(var_components, collapse=""))
    plt_models_df[, col_name] <- glb_models_df[, col_name]
    for (name in paste0(var_components[1], c("Upper", "Lower"), var_components[2]))
        pltCI_models_df[, name] <- glb_models_df[, name]
}

build_statsCI_data <- function(plt_models_df) {
    mltd_models_df <- melt(plt_models_df, id.vars="model_id")
    mltd_models_df$data <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) tail(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), "[.]")), 1))
    mltd_models_df$label <- sapply(1:nrow(mltd_models_df), 
        function(row_ix) head(unlist(strsplit(as.character(
            mltd_models_df[row_ix, "variable"]), paste0(".", mltd_models_df[row_ix, "data"]))), 1))
    #print(mltd_models_df)
    
    return(mltd_models_df)
}
mltd_models_df <- build_statsCI_data(plt_models_df)

mltdCI_models_df <- melt(pltCI_models_df, id.vars="model_id")
for (row_ix in 1:nrow(mltdCI_models_df)) {
    for (type in c("Upper", "Lower")) {
        if (length(var_components <- unlist(strsplit(
                as.character(mltdCI_models_df[row_ix, "variable"]), type))) > 1) {
            #print(sprintf("row_ix:%d; type:%s; ", row_ix, type))
            mltdCI_models_df[row_ix, "label"] <- var_components[1]
            mltdCI_models_df[row_ix, "data"] <- unlist(strsplit(var_components[2], "[.]"))[2]
            mltdCI_models_df[row_ix, "type"] <- type
            break
        }
    }    
}
#print(mltdCI_models_df)
# castCI_models_df <- dcast(mltdCI_models_df, value ~ type, fun.aggregate=sum)
# print(castCI_models_df)
wideCI_models_df <- reshape(subset(mltdCI_models_df, select=-variable), 
                            timevar="type", 
        idvar=setdiff(names(mltdCI_models_df), c("type", "value", "variable")), 
                            direction="wide")
#print(wideCI_models_df)
mrgdCI_models_df <- merge(wideCI_models_df, mltd_models_df, all.x=TRUE)
#print(mrgdCI_models_df)

# Merge stats back in if CIs don't exist
goback_vars <- c()
for (var in unique(mltd_models_df$label)) {
    for (type in unique(mltd_models_df$data)) {
        var_type <- paste0(var, ".", type)
        # if this data is already present, next
        if (var_type %in% unique(paste(mltd_models_df$label, mltd_models_df$data, sep=".")))
            next
        #print(sprintf("var_type:%s", var_type))
        goback_vars <- c(goback_vars, var_type)
    }
}

if (length(goback_vars) > 0) {
    mltd_goback_df <- build_statsCI_data(glb_models_df[, c("model_id", goback_vars)])
    mltd_models_df <- rbind(mltd_models_df, mltd_goback_df)
}

mltd_models_df <- merge(mltd_models_df, glb_models_df[, c("model_id", "model_method")], all.x=TRUE)

# print(myplot_bar(mltd_models_df, "model_id", "value", colorcol_name="data") + 
#         geom_errorbar(data=mrgdCI_models_df, 
#             mapping=aes(x=model_id, ymax=value.Upper, ymin=value.Lower), width=0.5) + 
#           facet_grid(label ~ data, scales="free") + 
#           theme(axis.text.x = element_text(angle = 45,vjust = 1)))
# mltd_models_df <- orderBy(~ value +variable +data +label + model_method + model_id, 
#                           mltd_models_df)
print(myplot_bar(mltd_models_df, "model_id", "value", colorcol_name="model_method") + 
        geom_errorbar(data=mrgdCI_models_df, 
            mapping=aes(x=model_id, ymax=value.Upper, ymin=value.Lower), width=0.5) + 
          facet_grid(label ~ data, scales="free") + 
          theme(axis.text.x = element_text(angle = 90,vjust = 0.5)))
```

```
## Warning: Stacking not well defined when ymin != 0
```

![](Michigan_Voters_files/figure-html/fit.models-71.png) 

```r
model_evl_terms <- c(NULL)
for (metric in glb_model_evl_criteria)
    model_evl_terms <- c(model_evl_terms, 
                    ifelse(length(grep("max", metric)) > 0, "-", "+"), metric)
model_sel_frmla <- as.formula(paste(c("~ ", model_evl_terms), collapse=" "))
print(tmp_models_df <- orderBy(model_sel_frmla, glb_models_df)[, c("model_id", glb_model_evl_criteria)])
```

```
##                     model_id max.Accuracy.OOB max.Kappa.OOB min.aic.fit
## 1          MFO.myMFO_classfr           0.6842    0.00000000          NA
## 3       Max.cor.Y.cv.0.rpart           0.6842    0.00000000          NA
## 5            Max.cor.Y.rpart           0.6842    0.00000000          NA
## 12          All.X.cp.0.rpart           0.3812    0.04400884          NA
## 7              Low.cor.X.glm           0.3416    0.01900987    6150.502
## 10                 All.X.glm           0.3416    0.01900987    6150.502
## 6              Max.cor.Y.glm           0.3158    0.00000000    6176.245
## 8               Groups.X.glm           0.3158    0.00000000    6214.036
## 9              GrpGndr.X.glm           0.3158    0.00000000    6214.066
## 2    Random.myrandom_classfr           0.3158    0.00000000          NA
## 4  Max.cor.Y.cv.0.cp.0.rpart           0.3158    0.00000000          NA
## 11               All.X.rpart           0.3158    0.00000000          NA
## 13                  All.X.rf           0.3158    0.00000000          NA
```

```r
print(myplot_radar(radar_inp_df=tmp_models_df))
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 13. Consider specifying shapes manually. if you must have them.
```

```
## Warning: Removed 5 rows containing missing values (geom_path).
```

```
## Warning: Removed 26 rows containing missing values (geom_point).
```

```
## Warning: Removed 8 rows containing missing values (geom_text).
```

```
## Warning in RColorBrewer::brewer.pal(n, pal): n too large, allowed maximum for palette Set1 is 9
## Returning the palette you asked for with that many colors
```

```
## Warning: The shape palette can deal with a maximum of 6 discrete values
## because more than 6 becomes difficult to discriminate; you have
## 13. Consider specifying shapes manually. if you must have them.
```

![](Michigan_Voters_files/figure-html/fit.models-72.png) 

```r
print("Metrics used for model selection:"); print(model_sel_frmla)
```

```
## [1] "Metrics used for model selection:"
```

```
## ~-max.Accuracy.OOB - max.Kappa.OOB + min.aic.fit
```

```r
print(sprintf("Best model id: %s", tmp_models_df[1, "model_id"]))
```

```
## [1] "Best model id: MFO.myMFO_classfr"
```

```r
if (is.null(glb_sel_mdl_id)) 
    { glb_sel_mdl_id <- tmp_models_df[1, "model_id"] } else 
        print(sprintf("User specified selection: %s", glb_sel_mdl_id))   
    
myprint_mdl(glb_sel_mdl <- glb_models_lst[[glb_sel_mdl_id]])
```

```
##             Length Class      Mode     
## unique.vals 2      factor     numeric  
## unique.prob 2      -none-     numeric  
## MFO.val     1      -none-     character
## x.names     1      -none-     character
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   2      -none-     character
```

```
## [1] TRUE
```

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "model.selected")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0
```

![](Michigan_Voters_files/figure-html/fit.models-73.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                    chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed8            fit.models                5                0  37.437
## elapsed9 fit.data.training.all                6                0 146.141
```

## Step `6`: fit.data.training.all

```r
if (!is.null(glb_fin_mdl_id) && (glb_fin_mdl_id %in% names(glb_models_lst))) {
    warning("Final model same as user selected model")
    glb_fin_mdl <- glb_sel_mdl
} else {    
    print(mdl_feats_df <- myextract_mdl_feats(sel_mdl=glb_sel_mdl, entity_df=glb_trnent_df))
    
    if ((model_method <- glb_sel_mdl$method) == "custom")
        # get actual method from the model_id
        model_method <- tail(unlist(strsplit(glb_sel_mdl_id, "[.]")), 1)
        
    # Sync with parameters in mydsutils.R    
    ret_lst <- myfit_mdl(model_id="Final", model_method=model_method,
                            indep_vars_vctr=mdl_feats_df$id, model_type=glb_model_type,
                            rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out, 
                            fit_df=glb_trnent_df, OOB_df=NULL,
                         # Automate from here
                         #  Issues if glb_sel_mdl$method == "rf" b/c trainControl is "oob"; not "cv"
                         n_cv_folds=glb_n_cv_folds, tune_models_df=NULL,
                            model_loss_mtrx=glb_model_metric_terms,
                            model_summaryFunction=glb_sel_mdl$control$summaryFunction,
                            model_metric=glb_sel_mdl$metric,
                            model_metric_maximize=glb_sel_mdl$maximize)
    glb_fin_mdl <- glb_models_lst[[length(glb_models_lst)]]
    glb_fin_mdl_id <- glb_models_df[length(glb_models_lst), "model_id"]
}
```

```
## [1] "in MFO.Classifier$varImp"
##        Overall
## .rnorm       0
##        importance     id fit.feat
## .rnorm        NaN .rnorm     TRUE
## [1] "fitting model: Final.myMFO_classfr"
## [1] "    indep_vars: .rnorm"
## + Fold1: parameter=none 
## [1] "in MFO.Classifier$fit"
## [1] "unique.vals:"
## [1] N Y
## Levels: N Y
## [1] "unique.prob:"
## y
##         N         Y 
## 0.6840684 0.3159316 
## [1] "MFO.val:"
## [1] "N"
## [1] "in MFO.Classifier$predict"
## - Fold1: parameter=none 
## + Fold2: parameter=none 
## [1] "in MFO.Classifier$fit"
## [1] "unique.vals:"
## [1] N Y
## Levels: N Y
## [1] "unique.prob:"
## y
##         N         Y 
## 0.6838632 0.3161368 
## [1] "MFO.val:"
## [1] "N"
## [1] "in MFO.Classifier$predict"
## - Fold2: parameter=none 
## + Fold3: parameter=none 
## [1] "in MFO.Classifier$fit"
## [1] "unique.vals:"
## [1] N Y
## Levels: N Y
## [1] "unique.prob:"
## y
##         N         Y 
## 0.6840684 0.3159316 
## [1] "MFO.val:"
## [1] "N"
## [1] "in MFO.Classifier$predict"
## - Fold3: parameter=none 
## Aggregating results
## Fitting final model on full training set
## [1] "in MFO.Classifier$fit"
## [1] "unique.vals:"
## [1] N Y
## Levels: N Y
## [1] "unique.prob:"
## y
##     N     Y 
## 0.684 0.316 
## [1] "MFO.val:"
## [1] "N"
##             Length Class      Mode     
## unique.vals 2      factor     numeric  
## unique.prob 2      -none-     numeric  
## MFO.val     1      -none-     character
## x.names     1      -none-     character
## xNames      1      -none-     character
## problemType 1      -none-     character
## tuneValue   1      data.frame list     
## obsLevels   2      -none-     character
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##       N     Y
## 1 0.684 0.316
## 2 0.684 0.316
## 3 0.684 0.316
## 4 0.684 0.316
## 5 0.684 0.316
## 6 0.684 0.316
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.fit"
##   voting.fctr voting.fctr.predict.Final.myMFO_classfr.N
## 1           N                                      3420
## 2           Y                                      1580
##           Reference
## Prediction    N    Y
##          N 3420 1580
##          Y    0    0
##   voting.fctr voting.fctr.predict.Final.myMFO_classfr.N
## 1           N                                      3420
## 2           Y                                      1580
##   voting.fctr.predict.Final.myMFO_classfr.Y
## 1                                         0
## 2                                         0
##          Prediction
## Reference    N    Y
##         N 3420    0
##         Y 1580    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.6840000      0.0000000      0.6709089      0.6968760      0.6840000 
## AccuracyPValue  McnemarPValue 
##      0.5068114      0.0000000
```

```
## Warning in mypredict_mdl(mdl, df = fit_df, rsp_var, rsp_var_out,
## model_id_method, : Expecting 1 metric: Accuracy; recd: Accuracy, Kappa;
## retaining Accuracy only
```

```
##              model_id  model_method  feats max.nTuningRuns
## 1 Final.myMFO_classfr myMFO_classfr .rnorm               1
##   min.elapsedtime.everything min.elapsedtime.final max.auc.fit
## 1                      0.888                 0.003         0.5
##   opt.prob.threshold.fit max.f.score.fit max.Accuracy.fit
## 1                    0.5               0        0.6840001
##   max.AccuracyLower.fit max.AccuracyUpper.fit max.Kappa.fit
## 1             0.6709089              0.696876             0
##   max.AccuracySD.fit max.KappaSD.fit
## 1        0.000236992               0
```

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="fit.data.training.all", 
    chunk_step_major=glb_script_df[nrow(glb_script_df), "chunk_step_major"], 
    chunk_step_minor=glb_script_df[nrow(glb_script_df), "chunk_step_minor"]+1,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed9  fit.data.training.all                6                0 146.141
## elapsed10 fit.data.training.all                6                1 161.115
```


```r
glb_rsp_var_out <- paste0(glb_rsp_var_out, tail(names(glb_models_lst), 1))
if (glb_is_regression) {
    glb_trnent_df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=glb_trnent_df, type="raw")
    print(myplot_scatter(glb_trnent_df, glb_rsp_var, glb_rsp_var_out, 
                         smooth=TRUE))
    glb_trnent_df[, paste0(glb_rsp_var_out, ".err")] <- 
        abs(glb_trnent_df[, glb_rsp_var_out] - glb_trnent_df[, glb_rsp_var])
    print(head(orderBy(reformulate(c("-", paste0(glb_rsp_var_out, ".err"))), 
                       glb_trnent_df)))                             
}    

if (glb_is_classification && glb_is_binomial) {
    # incorporate glb_clf_proba_threshold
    #   shd it only be for glb_fin_mdl or for earlier models ?
    if ((prob_threshold <- 
            glb_models_df[glb_models_df$model_id == glb_fin_mdl_id, "opt.prob.threshold.fit"]) != 
        glb_models_df[glb_models_df$model_id == glb_sel_mdl_id, "opt.prob.threshold.fit"])
        stop("user specification for probability threshold required")
    
    glb_trnent_df[, paste0(glb_rsp_var_out, ".prob")] <- 
        predict(glb_fin_mdl, newdata=glb_trnent_df, type="prob")[, 2]
    glb_trnent_df[, glb_rsp_var_out] <- 
			factor(levels(glb_trnent_df[, glb_rsp_var])[
				(glb_trnent_df[, paste0(glb_rsp_var_out, ".prob")] >=
					prob_threshold) * 1 + 1], levels(glb_trnent_df[, glb_rsp_var]))

    # prediction stats already reported by myfit_mdl ???
}    
```

```
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##       N     Y
## 1 0.684 0.316
## 2 0.684 0.316
## 3 0.684 0.316
## 4 0.684 0.316
## 5 0.684 0.316
## 6 0.684 0.316
```

```r
if (glb_is_classification && !glb_is_binomial) {
    glb_trnent_df[, glb_rsp_var_out] <- predict(glb_fin_mdl, newdata=glb_trnent_df, type="raw")
}    

print(glb_feats_df <- mymerge_feats_importance(feats_df=glb_feats_df, sel_mdl=glb_fin_mdl, 
                                               entity_df=glb_trnent_df))
```

```
## [1] "in MFO.Classifier$varImp"
##        Overall
## .rnorm       0
##          id        cor.y exclude.as.feat   cor.y.abs cor.low importance
## 1    .rnorm  0.007838379               0 0.007838379       1        NaN
## 2 civicduty  0.005217928               0 0.005217928       1         NA
## 3   control -0.076267307               0 0.076267307       1         NA
## 4 hawthorne  0.020770759               0 0.020770759       1         NA
## 5 neighbors  0.061253311               0 0.061253311       1         NA
## 6      self  0.033691805               0 0.033691805       1         NA
## 7       sex -0.017971837               0 0.017971837       1         NA
## 8    voting  1.000000000               1 1.000000000       0         NA
## 9       yob -0.114539250               0 0.114539250       1         NA
```

```r
# Most of this code is used again in predict.data.new chunk
glb_analytics_diag_plots <- function(obs_df) {
    for (var in subset(glb_feats_df, !is.na(importance))$id) {
        plot_df <- melt(obs_df, id.vars=var, 
                        measure.vars=c(glb_rsp_var, glb_rsp_var_out))
#         if (var == "<feat_name>") print(myplot_scatter(plot_df, var, "value", 
#                                              facet_colcol_name="variable") + 
#                       geom_vline(xintercept=<divider_val>, linetype="dotted")) else     
            print(myplot_scatter(plot_df, var, "value", colorcol_name="variable",
                                 facet_colcol_name="variable", jitter=TRUE) + 
                      guides(color=FALSE))
    }
    
    if (glb_is_regression) {
        plot_vars_df <- subset(glb_feats_df, importance > glb_feats_df[glb_feats_df$id == ".rnorm", "importance"])
        print(myplot_prediction_regression(df=obs_df, 
                    feat_x=ifelse(nrow(plot_vars_df) > 1, plot_vars_df$id[2], ".rownames"), 
                                           feat_y=plot_vars_df$id[1],
                    rsp_var=glb_rsp_var, rsp_var_out=glb_rsp_var_out,
                    id_vars=glb_id_vars)
#               + facet_wrap(reformulate(plot_vars_df$id[2])) # if [1 or 2] is a factor                                                         
#               + geom_point(aes_string(color="<col_name>.fctr")) #  to color the plot
              )
    }    
    
    if (glb_is_classification) {
        if (nrow(plot_vars_df <- subset(glb_feats_df, !is.na(importance))) == 0)
            warning("No features in selected model are statistically important")
        else print(myplot_prediction_classification(df=obs_df, 
                feat_x=ifelse(nrow(plot_vars_df) > 1, plot_vars_df$id[2], 
                              ".rownames"),
                                               feat_y=plot_vars_df$id[1],
                     rsp_var=glb_rsp_var, 
                     rsp_var_out=glb_rsp_var_out, 
                     id_vars=glb_id_vars)
#               + geom_hline(yintercept=<divider_val>, linetype = "dotted")
                )
    }    
}
glb_analytics_diag_plots(obs_df=glb_trnent_df)
```

```
## Warning in glb_analytics_diag_plots(obs_df = glb_trnent_df): No features
## in selected model are statistically important
```

```r
replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.training.all.prediction","model.final")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1
```

![](Michigan_Voters_files/figure-html/fit.data.training.all_1-1.png) 

```r
glb_script_df <- rbind(glb_script_df, 
                   data.frame(chunk_label="predict.data.new", 
                              chunk_step_major=max(glb_script_df$chunk_step_major)+1, 
                              chunk_step_minor=0,
                              elapsed=(proc.time() - glb_script_tm)["elapsed"]))
print(tail(glb_script_df, 2))
```

```
##                     chunk_label chunk_step_major chunk_step_minor elapsed
## elapsed10 fit.data.training.all                6                1 161.115
## elapsed11      predict.data.new                7                0 162.270
```

## Step `7`: predict data.new

```r
# Compute final model predictions
glb_newent_df[, glb_rsp_var_out] <- 
    mypredict_mdl(glb_fin_mdl, glb_newent_df, glb_rsp_var, glb_rsp_var_out, 
                  "Final", "Final",
                  glb_model_metric_smmry, glb_model_metric, 
                  glb_model_metric_maximize, ret_type="raw")
```

```
## [1] "in MFO.Classifier$predict"
## [1] "in MFO.Classifier$prob"
##       N     Y
## 1 0.684 0.316
## 2 0.684 0.316
## 3 0.684 0.316
## 4 0.684 0.316
## 5 0.684 0.316
## 6 0.684 0.316
## [1] "Classifier Probability Threshold: 0.5000 to maximize f.score.Final"
##   voting.fctr voting.fctr.predict.Final.myMFO_classfr.N
## 1           N                                      3421
## 2           Y                                      1579
##           Reference
## Prediction    N    Y
##          N 3421 1579
##          Y    0    0
##   voting.fctr voting.fctr.predict.Final.myMFO_classfr.N
## 1           N                                      3421
## 2           Y                                      1579
##   voting.fctr.predict.Final.myMFO_classfr.Y
## 1                                         0
## 2                                         0
##          Prediction
## Reference    N    Y
##         N 3421    0
##         Y 1579    0
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.6842000      0.0000000      0.6711109      0.6970737      0.6842000 
## AccuracyPValue  McnemarPValue 
##      0.5068134      0.0000000
```

```r
glb_analytics_diag_plots(obs_df=glb_newent_df)
```

```
## Warning in glb_analytics_diag_plots(obs_df = glb_newent_df): No features
## in selected model are statistically important
```

```r
tmp_replay_lst <- replay.petrisim(pn=glb_analytics_pn, 
    replay.trans=(glb_analytics_avl_objs <- c(glb_analytics_avl_objs, 
        "data.new.prediction")), flip_coord=TRUE)
```

```
## time	trans	 "bgn " "fit.data.training.all " "predict.data.new " "end " 
## 0.0000 	multiple enabled transitions:  data.training.all data.new model.selected 	firing:  data.training.all 
## 1.0000 	 1 	 2 1 0 0 
## 1.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction 	firing:  data.new 
## 2.0000 	 2 	 1 1 1 0 
## 2.0000 	multiple enabled transitions:  data.training.all data.new model.selected model.final data.training.all.prediction data.new.prediction 	firing:  model.selected 
## 3.0000 	 3 	 0 2 1 0 
## 3.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  data.training.all.prediction 
## 4.0000 	 5 	 0 1 1 1 
## 4.0000 	multiple enabled transitions:  model.final data.training.all.prediction data.new.prediction 	firing:  model.final 
## 5.0000 	 4 	 0 0 2 1 
## 6.0000 	 6 	 0 0 1 2
```

![](Michigan_Voters_files/figure-html/predict.data.new-1.png) 

```r
print(ggplot.petrinet(tmp_replay_lst[["pn"]]) + coord_flip())
```

![](Michigan_Voters_files/figure-html/predict.data.new-2.png) 

Null Hypothesis ($\sf{H_{0}}$): mpg is not impacted by am_fctr.  
The variance by am_fctr appears to be independent. 
#```{r q1, cache=FALSE}
# print(t.test(subset(cars_df, am_fctr == "automatic")$mpg, 
#              subset(cars_df, am_fctr == "manual")$mpg, 
#              var.equal=FALSE)$conf)
#```
We reject the null hypothesis i.e. we have evidence to conclude that am_fctr impacts mpg (95% confidence). Manual transmission is better for miles per gallon versus automatic transmission.


```
##                   chunk_label chunk_step_major chunk_step_minor elapsed
## 10      fit.data.training.all                6                0 146.141
## 11      fit.data.training.all                6                1 161.115
## 4         manage_missing_data                2                2  19.040
## 5          encode_retype_data                2                3  30.736
## 2                cleanse_data                2                0   5.886
## 6            extract_features                3                0  35.739
## 7             select_features                4                0  37.151
## 12           predict.data.new                7                0 162.270
## 8  remove_correlated_features                4                1  37.388
## 9                  fit.models                5                0  37.437
## 3       inspectORexplore.data                2                1   5.918
## 1                 import_data                1                0   0.002
##    elapsed_diff
## 10      108.704
## 11       14.974
## 4        13.122
## 5        11.696
## 2         5.884
## 6         5.003
## 7         1.412
## 12        1.155
## 8         0.237
## 9         0.049
## 3         0.032
## 1         0.000
```

```
## [1] "Total Elapsed Time: 162.27 secs"
```

![](Michigan_Voters_files/figure-html/print_sessionInfo-1.png) 

```
## R version 3.1.3 (2015-03-09)
## Platform: x86_64-apple-darwin13.4.0 (64-bit)
## Running under: OS X 10.10.3 (Yosemite)
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## attached base packages:
## [1] tcltk     grid      stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] randomForest_4.6-10 rpart.plot_1.5.2    rpart_4.1-9        
##  [4] ROCR_1.0-7          gplots_2.16.0       caret_6.0-41       
##  [7] lattice_0.20-31     sqldf_0.4-10        RSQLite_1.0.0      
## [10] DBI_0.3.1           gsubfn_0.6-6        proto_0.3-10       
## [13] reshape2_1.4.1      plyr_1.8.1          caTools_1.17.1     
## [16] doBy_4.5-13         survival_2.38-1     ggplot2_1.0.1      
## 
## loaded via a namespace (and not attached):
##  [1] bitops_1.0-6        BradleyTerry2_1.0-6 brglm_0.5-9        
##  [4] car_2.0-25          chron_2.3-45        class_7.3-12       
##  [7] codetools_0.2-11    colorspace_1.2-6    compiler_3.1.3     
## [10] digest_0.6.8        e1071_1.6-4         evaluate_0.5.5     
## [13] foreach_1.4.2       formatR_1.1         gdata_2.13.3       
## [16] gtable_0.1.2        gtools_3.4.1        htmltools_0.2.6    
## [19] iterators_1.0.7     KernSmooth_2.23-14  knitr_1.9          
## [22] labeling_0.3        lme4_1.1-7          MASS_7.3-40        
## [25] Matrix_1.2-0        mgcv_1.8-6          minqa_1.2.4        
## [28] munsell_0.4.2       nlme_3.1-120        nloptr_1.0.4       
## [31] nnet_7.3-9          parallel_3.1.3      pbkrtest_0.4-2     
## [34] quantreg_5.11       RColorBrewer_1.1-2  Rcpp_0.11.5        
## [37] rmarkdown_0.5.1     scales_0.2.4        SparseM_1.6        
## [40] splines_3.1.3       stringr_0.6.2       tools_3.1.3        
## [43] yaml_2.1.13
```
