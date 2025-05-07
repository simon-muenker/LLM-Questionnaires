library(tidyverse)
library(psych)
library(EGAnet)

DATA_PATH <- "./experiments/big_5/02--with-history/reports"
CONSTRUCTS <- tibble('E','N','C','A','O')

files = tibble(name=list.files(DATA_PATH, pattern="^raw\\.\\S+\\.csv$"))
data = apply(files, 1, function(x) read_csv(paste(DATA_PATH, "/", x, sep=""), show_col_types = FALSE))

calc_alpha = function(dataset) {
    return (
        apply(
            CONSTRUCTS, 
            2, 
            function(c) { 
                psych::alpha(
                    data[[i]] %>% select(starts_with(c)), 
                    check.keys=TRUE,
                    warnings=FALSE
                )$total$raw_alpha
            } 
        ) %>% round(3)
    )
}

for (i in 1:length(data)) { 

    print(files$name[[i]])
    tryCatch(
        { print(calc_alpha(data[i])) },
        warning = function(e) { print("No Alpha calculated") },
        error = function(e) { print("No Alpha calculated") }
    )

    tryCatch(
        {
            EGA(as.data.frame(data[i]))$plot.EGA
            ggsave(sprintf("%s/ega.%s.pdf", DATA_PATH, files$name[[i]]))
            while (!is.null(dev.list())) dev.off()
        },
        warning = function(e) { print("No graph exported") },
        error = function(e) { print("No graph exported") }
    )
}