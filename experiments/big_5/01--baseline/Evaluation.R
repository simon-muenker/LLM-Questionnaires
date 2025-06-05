library(fs)
library(purrr)
library(progress)

library(tibble)
library(dplyr)
library(readr)
library(tidyr)
library(magrittr)
library(ggplot2)

library(psych)
library(EGAnet)


# constants and configuration
data_path <- "./experiments/big_5/01--baseline/reports"
output_path <- paste0(data_path, "/R")

constructs <- c("E", "N", "C", "A", "O")

# validate data path
validate_path <- function(path) {
  if (!dir.exists(path)) {
    stop(sprintf("Data directory '%s' does not exist", path))
  }
  invisible(TRUE)
}

# load all relevant data files
load_data_files <- function(path, pattern = "^raw\\.\\S+\\.csv$") {
  validate_path(path)

  files <- tibble(
    name = list.files(path, pattern = pattern),
    path = fs::path(path, name)
  )

  if (nrow(files) == 0) {
    warning("No matching files found in the specified directory")
    return(NULL)
  }

  pb <- progress_bar$new(
    format = "Loading data [:bar] :percent (:current/:total) :eta",
    total = nrow(files)
  )

  # load each file and associate with its filename
  data_list <- files %>%
    mutate(data = map(path, function(p) {
      pb$tick()
      tryCatch(
        read_csv(p, show_col_types = FALSE),
        error = function(e) {
          warning(sprintf("Failed to read file %s: %s", p, e$message))
          return(NULL)
        }
      )
    }))

  # filter out any NULL data entries
  valid_data <- data_list %>% 
    filter(!map_lgl(data, is.null))

  message(sprintf("Successfully loaded %d/%d files", nrow(valid_data), nrow(files)))
  return(valid_data)
}

# calculate Cronbach's alpha for each construct
calc_alpha <- function(dataset, constructs) {
  if (is.null(dataset) || nrow(dataset) == 0) {
    warning("Empty dataset provided")
    return(NULL)
  }

  alpha_results <- map_dbl(constructs, function(c) {
    cols <- dataset %>% 
      select(starts_with(c)) %>%
      names()

    if (length(cols) == 0) {
      warning(sprintf("No columns found for construct '%s'", c))
      return(NA_real_)
    }

    tryCatch(
      psych::alpha(
        dataset %>% select(all_of(cols)),
        check.keys = TRUE,
        warnings = FALSE
      )$total$raw_alpha,
      error = function(e) {
        warning(sprintf("Alpha calculation failed for construct '%s': %s", c, e$message))
        return(NA_real_)
      }
    )
  })

  names(alpha_results) <- constructs
  return(round(alpha_results, 3))
}

# generate EGA plot
generate_ega <- function(dataset, output_path, filename) {
  if (is.null(dataset) || nrow(dataset) == 0) {
    warning("Empty dataset provided")
    return(FALSE)
  }

  # Ensure output directory exists
  fs::dir_create(dirname(output_path), recurse = TRUE)

  tryCatch({
    # Create and save the EGA plot
    ega_result <- EGAnet::EGA(as.data.frame(dataset))

    # Save the plot directly
    pdf(output_path)
    print(ega_result$plot.EGA)
    dev.off()

    # Return success
    message(sprintf("EGA plot saved to %s", output_path))
    return(TRUE)
  },
  error = function(e) {
    warning(sprintf("Error in EGA for %s: %s", filename, e$message))
    return(FALSE)
  })
}

# main function
analyze_big5_data <- function(data_path, constructs) {
  # Load all data files
  message("Starting Big 5 personality analysis...")
  data_files <- load_data_files(data_path)

  if (is.null(data_files) || nrow(data_files) == 0) {
    stop("No valid data files to process")
  }

  # Create results structure
  results <- data_files %>%
    mutate(
      alpha = NA,
      ega_success = FALSE
    )

  # Process each dataset with progress reporting
  pb <- progress_bar$new(
    format = "Analyzing datasets [:bar] :percent (:current/:total) :eta",
    total = nrow(data_files)
  )

  for (i in seq_len(nrow(data_files))) {
    current_file <- data_files$name[i]
    current_data <- data_files$data[[i]]

    message(sprintf("\nProcessing file: %s", current_file))

    # Calculate Cronbach's alpha
    alpha_result <- calc_alpha(current_data, constructs)
    if (!is.null(alpha_result)) {
      results$alpha[i] <- list(alpha_result)
      print(alpha_result)
    } else {
      message("No Cronbach's alpha calculated!")
    }

    # Generate EGA plot
    output_file <- fs::path(output_path, paste0("ega.", current_file, ".pdf"))
    generate_ega(current_data, output_file, current_file)

    pb$tick()
  }

  message("\nAnalysis complete!")
  return(results)
}

results <- analyze_big5_data(data_path, constructs)

summary_file <- fs::path(output_path, "alpha_values.csv")

results %>%
  select(name, alpha) %>%
  unnest_wider(alpha) %>%
  write_csv(summary_file)

message(sprintf("Summary results saved to %s", summary_file))