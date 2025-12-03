# Run R oreo SPP on golden_data inputs.
# Execute with: Rscript scripts/run_oreo.R

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

base_dir <- file.path(getwd(), "golden_data")
input_dir <- file.path(base_dir, "input")
out_dir <- file.path(base_dir, "outputs", "r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- c("sin_fundamental", "sin_noisy")

source("/Users/b80985/Documents/MATLAB/oreo/R/RPPplus_fourier_v3.r")
source("/Users/b80985/Documents/MATLAB/oreo/R/RPPplus_numerical.r")

omega <- 2 * pi

run_ds <- function(ds) {
  df <- read_csv(file.path(input_dir, paste0(ds, ".csv")), show_col_types = FALSE)
  time_wave <- df$t
  strain <- df$gamma
  rate <- rep(NA_real_, length(strain))
  stress <- df$sigma
  resp_wave <- cbind(strain, rate, stress)
  L <- nrow(resp_wave)

  # Fourier
  out_f <- RPPplus_fourier_v3(time_wave = time_wave, resp_wave = resp_wave, L = L, omega = omega, M = 39, p = 1)
  write_csv(as.data.frame(out_f$spp_data_out), file.path(out_dir, paste0(ds, "_spp_data_out_fourier.csv")))
  write_csv(as.data.frame(out_f$fsf_data_out), file.path(out_dir, paste0(ds, "_fsf_data_out_fourier.csv")))
  write_csv(as.data.frame(out_f$ft_out), file.path(out_dir, paste0(ds, "_ft_out_fourier.csv")))

  # Numerical
  out_n <- RPPplus_numerical(time_wave = time_wave, resp_wave = resp_wave, L = L, k = 8, num_mode = 2)
  write_csv(as.data.frame(out_n$spp_data_out), file.path(out_dir, paste0(ds, "_spp_data_out_numerical.csv")))
  write_csv(as.data.frame(out_n$fsf_data_out), file.path(out_dir, paste0(ds, "_fsf_data_out_numerical.csv")))
}

lapply(datasets, run_ds)
cat("R oreo runs complete. Outputs in", out_dir, "\n")
