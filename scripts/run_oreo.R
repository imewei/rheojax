# Run R oreo SPP on golden_data inputs.
# Execute with: Rscript scripts/run_oreo.R
# Requires: fftwtools, pracma, spectral packages
# Install with: install.packages(c("fftwtools", "pracma", "spectral"))

suppressPackageStartupMessages({
  library(fftwtools)
  library(pracma)
  library(spectral)
})

# Get script directory (works with Rscript and source())
get_script_dir <- function() {
  # Try commandArgs for Rscript
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  # Fallback: assume running from repo root
  if (dir.exists("scripts/golden_data")) {
    return(normalizePath("scripts"))
  }
  # Last resort: current directory
  return(getwd())
}

script_dir <- get_script_dir()
base_dir <- file.path(script_dir, "golden_data")
input_dir <- file.path(base_dir, "input")
out_dir <- file.path(base_dir, "outputs", "r")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- c("sin_fundamental", "sin_noisy")

source("/Users/b80985/Documents/MATLAB/oreo/R/RPPplus_fourier_v3.r")
source("/Users/b80985/Documents/MATLAB/oreo/R/RPPplus_numerical.r")

omega <- 2 * pi

run_ds <- function(ds) {
  # Use base R read.csv
  df <- read.csv(file.path(input_dir, paste0(ds, ".csv")))
  time_wave <- df$t
  strain <- df$gamma
  stress <- df$sigma
  L <- length(strain)

  # Compute strain rate numerically (central difference, periodic boundaries)
  # This matches the 8-point 4th-order stencil used by SPPplus
  dt <- mean(diff(time_wave))
  rate <- numeric(L)
  for (i in 1:L) {
    # Periodic index wrapping
    ip1 <- ((i) %% L) + 1
    ip2 <- ((i + 1) %% L) + 1
    ip3 <- ((i + 2) %% L) + 1
    ip4 <- ((i + 3) %% L) + 1
    im1 <- ((i - 2) %% L) + 1
    im2 <- ((i - 3) %% L) + 1
    im3 <- ((i - 4) %% L) + 1
    im4 <- ((i - 5) %% L) + 1
    # 8-point 4th-order central difference
    rate[i] <- (-3*strain[ip4] + 32*strain[ip3] - 168*strain[ip2] + 672*strain[ip1]
                - 672*strain[im1] + 168*strain[im2] - 32*strain[im3] + 3*strain[im4]) / (840 * dt)
  }

  resp_wave <- cbind(strain, rate, stress)

  # Fourier (function is rpp_fft, not RPPplus_fourier_v3)
  # p=3 because input data has 3 cycles (from gen_inputs.py n_cycles=3)
  out_f <- rpp_fft(time_wave = time_wave, resp_wave = resp_wave, L = L, omega = omega, M = 39, p = 3)
  write.csv(as.data.frame(out_f$spp_data_out), file.path(out_dir, paste0(ds, "_spp_data_out_fourier.csv")), row.names = FALSE)
  write.csv(as.data.frame(out_f$fsf_data_out), file.path(out_dir, paste0(ds, "_fsf_data_out_fourier.csv")), row.names = FALSE)
  write.csv(as.data.frame(out_f$ft_out), file.path(out_dir, paste0(ds, "_ft_out_fourier.csv")), row.names = FALSE)

  # Numerical (function is Rpp_num, not RPPplus_numerical)
  out_n <- Rpp_num(time_wave = time_wave, resp_wave = resp_wave, L = L, k = 8, num_mode = 2)
  write.csv(as.data.frame(out_n$spp_data_out), file.path(out_dir, paste0(ds, "_spp_data_out_numerical.csv")), row.names = FALSE)
  write.csv(as.data.frame(out_n$fsf_data_out), file.path(out_dir, paste0(ds, "_fsf_data_out_numerical.csv")), row.names = FALSE)
}

lapply(datasets, run_ds)
cat("R oreo runs complete. Outputs in", out_dir, "\n")
