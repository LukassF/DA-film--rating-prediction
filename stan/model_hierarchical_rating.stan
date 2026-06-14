data {
  int<lower=0> N;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> genre_id;
  vector[N] y;
  vector[N] is_pg13;
  vector[N] budget;
  vector[N] num_ratings;
}
transformed data {
  vector[N] inv_sqrt_ratings;
  for (i in 1:N) {
    inv_sqrt_ratings[i] = 1.0 / sqrt(num_ratings[i] + 0.001);
  }
}
parameters {
  real mu_alpha;
  real<lower=0> sigma_alpha;
  real mu_pg13;
  real<lower=0> sigma_pg13;
  vector[J] alpha_raw;
  vector[J] beta_pg13_raw;
  real b_budget;
  real<lower=0> sigma;
}
transformed parameters {
  vector[J] alpha = mu_alpha + sigma_alpha * alpha_raw;
  vector[J] beta_pg13 = mu_pg13 + sigma_pg13 * beta_pg13_raw;
  vector[N] mu;
  vector[N] weighted_sigma;
  for (i in 1:N) {
    mu[i] = alpha[genre_id[i]] + beta_pg13[genre_id[i]] * is_pg13[i] + b_budget * budget[i];
  }
  weighted_sigma = sigma * inv_sqrt_ratings;
}
model {
  mu_alpha ~ normal(6.5, 1.5);
  sigma_alpha ~ exponential(1);
  mu_pg13 ~ normal(0, 1);
  sigma_pg13 ~ exponential(1);
  alpha_raw ~ normal(0, 1);
  beta_pg13_raw ~ normal(0, 1);
  b_budget ~ normal(0, 1);
  sigma ~ exponential(1);
  y ~ normal(mu, weighted_sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | mu[i], weighted_sigma[i]);
    y_rep[i] = normal_rng(mu[i], weighted_sigma[i]);
  }
}