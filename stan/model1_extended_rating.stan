data {
  int<lower=0> N;
  vector[N] y;
  vector[N] is_pg13;
  vector[N] is_dark_genre;
  vector[N] budget;
}
parameters {
  real alpha;
  real b_pg13; real b_dark; real b_budget;
  real b_pg13_dark; real b_pg13_budget; real b_dark_budget;
  real b_triple;
  real<lower=0> sigma;
}
model {
  // weakly informative priors
  // only the mean of alpha is informed by domain knowledge, while the other parameters are centered around zero with a standard deviation of 1, reflecting our uncertainty about their effects.
  alpha ~ normal(6.5, 1.5);
  // regulizing priors - there is no strong prior belief about the direction or magnitude of the effects, but we want to prevent extreme values that could lead to overfitting.
  b_pg13 ~ normal(0, 1); b_dark ~ normal(0, 1); b_budget ~ normal(0, 1);
  b_pg13_dark ~ normal(0, 1); b_pg13_budget ~ normal(0, 1); b_dark_budget ~ normal(0, 1);
  b_triple ~ normal(0, 1);
  sigma ~ exponential(1);

    // add pg_13 prediction and interaction terms
  vector[N] mu = alpha 
                 + b_pg13 * is_pg13 
                 + b_dark * is_dark_genre 
                 + b_budget * budget 
                 + b_pg13_dark * (is_pg13 .* is_dark_genre) 
                 + b_pg13_budget * (is_pg13 .* budget) 
                 + b_dark_budget * (is_dark_genre .* budget) 
                 + b_triple * (is_pg13 .* is_dark_genre .* budget);

  // add likelihood with error term
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
    vector[N] y_rep; // Posterior predictive distribution
    
  for (n in 1:N) {
    real mu_n = alpha + b_pg13 * is_pg13[n] + b_dark * is_dark_genre[n] + b_budget * budget[n] 
                + b_pg13_dark * is_pg13[n] * is_dark_genre[n] + b_pg13_budget * is_pg13[n] * budget[n] 
                + b_dark_budget * is_dark_genre[n] * budget[n] + b_triple * is_pg13[n] * is_dark_genre[n] * budget[n];
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma);
    y_rep[n] = normal_rng(mu_n, sigma);

  }
}