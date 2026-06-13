data {
  int<lower=0> N;
  vector[N] y;
  vector[N] is_dark_genre;
  vector[N] budget;
}
parameters {
  real alpha;
  real b_dark; 
  real b_budget;
  real b_dark_budget;
  real<lower=0> sigma;
}
model {
  // Weakly informative priors
  alpha ~ normal(0, 1);
  // regulizing priors - there is no strong prior belief about the direction or magnitude of the effects, but we want to prevent extreme values that could lead to overfitting.
  b_dark ~ normal(0, 1); 
  b_budget ~ normal(0.5, 0.5); // Assuming a positive effect of budget on revenue, but allowing for uncertainty
  b_dark_budget ~ normal(0, 1);
  sigma ~ exponential(1);

  vector[N] mu = alpha 
                 + b_dark * is_dark_genre 
                 + b_budget * budget 
                 + b_dark_budget * (is_dark_genre .* budget);

  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep; // Posterior predictive distribution
  for (n in 1:N) {
    real mu_n = alpha + b_dark * is_dark_genre[n] + b_budget * budget[n] + b_dark_budget * is_dark_genre[n] * budget[n];
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma);
    y_rep[n] = normal_rng(mu_n, sigma);
  }
}