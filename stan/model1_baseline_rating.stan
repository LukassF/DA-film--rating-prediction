data {
  int<lower=0> N;
  vector[N] y;
  vector[N] is_dark_genre;
  vector[N] budget;
  vector[N] num_ratings;
}

// runs only once
transformed data {
  vector[N] inv_sqrt_ratings;
  for (i in 1:N) {
    inv_sqrt_ratings[i] = 1.0 / sqrt(num_ratings[i] + 0.001);
  }
}

parameters {
  real alpha;
  real b_dark; 
  real b_budget;
  real b_dark_budget;
  real<lower=0> sigma;
}


transformed parameters {
  vector[N] mu;
  vector[N] weighted_sigma;
  
  mu = alpha 
    + b_dark * is_dark_genre 
    + b_budget * budget 
    + b_dark_budget * (is_dark_genre .* budget);

  weighted_sigma = sigma * inv_sqrt_ratings;
}

model {
  // Weakly informative priors
  // only the mean of alpha is informed by domain knowledge, while the other parameters are centered around zero with a standard deviation of 1, reflecting our uncertainty about their effects.
  // https://pmc.ncbi.nlm.nih.gov/articles/PMC2929029/ - the found value is in the scale of 0-5 and the mean is around 3,57 but in our case the rating is from 0 to 10, so we can expect the mean to be around 6.5
  alpha ~ normal(6.5, 1.5);
  // regulizing priors - there is no strong prior belief about the direction or magnitude of the effects, but we want to prevent extreme values that could lead to overfitting.
  b_dark ~ normal(0, 1); 
  b_budget ~ normal(0, 1);
  b_dark_budget ~ normal(0, 1);
  sigma ~ exponential(1);

  y ~ normal(mu, weighted_sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep; // Posterior predictive distribution
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu[n], weighted_sigma[n]);
    y_rep[n] = normal_rng(mu[n], weighted_sigma[n]);
  }
}