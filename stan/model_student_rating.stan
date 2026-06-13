data {
  int<lower=0> N;
  vector[N] y;
  vector[N] is_pg13;
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
  real b_pg13; real b_dark; real b_budget;
  real b_pg13_dark; real b_pg13_budget; real b_dark_budget;
  real b_triple;
  real<lower=0> sigma;
  real<lower=2> nu;
}

transformed parameters {
  vector[N] mu;
  vector[N] weighted_sigma;
  
  mu = alpha 
       + b_pg13 * is_pg13 
       + b_dark * is_dark_genre 
       + b_budget * budget 
       + b_pg13_dark * (is_pg13 .* is_dark_genre) 
       + b_pg13_budget * (is_pg13 .* budget) 
       + b_dark_budget * (is_dark_genre .* budget) 
       + b_triple * (is_pg13 .* is_dark_genre .* budget);

  weighted_sigma = sigma * inv_sqrt_ratings;
}

model {
  // weakly informative priors
  alpha ~ normal(6.5, 1.5);
  
  // regularizing priors
  b_pg13 ~ normal(0, 1); b_dark ~ normal(0, 1); b_budget ~ normal(0, 1);
  b_pg13_dark ~ normal(0, 1); b_pg13_budget ~ normal(0, 1); b_dark_budget ~ normal(0, 1);
  b_triple ~ normal(0, 1);
  sigma ~ exponential(1);
  nu ~ gamma(2, 0.1);

  y ~ student_t(nu, mu, weighted_sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep; // Posterior predictive distribution
    
  for (n in 1:N) {
    log_lik[n] = student_t_lpdf(y[n] | nu, mu[n], weighted_sigma[n]);
    y_rep[n] = student_t_rng(nu, mu[n], weighted_sigma[n]);
  }
}