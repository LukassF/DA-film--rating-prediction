data {
  int<lower=0> N;
  vector[N] is_pg13;
  vector[N] is_dark_genre;
  vector[N] budget;
}

generated quantities {
  real alpha = normal_rng(0, 1);
  real b_pg13 = normal_rng(0, 1);
  real b_dark = normal_rng(0, 1);
  real b_budget = normal_rng(0, 1);
  real b_pg13_dark = normal_rng(0, 1);
  real b_pg13_budget = normal_rng(0, 1);
  real b_dark_budget = normal_rng(0, 1);
  real b_triple = normal_rng(0, 1);
  real sigma = exponential_rng(1);

  vector[N] y_prior;
  for (n in 1:N) {
    real mu_n = alpha 
                + b_pg13 * is_pg13[n] 
                + b_dark * is_dark_genre[n] 
                + b_budget * budget[n] 
                + b_pg13_dark * is_pg13[n] * is_dark_genre[n] 
                + b_pg13_budget * is_pg13[n] * budget[n] 
                + b_dark_budget * is_dark_genre[n] * budget[n] 
                + b_triple * is_pg13[n] * is_dark_genre[n] * budget[n];
                
    y_prior[n] = normal_rng(mu_n, sigma);
  }
}