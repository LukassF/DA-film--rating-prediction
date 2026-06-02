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
  real<lower=1> nu;
}
model {
  alpha ~ normal(0, 1);
  b_pg13 ~ normal(0, 1); b_dark ~ normal(0, 1); b_budget ~ normal(0, 1);
  b_pg13_dark ~ normal(0, 1); b_pg13_budget ~ normal(0, 1); b_dark_budget ~ normal(0, 1);
  b_triple ~ normal(0, 1);
  sigma ~ exponential(1);
  nu ~ gamma(2, 0.1);            

  vector[N] mu = alpha 
                 + b_pg13 * is_pg13 
                 + b_dark * is_dark_genre 
                 + b_budget * budget 
                 + b_pg13_dark * (is_pg13 .* is_dark_genre) 
                 + b_pg13_budget * (is_pg13 .* budget) 
                 + b_dark_budget * (is_dark_genre .* budget) 
                 + b_triple * (is_pg13 .* is_dark_genre .* budget);

  y ~ student_t(nu, mu, sigma);
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep; 

  for (n in 1:N) {
    real mu_n = alpha + b_pg13 * is_pg13[n] + b_dark * is_dark_genre[n] + b_budget * budget[n] 
                + b_pg13_dark * is_pg13[n] * is_dark_genre[n] + b_pg13_budget * is_pg13[n] * budget[n] 
                + b_dark_budget * is_dark_genre[n] * budget[n] + b_triple * is_pg13[n] * is_dark_genre[n] * budget[n];
    
    log_lik[n] = student_t_lpdf(y[n] | nu, mu_n, sigma);
    y_rep[n] = student_t_rng(nu, mu_n, sigma);
  }
}