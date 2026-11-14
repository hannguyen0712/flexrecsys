from bocpd import compute_residuals, detect_user_changepoints, generate_weights
from mf import MF                                                                                                                                                     
from data import load_split, set_seed, RATING_SCALE                                                                                                                   
from metrics import rmse, evaluate_ranking 

set_seed()                                                                                                                                                            
dataset = "netflix"                                                                                                                                                 
split = "dev"                                                                                                                                                         
scale = RATING_SCALE[dataset]                                                                                                                                         
train, val, test = load_split(dataset, split)                                                                                                                         
                                                                                                                                                                                                                                                                                                                                        
print("\nTraining initial MF...")                                                                                                                           
mf_initial = MF(n_factors=64, rating_scale=scale).fit(train, val)                                                                                                     
                                                                                                                                                                    
print("\nDetecting changepoints...")                                                                                                                        
residuals = compute_residuals(mf_initial, train)                                                                                                                      
changepoints = detect_user_changepoints(residuals)                                                                                                                    
n_users_with_cp = sum(1 for cp in changepoints.values() if cp)                                                                                                        
                                                                                                                                                                    
print("\nTraining weighted MF...")
weights = generate_weights(train, changepoints)                                                                                                                       
mf_weighted = MF(n_factors=64, rating_scale=scale).fit(train, val, weights=weights)                                                                                   
                                                                                                                                                                    
print("\nResults:")
initial_rmse = rmse(val.rating, mf_initial.predict(val))
weighted_rmse = rmse(val.rating, mf_weighted.predict(val))
print(f"Initial MF val_rmse: {initial_rmse:.4f}")
print(f"Weighted MF val_rmse: {weighted_rmse:.4f}")


# ==== Hyperparameter tuning ==== # 
print("\nBase weight tuning:")
results = []
for base in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    weights = generate_weights(train, changepoints, base_weight=base, post_cp_weight=1.0)
    model = MF(n_factors=64, rating_scale=scale, epochs=15, verbose=False).fit(train, val, weights=weights)
    val_rmse = rmse(val.rating, model.predict(val))
    results.append((base, val_rmse))
    improvement = (initial_rmse - val_rmse) / initial_rmse * 100
    print(f"  base={base:.2f}: val_rmse={val_rmse:.4f} ({improvement:+.2f}% vs baseline)")

best_base, best_rmse = min(results, key=lambda x: x[1])
print(f"\n  Best: base_weight={best_base} with val_rmse={best_rmse:.4f}")

print("\nHazard rate tuning:")
hazard_results = []
for hazard in [1/20, 1/30, 1/50, 1/75, 1/100]:
    cp = detect_user_changepoints(residuals, hazard=hazard)
    n_cp = sum(1 for c in cp.values() if c)
    weights = generate_weights(train, cp, base_weight=best_base, post_cp_weight=1.0)
    model = MF(n_factors=64, rating_scale=scale, epochs=15, verbose=False).fit(train, val, weights=weights)
    val_rmse = rmse(val.rating, model.predict(val))
    hazard_results.append((hazard, val_rmse, n_cp))
    print(f"  hazard=1/{int(1/hazard):d}: val_rmse={val_rmse:.4f} (users_with_cp={n_cp})")

best_hazard, best_hz_rmse, _ = min(hazard_results, key=lambda x: x[1])
print(f"\n  Best: hazard=1/{int(1/best_hazard)} with val_rmse={best_hz_rmse:.4f}")

print(f"\nBaseline: {initial_rmse:.4f} → BOCPD-MF: {best_hz_rmse:.4f}")
print(f"Best params: base_weight={best_base}, hazard=1/{int(1/best_hazard)}")