# 1. Create merged dataset with 8 path features
./merge_adaptive_path_features.sh my_experiment 8 true

# 2. Train adaptive-only model (your 53.6% winner)
./train_adaptive_path_model.sh my_experiment adaptive_only 64 0.05

# 3. Train experimental model (adaptive + path features)
./train_adaptive_path_model.sh my_experiment experimental 64 0.05

# 4. Benchmark both models
./benchmark_adaptive_path.sh my_experiment adaptive_only
./benchmark_adaptive_path.sh my_experiment experimental

# 5. Compare results
python compare_model_performance.py --experiments my_experiment