import numpy as np
import pandas as pd

# -----------------------------
# 1. LOAD DATA
# -----------------------------
file_path = "/Users/Elisha/Desktop/Python learning/JPMorgan/Task 3 and 4_Loan_Data.csv"
df = pd.read_csv(file_path)

# Keep only the needed columns
df = df[['fico_score', 'default']].copy()

# -----------------------------
# 2. AGGREGATE BY FICO SCORE
# -----------------------------
# For each unique FICO score:
#   n = number of borrowers
#   k = number of defaults
agg = (
    df.groupby('fico_score')
      .agg(n=('default', 'count'),
           k=('default', 'sum'))
      .reset_index()
      .sort_values('fico_score')
      .reset_index(drop=True)
)

scores = agg['fico_score'].to_numpy()
n = agg['n'].to_numpy()
k = agg['k'].to_numpy()

m = len(scores)

# Prefix sums for fast interval queries
cum_n = np.concatenate(([0], np.cumsum(n)))
cum_k = np.concatenate(([0], np.cumsum(k)))

# For MSE approach
cum_x = np.concatenate(([0], np.cumsum(scores * n)))
cum_x2 = np.concatenate(([0], np.cumsum((scores ** 2) * n)))

# -----------------------------
# 3. HELPERS
# -----------------------------
EPS = 1e-12

def interval_counts(i, j):
    """
    Returns total borrowers and defaults for interval [i, j] inclusive.
    """
    total_n = cum_n[j + 1] - cum_n[i]
    total_k = cum_k[j + 1] - cum_k[i]
    return total_n, total_k

def bucket_loglik(i, j):
    """
    Log-likelihood of assigning FICO scores from index i to j into one bucket.
    Each bucket gets one PD estimate p = k / n.
    """
    total_n, total_k = interval_counts(i, j)
    if total_n == 0:
        return -np.inf

    p = total_k / total_n

    # Avoid log(0)
    p = min(max(p, EPS), 1 - EPS)

    return total_k * np.log(p) + (total_n - total_k) * np.log(1 - p)

def bucket_sse(i, j):
    """
    Weighted sum of squared errors if scores in [i, j] are represented by one mean.
    Weighted by borrower count n at each fico score.
    """
    total_n = cum_n[j + 1] - cum_n[i]
    total_x = cum_x[j + 1] - cum_x[i]
    total_x2 = cum_x2[j + 1] - cum_x2[i]

    if total_n == 0:
        return np.inf

    mean_x = total_x / total_n
    sse = total_x2 - 2 * mean_x * total_x + total_n * mean_x ** 2
    return sse

# Precompute bucket objective matrices
loglik_matrix = np.full((m, m), -np.inf)
sse_matrix = np.full((m, m), np.inf)

for i in range(m):
    for j in range(i, m):
        loglik_matrix[i, j] = bucket_loglik(i, j)
        sse_matrix[i, j] = bucket_sse(i, j)

# -----------------------------
# 4. DYNAMIC PROGRAMMING
# -----------------------------
def optimal_buckets_loglik(num_buckets):
    """
    Maximise total log-likelihood over contiguous FICO buckets.
    Returns bucket boundaries in terms of score intervals.
    """
    dp = np.full((num_buckets + 1, m), -np.inf)
    prev = np.full((num_buckets + 1, m), -1, dtype=int)

    # Base case: 1 bucket covering 0..j
    for j in range(m):
        dp[1, j] = loglik_matrix[0, j]

    # Fill DP
    for b in range(2, num_buckets + 1):
        for j in range(b - 1, m):
            best_val = -np.inf
            best_i = -1
            for i in range(b - 2, j):
                candidate = dp[b - 1, i] + loglik_matrix[i + 1, j]
                if candidate > best_val:
                    best_val = candidate
                    best_i = i
            dp[b, j] = best_val
            prev[b, j] = best_i

    # Backtrack
    boundaries = []
    b = num_buckets
    j = m - 1

    while b > 1:
        i = prev[b, j]
        boundaries.append((i + 1, j))
        j = i
        b -= 1
    boundaries.append((0, j))
    boundaries.reverse()

    return boundaries, dp[num_buckets, m - 1]

def optimal_buckets_mse(num_buckets):
    """
    Minimise total weighted SSE over contiguous FICO buckets.
    """
    dp = np.full((num_buckets + 1, m), np.inf)
    prev = np.full((num_buckets + 1, m), -1, dtype=int)

    # Base case
    for j in range(m):
        dp[1, j] = sse_matrix[0, j]

    # Fill DP
    for b in range(2, num_buckets + 1):
        for j in range(b - 1, m):
            best_val = np.inf
            best_i = -1
            for i in range(b - 2, j):
                candidate = dp[b - 1, i] + sse_matrix[i + 1, j]
                if candidate < best_val:
                    best_val = candidate
                    best_i = i
            dp[b, j] = best_val
            prev[b, j] = best_i

    # Backtrack
    boundaries = []
    b = num_buckets
    j = m - 1

    while b > 1:
        i = prev[b, j]
        boundaries.append((i + 1, j))
        j = i
        b -= 1
    boundaries.append((0, j))
    boundaries.reverse()

    return boundaries, dp[num_buckets, m - 1]

# -----------------------------
# 5. CREATE RATING MAP
# -----------------------------
def build_rating_map(boundaries):
    """
    Lower rating = better credit score.
    Since higher FICO is better, the highest FICO bucket gets rating 1.
    """
    rows = []

    # boundaries are in ascending FICO order
    # so reverse ratings
    num_buckets = len(boundaries)

    for idx, (start, end) in enumerate(boundaries):
        fico_min = scores[start]
        fico_max = scores[end]
        total_n, total_k = interval_counts(start, end)
        pd_bucket = total_k / total_n if total_n > 0 else np.nan

        # Ascending FICO buckets: worst to best
        # Need lower rating = better score
        rating = num_buckets - idx

        rows.append({
            'rating': rating,
            'fico_min': fico_min,
            'fico_max': fico_max,
            'borrowers': total_n,
            'defaults': total_k,
            'pd': pd_bucket
        })

    rating_map = pd.DataFrame(rows).sort_values('rating').reset_index(drop=True)
    return rating_map

def assign_ratings(df, rating_map):
    """
    Assign rating and bucket PD to each borrower.
    """
    df_out = df.copy()
    df_out['rating'] = np.nan
    df_out['bucket_pd'] = np.nan

    for _, row in rating_map.iterrows():
        mask = (df_out['fico_score'] >= row['fico_min']) & (df_out['fico_score'] <= row['fico_max'])
        df_out.loc[mask, 'rating'] = int(row['rating'])
        df_out.loc[mask, 'bucket_pd'] = row['pd']

    return df_out

# -----------------------------
# 6. RUN EXAMPLE
# -----------------------------
NUM_BUCKETS = 10

# Likelihood-optimal buckets
ll_boundaries, ll_value = optimal_buckets_loglik(NUM_BUCKETS)
rating_map_ll = build_rating_map(ll_boundaries)
df_ll = assign_ratings(df, rating_map_ll)

print("=== Log-Likelihood Optimal Rating Map ===")
print(rating_map_ll)
print(f"\nTotal log-likelihood: {ll_value:.4f}")

# MSE-optimal buckets
mse_boundaries, mse_value = optimal_buckets_mse(NUM_BUCKETS)
rating_map_mse = build_rating_map(mse_boundaries)
df_mse = assign_ratings(df, rating_map_mse)

print("\n=== MSE Optimal Rating Map ===")
print(rating_map_mse)
print(f"\nTotal weighted SSE: {mse_value:.4f}")