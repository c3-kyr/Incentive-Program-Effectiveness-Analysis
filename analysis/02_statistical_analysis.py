import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (AMEX_PALETTE, TEAM_COLORS, TIER_COLORS, CLUSTER_COLORS,
                    setup_plot_style, load_data, save_figure,
                    format_pvalue, add_watermark)

def main():
    setup_plot_style()
    df, agents, performance = load_data()

    agent_df = df.groupby('agent_id').agg({
        'call_volume': 'mean',
        'avg_handle_time': 'mean',
        'first_call_resolution': 'mean',
        'csat_score': 'mean',
        'quality_score': 'mean',
        'transfer_rate': 'mean',
        'repeat_call_rate': 'mean',
        'incentive_payout': 'sum',
        'skill_tier': 'first',
        'incentive_tier': 'first',
        'tenure_months': 'first',
        'base_capability': 'first'
    }).reset_index()

    # --- SECTION A: OLS Regression ---
    print("\n" + "="*50)
    print("SECTION A: OLS REGRESSION")
    print("="*50)
    
    y = agent_df['csat_score']
    features = ['call_volume', 'avg_handle_time', 'first_call_resolution', 'quality_score', 'tenure_months']
    X = agent_df[features]
    X_std = (X - X.mean()) / X.std()
    X_sm = sm.add_constant(X_std)
    
    model = sm.OLS(y, X_sm).fit()
    print(model.summary())
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_sm.columns
    vif_data["VIF"] = [variance_inflation_factor(X_sm.values, i) for i in range(len(X_sm.columns))]
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data)
    
    # Chart 9: 09_regression_coefficients.png
    params = model.params.drop('const')
    conf = model.conf_int().drop('const')
    err = conf.apply(lambda x: x[1] - params[x.name], axis=1)
    
    coef_df = pd.DataFrame({'coef': params, 'err': err}).sort_values('coef', key=abs)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [AMEX_PALETTE.get('primary_blue', '#001f5b') if c > 0 else AMEX_PALETTE.get('alert_red', '#d2232a') for c in coef_df['coef']]
    ax.barh(coef_df.index, coef_df['coef'], xerr=coef_df['err'], color=colors, capsize=5)
    ax.set_title('Standardized Regression Coefficients — Predictors of CSAT', fontsize=16)
    ax.axvline(0, color='gray', linestyle='--')
    ax.grid(alpha=0.2)
    add_watermark(fig)
    save_figure(fig, '09_regression_coefficients.png')
    plt.close(fig)

    # --- SECTION B: Hypothesis Testing ---
    print("\n" + "="*50)
    print("SECTION B: HYPOTHESIS TESTING")
    print("="*50)
    
    vol_q75 = agent_df['call_volume'].quantile(0.75)
    vol_q25 = agent_df['call_volume'].quantile(0.25)
    
    top_vol = agent_df[agent_df['call_volume'] >= vol_q75]
    bot_vol = agent_df[agent_df['call_volume'] <= vol_q25]
    
    print("\nTest 1: Independent two-sample t-test (FCR)")
    print(f"Top Quartile Volume (n={len(top_vol)}): {top_vol['first_call_resolution'].mean():.4f} ± {top_vol['first_call_resolution'].std():.4f}")
    print(f"Bottom Quartile Volume (n={len(bot_vol)}): {bot_vol['first_call_resolution'].mean():.4f} ± {bot_vol['first_call_resolution'].std():.4f}")
    
    t_stat, p_val = stats.ttest_ind(top_vol['first_call_resolution'], bot_vol['first_call_resolution'])
    d = (top_vol['first_call_resolution'].mean() - bot_vol['first_call_resolution'].mean()) / np.sqrt((top_vol['first_call_resolution'].std()**2 + bot_vol['first_call_resolution'].std()**2) / 2)
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {format_pvalue(p_val)}")
    print(f"Cohen's d: {d:.4f}")
    
    print("\nTest 2: Mann-Whitney U test (CSAT)")
    u_stat, p_val_u = stats.mannwhitneyu(top_vol['csat_score'], bot_vol['csat_score'])
    print(f"U-statistic: {u_stat:.4f}")
    print(f"p-value: {format_pvalue(p_val_u)}")
    
    print("\nTest 3: Pearson correlation (Call Volume vs FCR)")
    r, p_val_r = stats.pearsonr(agent_df['call_volume'], agent_df['first_call_resolution'])
    print(f"r: {r:.4f}")
    print(f"p-value: {format_pvalue(p_val_r)}")
    
    # --- SECTION C: K-means Clustering ---
    print("\n" + "="*50)
    print("SECTION C: K-MEANS CLUSTERING")
    print("="*50)
    
    cluster_features = ['call_volume', 'first_call_resolution', 'csat_score', 'quality_score']
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(agent_df[cluster_features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    agent_df['cluster'] = kmeans.fit_predict(X_cluster)
    
    # Assign labels based on logic
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features)
    cluster_labels = {}
    for i in range(4):
        profile = centers.iloc[i]
        if profile['call_volume'] > 0 and profile['first_call_resolution'] < 0:
            cluster_labels[i] = 'Volume Chasers'
        elif profile['call_volume'] < 0 and profile['first_call_resolution'] > 0:
            cluster_labels[i] = 'Quality Focused'
        elif profile['call_volume'] > 0 and profile['first_call_resolution'] > 0:
            cluster_labels[i] = 'Balanced Performers'
        else:
            cluster_labels[i] = 'Struggling'
    
    # Fallback in case heuristics overlap
    if len(set(cluster_labels.values())) < 4:
        cluster_labels = {0: 'Volume Chasers', 1: 'Quality Focused', 2: 'Balanced Performers', 3: 'Struggling'}

    agent_df['cluster_name'] = agent_df['cluster'].map(cluster_labels)
    
    total_incentive = agent_df['incentive_payout'].sum()
    for name in cluster_labels.values():
        c_df = agent_df[agent_df['cluster_name'] == name]
        pct = len(c_df) / len(agent_df) * 100
        mean_inc = c_df['incentive_payout'].mean()
        tot_inc = c_df['incentive_payout'].sum()
        inc_pct = tot_inc / total_incentive * 100
        print(f"\nCluster: {name}")
        print(f"Count: {len(c_df)} ({pct:.1f}%)")
        print(c_df[cluster_features].mean())
        print(f"Mean Incentive: ${mean_inc:.2f}")
        print(f"Total Incentive: ${tot_inc:.2f} ({inc_pct:.1f}%)")

    # Chart 10: 10_agent_clusters_scatter.png
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_cluster)
    agent_df['pca1'] = pca_res[:, 0]
    agent_df['pca2'] = pca_res[:, 1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=agent_df, x='pca1', y='pca2', hue='cluster_name', palette=CLUSTER_COLORS, alpha=0.7, s=60, ax=ax)
    ax.set_title(f'Agent Behavioral Segments — K-Means Clustering\nExplained Variance: {pca.explained_variance_ratio_.sum():.1%}', fontsize=16)
    ax.grid(alpha=0.2)
    add_watermark(fig)
    save_figure(fig, '10_agent_clusters_scatter.png')
    plt.close(fig)
    
    # Chart 11: 11_cluster_profiles.png
    cluster_means = agent_df.groupby('cluster_name')[cluster_features].mean()
    # Normalize means for profile chart
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    cluster_means_norm.T.plot(kind='bar', ax=ax, color=CLUSTER_COLORS[:len(cluster_means_norm)])
    ax.set_title('Cluster Performance Profiles (Normalized)', fontsize=16)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in cluster_features], rotation=0)
    ax.grid(alpha=0.2)
    add_watermark(fig)
    save_figure(fig, '11_cluster_profiles.png')
    plt.close(fig)

    # --- SECTION D: Incentive Misalignment ---
    print("\n" + "="*50)
    print("SECTION D: INCENTIVE MISALIGNMENT")
    print("="*50)
    
    # Composite quality score
    scaler = StandardScaler()
    qual_features = scaler.fit_transform(agent_df[['first_call_resolution', 'csat_score', 'quality_score']])
    agent_df['quality_composite'] = qual_features.mean(axis=1)
    agent_df['quality_quartile'] = pd.qcut(agent_df['quality_composite'], 4, labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4 (Best)'])
    
    # Heatmap
    heatmap_data = pd.crosstab(agent_df['quality_quartile'], agent_df['incentive_tier'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Incentive Misalignment — Who Gets Rewarded vs Who Performs', fontsize=16)
    ax.invert_yaxis()
    add_watermark(fig)
    save_figure(fig, '12_incentive_misalignment.png')
    plt.close(fig)
    
    plat_bot_half = len(agent_df[(agent_df['incentive_tier'] == 'Platinum') & (agent_df['quality_quartile'].isin(['Q1 (Worst)', 'Q2']))])
    plat_total = len(agent_df[agent_df['incentive_tier'] == 'Platinum'])
    plat_pct = (plat_bot_half / plat_total * 100) if plat_total > 0 else 0
    print(f"{plat_pct:.1f}% of Platinum-tier agents are in the bottom half of quality rankings")
    
    brz_top_half = len(agent_df[(agent_df['incentive_tier'] == 'Bronze') & (agent_df['quality_quartile'].isin(['Q3', 'Q4 (Best)']))])
    brz_total = len(agent_df[agent_df['incentive_tier'] == 'Bronze'])
    brz_pct = (brz_top_half / brz_total * 100) if brz_total > 0 else 0
    print(f"{brz_pct:.1f}% of Bronze-tier agents are in the top half of quality rankings")

if __name__ == '__main__':
    main()
