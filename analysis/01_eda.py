import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (AMEX_PALETTE, TEAM_COLORS, TIER_COLORS, CLUSTER_COLORS,
                    setup_plot_style, load_data, save_figure,
                    format_pvalue, add_watermark)

def main():
    setup_plot_style()
    df, agents, performance = load_data()

    # Pre-processing: aggregate to agent-level
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
    
    # Skill tier order
    skill_order = ['Junior', 'Mid', 'Senior', 'Expert']
    agent_df['skill_tier'] = pd.Categorical(agent_df['skill_tier'], categories=skill_order, ordered=True)
    tier_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
    agent_df['incentive_tier'] = pd.Categorical(agent_df['incentive_tier'], categories=tier_order, ordered=True)

    print("================ Summary Statistics ================")
    print(agent_df.describe())
    
    # Chart 1: 01_metric_distributions.png
    metrics = ['call_volume', 'avg_handle_time', 'first_call_resolution', 'csat_score', 'quality_score', 'transfer_rate']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle('Distribution of Key Performance Metrics', fontsize=20)
    for i, metric in enumerate(metrics):
        ax = axes[i//3, i%3]
        sns.histplot(data=agent_df, x=metric, kde=True, ax=ax, color=AMEX_PALETTE.get('primary_blue', '#001f5b'),
                     line_kws={'color': AMEX_PALETTE.get('accent_teal', '#007a86')})
        mean_val = agent_df[metric].mean()
        ax.axvline(mean_val, color=AMEX_PALETTE.get('warning_amber', '#d48600'), linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.set_title(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(alpha=0.2)
    add_watermark(fig)
    save_figure(fig, '01_metric_distributions.png')
    plt.close(fig)

    # Chart 2: 02_correlation_heatmap.png
    corr_cols = ['call_volume', 'avg_handle_time', 'first_call_resolution', 'csat_score', 'quality_score', 'transfer_rate', 'repeat_call_rate', 'incentive_payout']
    corr_matrix = agent_df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = LinearSegmentedColormap.from_list('custom_amex', [AMEX_PALETTE.get('primary_blue', '#001f5b'), '#ffffff', AMEX_PALETTE.get('alert_red', '#d2232a')])
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap, center=0, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Performance Metric Correlations — The Hidden Tensions', fontsize=18)
    add_watermark(fig)
    save_figure(fig, '02_correlation_heatmap.png')
    plt.close(fig)

    # Chart 3: 03_volume_vs_fcr.png
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=agent_df, x='call_volume', y='first_call_resolution', hue='skill_tier', palette=TIER_COLORS, alpha=0.4, ax=ax)
    sns.regplot(data=agent_df, x='call_volume', y='first_call_resolution', scatter=False, ax=ax, color=AMEX_PALETTE.get('alert_red', '#d2232a'))
    # R2 and equation
    X = sm.add_constant(agent_df['call_volume'])
    model = sm.OLS(agent_df['first_call_resolution'], X).fit()
    r2 = model.rsquared
    b0, b1 = model.params
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\ny = {b1:.4f}x + {b0:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.set_title('Call Volume vs First Call Resolution', fontsize=16)
    ax.grid(alpha=0.2)
    add_watermark(fig)
    save_figure(fig, '03_volume_vs_fcr.png')
    plt.close(fig)

    # Chart 4: 04_volume_vs_csat.png
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=agent_df, x='call_volume', y='csat_score', hue='skill_tier', palette=TIER_COLORS, alpha=0.4, ax=ax)
    sns.regplot(data=agent_df, x='call_volume', y='csat_score', scatter=False, ax=ax, color=AMEX_PALETTE.get('alert_red', '#d2232a'))
    X = sm.add_constant(agent_df['call_volume'])
    model = sm.OLS(agent_df['csat_score'], X).fit()
    ax.text(0.05, 0.95, f'R² = {model.rsquared:.3f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.set_title('Call Volume vs Customer Satisfaction', fontsize=16)
    ax.grid(alpha=0.2)
    add_watermark(fig)
    save_figure(fig, '04_volume_vs_csat.png')
    plt.close(fig)

    # Chart 5: 05_fcr_vs_csat.png
    fig, ax = plt.subplots(figsize=(12, 8))
    tier_colors = {t: TIER_COLORS.get(t, '#888') for t in tier_order} if isinstance(TIER_COLORS, dict) else TIER_COLORS
    sns.scatterplot(data=agent_df, x='first_call_resolution', y='csat_score', hue='incentive_tier', hue_order=tier_order, palette=tier_colors, alpha=0.6, ax=ax)
    ax.set_title('FCR Drives Customer Satisfaction — The Right Metric to Incentivize', fontsize=16)
    ax.grid(alpha=0.2)
    add_watermark(fig)
    save_figure(fig, '05_fcr_vs_csat.png')
    plt.close(fig)

    # Chart 6: 06_monthly_trends.png
    # Aggregate by month
    monthly_df = df.groupby('month').agg({
        'first_call_resolution': 'mean',
        'csat_score': 'mean',
        'call_volume': 'mean',
        'quality_score': 'mean'
    }).reset_index()
    
    # Normalize 0-1
    for col in ['csat_score', 'call_volume', 'quality_score']:
        monthly_df[f'{col}_norm'] = (monthly_df[col] - monthly_df[col].min()) / (monthly_df[col].max() - monthly_df[col].min())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(monthly_df['month'], monthly_df['first_call_resolution'], marker='o', label='FCR (Actual)', color=AMEX_PALETTE.get('primary_blue', '#001f5b'))
    ax.plot(monthly_df['month'], monthly_df['csat_score_norm'], marker='s', label='CSAT (Norm)', color=AMEX_PALETTE.get('accent_teal', '#007a86'))
    ax.plot(monthly_df['month'], monthly_df['call_volume_norm'], marker='^', label='Volume (Norm)', color=AMEX_PALETTE.get('alert_red', '#d2232a'))
    ax.plot(monthly_df['month'], monthly_df['quality_score_norm'], marker='d', label='Quality (Norm)', color=AMEX_PALETTE.get('warning_amber', '#d48600'))
    ax.set_title('Monthly Performance Trends', fontsize=16)
    ax.set_xlabel('Months 1-12')
    ax.legend()
    ax.grid(alpha=0.2)
    add_watermark(fig)
    save_figure(fig, '06_monthly_trends.png')
    plt.close(fig)

    # Chart 7: 07_skill_tier_boxplots.png
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle('Performance by Experience Level', fontsize=18)
    
    sns.boxplot(data=agent_df, x='skill_tier', y='first_call_resolution', ax=axes[0], palette=TIER_COLORS, order=skill_order, showfliers=False)
    sns.stripplot(data=agent_df, x='skill_tier', y='first_call_resolution', ax=axes[0], color='black', alpha=0.3, order=skill_order)
    axes[0].set_title('FCR by Skill Tier')
    axes[0].grid(alpha=0.2)
    
    sns.boxplot(data=agent_df, x='skill_tier', y='csat_score', ax=axes[1], palette=TIER_COLORS, order=skill_order, showfliers=False)
    sns.stripplot(data=agent_df, x='skill_tier', y='csat_score', ax=axes[1], color='black', alpha=0.3, order=skill_order)
    axes[1].set_title('CSAT by Skill Tier')
    axes[1].grid(alpha=0.2)
    
    add_watermark(fig)
    save_figure(fig, '07_skill_tier_boxplots.png')
    plt.close(fig)

    # Chart 8: 08_incentive_distribution.png
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    fig.suptitle('Incentive Program Distribution', fontsize=18)
    
    inc_colors = {'Bronze': AMEX_PALETTE.get('alert_red', '#d2232a'), 
                  'Silver': AMEX_PALETTE.get('warning_amber', '#d48600'),
                  'Gold': AMEX_PALETTE.get('accent_teal', '#007a86'),
                  'Platinum': AMEX_PALETTE.get('primary_blue', '#001f5b')}
    
    sns.countplot(data=agent_df, x='incentive_tier', order=tier_order, palette=inc_colors, ax=axes[0])
    axes[0].set_title('Incentive Tier Counts')
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    axes[0].grid(alpha=0.2)
    
    sns.boxplot(data=agent_df, x='incentive_tier', y='incentive_payout', order=tier_order, palette=inc_colors, ax=axes[1])
    axes[1].set_title('Total Incentive Payouts by Tier')
    axes[1].grid(alpha=0.2)
    
    add_watermark(fig)
    save_figure(fig, '08_incentive_distribution.png')
    plt.close(fig)

if __name__ == '__main__':
    main()
