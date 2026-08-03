"""
03_incentive_simulation.py
Compares the CURRENT incentive structure vs a PROPOSED redesign and runs Monte Carlo simulation to project impact.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (AMEX_PALETTE, TEAM_COLORS, TIER_COLORS,
                    setup_plot_style, load_data, save_figure,
                    format_pvalue, add_watermark, OUTPUT_DIR)

def main():
    setup_plot_style()
    df, agents, performance = load_data()
    
    # --- STEP 1: Recalculate Payouts Under Proposed Structure ---
    
    # Current incentive weights
    current_weights = {
        'volume': 0.40,
        'aht': 0.30,
        'fcr': 0.15,
        'csat': 0.15,
        'quality': 0.0
    }
    
    # Proposed incentive weights
    proposed_weights = {
        'volume': 0.15,
        'aht': 0.10,
        'fcr': 0.40,
        'csat': 0.25,
        'quality': 0.10
    }
    
    df_sim = df.copy()
    
    # Recalculate thresholds for each month
    proposed_scores = []
    
    for month, group in df_sim.groupby('month'):
        # volume_hit = 1 if call_volume >= 75th percentile for that month
        vol_p75 = group['call_volume'].quantile(0.75)
        volume_hit = (group['call_volume'] >= vol_p75).astype(int)
        
        # aht_hit = 1 if avg_handle_time <= median for that month
        aht_med = group['avg_handle_time'].median()
        aht_hit = (group['avg_handle_time'] <= aht_med).astype(int)
        
        # fcr_hit = 1 if first_call_resolution >= 0.75
        fcr_hit = (group['first_call_resolution'] >= 0.75).astype(int)
        
        # csat_hit = 1 if csat_score >= 4.0
        csat_hit = (group['csat_score'] >= 4.0).astype(int)
        
        # quality_hit = 1 if quality_score >= 75th percentile for that month (NEW)
        qual_p75 = group['quality_score'].quantile(0.75)
        quality_hit = (group['quality_score'] >= qual_p75).astype(int)
        
        # proposed_score = 0.15*volume_hit + 0.10*aht_hit + 0.40*fcr_hit + 0.25*csat_hit + 0.10*quality_hit
        score = (0.15 * volume_hit + 
                 0.10 * aht_hit + 
                 0.40 * fcr_hit + 
                 0.25 * csat_hit + 
                 0.10 * quality_hit)
        
        group['proposed_score'] = score
        proposed_scores.append(group['proposed_score'])
        
    df_sim['proposed_score'] = pd.concat(proposed_scores)
    df_sim['proposed_payout'] = df_sim['proposed_score'] * 500
    
    total_current_payout = df_sim['incentive_payout'].sum()
    total_proposed_payout = df_sim['proposed_payout'].sum()
    
    print("--- Payout Comparison ---")
    print(f"Total Current Payouts: ${total_current_payout:,.2f}")
    print(f"Total Proposed Payouts: ${total_proposed_payout:,.2f}")
    print(f"Difference: ${total_proposed_payout - total_current_payout:,.2f}\n")
    
    # Chart 13: 13_incentive_weight_comparison.png
    fig, ax = plt.subplots(figsize=(12, 7))
    metrics = ['Volume', 'AHT', 'FCR', 'CSAT', 'Quality']
    curr_w = [current_weights['volume'], current_weights['aht'], current_weights['fcr'], current_weights['csat'], current_weights['quality']]
    prop_w = [proposed_weights['volume'], proposed_weights['aht'], proposed_weights['fcr'], proposed_weights['csat'], proposed_weights['quality']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, curr_w, width, label='Current', color=AMEX_PALETTE.get('primary_blue', '#00175a'))
    rects2 = ax.bar(x + width/2, prop_w, width, label='Proposed', color=AMEX_PALETTE.get('accent_green', '#007a33'))
    
    ax.set_ylabel('Weight')
    ax.set_title('Incentive Structure Redesign — Shifting Weight to Quality', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)
    
    add_watermark(ax)
    save_figure(fig, '13_incentive_weight_comparison.png')
    plt.close()
    
    # --- STEP 2: Behavioral Response Model ---
    base_elasticity = 0.20
    
    current_means = {
        'volume': df_sim['call_volume'].mean(),
        'aht': df_sim['avg_handle_time'].mean(),
        'fcr': df_sim['first_call_resolution'].mean(),
        'csat': df_sim['csat_score'].mean(),
        'quality': df_sim['quality_score'].mean()
    }
    
    weight_changes = {
        k: proposed_weights[k] - current_weights[k] for k in current_weights.keys()
    }
    
    # --- STEP 3: Monte Carlo Simulation (1000 iterations) ---
    np.random.seed(42)
    n_iterations = 1000
    
    sim_results = {
        'fcr': [],
        'csat': [],
        'volume': [],
        'quality': [],
        'repeat_rate': []
    }
    
    for _ in range(n_iterations):
        elasticity = np.clip(np.random.normal(0.20, 0.05), 0.05, 0.40)
        
        proj_fcr = current_means['fcr'] + current_means['fcr'] * elasticity * weight_changes['fcr']
        proj_csat = current_means['csat'] + current_means['csat'] * elasticity * weight_changes['csat']
        proj_vol = current_means['volume'] + current_means['volume'] * elasticity * weight_changes['volume']
        proj_qual = current_means['quality'] + current_means['quality'] * elasticity * weight_changes['quality']
        
        # Add noise: normal(0, 0.02 * current_value)
        proj_fcr += np.random.normal(0, 0.02 * current_means['fcr'])
        proj_csat += np.random.normal(0, 0.02 * current_means['csat'])
        proj_vol += np.random.normal(0, 0.02 * current_means['volume'])
        proj_qual += np.random.normal(0, 0.02 * current_means['quality'])
        
        proj_repeat_rate = 0.50 - 0.40 * proj_fcr
        
        sim_results['fcr'].append(proj_fcr)
        sim_results['csat'].append(proj_csat)
        sim_results['volume'].append(proj_vol)
        sim_results['quality'].append(proj_qual)
        sim_results['repeat_rate'].append(proj_repeat_rate)
        
    sim_df = pd.DataFrame(sim_results)
    
    # Chart 14: 14_monte_carlo_csat.png
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(sim_df['csat'], kde=True, color=AMEX_PALETTE.get('primary_blue', '#00175a'), ax=ax)
    
    ax.axvline(current_means['csat'], color=AMEX_PALETTE.get('alert_red', '#d9261c'), linestyle='--', label=f"Current Mean ({current_means['csat']:.2f})")
    
    mean_proj_csat = sim_df['csat'].mean()
    ci_lower = np.percentile(sim_df['csat'], 2.5)
    ci_upper = np.percentile(sim_df['csat'], 97.5)
    
    ax.axvline(mean_proj_csat, color=AMEX_PALETTE.get('accent_green', '#007a33'), linestyle='-', label=f"Projected Mean ({mean_proj_csat:.2f})")
    ax.axvspan(ci_lower, ci_upper, color=AMEX_PALETTE.get('accent_green', '#007a33'), alpha=0.2, label='95% CI')
    
    if mean_proj_csat > current_means['csat']:
        ax.axvspan(current_means['csat'], ax.get_xlim()[1], color=AMEX_PALETTE.get('accent_green', '#007a33'), alpha=0.1)
        
    ax.set_title('Monte Carlo Simulation: Projected CSAT Under Redesigned Incentives (n=1,000)', pad=20)
    ax.set_xlabel('CSAT Score')
    ax.legend()
    
    add_watermark(ax)
    save_figure(fig, '14_monte_carlo_csat.png')
    plt.close()
    
    # Chart 15: 15_projected_changes.png
    current_repeat_rate = df_sim['repeat_call_rate'].mean()
    
    metrics_display = ['FCR', 'CSAT', 'Call Volume', 'Quality Score', 'Repeat Call Rate']
    
    changes = [
        (sim_df['fcr'].mean() - current_means['fcr']) / current_means['fcr'] * 100,
        (sim_df['csat'].mean() - current_means['csat']) / current_means['csat'] * 100,
        (sim_df['volume'].mean() - current_means['volume']) / current_means['volume'] * 100,
        (sim_df['quality'].mean() - current_means['quality']) / current_means['quality'] * 100,
        (sim_df['repeat_rate'].mean() - current_repeat_rate) / current_repeat_rate * 100
    ]
    
    ci_bounds = [
        np.percentile((sim_df['fcr'] - current_means['fcr']) / current_means['fcr'] * 100, [2.5, 97.5]),
        np.percentile((sim_df['csat'] - current_means['csat']) / current_means['csat'] * 100, [2.5, 97.5]),
        np.percentile((sim_df['volume'] - current_means['volume']) / current_means['volume'] * 100, [2.5, 97.5]),
        np.percentile((sim_df['quality'] - current_means['quality']) / current_means['quality'] * 100, [2.5, 97.5]),
        np.percentile((sim_df['repeat_rate'] - current_repeat_rate) / current_repeat_rate * 100, [2.5, 97.5])
    ]
    
    errors = np.array([[(c - b[0]), (b[1] - c)] for c, b in zip(changes, ci_bounds)]).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(metrics_display))
    
    is_improvement = [
        changes[0] > 0, # FCR (higher is better)
        changes[1] > 0, # CSAT (higher is better)
        changes[2] < 0, # Vol (lower is better in this context of shifting away from volume)
        changes[3] > 0, # Quality (higher is better)
        changes[4] < 0  # Repeat (lower is better)
    ]
    
    colors = [AMEX_PALETTE.get('accent_green', '#007a33') if imp else AMEX_PALETTE.get('alert_red', '#d9261c') for imp in is_improvement]
    
    bars = ax.barh(y_pos, changes, xerr=errors, align='center', color=colors, ecolor='black', capsize=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics_display)
    ax.invert_yaxis()
    ax.set_xlabel('Projected % Change')
    ax.set_title('Projected Impact of Incentive Redesign', pad=20)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 5 if width > 0 else width - 15
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center')
        
    add_watermark(ax)
    save_figure(fig, '15_projected_changes.png')
    plt.close()
    
    # --- STEP 4: ROI Analysis ---
    cost_per_repeat_call = 10
    
    avg_call_vol = current_means['volume']
    avg_proj_vol = sim_df['volume'].mean()
    
    current_repeats = current_repeat_rate * avg_call_vol
    projected_repeats = sim_df['repeat_rate'].mean() * avg_proj_vol
    
    savings_per_agent = (current_repeats - projected_repeats) * cost_per_repeat_call
    total_agents = 200
    monthly_savings = savings_per_agent * total_agents
    annual_savings = monthly_savings * 12
    
    total_bonus_cost_diff_annual = total_proposed_payout - total_current_payout
    
    net_annual_impact = annual_savings - total_bonus_cost_diff_annual
    
    # Chart 16: 16_roi_analysis.png
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Current Repeat Call Cost\n(Annual)', 'Projected Repeat Call Cost\n(Annual)', 'Gross Savings\n(Annual)', 'Bonus Cost Change\n(Annual)', 'NET ROI\n(Annual)']
    
    current_repeat_annual = current_repeats * cost_per_repeat_call * total_agents * 12
    proj_repeat_annual = projected_repeats * cost_per_repeat_call * total_agents * 12
    
    values = [
        current_repeat_annual,
        proj_repeat_annual,
        annual_savings,
        total_bonus_cost_diff_annual,
        net_annual_impact
    ]
    
    bar_colors = [
        AMEX_PALETTE.get('secondary_blue', '#005fb2'),
        AMEX_PALETTE.get('secondary_blue', '#005fb2'),
        AMEX_PALETTE.get('accent_green', '#007a33'),
        AMEX_PALETTE.get('alert_red', '#d9261c') if total_bonus_cost_diff_annual > 0 else AMEX_PALETTE.get('accent_green', '#007a33'),
        AMEX_PALETTE.get('accent_green', '#007a33') if net_annual_impact > 0 else AMEX_PALETTE.get('alert_red', '#d9261c')
    ]
    
    bars = ax.bar(categories, values, color=bar_colors)
    
    ax.set_ylabel('Annual Cost / Savings ($)')
    ax.set_title('ROI Analysis — Incentive Redesign Financial Impact', pad=20)
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"${height:,.0f}",
                ha='center', va='bottom' if height >= 0 else 'top')
                
    add_watermark(ax)
    save_figure(fig, '16_roi_analysis.png')
    plt.close()
    
    # --- FINAL SUMMARY ---
    print("=" * 60)
    print("EXECUTIVE SUMMARY: INCENTIVE REDESIGN PROJECTED IMPACT")
    print("=" * 60)
    print(f"Current Repeat Calls / Agent / Month:   {current_repeats:.2f}")
    print(f"Projected Repeat Calls / Agent / Month: {projected_repeats:.2f}")
    print(f"Savings / Agent / Month:                ${savings_per_agent:,.2f}")
    print(f"Total Monthly Savings (200 agents):     ${monthly_savings:,.2f}")
    print(f"Annual Savings Projection:              ${annual_savings:,.2f}")
    print(f"Bonus Cost Difference (Annualized):     ${total_bonus_cost_diff_annual:,.2f}")
    print("-" * 60)
    print(f"NET ANNUAL IMPACT:                      ${net_annual_impact:,.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
