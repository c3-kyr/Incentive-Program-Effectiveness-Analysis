import pandas as pd
import numpy as np
from pathlib import Path
import os

def generate_data():
    np.random.seed(42)
    
    num_agents = 200
    teams = ['Alpha', 'Beta', 'Gamma', 'Delta']
    
    # Generate Agent Profiles
    agent_ids = [f'A{i:03d}' for i in range(1, num_agents + 1)]
    agent_teams = np.random.choice(teams, num_agents)
    
    # tenure_months: lognormal(3, 0.8) clipped to [1, 120]
    tenure_months = np.clip(np.random.lognormal(3, 0.8, num_agents), 1, 120).astype(int)
    
    # skill_tier
    skill_tier = []
    for t in tenure_months:
        if t < 12:
            skill_tier.append('Junior')
        elif t < 36:
            skill_tier.append('Mid')
        elif t < 72:
            skill_tier.append('Senior')
        else:
            skill_tier.append('Expert')
            
    # base_capability: 0.3*(tenure/120) + 0.5*beta(2,3) + 0.2*uniform(0,1), clipped to [0.15, 0.98]
    base_capability = np.clip(
        0.3 * (tenure_months / 120) + 0.5 * np.random.beta(2, 3, num_agents) + 0.2 * np.random.uniform(0, 1, num_agents),
        0.15, 0.98
    )
    
    # volume_tendency: beta(2,2)
    volume_tendency = np.clip(np.random.beta(2, 2, num_agents), 0.1, 0.9)
    
    agents_df = pd.DataFrame({
        'agent_id': agent_ids,
        'team': agent_teams,
        'tenure_months': tenure_months,
        'skill_tier': skill_tier,
        'base_capability': base_capability,
        'volume_tendency': volume_tendency
    })
    
    # Generate Monthly Performance
    perf_records = []
    for month in range(1, 13):
        # Calculate for each agent in this month
        call_vol = np.clip(350 + 400 * volume_tendency + 50 * base_capability + np.random.normal(0, 30, num_agents), 250, 850).astype(int)
        aht = np.clip(500 - 250 * volume_tendency + 100 * (1 - base_capability) + np.random.normal(0, 30, num_agents), 150, 650)
        fcr = np.clip(0.50 + 0.35 * base_capability - 0.18 * volume_tendency + 0.08 * (tenure_months / 120) + np.random.normal(0, 0.04, num_agents), 0.40, 0.98)
        
        aht_norm = (aht - 150) / (650 - 150)
        csat = np.clip(1.5 + 2.8 * fcr + 0.4 * aht_norm + 0.3 * base_capability + np.random.normal(0, 0.15, num_agents), 1.5, 5.0)
        
        qual = np.clip(35 + 50 * base_capability - 15 * volume_tendency + np.random.normal(0, 5, num_agents), 25, 100)
        trans_rate = np.clip(0.05 + 0.25 * volume_tendency - 0.10 * base_capability + np.random.normal(0, 0.03, num_agents), 0.02, 0.45)
        rep_call = np.clip(0.50 - 0.40 * fcr + np.random.normal(0, 0.03, num_agents), 0.03, 0.50)
        
        # Current month df
        month_df = pd.DataFrame({
            'agent_id': agent_ids,
            'month': month,
            'call_volume': call_vol,
            'avg_handle_time': aht,
            'first_call_resolution': fcr,
            'csat_score': csat,
            'quality_score': qual,
            'transfer_rate': trans_rate,
            'repeat_call_rate': rep_call
        })
        
        # Calculate thresholds
        vol_75 = month_df['call_volume'].quantile(0.75)
        aht_50 = month_df['avg_handle_time'].median()
        
        volume_hit = (month_df['call_volume'] >= vol_75).astype(int)
        aht_hit = (month_df['avg_handle_time'] <= aht_50).astype(int)
        fcr_hit = (month_df['first_call_resolution'] >= 0.75).astype(int)
        csat_hit = (month_df['csat_score'] >= 4.0).astype(int)
        
        weighted_score = 0.40 * volume_hit + 0.30 * aht_hit + 0.15 * fcr_hit + 0.15 * csat_hit
        base_bonus = 500
        
        month_df['incentive_payout'] = np.round(weighted_score * base_bonus, 2)
        
        tier = []
        for s in weighted_score:
            if s >= 0.85:
                tier.append('Platinum')
            elif s >= 0.55:
                tier.append('Gold')
            elif s >= 0.30:
                tier.append('Silver')
            else:
                tier.append('Bronze')
                
        month_df['incentive_tier'] = tier
        
        perf_records.append(month_df)
        
    perf_df = pd.concat(perf_records, ignore_index=True)
    
    # Save
    data_dir = Path(__file__).resolve().parent
    data_dir.mkdir(parents=True, exist_ok=True)
    
    agents_path = data_dir / 'agents.csv'
    perf_path = data_dir / 'monthly_performance.csv'
    
    agents_df.to_csv(agents_path, index=False)
    perf_df.to_csv(perf_path, index=False)
    
    # Summarize
    print(f"Total records generated: {len(perf_df)} (agent-months)")
    print("\nMean of each metric:")
    metrics = ['call_volume', 'avg_handle_time', 'first_call_resolution', 'csat_score', 'quality_score', 'transfer_rate', 'repeat_call_rate', 'incentive_payout']
    print(perf_df[metrics].mean())
    print("\nStd of each metric:")
    print(perf_df[metrics].std())
    
    corr_vol_fcr = perf_df['call_volume'].corr(perf_df['first_call_resolution'])
    print(f"\nCorrelation between call_volume and first_call_resolution: {corr_vol_fcr:.3f}")
    
    print("\nDistribution of incentive tiers:")
    print(perf_df['incentive_tier'].value_counts(normalize=True))
    
    merged_df = pd.merge(perf_df, agents_df, on='agent_id')
    print("\nCorrelations with volume_tendency (Verification of causal structure):")
    print(f"  call_volume: {merged_df['volume_tendency'].corr(merged_df['call_volume']):.3f}")
    print(f"  first_call_resolution: {merged_df['volume_tendency'].corr(merged_df['first_call_resolution']):.3f}")
    print(f"  avg_handle_time: {merged_df['volume_tendency'].corr(merged_df['avg_handle_time']):.3f}")
    print(f"  csat_score: {merged_df['volume_tendency'].corr(merged_df['csat_score']):.3f}")

if __name__ == '__main__':
    generate_data()
