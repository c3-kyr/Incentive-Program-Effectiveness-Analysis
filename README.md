# Incentive Program Effectiveness Analysis

> "Our incentive program is optimizing for the wrong metric — here's the data, and here's a redesigned structure with projected impact."

## Business Problem
Many call centers design incentive structures heavily weighted toward Call Volume and Average Handle Time (AHT) to maximize operational throughput. However, this often backfires, creating a perverse incentive where agents rush through calls. This leads to unresolved issues, plummeting Customer Satisfaction (CSAT) scores, and a surge in repeat calls—ultimately costing the business more than the original time savings. This project analyzes a simulated 200-agent call center to quantify these negative effects and proposes a redesigned incentive program optimized for quality and First Call Resolution (FCR).

## Methodology
- **Data Generation**: Simulated 200-agent call center with 12 months of performance data, capturing agent archetypes, base capabilities, and monthly metrics.
- **Statistical Analysis**: Regression analysis to isolate the effect of call volume on customer outcomes (CSAT, FCR).
- **Segmentation**: K-means clustering to identify agent behavioral archetypes (e.g., Volume Chasers vs. Quality Leaders).
- **Simulation**: Monte Carlo simulation (1,000 iterations) to project the financial and behavioral impact of a redesigned incentive structure.

## Key Findings
1. **Volume-based incentives produce perverse outcomes**: High call volume is strongly negatively correlated with CSAT and FCR (r = -0.401 for FCR), while driving higher repeat call rates.
2. **FCR is the strongest predictor of CSAT**: Improving First Call Resolution is the single most effective lever for increasing customer satisfaction. Regression showed the FCR coefficient (0.2165) is 16.7x larger than the call volume coefficient (0.0130).
3. **Agent segmentation reveals misaligned payouts**: The current payout structure disproportionately rewards "Volume Chasers" over "Quality Leaders". Notably, **63.5% of Bronze-tier agents** (lowest payouts) actually rank in the top half of overall quality metrics.
4. **Redesigned incentive structure projects significant improvement**: By shifting the incentive weight from volume to FCR and CSAT, Monte Carlo simulation projects an annual net financial impact of **$495,190.18** primarily from reducing repeat calls, at equivalent or lower total bonus cost.

## Recommendations
1. **Shift incentive weight from volume to FCR**: Reduce the volume weight from 40% to 15% and increase FCR from 15% to 40%.
2. **Add quality score component**: Introduce a 10% weight for quality assurance scores to reinforce comprehensive problem-solving.
3. **Implement balanced scorecard approach**: Ensure agents must meet minimum quality thresholds before volume multipliers apply.

## Tech Stack
- **Python** (pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn)
- **SQL** (BigQuery-compatible analytical queries using window functions, NTILE, CTEs)
- **Statistical Modeling** (OLS Regression, Hypothesis Testing, K-Means Clustering)
- **Monte Carlo Simulation** (Behavioral elasticity and ROI projection)

## Project Structure
```text
incentive-program-analysis/
├── README.md
├── requirements.txt
├── utils.py
├── data/
│   ├── generate_data.py
│   ├── agents.csv
│   └── monthly_performance.csv
├── sql/
│   └── bigquery_queries.sql
├── analysis/
│   ├── 01_eda.py
│   ├── 02_statistical_analysis.py
│   └── 03_incentive_simulation.py
└── output/
    └── [Generated visualizations 1-16]
```

## How to Run
```bash
pip install -r requirements.txt
python data/generate_data.py
python analysis/01_eda.py
python analysis/02_statistical_analysis.py
python analysis/03_incentive_simulation.py
```

## Resume Bullet Points
> **Incentive Program Effectiveness Analysis | Python, BigQuery SQL, Statistical Modeling**
> - Simulated 12 months of agent performance data (200 agents) and utilized regression analysis to identify perverse outcomes of volume-based incentive structures.
> - Found strong negative correlation between call volume and FCR (r = -0.401), revealing that 63.5% of lowest-paid (Bronze tier) agents were actually in the top half of quality performance.
> - Modeled a redesigned incentive structure using Monte Carlo simulations (1,000 iterations), projecting a substantial improvement in CSAT and **$495K+** in annual net cost savings from reduced repeat call volume.
> - Wrote 8 production-grade BigQuery SQL queries utilizing window functions, NTILE, and CTEs to analyze KPI distribution and team rankings.
