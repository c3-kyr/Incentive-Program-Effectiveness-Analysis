-- ============================================================
-- QUERY 1: Monthly KPI Summary by Team with Month-over-Month Change
-- ============================================================
-- Purpose: Calculate average KPIs by team and month, and use LAG() to compute MoM percent changes.
WITH monthly_team_kpi AS (
  SELECT
    ap.team,
    mp.month,
    AVG(mp.first_call_resolution) AS avg_fcr,
    AVG(mp.csat_score) AS avg_csat,
    SUM(mp.call_volume) AS total_call_volume
  FROM `servicing.monthly_performance` mp
  JOIN `servicing.agent_profiles` ap
    ON mp.agent_id = ap.agent_id
  GROUP BY ap.team, mp.month
)
SELECT
  team,
  month,
  avg_fcr,
  LAG(avg_fcr) OVER (PARTITION BY team ORDER BY month) AS prev_month_avg_fcr,
  SAFE_DIVIDE(avg_fcr - LAG(avg_fcr) OVER (PARTITION BY team ORDER BY month), LAG(avg_fcr) OVER (PARTITION BY team ORDER BY month)) AS fcr_mom_pct_change,
  avg_csat,
  LAG(avg_csat) OVER (PARTITION BY team ORDER BY month) AS prev_month_avg_csat,
  SAFE_DIVIDE(avg_csat - LAG(avg_csat) OVER (PARTITION BY team ORDER BY month), LAG(avg_csat) OVER (PARTITION BY team ORDER BY month)) AS csat_mom_pct_change,
  total_call_volume,
  LAG(total_call_volume) OVER (PARTITION BY team ORDER BY month) AS prev_month_call_volume,
  SAFE_DIVIDE(total_call_volume - LAG(total_call_volume) OVER (PARTITION BY team ORDER BY month), LAG(total_call_volume) OVER (PARTITION BY team ORDER BY month)) AS volume_mom_pct_change
FROM monthly_team_kpi
ORDER BY team, month;

-- ============================================================
-- QUERY 2: Incentive Payout Distribution Analysis
-- ============================================================
-- Purpose: Create quartiles of incentive_payout using NTILE and analyze quality metrics per quartile.
WITH payout_quartiles AS (
  SELECT
    agent_id,
    month,
    incentive_payout,
    NTILE(4) OVER (PARTITION BY month ORDER BY incentive_payout ASC) AS payout_quartile,
    first_call_resolution,
    csat_score,
    quality_score,
    call_volume
  FROM `servicing.monthly_performance`
  WHERE incentive_payout IS NOT NULL
)
SELECT
  month,
  payout_quartile,
  MIN(incentive_payout) AS min_payout,
  MAX(incentive_payout) AS max_payout,
  AVG(first_call_resolution) AS mean_fcr,
  AVG(csat_score) AS mean_csat,
  AVG(quality_score) AS mean_quality_score,
  AVG(call_volume) AS mean_call_volume
FROM payout_quartiles
GROUP BY month, payout_quartile
ORDER BY month, payout_quartile;

-- ============================================================
-- QUERY 3: Volume Quartile vs Quality Metrics — The Smoking Gun
-- ============================================================
-- Purpose: Create volume quartiles and compare quality metrics to expose the inverse correlation between volume and quality.
WITH volume_quartiles AS (
  SELECT
    agent_id,
    month,
    call_volume,
    NTILE(4) OVER (PARTITION BY month ORDER BY call_volume ASC) AS volume_quartile,
    first_call_resolution,
    csat_score,
    quality_score,
    transfer_rate
  FROM `servicing.monthly_performance`
)
SELECT
  month,
  volume_quartile,
  AVG(call_volume) AS mean_volume,
  AVG(first_call_resolution) AS mean_fcr,
  AVG(csat_score) AS mean_csat,
  AVG(quality_score) AS mean_quality_score,
  AVG(transfer_rate) AS mean_transfer_rate
FROM volume_quartiles
GROUP BY month, volume_quartile
ORDER BY month, volume_quartile;

-- ============================================================
-- QUERY 4: Statistical Correlation Analysis
-- ============================================================
-- Purpose: Calculate pairwise correlations between key performance metrics to find statistical dependencies.
SELECT
  'first_call_resolution' AS metric,
  CORR(first_call_resolution, first_call_resolution) AS corr_fcr,
  CORR(first_call_resolution, csat_score) AS corr_csat,
  CORR(first_call_resolution, quality_score) AS corr_quality,
  CORR(first_call_resolution, call_volume) AS corr_volume,
  CORR(first_call_resolution, incentive_payout) AS corr_incentive
FROM `servicing.monthly_performance`
UNION ALL
SELECT
  'csat_score' AS metric,
  CORR(csat_score, first_call_resolution) AS corr_fcr,
  CORR(csat_score, csat_score) AS corr_csat,
  CORR(csat_score, quality_score) AS corr_quality,
  CORR(csat_score, call_volume) AS corr_volume,
  CORR(csat_score, incentive_payout) AS corr_incentive
FROM `servicing.monthly_performance`
UNION ALL
SELECT
  'quality_score' AS metric,
  CORR(quality_score, first_call_resolution) AS corr_fcr,
  CORR(quality_score, csat_score) AS corr_csat,
  CORR(quality_score, quality_score) AS corr_quality,
  CORR(quality_score, call_volume) AS corr_volume,
  CORR(quality_score, incentive_payout) AS corr_incentive
FROM `servicing.monthly_performance`
UNION ALL
SELECT
  'call_volume' AS metric,
  CORR(call_volume, first_call_resolution) AS corr_fcr,
  CORR(call_volume, csat_score) AS corr_csat,
  CORR(call_volume, quality_score) AS corr_quality,
  CORR(call_volume, call_volume) AS corr_volume,
  CORR(call_volume, incentive_payout) AS corr_incentive
FROM `servicing.monthly_performance`;

-- ============================================================
-- QUERY 5: Team Performance Comparison with Rankings
-- ============================================================
-- Purpose: Rank teams based on composite performance metrics (FCR, CSAT, Volume, Quality).
WITH team_aggregated AS (
  SELECT
    ap.team,
    AVG(mp.first_call_resolution) AS avg_fcr,
    AVG(mp.csat_score) AS avg_csat,
    SUM(mp.call_volume) AS total_volume,
    AVG(mp.quality_score) AS avg_quality,
    (AVG(mp.first_call_resolution) + AVG(mp.csat_score) + AVG(mp.quality_score)) AS composite_score
  FROM `servicing.monthly_performance` mp
  JOIN `servicing.agent_profiles` ap
    ON mp.agent_id = ap.agent_id
  GROUP BY ap.team
)
SELECT
  team,
  avg_fcr,
  avg_csat,
  total_volume,
  avg_quality,
  composite_score,
  RANK() OVER (ORDER BY composite_score DESC) AS composite_rank,
  PERCENT_RANK() OVER (ORDER BY composite_score ASC) AS composite_percentile
FROM team_aggregated
ORDER BY composite_rank;

-- ============================================================
-- QUERY 6: Top vs Bottom Performer Comparison
-- ============================================================
-- Purpose: Identify top/bottom 20% by composite score and compare their volume, AHT, and payouts.
WITH agent_scores AS (
  SELECT
    agent_id,
    AVG(first_call_resolution) + AVG(csat_score) + AVG(quality_score) AS composite_score,
    AVG(call_volume) AS avg_volume,
    AVG(avg_handle_time) AS avg_aht,
    AVG(incentive_payout) AS avg_payout,
    PERCENT_RANK() OVER (ORDER BY (AVG(first_call_resolution) + AVG(csat_score) + AVG(quality_score)) ASC) AS score_percentile
  FROM `servicing.monthly_performance`
  GROUP BY agent_id
),
classified_agents AS (
  SELECT
    agent_id,
    composite_score,
    avg_volume,
    avg_aht,
    avg_payout,
    CASE 
      WHEN score_percentile >= 0.8 THEN 'Top 20%'
      WHEN score_percentile <= 0.2 THEN 'Bottom 20%'
      ELSE 'Middle 60%'
    END AS performance_group
  FROM agent_scores
)
SELECT
  performance_group,
  COUNT(agent_id) AS agent_count,
  AVG(composite_score) AS avg_composite_score,
  AVG(avg_volume) AS avg_volume,
  AVG(avg_aht) AS avg_aht,
  AVG(avg_payout) AS avg_payout
FROM classified_agents
WHERE performance_group IN ('Top 20%', 'Bottom 20%')
GROUP BY performance_group
ORDER BY performance_group DESC;

-- ============================================================
-- QUERY 7: Incentive ROI Analysis — Cost Per Quality Point
-- ============================================================
-- Purpose: Calculate how much incentive spend is used per unit of quality metric (FCR/CSAT) by team.
WITH team_payout_quality AS (
  SELECT
    ap.team,
    ap.skill_tier,
    SUM(mp.incentive_payout) AS total_incentive_payout,
    AVG(mp.first_call_resolution) AS avg_fcr,
    AVG(mp.csat_score) AS avg_csat
  FROM `servicing.monthly_performance` mp
  JOIN `servicing.agent_profiles` ap
    ON mp.agent_id = ap.agent_id
  GROUP BY ap.team, ap.skill_tier
)
SELECT
  team,
  skill_tier,
  total_incentive_payout,
  avg_fcr,
  avg_csat,
  SAFE_DIVIDE(total_incentive_payout, avg_fcr) AS cost_per_fcr_point,
  SAFE_DIVIDE(total_incentive_payout, avg_csat) AS cost_per_csat_point
FROM team_payout_quality
ORDER BY cost_per_fcr_point DESC;

-- ============================================================
-- QUERY 8: Rolling Performance Trends with Moving Averages
-- ============================================================
-- Purpose: Calculate 3-month rolling averages for key metrics using window functions to identify trends.
WITH rolling_metrics AS (
  SELECT
    agent_id,
    month,
    csat_score,
    AVG(csat_score) OVER (
      PARTITION BY agent_id 
      ORDER BY month 
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3m_csat,
    first_call_resolution,
    AVG(first_call_resolution) OVER (
      PARTITION BY agent_id 
      ORDER BY month 
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3m_fcr
  FROM `servicing.monthly_performance`
)
SELECT
  agent_id,
  month,
  csat_score,
  rolling_3m_csat,
  CASE 
    WHEN csat_score > LAG(csat_score) OVER (PARTITION BY agent_id ORDER BY month) THEN 'Improving'
    WHEN csat_score < LAG(csat_score) OVER (PARTITION BY agent_id ORDER BY month) THEN 'Declining'
    ELSE 'Neutral'
  END AS csat_trend,
  first_call_resolution,
  rolling_3m_fcr
FROM rolling_metrics
ORDER BY agent_id, month;
