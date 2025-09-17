# Databricks notebook source
# MAGIC %md
# MAGIC # Chipotle Analytics - Business Aggregations
# MAGIC 
# MAGIC **Purpose**: Compute pre-aggregated business metrics for fast analytics and Genie spaces
# MAGIC 
# MAGIC **Outputs**: 
# MAGIC - `gold.store_performance_monthly` - Monthly store KPIs with comparisons
# MAGIC - `gold.customer_segments_monthly` - Customer segment behavior and value metrics
# MAGIC 
# MAGIC **Assumptions**:
# MAGIC - Synthetic data already generated
# MAGIC - Unity Catalog permissions in place
# MAGIC - Runs incrementally (processes new months only)
# MAGIC 
# MAGIC **Parameters**:
# MAGIC - processing_mode: full_refresh or incremental (default: incremental)
# MAGIC - months_lookback: Number of months to process in incremental mode (default: 3)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Dict

# Widget parameters
dbutils.widgets.text("catalog_name", "chipotle_analytics", "Catalog Name")
dbutils.widgets.dropdown("processing_mode", "incremental", ["full_refresh", "incremental"], "Processing Mode")
dbutils.widgets.text("months_lookback", "3", "Months Lookback (for incremental)")

# Get parameters
CATALOG = dbutils.widgets.get("catalog_name")
PROCESSING_MODE = dbutils.widgets.get("processing_mode")
MONTHS_LOOKBACK = int(dbutils.widgets.get("months_lookback"))

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md ## Step 1: Determine Processing Range

# COMMAND ----------

# Get current max dates from aggregation tables
def get_processing_dates():
    """Determine date range to process based on mode and existing data."""
    
    # Get latest transaction date
    max_trans_date = spark.sql(f"""
        SELECT MAX(order_date) as max_date 
        FROM {CATALOG}.gold.transactions
    """).collect()[0]['max_date']
    
    if PROCESSING_MODE == "full_refresh":
        # Process all available data
        start_date = spark.sql(f"""
            SELECT MIN(order_date) as min_date 
            FROM {CATALOG}.gold.transactions
        """).collect()[0]['min_date']
        
        print(f"Full refresh mode - processing from {start_date} to {max_trans_date}")
        return start_date, max_trans_date
    
    else:  # incremental
        # Check what's already been processed
        try:
            last_processed = spark.sql(f"""
                SELECT MAX(year_month) as last_month 
                FROM {CATALOG}.gold.store_performance_monthly
            """).collect()[0]['last_month']
            
            if last_processed:
                # Start from next month after last processed
                year, month = map(int, last_processed.split('-'))
                start_date = datetime(year, month, 1) + relativedelta(months=1)
                start_date = start_date.date()
            else:
                # No data yet, start from beginning
                start_date = spark.sql(f"""
                    SELECT MIN(order_date) as min_date 
                    FROM {CATALOG}.gold.transactions
                """).collect()[0]['min_date']
        except:
            # Table doesn't exist, start from beginning
            start_date = spark.sql(f"""
                SELECT MIN(order_date) as min_date 
                FROM {CATALOG}.gold.transactions
            """).collect()[0]['min_date']
        
        # But don't go back more than MONTHS_LOOKBACK
        lookback_date = (datetime.now() - relativedelta(months=MONTHS_LOOKBACK)).date()
        if start_date < lookback_date:
            start_date = lookback_date.replace(day=1)
        
        print(f"Incremental mode - processing from {start_date} to {max_trans_date}")
        return start_date, max_trans_date

START_DATE, END_DATE = get_processing_dates()

# COMMAND ----------

# MAGIC %md ## Step 2: Monthly Store Performance Aggregation

# COMMAND ----------

# Build monthly store performance metrics
store_monthly_df = spark.sql(f"""
    WITH monthly_revenue AS (
        SELECT 
            t.store_id,
            DATE_FORMAT(t.order_date, 'yyyy-MM') as year_month,
            COUNT(*) as transaction_count,
            COUNT(DISTINCT t.customer_id) as unique_customers,
            SUM(t.total_amount) as total_revenue,
            AVG(t.total_amount) as avg_ticket,
            
            -- Channel breakdown
            SUM(CASE WHEN t.channel IN ('app', 'web') THEN t.total_amount ELSE 0 END) / SUM(t.total_amount) as digital_mix_pct,
            
            -- Customer metrics
            COUNT(DISTINCT CASE 
                WHEN t.customer_id IN (
                    SELECT DISTINCT customer_id 
                    FROM {CATALOG}.gold.transactions 
                    WHERE order_date < DATE_TRUNC('month', t.order_date)
                ) THEN NULL 
                ELSE t.customer_id 
            END) as new_customers,
            
            -- Promotional impact
            SUM(t.discount_amount) as total_discounts
        FROM {CATALOG}.gold.transactions t
        WHERE t.order_date >= '{START_DATE}' 
          AND t.order_date <= '{END_DATE}'
        GROUP BY t.store_id, DATE_FORMAT(t.order_date, 'yyyy-MM')
    ),
    
    monthly_operations AS (
        SELECT
            store_id,
            DATE_FORMAT(business_date, 'yyyy-MM') as year_month,
            AVG(food_cost_pct) as food_cost_pct,
            AVG(avg_satisfaction_score) as nps_score,
            AVG(staff_hours_actual / NULLIF(staff_hours_scheduled, 0)) as labor_efficiency,
            SUM(waste_amount) as total_waste
        FROM {CATALOG}.gold.daily_store_performance
        WHERE business_date >= '{START_DATE}' 
          AND business_date <= '{END_DATE}'
        GROUP BY store_id, DATE_FORMAT(business_date, 'yyyy-MM')
    ),
    
    prior_month_metrics AS (
        SELECT
            store_id,
            year_month,
            LAG(total_revenue, 1) OVER (PARTITION BY store_id ORDER BY year_month) as prev_revenue,
            LAG(unique_customers, 1) OVER (PARTITION BY store_id ORDER BY year_month) as prev_customers
        FROM monthly_revenue
    ),
    
    regional_benchmarks AS (
        SELECT 
            s.state,
            r.year_month,
            AVG(r.total_revenue) as region_avg_revenue,
            PERCENTILE(r.total_revenue, 0.5) as region_median_revenue
        FROM monthly_revenue r
        JOIN {CATALOG}.gold.store_locations s ON r.store_id = s.store_id
        GROUP BY s.state, r.year_month
    ),
    
    national_benchmarks AS (
        SELECT
            year_month,
            AVG(total_revenue) as national_avg_revenue,
            PERCENTILE(total_revenue, 0.5) as national_median_revenue
        FROM monthly_revenue
        GROUP BY year_month
    )
    
    SELECT
        r.store_id,
        r.year_month,
        
        -- Revenue metrics
        r.total_revenue,
        ROUND((r.total_revenue - p.prev_revenue) / NULLIF(p.prev_revenue, 0) * 100, 2) as revenue_growth_pct,
        ROUND(r.total_revenue / (DATEDIFF(LAST_DAY(CONCAT(r.year_month, '-01')), CONCAT(r.year_month, '-01')) + 1) / 150 * 100, 2) as revenue_vs_budget_pct,
        0.0 as market_share_est_pct,  -- Simplified
        
        -- Customer metrics  
        r.unique_customers,
        r.new_customers,
        ROUND((r.unique_customers - r.new_customers) / NULLIF(p.prev_customers, 0) * 100, 2) as customer_retention_rate,
        ROUND(r.transaction_count / NULLIF(r.unique_customers, 0), 2) as avg_visits_per_customer,
        o.nps_score,
        
        -- Operational metrics
        r.avg_ticket,
        ROUND(r.transaction_count / (DATEDIFF(LAST_DAY(CONCAT(r.year_month, '-01')), CONCAT(r.year_month, '-01')) + 1), 2) as transactions_per_day,
        ROUND(r.digital_mix_pct * 100, 2) as digital_mix_pct,
        ROUND(o.food_cost_pct * 100, 2) as food_cost_pct,
        ROUND((1 - o.labor_efficiency) * 100, 2) as labor_cost_pct,
        
        -- Benchmark comparisons
        ROUND((r.total_revenue - rb.region_avg_revenue) / NULLIF(rb.region_avg_revenue, 0) * 100, 2) as vs_region_avg_pct,
        0.0 as vs_similar_stores_pct,  -- Simplified
        ROUND((r.total_revenue - nb.national_avg_revenue) / NULLIF(nb.national_avg_revenue, 0) * 100, 2) as vs_national_avg_pct,
        CAST(PERCENT_RANK() OVER (PARTITION BY r.year_month ORDER BY r.total_revenue) * 100 AS INT) as percentile_ranking,
        
        CURRENT_TIMESTAMP() as calculated_timestamp
        
    FROM monthly_revenue r
    LEFT JOIN monthly_operations o ON r.store_id = o.store_id AND r.year_month = o.year_month
    LEFT JOIN prior_month_metrics p ON r.store_id = p.store_id AND r.year_month = p.year_month
    LEFT JOIN {CATALOG}.gold.store_locations s ON r.store_id = s.store_id
    LEFT JOIN regional_benchmarks rb ON s.state = rb.state AND r.year_month = rb.year_month
    LEFT JOIN national_benchmarks nb ON r.year_month = nb.year_month
""")

# Write results
if PROCESSING_MODE == "full_refresh":
    store_monthly_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.gold.store_performance_monthly")
else:
    # Merge for incremental
    store_monthly_df.createOrReplaceTempView("store_monthly_updates")
    spark.sql(f"""
        MERGE INTO {CATALOG}.gold.store_performance_monthly t
        USING store_monthly_updates s
        ON t.store_id = s.store_id AND t.year_month = s.year_month
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

row_count = store_monthly_df.count()
print(f"✓ Processed {row_count} store-month combinations")

# COMMAND ----------

# MAGIC %md ## Step 3: Customer Segments Monthly Aggregation

# COMMAND ----------

# Build customer segment monthly metrics
segments_monthly_df = spark.sql(f"""
    WITH customer_monthly_activity AS (
        SELECT
            c.customer_segment as segment_name,
            DATE_FORMAT(t.order_date, 'yyyy-MM') as year_month,
            c.customer_id,
            COUNT(*) as visits,
            SUM(t.total_amount) as spend,
            AVG(t.total_amount) as avg_order_value,
            
            -- Channel preferences
            SUM(CASE WHEN t.channel = 'app' THEN 1 ELSE 0 END) / COUNT(*) as app_pct,
            SUM(CASE WHEN t.channel = 'web' THEN 1 ELSE 0 END) / COUNT(*) as web_pct,
            SUM(CASE WHEN t.channel = 'in_store' THEN 1 ELSE 0 END) / COUNT(*) as in_store_pct,
            
            -- Daypart preferences
            SUM(CASE WHEN HOUR(t.order_timestamp) BETWEEN 7 AND 10 THEN 1 ELSE 0 END) / COUNT(*) as breakfast_pct,
            SUM(CASE WHEN HOUR(t.order_timestamp) BETWEEN 11 AND 14 THEN 1 ELSE 0 END) / COUNT(*) as lunch_pct,
            SUM(CASE WHEN HOUR(t.order_timestamp) BETWEEN 17 AND 20 THEN 1 ELSE 0 END) / COUNT(*) as dinner_pct,
            
            -- Promotional response
            SUM(CASE WHEN SIZE(t.promotion_codes) > 0 THEN 1 ELSE 0 END) / COUNT(*) as promo_usage_rate
            
        FROM {CATALOG}.gold.transactions t
        JOIN {CATALOG}.gold.customer_profiles c ON t.customer_id = c.customer_id
        WHERE t.order_date >= '{START_DATE}' 
          AND t.order_date <= '{END_DATE}'
        GROUP BY c.customer_segment, DATE_FORMAT(t.order_date, 'yyyy-MM'), c.customer_id
    ),
    
    segment_aggregates AS (
        SELECT
            segment_name,
            year_month,
            COUNT(DISTINCT customer_id) as segment_size,
            AVG(visits) as avg_monthly_visits,
            AVG(avg_order_value) as avg_order_value,
            AVG(spend * 12) as lifetime_value,  -- Annualized estimate
            
            -- Build channel mix map
            MAP(
                'app', ROUND(AVG(app_pct), 3),
                'web', ROUND(AVG(web_pct), 3),
                'in_store', ROUND(AVG(in_store_pct), 3)
            ) as channel_mix,
            
            -- Determine preferred dayparts
            CASE
                WHEN AVG(lunch_pct) > 0.5 THEN ARRAY('lunch')
                WHEN AVG(dinner_pct) > 0.4 THEN ARRAY('dinner')
                ELSE ARRAY('lunch', 'dinner')
            END as preferred_dayparts,
            
            AVG(promo_usage_rate) as promotion_response_rate
            
        FROM customer_monthly_activity
        GROUP BY segment_name, year_month
    ),
    
    segment_growth AS (
        SELECT
            segment_name,
            year_month,
            segment_size,
            LAG(segment_size, 1) OVER (PARTITION BY segment_name ORDER BY year_month) as prev_size
        FROM segment_aggregates
    ),
    
    protein_preferences AS (
        -- Simplified protein preference calculation
        SELECT
            c.customer_segment as segment_name,
            MAP(
                'chicken', 0.45,
                'steak', 0.25,
                'carnitas', 0.15,
                'barbacoa', 0.10,
                'sofritas', 0.05
            ) as protein_preferences
        FROM {CATALOG}.gold.customer_profiles c
        GROUP BY c.customer_segment
    )
    
    SELECT
        a.segment_name,
        a.year_month,
        a.segment_size,
        ROUND(a.avg_monthly_visits, 2) as avg_monthly_visits,
        ROUND(a.avg_order_value, 2) as avg_order_value,
        ROUND(a.lifetime_value, 2) as lifetime_value,
        
        -- Calculate churn (simplified)
        CASE 
            WHEN a.segment_name = 'power_user' THEN 2.0
            WHEN a.segment_name = 'loyal_regular' THEN 5.0
            WHEN a.segment_name = 'occasional' THEN 15.0
            ELSE 25.0
        END as churn_rate_pct,
        
        a.preferred_dayparts,
        a.channel_mix,
        p.protein_preferences,
        ROUND(a.promotion_response_rate, 3) as promotion_response_rate,
        
        -- Growth metrics
        ROUND((a.segment_size - g.prev_size) / NULLIF(g.prev_size, 0) * 100, 2) as segment_growth_pct,
        
        -- Revenue contribution (simplified)
        ROUND(a.segment_size * a.avg_monthly_visits * a.avg_order_value / 
            (SELECT SUM(segment_size * avg_monthly_visits * avg_order_value) 
             FROM segment_aggregates WHERE year_month = a.year_month) * 100, 2) as revenue_contribution_pct,
        
        -- Acquisition cost (simplified estimate)
        CASE 
            WHEN a.segment_name = 'power_user' THEN 25.00
            WHEN a.segment_name = 'loyal_regular' THEN 18.00
            WHEN a.segment_name = 'occasional' THEN 12.00
            ELSE 8.00
        END as acquisition_cost
        
    FROM segment_aggregates a
    LEFT JOIN segment_growth g ON a.segment_name = g.segment_name AND a.year_month = g.year_month
    LEFT JOIN protein_preferences p ON a.segment_name = p.segment_name
""")

# Write results
if PROCESSING_MODE == "full_refresh":
    segments_monthly_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.gold.customer_segments_monthly")
else:
    # Merge for incremental
    segments_monthly_df.createOrReplaceTempView("segments_monthly_updates")
    spark.sql(f"""
        MERGE INTO {CATALOG}.gold.customer_segments_monthly t
        USING segments_monthly_updates s
        ON t.segment_name = s.segment_name AND t.year_month = s.year_month
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

row_count = segments_monthly_df.count()
print(f"✓ Processed {row_count} segment-month combinations")

# COMMAND ----------

# MAGIC %md ## Step 4: Create Supporting Views for Genie Spaces

# COMMAND ----------

# Create denormalized view for easier Genie access
spark.sql(f"""
    CREATE OR REPLACE VIEW {CATALOG}.gold.v_store_performance_enriched AS
    SELECT
        p.*,
        s.city,
        s.state,
        s.trade_area_type,
        s.store_format,
        s.seating_capacity,
        s.population_3mi,
        s.median_income_3mi,
        
        -- Add YoY comparisons
        LAG(p.total_revenue, 12) OVER (PARTITION BY p.store_id ORDER BY p.year_month) as revenue_yoy_prev,
        LAG(p.unique_customers, 12) OVER (PARTITION BY p.store_id ORDER BY p.year_month) as customers_yoy_prev,
        
        -- Add quarter aggregation
        CONCAT(SUBSTRING(p.year_month, 1, 4), '-Q', QUARTER(CONCAT(p.year_month, '-01'))) as year_quarter
        
    FROM {CATALOG}.gold.store_performance_monthly p
    JOIN {CATALOG}.gold.store_locations s ON p.store_id = s.store_id
""")

# Create customer lifetime value view
spark.sql(f"""
    CREATE OR REPLACE VIEW {CATALOG}.gold.v_customer_lifetime_value AS
    WITH customer_history AS (
        SELECT
            c.customer_id,
            c.customer_segment,
            c.registration_date,
            COUNT(DISTINCT t.order_id) as total_orders,
            SUM(t.total_amount) as total_spend,
            MIN(t.order_date) as first_order_date,
            MAX(t.order_date) as last_order_date,
            DATEDIFF(MAX(t.order_date), MIN(t.order_date)) + 1 as customer_tenure_days
        FROM {CATALOG}.gold.customer_profiles c
        LEFT JOIN {CATALOG}.gold.transactions t ON c.customer_id = t.customer_id
        GROUP BY c.customer_id, c.customer_segment, c.registration_date
    )
    SELECT
        customer_id,
        customer_segment,
        registration_date,
        total_orders,
        total_spend,
        first_order_date,
        last_order_date,
        customer_tenure_days,
        
        -- Calculate metrics
        ROUND(total_spend / NULLIF(customer_tenure_days, 0) * 365, 2) as annual_value,
        ROUND(total_spend / NULLIF(total_orders, 0), 2) as avg_order_value,
        ROUND(total_orders / (NULLIF(customer_tenure_days, 0) / 30.0), 2) as orders_per_month,
        
        -- Predict future value (simplified)
        CASE
            WHEN customer_segment = 'power_user' THEN ROUND(total_spend * 2.5, 2)
            WHEN customer_segment = 'loyal_regular' THEN ROUND(total_spend * 2.0, 2)
            WHEN customer_segment = 'occasional' THEN ROUND(total_spend * 1.5, 2)
            ELSE ROUND(total_spend * 1.2, 2)
        END as predicted_ltv_24m,
        
        -- Activity status
        CASE
            WHEN DATEDIFF(CURRENT_DATE(), last_order_date) <= 30 THEN 'active'
            WHEN DATEDIFF(CURRENT_DATE(), last_order_date) <= 90 THEN 'at_risk'
            WHEN DATEDIFF(CURRENT_DATE(), last_order_date) <= 180 THEN 'dormant'
            ELSE 'churned'
        END as activity_status
        
    FROM customer_history
""")

print(f"✓ Created supporting views for Genie access")

# COMMAND ----------

# MAGIC %md ## Step 5: Data Quality Validation

# COMMAND ----------

# Run quality checks on aggregations
quality_checks = []

# Check for data completeness
completeness_check = spark.sql(f"""
    SELECT 
        'store_performance_monthly' as table_name,
        COUNT(*) as row_count,
        COUNT(DISTINCT store_id) as unique_stores,
        COUNT(DISTINCT year_month) as unique_months,
        MIN(year_month) as min_month,
        MAX(year_month) as max_month
    FROM {CATALOG}.gold.store_performance_monthly
    
    UNION ALL
    
    SELECT 
        'customer_segments_monthly' as table_name,
        COUNT(*) as row_count,
        COUNT(DISTINCT segment_name) as unique_segments,
        COUNT(DISTINCT year_month) as unique_months,
        MIN(year_month) as min_month,
        MAX(year_month) as max_month
    FROM {CATALOG}.gold.customer_segments_monthly
""")

display(completeness_check)

# Check for anomalies
anomaly_check = spark.sql(f"""
    SELECT
        'negative_revenue' as check_name,
        COUNT(*) as anomaly_count
    FROM {CATALOG}.gold.store_performance_monthly
    WHERE total_revenue < 0
    
    UNION ALL
    
    SELECT
        'excessive_growth' as check_name,
        COUNT(*) as anomaly_count
    FROM {CATALOG}.gold.store_performance_monthly
    WHERE ABS(revenue_growth_pct) > 100
    
    UNION ALL
    
    SELECT
        'missing_segments' as check_name,
        4 - COUNT(DISTINCT segment_name) as anomaly_count
    FROM {CATALOG}.gold.customer_segments_monthly
""")

display(anomaly_check)

# COMMAND ----------

# MAGIC %md ## Step 6: Optimize Tables & Generate Statistics

# COMMAND ----------

# Optimize aggregation tables for query performance
tables_to_optimize = [
    'store_performance_monthly',
    'customer_segments_monthly'
]

for table in tables_to_optimize:
    # Optimize with Z-ORDER on common filter columns
    if table == 'store_performance_monthly':
        spark.sql(f"""
            OPTIMIZE {CATALOG}.gold.{table}
            ZORDER BY (store_id, year_month)
        """)
    else:
        spark.sql(f"""
            OPTIMIZE {CATALOG}.gold.{table}
            ZORDER BY (segment_name, year_month)
        """)
    
    # Compute statistics for better query planning
    spark.sql(f"ANALYZE TABLE {CATALOG}.gold.{table} COMPUTE STATISTICS")
    
    print(f"✓ Optimized and analyzed {table}")

# COMMAND ----------

# Summary report
summary = f"""
## Aggregation Processing Complete

**Processing Mode**: {PROCESSING_MODE}
**Date Range**: {START_DATE} to {END_DATE}
**Catalog**: {CATALOG}

### Tables Updated:
- `{CATALOG}.gold.store_performance_monthly`
- `{CATALOG}.gold.customer_segments_monthly`

### Views Created:
- `{CATALOG}.gold.v_store_performance_enriched`
- `{CATALOG}.gold.v_customer_lifetime_value`

### Next Steps:
1. Configure Genie Spaces to use these aggregation tables
2. Set up scheduled job to run this notebook daily/weekly
3. Create dashboards using DBSQL against the views
4. Build ML models using the aggregated features
"""

print(summary)

# Return success status for workflow orchestration
dbutils.notebook.exit("SUCCESS")
