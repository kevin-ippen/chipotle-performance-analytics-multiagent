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
    max_trans_date_query = spark.sql(f"SELECT MAX(order_date) as max_date FROM {CATALOG}.gold.transactions")
    max_trans_date = max_trans_date_query.collect()[0]['max_date']
    
    if PROCESSING_MODE == "full_refresh":
        # Process all available data
        min_trans_date_query = spark.sql(f"SELECT MIN(order_date) as min_date FROM {CATALOG}.gold.transactions")
        start_date = min_trans_date_query.collect()[0]['min_date']
        
        print(f"Full refresh mode - processing from {start_date} to {max_trans_date}")
        return start_date, max_trans_date
    
    else:  # incremental
        # Check what's already been processed
        try:
            last_processed_query = spark.sql(f"SELECT MAX(year_month) as last_month FROM {CATALOG}.gold.store_performance_monthly")
            last_processed = last_processed_query.collect()[0]['last_month']
            
            if last_processed:
                # Start from next month after last processed
                year, month = map(int, last_processed.split('-'))
                start_date = (datetime(year, month, 1) + relativedelta(months=1)).date()
            else:
                # No data yet, start from beginning
                min_trans_date_query = spark.sql(f"SELECT MIN(order_date) as min_date FROM {CATALOG}.gold.transactions")
                start_date = min_trans_date_query.collect()[0]['min_date']
        except Exception:
            # Table doesn't exist, start from beginning
            min_trans_date_query = spark.sql(f"SELECT MIN(order_date) as min_date FROM {CATALOG}.gold.transactions")
            start_date = min_trans_date_query.collect()[0]['min_date']
        
        # But don't go back more than MONTHS_LOOKBACK
        lookback_date = (datetime.now() - relativedelta(months=MONTHS_LOOKBACK)).date()
        if start_date < lookback_date:
            start_date = lookback_date.replace(day=1)
        
        print(f"Incremental mode - processing from {start_date} to {max_trans_date}")
        return start_date, max_trans_date

START_DATE, END_DATE = get_processing_dates()

# COMMAND ----------

def upsert_delta(df, target_table, key_cols, cast_to_target=True):
    # Create if missing
    if not spark.catalog.tableExists(target_table):
        df.write.mode("overwrite").saveAsTable(target_table)
        return

    # (Optional) cast source to match target table schema exactly
    if cast_to_target:
        tgt = spark.table(target_table)
        tgt_types = dict(tgt.dtypes)          # {'col': 'decimal(18,2)', 'year_month':'string', ...}
        select_exprs = []
        for c in tgt.columns:                  # preserve target column order
            if c in df.columns:
                select_exprs.append(f"CAST(`{c}` AS {tgt_types[c]}) AS `{c}`")
        df = df.selectExpr(*select_exprs)

    # MERGE with explicit SET (works across Delta versions)
    temp_view = "__upsert_src__"
    df.createOrReplaceTempView(temp_view)
    set_clause = ", ".join([f"t.`{c}` = s.`{c}`" for c in spark.table(target_table).columns])
    on_clause  = " AND ".join([f"t.`{k}` = s.`{k}`" for k in key_cols])
    spark.sql(f"""
        MERGE INTO {target_table} t
        USING {temp_view} s
        ON {on_clause}
        WHEN MATCHED THEN UPDATE SET {set_clause}
        WHEN NOT MATCHED THEN INSERT *
    """)


# COMMAND ----------

# Quick sanity: required sources exist
req = [
    f"{CATALOG}.gold.transactions",
    f"{CATALOG}.gold.daily_store_performance",
    f"{CATALOG}.gold.customer_profiles",
]
for t in req:
    assert spark.catalog.tableExists(t), f"Missing required table: {t}"

HAS_STORE_LOCATIONS = spark.catalog.tableExists(f"{CATALOG}.gold.store_locations")
print("HAS_STORE_LOCATIONS =", HAS_STORE_LOCATIONS)


# COMMAND ----------

# MAGIC %md ## Step 2: Monthly Store Performance Aggregation

# COMMAND ----------

# STEP 2 — Store monthly aggregates (robust WITH/CTE assembly + optional regional joins)

# Preconditions (light sanity)
assert spark.catalog.tableExists(f"{CATALOG}.gold.transactions"), "Missing table: transactions"
assert spark.catalog.tableExists(f"{CATALOG}.gold.daily_store_performance"), "Missing table: daily_store_performance"

HAS_STORE_LOCATIONS = spark.catalog.tableExists(f"{CATALOG}.gold.store_locations")

# ---- Build CTEs safely as a list, then join with commas
ctes = []

ctes.append(f"""
customer_first_month AS (
  SELECT customer_id, date_format(MIN(order_date),'yyyy-MM') AS first_month
  FROM {CATALOG}.gold.transactions
  GROUP BY customer_id
)
""")

ctes.append(f"""
monthly_revenue AS (
  SELECT 
      t.store_id,
      date_format(t.order_date, 'yyyy-MM') AS year_month,
      COUNT(*) AS transaction_count,
      COUNT(DISTINCT t.customer_id) AS unique_customers,
      CAST(SUM(t.total_amount) AS DECIMAL(18,2)) AS total_revenue,
      CAST(AVG(t.total_amount) AS DECIMAL(18,2)) AS avg_ticket,
      CASE WHEN SUM(t.total_amount) IS NULL OR SUM(t.total_amount)=0
           THEN 0.0
           ELSE SUM(CASE WHEN t.channel IN ('app','web') THEN t.total_amount ELSE 0 END) / SUM(t.total_amount)
      END AS digital_mix_frac,
      SUM(CASE WHEN cf.first_month = date_format(t.order_date,'yyyy-MM') THEN 1 ELSE 0 END) AS new_customers,
      SUM(t.discount_amount) AS total_discounts
  FROM {CATALOG}.gold.transactions t
  LEFT JOIN customer_first_month cf ON t.customer_id = cf.customer_id
  WHERE t.order_date >= '{START_DATE}' AND t.order_date <= '{END_DATE}'
  GROUP BY t.store_id, date_format(t.order_date, 'yyyy-MM')
)
""")

ctes.append(f"""
monthly_operations AS (
  SELECT
      store_id,
      date_format(business_date, 'yyyy-MM') AS year_month,
      AVG(food_cost_pct) AS food_cost_pct,
      AVG(avg_satisfaction_score) AS nps_score,
      AVG(CASE WHEN staff_hours_scheduled IS NULL OR staff_hours_scheduled = 0
               THEN NULL
               ELSE staff_hours_actual / staff_hours_scheduled END) AS labor_efficiency,
      SUM(waste_amount) AS total_waste
  FROM {CATALOG}.gold.daily_store_performance
  WHERE business_date >= '{START_DATE}' AND business_date <= '{END_DATE}'
  GROUP BY store_id, date_format(business_date, 'yyyy-MM')
)
""")

ctes.append("""
prior_month AS (
  SELECT
      store_id,
      year_month,
      LAG(total_revenue)  OVER (PARTITION BY store_id ORDER BY year_month) AS prev_revenue,
      LAG(unique_customers) OVER (PARTITION BY store_id ORDER BY year_month) AS prev_customers
  FROM monthly_revenue
)
""")

if HAS_STORE_LOCATIONS:
    ctes.append(f"""
regional_benchmarks AS (
  SELECT 
      s.state,
      r.year_month,
      AVG(r.total_revenue) AS region_avg_revenue,
      percentile_approx(r.total_revenue, 0.5) AS region_median_revenue
  FROM monthly_revenue r
  JOIN {CATALOG}.gold.store_locations s ON r.store_id = s.store_id
  GROUP BY s.state, r.year_month
)
""")

ctes.append("""
national_benchmarks AS (
  SELECT
      year_month,
      AVG(total_revenue) AS national_avg_revenue,
      percentile_approx(total_revenue, 0.5) AS national_median_revenue
  FROM monthly_revenue
  GROUP BY year_month
)
""")

with_clause = "WITH\n" + ",\n".join(cte.strip() for cte in ctes)

# ---- Build optional pieces first (to avoid backslashes in f-string expressions)
region_select_expr = (
    "ROUND((r.total_revenue - rb.region_avg_revenue) / rb.region_avg_revenue * 100, 2) AS vs_region_avg_pct,"
    if HAS_STORE_LOCATIONS
    else "CAST(NULL AS DOUBLE) AS vs_region_avg_pct,"
)

region_join_clause = (
    f"LEFT JOIN {CATALOG}.gold.store_locations s ON r.store_id = s.store_id\n"
    "LEFT JOIN regional_benchmarks rb ON s.state = rb.state AND r.year_month = rb.year_month\n"
    if HAS_STORE_LOCATIONS
    else ""
)

# ---- Main SELECT
select_clause = f"""
SELECT
    r.store_id,
    r.year_month,

    -- Revenue metrics
    r.total_revenue,
    CASE WHEN p.prev_revenue IS NULL OR p.prev_revenue = 0 THEN NULL
         ELSE ROUND((r.total_revenue - p.prev_revenue) / p.prev_revenue * 100, 2) END AS revenue_growth_pct,
    ROUND( r.total_revenue / (datediff(last_day(concat(r.year_month,'-01')), concat(r.year_month,'-01')) + 1) / 150 * 100, 2) AS revenue_vs_budget_pct,
    0.0 AS market_share_est_pct,

    -- Customer metrics
    r.unique_customers,
    r.new_customers,
    CASE WHEN p.prev_customers IS NULL OR p.prev_customers = 0 THEN NULL
         ELSE ROUND( (r.unique_customers - r.new_customers) / p.prev_customers * 100, 2) END AS customer_retention_rate,
    CASE WHEN r.unique_customers IS NULL OR r.unique_customers = 0 THEN NULL
         ELSE ROUND( r.transaction_count / r.unique_customers, 2) END AS avg_visits_per_customer,
    o.nps_score,

    -- Operational metrics
    r.avg_ticket,
    ROUND( r.transaction_count / (datediff(last_day(concat(r.year_month,'-01')), concat(r.year_month,'-01')) + 1), 2) AS transactions_per_day,
    ROUND( r.digital_mix_frac * 100, 2) AS digital_mix_pct,
    ROUND( o.food_cost_pct * 100, 2) AS food_cost_pct,
    ROUND( (1 - o.labor_efficiency) * 100, 2) AS labor_cost_pct,

    -- Benchmarks
    {region_select_expr}
    0.0 AS vs_similar_stores_pct,
    ROUND((r.total_revenue - nb.national_avg_revenue) / nb.national_avg_revenue * 100, 2) AS vs_national_avg_pct,
    CAST(percent_rank() OVER (PARTITION BY r.year_month ORDER BY r.total_revenue) * 100 AS INT) AS percentile_ranking,

    current_timestamp() AS calculated_timestamp

FROM monthly_revenue r
LEFT JOIN monthly_operations o ON r.store_id=o.store_id AND r.year_month=o.year_month
LEFT JOIN prior_month p ON r.store_id=p.store_id AND r.year_month=p.year_month
{region_join_clause}LEFT JOIN national_benchmarks nb ON r.year_month=nb.year_month
"""

store_monthly_sql = with_clause + "\n" + select_clause
# print(store_monthly_sql)  # <- uncomment to inspect

store_monthly_df = spark.sql(store_monthly_sql)

# ---- Write results
target_table = f"{CATALOG}.gold.store_performance_monthly"

# Get the schema of the target table if it exists
if spark.catalog.tableExists(target_table):
    target_schema = spark.table(target_table).schema
    # Cast DataFrame columns to match target schema
    for field in target_schema:
        if field.name in store_monthly_df.columns:
            store_monthly_df = store_monthly_df.withColumn(
                field.name,
                store_monthly_df[field.name].cast(field.dataType)
            )

if PROCESSING_MODE.lower() == "full_refresh":
    store_monthly_df.write.mode("overwrite").saveAsTable(target_table)
else:
    if not spark.catalog.tableExists(target_table):
        store_monthly_df.write.mode("overwrite").saveAsTable(target_table)
    else:
        store_monthly_df.createOrReplaceTempView("store_monthly_updates")
        cols = store_monthly_df.columns
        set_clause = ", ".join([f"t.`{c}` = s.`{c}`" for c in cols])
        merge_sql = f"""
        MERGE INTO {target_table} t
        USING store_monthly_updates s
        ON t.store_id = s.store_id AND t.year_month = s.year_month
        WHEN MATCHED THEN UPDATE SET {set_clause}
        WHEN NOT MATCHED THEN INSERT *
        """
        spark.sql(merge_sql)

print(f"✓ store_performance_monthly: {spark.table(target_table).count()} rows")

# COMMAND ----------

# MAGIC %md ## Step 3: Customer Segments Monthly Aggregation

# COMMAND ----------

# COMMAND ----------
# MAGIC %md ## Step 3: Customer Segments Monthly Aggregation

# COMMAND ----------

# Build customer segment monthly metrics
segments_monthly_df = spark.sql(f"""
    WITH customer_monthly_activity AS (
        SELECT
            c.customer_segment AS segment_name,
            date_format(t.order_date, 'yyyy-MM') AS year_month,
            c.customer_id,
            COUNT(*) AS visits,
            SUM(t.total_amount) AS spend,
            AVG(t.total_amount) AS avg_order_value,

            -- Channel preferences (force DOUBLE denom to avoid integer division)
            SUM(CASE WHEN t.channel = 'app' THEN 1 ELSE 0 END) / CAST(COUNT(*) AS DOUBLE) AS app_pct,
            SUM(CASE WHEN t.channel = 'web' THEN 1 ELSE 0 END) / CAST(COUNT(*) AS DOUBLE) AS web_pct,
            SUM(CASE WHEN t.channel = 'in_store' THEN 1 ELSE 0 END) / CAST(COUNT(*) AS DOUBLE) AS in_store_pct,

            -- Daypart preferences
            SUM(CASE WHEN HOUR(t.order_timestamp) BETWEEN 7 AND 10 THEN 1 ELSE 0 END) / CAST(COUNT(*) AS DOUBLE) AS breakfast_pct,
            SUM(CASE WHEN HOUR(t.order_timestamp) BETWEEN 11 AND 14 THEN 1 ELSE 0 END) / CAST(COUNT(*) AS DOUBLE) AS lunch_pct,
            SUM(CASE WHEN HOUR(t.order_timestamp) BETWEEN 17 AND 20 THEN 1 ELSE 0 END) / CAST(COUNT(*) AS DOUBLE) AS dinner_pct,

            -- Promotional response
            SUM(CASE WHEN size(t.promotion_codes) > 0 THEN 1 ELSE 0 END) / CAST(COUNT(*) AS DOUBLE) AS promo_usage_rate

        FROM {CATALOG}.gold.transactions t
        JOIN {CATALOG}.gold.customer_profiles c ON t.customer_id = c.customer_id
        WHERE t.order_date >= '{START_DATE}' AND t.order_date <= '{END_DATE}'
        GROUP BY c.customer_segment, date_format(t.order_date, 'yyyy-MM'), c.customer_id
    ),

    segment_aggregates AS (
        SELECT
            segment_name,
            year_month,
            CAST(COUNT(DISTINCT customer_id) AS INT) AS segment_size, -- Corrected type to INT
            AVG(visits) AS avg_monthly_visits,
            CAST(AVG(avg_order_value) AS DOUBLE) AS avg_order_value,  -- Corrected type to DOUBLE
            CAST(AVG(spend * 12)      AS DOUBLE) AS lifetime_value,  -- Corrected type to DOUBLE

            -- Channel mix as MAP<string,double>
            MAP(
                'app',      CAST(ROUND(AVG(app_pct), 3) AS DOUBLE),
                'web',      CAST(ROUND(AVG(web_pct), 3) AS DOUBLE),
                'in_store', CAST(ROUND(AVG(in_store_pct), 3) AS DOUBLE)
            ) AS channel_mix,

            -- Preferred dayparts
            CASE
                WHEN AVG(lunch_pct)  > 0.5 THEN array('lunch')
                WHEN AVG(dinner_pct) > 0.4 THEN array('dinner')
                ELSE array('lunch', 'dinner')
            END AS preferred_dayparts,

            AVG(promo_usage_rate) AS promotion_response_rate
        FROM customer_monthly_activity
        GROUP BY segment_name, year_month
    ),

    segment_growth AS (
        SELECT
            segment_name,
            year_month,
            segment_size,
            LAG(segment_size) OVER (PARTITION BY segment_name ORDER BY year_month) AS prev_size
        FROM segment_aggregates
    ),

    protein_preferences AS (
        -- Placeholder mix; replace with mined prefs from order_items if desired
        SELECT
            c.customer_segment AS segment_name,
            MAP('chicken',0.45,'steak',0.25,'carnitas',0.15,'barbacoa',0.10,'sofritas',0.05) AS protein_preferences
        FROM {CATALOG}.gold.customer_profiles c
        GROUP BY c.customer_segment
    ),

    month_totals AS (
        SELECT
            year_month,
            SUM(CAST(segment_size AS DOUBLE) * avg_monthly_visits * avg_order_value) AS month_total_value
        FROM segment_aggregates
        GROUP BY year_month
    )

    SELECT
        a.segment_name,
        a.year_month,
        a.segment_size,
        CAST(ROUND(a.avg_monthly_visits, 2) AS DOUBLE) AS avg_monthly_visits,
        a.avg_order_value,
        a.lifetime_value,

        -- Simplified churn assumption
        CASE 
            WHEN a.segment_name = 'power_user'    THEN 2.0
            WHEN a.segment_name = 'loyal_regular' THEN 5.0
            WHEN a.segment_name = 'occasional'    THEN 15.0
            ELSE 25.0
        END AS churn_rate_pct,

        a.preferred_dayparts,
        a.channel_mix,
        p.protein_preferences,
        CAST(ROUND(a.promotion_response_rate, 3) AS DOUBLE) AS promotion_response_rate,

        -- Growth vs prior month (safe divide)
        CAST(CASE WHEN g.prev_size IS NULL OR g.prev_size = 0 THEN NULL
             ELSE ROUND((CAST(a.segment_size AS DOUBLE) - CAST(g.prev_size AS DOUBLE)) / CAST(g.prev_size AS DOUBLE) * 100, 2)
        END AS DOUBLE) AS segment_growth_pct,

        -- Revenue contribution within month (safe divide via month_totals join)
        CAST(CASE WHEN mt.month_total_value IS NULL OR mt.month_total_value = 0 THEN NULL
             ELSE ROUND(CAST(a.segment_size AS DOUBLE) * a.avg_monthly_visits * a.avg_order_value / mt.month_total_value * 100, 2)
        END AS DOUBLE) AS revenue_contribution_pct,

        -- Simplified acquisition cost (kept as DOUBLE; cast to DECIMAL if you prefer currency)
        CAST(CASE 
            WHEN a.segment_name = 'power_user'    THEN 25.00
            WHEN a.segment_name = 'loyal_regular' THEN 18.00
            WHEN a.segment_name = 'occasional'    THEN 12.00
            ELSE 8.00
        END AS DOUBLE) AS acquisition_cost

    FROM segment_aggregates a
    LEFT JOIN segment_growth g      ON a.segment_name = g.segment_name AND a.year_month = g.year_month
    LEFT JOIN protein_preferences p ON a.segment_name = p.segment_name
    LEFT JOIN month_totals mt       ON a.year_month   = mt.year_month
""")

segments_monthly_df = spark.sql(segments_sql)

# ---- Write / Upsert with schema-stabilizing cast to target (avoids DELTA_FAILED_TO_MERGE_FIELDS)
target_table = f"{CATALOG}.gold.customer_segments_monthly"

if PROCESSING_MODE.lower() == "full_refresh":
    spark.sql(f"DROP TABLE IF EXISTS {target_table}")
    segments_monthly_df.write.mode("overwrite").saveAsTable(target_table)
else:
    if not spark.catalog.tableExists(target_table):
        segments_monthly_df.write.mode("overwrite").saveAsTable(target_table)
    else:
        segments_monthly_df.createOrReplaceTempView("segments_monthly_updates")
        
        # MERGE with safe UPDATE/INSERT
        spark.sql(f"""
            MERGE INTO {target_table} t
            USING segments_monthly_updates s
            ON t.segment_name = s.segment_name AND t.year_month = s.year_month
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)

print(f"✓ customer_segments_monthly: {spark.table(target_table).count()} rows")

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
