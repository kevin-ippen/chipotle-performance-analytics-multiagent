# Databricks notebook source
# MAGIC %md
# MAGIC # Chipotle Analytics - Configuration & Setup
# MAGIC
# MAGIC **Purpose**: Initialize Unity Catalog structure, create schemas, and define base table DDL
# MAGIC
# MAGIC **Outputs**: 
# MAGIC - Catalog: `chipotle_analytics`
# MAGIC - Schemas: `bronze`, `silver`, `gold`, `ml_models`
# MAGIC - Base table structures in gold layer
# MAGIC
# MAGIC **Assumptions**: 
# MAGIC - User has CREATE CATALOG privileges
# MAGIC - Unity Catalog enabled workspace
# MAGIC
# MAGIC **Parameters**: 
# MAGIC - catalog_name: Target catalog (default: chipotle_analytics)
# MAGIC - reset_catalog: Drop and recreate if True (default: False)

# COMMAND ----------

# MAGIC %pip install -q pydantic

# COMMAND ----------

from pydantic import BaseModel
from typing import Optional
import json

# Widget parameters
dbutils.widgets.text("catalog_name", "chipotle_analytics", "Catalog Name")
dbutils.widgets.dropdown("reset_catalog", "false", ["true", "false"], "Reset Catalog")
dbutils.widgets.text("storage_location", "", "Storage Location (optional)")

# Get parameters
CATALOG = dbutils.widgets.get("catalog_name")
RESET_CATALOG = dbutils.widgets.get("reset_catalog").lower() == "true"
STORAGE_LOCATION = dbutils.widgets.get("storage_location")

# COMMAND ----------

# MAGIC %md ## Step 1: Catalog and Schema Setup

# COMMAND ----------

# Reset catalog if requested
if RESET_CATALOG:
    spark.sql(f"DROP CATALOG IF EXISTS {CATALOG} CASCADE")
    print(f"Dropped catalog {CATALOG}")

# Create catalog with optional managed location
storage_clause = f"MANAGED LOCATION '{STORAGE_LOCATION}'" if STORAGE_LOCATION else ""
spark.sql(f"""
    CREATE CATALOG IF NOT EXISTS {CATALOG}
    {storage_clause}
    COMMENT 'Chipotle sales performance analytics data'
""")

# Create schemas
schemas = {
    "bronze": "Raw ingestion layer",
    "silver": "Cleaned and validated data",
    "gold": "Business-ready aggregated data",
    "ml_models": "Trained ML models and functions"
}

for schema, comment in schemas.items():
    spark.sql(f"""
        CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}
        COMMENT '{comment}'
    """)
    print(f"Created schema: {CATALOG}.{schema}")

# Set default catalog
spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md ## Step 2: Gold Layer - Core Tables DDL

# COMMAND ----------

# COMMAND ----------
# MAGIC %md ## Step 2: Gold Layer - Core Tables DDL

# COMMAND ----------

# Drop existing tables to ensure a clean slate and prevent schema mismatch errors from previous runs
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.transactions")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.customer_profiles")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.store_locations")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.daily_store_performance")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.menu_items")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.campaigns")

# Transactions table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.transactions (
        order_id STRING NOT NULL,
        store_id STRING NOT NULL,
        customer_id STRING,
        order_timestamp TIMESTAMP NOT NULL,
        order_date DATE NOT NULL,
        channel STRING NOT NULL,
        order_type STRING NOT NULL,
        total_amount DOUBLE NOT NULL,  -- Changed from DECIMAL
        tax_amount DOUBLE,             -- Changed from DECIMAL
        tip_amount DOUBLE,             -- Changed from DECIMAL
        discount_amount DOUBLE,        -- Changed from DECIMAL
        payment_method STRING,
        promotion_codes ARRAY<STRING>,
        delivery_partner STRING,
        order_prep_time_minutes INT,
        customer_wait_time_minutes INT,
        order_items ARRAY<STRUCT<
            item_id: STRING,
            item_name: STRING,
            category: STRING,
            base_price: DOUBLE,        -- Changed from DECIMAL
            modifications: ARRAY<STRING>,
            quantity: INT
        >>
    )
    USING DELTA
    PARTITIONED BY (order_date)
    CLUSTER BY (store_id, customer_id)
    COMMENT 'Core transaction data for all Chipotle orders'
    TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true'
    )
""")

# Customer profiles table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.customer_profiles (
        customer_id STRING NOT NULL,
        registration_date DATE,
        registration_channel STRING,
        age_range STRING,
        income_bracket STRING,
        household_size INT,
        zip_code STRING,
        lifestyle STRING,
        loyalty_tier STRING,
        points_balance INT,
        lifetime_spend DOUBLE,      -- Changed from DECIMAL
        visit_frequency STRING,
        avg_order_value DOUBLE,     -- Changed from DECIMAL
        preferred_proteins ARRAY<STRING>,
        dietary_preferences ARRAY<STRING>,
        app_user BOOLEAN,
        email_subscriber BOOLEAN,
        push_notifications BOOLEAN,
        social_media_follower BOOLEAN,
        referrals_made INT,
        churn_risk_score DOUBLE,    -- Changed from FLOAT
        customer_segment STRING
    )
    USING DELTA
    CLUSTER BY (customer_segment, zip_code)
    COMMENT 'Customer profiles with demographics and behavior'
""")

# Store locations table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.store_locations (
        store_id STRING NOT NULL,
        store_number INT,
        address STRING,
        city STRING,
        state STRING,
        zip_code STRING,
        latitude DOUBLE,
        longitude DOUBLE,
        trade_area_type STRING,
        drive_thru BOOLEAN,
        parking_spaces INT,
        population_3mi INT,
        median_income_3mi INT,
        college_students_pct DOUBLE,          -- Changed from FLOAT
        working_professionals_pct DOUBLE,     -- Changed from FLOAT
        families_pct DOUBLE,                  -- Changed from FLOAT
        open_date DATE,
        store_format STRING,
        seating_capacity INT,
        kitchen_capacity_score INT,
        staff_count_avg INT,
        manager_tenure_months INT,
        fast_casual_competitors_1mi INT,
        direct_competitors_3mi INT,
        restaurant_density_1mi INT,
        active_flag BOOLEAN
    )
    USING DELTA
    COMMENT 'Store locations with characteristics and demographics'
""")

# Daily store performance
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.daily_store_performance (
        store_id STRING NOT NULL,
        business_date DATE NOT NULL,
        total_revenue DOUBLE,         -- Changed from DECIMAL
        transaction_count INT,
        average_ticket DOUBLE,        -- Changed from DECIMAL
        revenue_per_seat DOUBLE,      -- Changed from DECIMAL
        in_store_revenue DOUBLE,      -- Changed from DECIMAL
        digital_revenue DOUBLE,       -- Changed from DECIMAL
        catering_revenue DOUBLE,      -- Changed from DECIMAL
        delivery_revenue DOUBLE,      -- Changed from DECIMAL
        avg_service_time DOUBLE,      -- Changed from FLOAT
        peak_hour_throughput INT,
        staff_hours_scheduled DOUBLE, -- Changed from FLOAT
        staff_hours_actual DOUBLE,    -- Changed from FLOAT
        food_cost_pct DOUBLE,         -- Changed from FLOAT
        waste_amount DOUBLE,          -- Changed from DECIMAL
        new_customers INT,
        returning_customers INT,
        loyalty_redemptions INT,
        avg_satisfaction_score DOUBLE,  -- Changed from FLOAT
        weather_condition STRING,
        temperature_high INT,
        precipitation_inches DOUBLE,  -- Changed from FLOAT
        local_events ARRAY<STRING>
    )
    USING DELTA
    PARTITIONED BY (business_date)
    CLUSTER BY (store_id)
    COMMENT 'Daily store performance metrics'
""")

# Menu items table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.menu_items (
        item_id STRING NOT NULL,
        item_name STRING NOT NULL,
        category STRING NOT NULL,
        subcategory STRING,
        base_price DOUBLE NOT NULL,   -- Changed from DECIMAL
        cost_of_goods DOUBLE,         -- Changed from DECIMAL
        margin_pct DOUBLE,            -- Changed from FLOAT
        calories INT,
        protein_g DOUBLE,             -- Changed from FLOAT
        carbs_g DOUBLE,               -- Changed from FLOAT
        fat_g DOUBLE,                 -- Changed from FLOAT
        sodium_mg INT,
        allergens ARRAY<STRING>,
        dietary_flags ARRAY<STRING>,
        active_flag BOOLEAN,
        last_updated TIMESTAMP
    )
    USING DELTA
    COMMENT 'Menu items with pricing and nutritional info'
""")

# Campaigns table  
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.campaigns (
        campaign_id STRING NOT NULL,
        campaign_name STRING,
        campaign_type STRING,
        promotion_type STRING,
        discount_amount DOUBLE,       -- Changed from DECIMAL
        minimum_purchase DOUBLE,      -- Changed from DECIMAL
        eligible_items ARRAY<STRING>,
        promo_code STRING,
        target_segments ARRAY<STRING>,
        geographic_scope ARRAY<STRING>,
        channel_restrictions ARRAY<STRING>,
        start_date DATE,
        end_date DATE,
        announcement_date DATE,
        total_budget DOUBLE,          -- Changed from DECIMAL
        media_spend DOUBLE,           -- Changed from DECIMAL
        discount_budget DOUBLE,       -- Changed from DECIMAL
        target_redemptions INT,
        target_roi DOUBLE,            -- Changed from FLOAT
        target_new_customers INT
    )
    USING DELTA
    COMMENT 'Marketing campaigns and promotions'
""")

# COMMAND ----------

# COMMAND ----------
# MAGIC %md ## Step 2: Gold Layer - Core Tables DDL

# COMMAND ----------

# Drop existing tables to ensure a clean slate and prevent schema mismatch errors from previous runs
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.transactions")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.customer_profiles")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.store_locations")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.daily_store_performance")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.menu_items")
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.campaigns")

# Transactions table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.transactions (
        order_id STRING NOT NULL,
        store_id STRING NOT NULL,
        customer_id STRING,
        order_timestamp TIMESTAMP NOT NULL,
        order_date DATE NOT NULL,
        channel STRING NOT NULL,
        order_type STRING NOT NULL,
        total_amount DOUBLE NOT NULL,
        tax_amount DOUBLE,
        tip_amount DOUBLE,
        discount_amount DOUBLE,
        payment_method STRING,
        promotion_codes ARRAY<STRING>,
        delivery_partner STRING,
        order_prep_time_minutes INT,
        customer_wait_time_minutes INT,
        order_items ARRAY<STRUCT<
            item_id: STRING,
            item_name: STRING,
            category: STRING,
            base_price: DOUBLE,
            modifications: ARRAY<STRING>,
            quantity: INT
        >>
    )
    USING DELTA
    CLUSTER BY (order_date, store_id, customer_id)
    COMMENT 'Core transaction data for all Chipotle orders'
    TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true'
    )
""")

# Customer profiles table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.customer_profiles (
        customer_id STRING NOT NULL,
        registration_date DATE,
        registration_channel STRING,
        age_range STRING,
        income_bracket STRING,
        household_size INT,
        zip_code STRING,
        lifestyle STRING,
        loyalty_tier STRING,
        points_balance INT,
        lifetime_spend DOUBLE,
        visit_frequency STRING,
        avg_order_value DOUBLE,
        preferred_proteins ARRAY<STRING>,
        dietary_preferences ARRAY<STRING>,
        app_user BOOLEAN,
        email_subscriber BOOLEAN,
        push_notifications BOOLEAN,
        social_media_follower BOOLEAN,
        referrals_made INT,
        churn_risk_score DOUBLE,
        customer_segment STRING
    )
    USING DELTA
    CLUSTER BY (customer_segment, zip_code)
    COMMENT 'Customer profiles with demographics and behavior'
""")

# Store locations table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.store_locations (
        store_id STRING NOT NULL,
        store_number INT,
        address STRING,
        city STRING,
        state STRING,
        zip_code STRING,
        latitude DOUBLE,
        longitude DOUBLE,
        trade_area_type STRING,
        drive_thru BOOLEAN,
        parking_spaces INT,
        population_3mi INT,
        median_income_3mi INT,
        college_students_pct DOUBLE,
        working_professionals_pct DOUBLE,
        families_pct DOUBLE,
        open_date DATE,
        store_format STRING,
        seating_capacity INT,
        kitchen_capacity_score INT,
        staff_count_avg INT,
        manager_tenure_months INT,
        fast_casual_competitors_1mi INT,
        direct_competitors_3mi INT,
        restaurant_density_1mi INT,
        active_flag BOOLEAN
    )
    USING DELTA
    COMMENT 'Store locations with characteristics and demographics'
""")

# Daily store performance
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.daily_store_performance (
        store_id STRING NOT NULL,
        business_date DATE NOT NULL,
        total_revenue DOUBLE,
        transaction_count INT,
        average_ticket DOUBLE,
        revenue_per_seat DOUBLE,
        in_store_revenue DOUBLE,
        digital_revenue DOUBLE,
        catering_revenue DOUBLE,
        delivery_revenue DOUBLE,
        avg_service_time DOUBLE,
        peak_hour_throughput INT,
        staff_hours_scheduled DOUBLE,
        staff_hours_actual DOUBLE,
        food_cost_pct DOUBLE,
        waste_amount DOUBLE,
        new_customers INT,
        returning_customers INT,
        loyalty_redemptions INT,
        avg_satisfaction_score DOUBLE,
        weather_condition STRING,
        temperature_high INT,
        precipitation_inches DOUBLE,
        local_events ARRAY<STRING>
    )
    USING DELTA
    CLUSTER BY (business_date, store_id)
    COMMENT 'Daily store performance metrics'
""")

# Menu items table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.menu_items (
        item_id STRING NOT NULL,
        item_name STRING NOT NULL,
        category STRING NOT NULL,
        subcategory STRING,
        base_price DOUBLE NOT NULL,
        cost_of_goods DOUBLE,
        margin_pct DOUBLE,
        calories INT,
        protein_g DOUBLE,
        carbs_g DOUBLE,
        fat_g DOUBLE,
        sodium_mg INT,
        allergens ARRAY<STRING>,
        dietary_flags ARRAY<STRING>,
        active_flag BOOLEAN,
        last_updated TIMESTAMP
    )
    USING DELTA
    COMMENT 'Menu items with pricing and nutritional info'
""")

# Campaigns table  
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.campaigns (
        campaign_id STRING NOT NULL,
        campaign_name STRING,
        campaign_type STRING,
        promotion_type STRING,
        discount_amount DOUBLE,
        minimum_purchase DOUBLE,
        eligible_items ARRAY<STRING>,
        promo_code STRING,
        target_segments ARRAY<STRING>,
        geographic_scope ARRAY<STRING>,
        channel_restrictions ARRAY<STRING>,
        start_date DATE,
        end_date DATE,
        announcement_date DATE,
        total_budget DOUBLE,
        media_spend DOUBLE,
        discount_budget DOUBLE,
        target_redemptions INT,
        target_roi DOUBLE,
        target_new_customers INT
    )
    USING DELTA
    COMMENT 'Marketing campaigns and promotions'
""")

# COMMAND ----------

# MAGIC %md ## Step 3: Aggregation Tables

# COMMAND ----------

# COMMAND ----------
# MAGIC %md ## Step 3: Aggregation Tables

# COMMAND ----------

# Monthly store performance aggregation
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.store_performance_monthly (
        store_id STRING NOT NULL,
        year_month STRING NOT NULL,
        total_revenue DOUBLE,
        revenue_growth_pct DOUBLE,
        revenue_vs_budget_pct DOUBLE,
        market_share_est_pct DOUBLE,
        unique_customers INT,
        new_customers INT,
        customer_retention_rate DOUBLE,
        avg_visits_per_customer DOUBLE,
        nps_score DOUBLE,
        avg_ticket DOUBLE,
        transactions_per_day DOUBLE,
        digital_mix_pct DOUBLE,
        food_cost_pct DOUBLE,
        labor_cost_pct DOUBLE,
        vs_region_avg_pct DOUBLE,
        vs_similar_stores_pct DOUBLE,
        vs_national_avg_pct DOUBLE,
        percentile_ranking INT,
        calculated_timestamp TIMESTAMP
    )
    USING DELTA
    CLUSTER BY (year_month, store_id)
    COMMENT 'Pre-aggregated monthly store performance'
""")

# Customer segments monthly aggregation
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.customer_segments_monthly (
        segment_name STRING NOT NULL,
        year_month STRING NOT NULL,
        segment_size INT,
        avg_monthly_visits DOUBLE,
        avg_order_value DOUBLE,
        lifetime_value DOUBLE,
        churn_rate_pct DOUBLE,
        preferred_dayparts ARRAY<STRING>,
        channel_mix MAP<STRING, DOUBLE>,
        protein_preferences MAP<STRING, DOUBLE>,
        promotion_response_rate DOUBLE,
        segment_growth_pct DOUBLE,
        revenue_contribution_pct DOUBLE,
        acquisition_cost DOUBLE
    )
    USING DELTA
    CLUSTER BY (year_month)
    COMMENT 'Pre-aggregated customer segment analysis'
""")

# COMMAND ----------

# Monthly store performance aggregation
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.store_performance_monthly (
        store_id STRING NOT NULL,
        year_month STRING NOT NULL,
        total_revenue DOUBLE,
        revenue_growth_pct DOUBLE,
        revenue_vs_budget_pct DOUBLE,
        market_share_est_pct DOUBLE,
        unique_customers INT,
        new_customers INT,
        customer_retention_rate DOUBLE,
        avg_visits_per_customer DOUBLE,
        nps_score DOUBLE,
        avg_ticket DOUBLE,
        transactions_per_day DOUBLE,
        digital_mix_pct DOUBLE,
        food_cost_pct DOUBLE,
        labor_cost_pct DOUBLE,
        vs_region_avg_pct DOUBLE,
        vs_similar_stores_pct DOUBLE,
        vs_national_avg_pct DOUBLE,
        percentile_ranking INT,
        calculated_timestamp TIMESTAMP
    )
    USING DELTA
    CLUSTER BY (year_month, store_id)
    COMMENT 'Pre-aggregated monthly store performance'
""")

# Customer segments monthly aggregation
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.customer_segments_monthly (
        segment_name STRING NOT NULL,
        year_month STRING NOT NULL,
        segment_size INT,
        avg_monthly_visits DOUBLE,
        avg_order_value DOUBLE,
        lifetime_value DOUBLE,
        churn_rate_pct DOUBLE,
        preferred_dayparts ARRAY<STRING>,
        channel_mix MAP<STRING, DOUBLE>,
        protein_preferences MAP<STRING, DOUBLE>,
        promotion_response_rate DOUBLE,
        segment_growth_pct DOUBLE,
        revenue_contribution_pct DOUBLE,
        acquisition_cost DOUBLE
    )
    USING DELTA
    CLUSTER BY (year_month)
    COMMENT 'Pre-aggregated customer segment analysis'
""")

# COMMAND ----------

# MAGIC %md ## Step 4: Verify Setup

# COMMAND ----------

# List all tables created
display(spark.sql(f"""
    SELECT 
        table_catalog,
        table_schema,
        table_name,
        table_type,
        comment
    FROM {CATALOG}.information_schema.tables
    WHERE table_catalog = '{CATALOG}'
    ORDER BY table_schema, table_name
"""))

# COMMAND ----------

# Configuration summary
config_summary = {
    "catalog": CATALOG,
    "schemas_created": list(schemas.keys()),
    "core_tables": [
        "transactions",
        "customer_profiles", 
        "store_locations",
        "daily_store_performance",
        "menu_items",
        "campaigns"
    ],
    "aggregation_tables": [
        "store_performance_monthly",
        "customer_segments_monthly"
    ],
    "reset_performed": RESET_CATALOG
}

print(json.dumps(config_summary, indent=2))

# COMMAND ----------

# Grant permissions template (uncomment and modify as needed)
"""
# Grant usage on catalog
spark.sql(f"GRANT USAGE ON CATALOG {CATALOG} TO `data_analysts`")

# Grant schema permissions
spark.sql(f"GRANT USAGE ON SCHEMA {CATALOG}.gold TO `data_analysts`")
spark.sql(f"GRANT SELECT ON SCHEMA {CATALOG}.gold TO `data_analysts`")

# Grant table permissions for specific roles
spark.sql(f"GRANT SELECT ON TABLE {CATALOG}.gold.transactions TO `sales_team`")
"""
print(f"✓ Configuration complete for {CATALOG}")
