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
        total_amount DECIMAL(8,2) NOT NULL,
        tax_amount DECIMAL(6,2),
        tip_amount DECIMAL(6,2),
        discount_amount DECIMAL(6,2),
        payment_method STRING,
        promotion_codes ARRAY<STRING>,
        delivery_partner STRING,
        order_prep_time_minutes INT,
        customer_wait_time_minutes INT,
        order_items ARRAY<STRUCT<
            item_id: STRING,
            item_name: STRING,
            category: STRING,
            base_price: DECIMAL(5,2),
            modifications: ARRAY<STRING>,
            quantity: INT
        >>,
        CONSTRAINT pk_order PRIMARY KEY (order_id),
        CONSTRAINT valid_channel CHECK (channel IN ('in_store', 'app', 'web', 'catering')),
        CONSTRAINT valid_order_type CHECK (order_type IN ('pickup', 'delivery', 'dine_in')),
        CONSTRAINT positive_amount CHECK (total_amount > 0)
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
        lifetime_spend DECIMAL(10,2),
        visit_frequency STRING,
        avg_order_value DECIMAL(6,2),
        preferred_proteins ARRAY<STRING>,
        dietary_preferences ARRAY<STRING>,
        app_user BOOLEAN,
        email_subscriber BOOLEAN,
        push_notifications BOOLEAN,
        social_media_follower BOOLEAN,
        referrals_made INT,
        churn_risk_score FLOAT,
        customer_segment STRING,
        CONSTRAINT pk_customer PRIMARY KEY (customer_id),
        CONSTRAINT valid_segment CHECK (customer_segment IN ('power_user', 'loyal_regular', 'price_sensitive', 'occasional', 'at_risk')),
        CONSTRAINT valid_loyalty CHECK (loyalty_tier IN ('bronze', 'silver', 'gold', 'platinum'))
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
        college_students_pct FLOAT,
        working_professionals_pct FLOAT,
        families_pct FLOAT,
        open_date DATE,
        store_format STRING,
        seating_capacity INT,
        kitchen_capacity_score INT,
        staff_count_avg INT,
        manager_tenure_months INT,
        fast_casual_competitors_1mi INT,
        direct_competitors_3mi INT,
        restaurant_density_1mi INT,
        active_flag BOOLEAN,
        CONSTRAINT pk_store PRIMARY KEY (store_id),
        CONSTRAINT valid_format CHECK (store_format IN ('standard', 'drive_thru', 'urban_compact')),
        CONSTRAINT valid_trade_area CHECK (trade_area_type IN ('urban', 'suburban', 'university', 'mall'))
    )
    USING DELTA
    COMMENT 'Store locations with characteristics and demographics'
""")

# Daily store performance
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.daily_store_performance (
        store_id STRING NOT NULL,
        business_date DATE NOT NULL,
        total_revenue DECIMAL(10,2),
        transaction_count INT,
        average_ticket DECIMAL(6,2),
        revenue_per_seat DECIMAL(8,2),
        in_store_revenue DECIMAL(8,2),
        digital_revenue DECIMAL(8,2),
        catering_revenue DECIMAL(8,2),
        delivery_revenue DECIMAL(8,2),
        avg_service_time FLOAT,
        peak_hour_throughput INT,
        staff_hours_scheduled FLOAT,
        staff_hours_actual FLOAT,
        food_cost_pct FLOAT,
        waste_amount DECIMAL(6,2),
        new_customers INT,
        returning_customers INT,
        loyalty_redemptions INT,
        avg_satisfaction_score FLOAT,
        weather_condition STRING,
        temperature_high INT,
        precipitation_inches FLOAT,
        local_events ARRAY<STRING>,
        CONSTRAINT pk_daily_perf PRIMARY KEY (store_id, business_date)
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
        base_price DECIMAL(5,2) NOT NULL,
        cost_of_goods DECIMAL(5,2),
        margin_pct FLOAT,
        calories INT,
        protein_g FLOAT,
        carbs_g FLOAT,
        fat_g FLOAT,
        sodium_mg INT,
        allergens ARRAY<STRING>,
        dietary_flags ARRAY<STRING>,
        active_flag BOOLEAN,
        last_updated TIMESTAMP,
        CONSTRAINT pk_item PRIMARY KEY (item_id),
        CONSTRAINT valid_category CHECK (category IN ('entree', 'protein', 'side', 'beverage', 'dessert'))
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
        discount_amount DECIMAL(5,2),
        minimum_purchase DECIMAL(5,2),
        eligible_items ARRAY<STRING>,
        promo_code STRING,
        target_segments ARRAY<STRING>,
        geographic_scope ARRAY<STRING>,
        channel_restrictions ARRAY<STRING>,
        start_date DATE,
        end_date DATE,
        announcement_date DATE,
        total_budget DECIMAL(10,2),
        media_spend DECIMAL(10,2),
        discount_budget DECIMAL(10,2),
        target_redemptions INT,
        target_roi FLOAT,
        target_new_customers INT,
        CONSTRAINT pk_campaign PRIMARY KEY (campaign_id),
        CONSTRAINT valid_campaign_type CHECK (campaign_type IN ('national', 'regional', 'local', 'digital_only')),
        CONSTRAINT valid_promo_type CHECK (promotion_type IN ('discount', 'bogo', 'free_item', 'points_multiplier'))
    )
    USING DELTA
    COMMENT 'Marketing campaigns and promotions'
""")

# COMMAND ----------

# MAGIC %md ## Step 3: Aggregation Tables

# COMMAND ----------

# Monthly store performance aggregation
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.store_performance_monthly (
        store_id STRING NOT NULL,
        year_month STRING NOT NULL,
        total_revenue DECIMAL(10,2),
        revenue_growth_pct FLOAT,
        revenue_vs_budget_pct FLOAT,
        market_share_est_pct FLOAT,
        unique_customers INT,
        new_customers INT,
        customer_retention_rate FLOAT,
        avg_visits_per_customer FLOAT,
        nps_score FLOAT,
        avg_ticket DECIMAL(6,2),
        transactions_per_day FLOAT,
        digital_mix_pct FLOAT,
        food_cost_pct FLOAT,
        labor_cost_pct FLOAT,
        vs_region_avg_pct FLOAT,
        vs_similar_stores_pct FLOAT,
        vs_national_avg_pct FLOAT,
        percentile_ranking INT,
        calculated_timestamp TIMESTAMP,
        CONSTRAINT pk_monthly_perf PRIMARY KEY (store_id, year_month)
    )
    USING DELTA
    PARTITIONED BY (year_month)
    CLUSTER BY (store_id)
    COMMENT 'Pre-aggregated monthly store performance'
""")

# Customer segments monthly aggregation
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.customer_segments_monthly (
        segment_name STRING NOT NULL,
        year_month STRING NOT NULL,
        segment_size INT,
        avg_monthly_visits FLOAT,
        avg_order_value DECIMAL(6,2),
        lifetime_value DECIMAL(8,2),
        churn_rate_pct FLOAT,
        preferred_dayparts ARRAY<STRING>,
        channel_mix MAP<STRING, FLOAT>,
        protein_preferences MAP<STRING, FLOAT>,
        promotion_response_rate FLOAT,
        segment_growth_pct FLOAT,
        revenue_contribution_pct FLOAT,
        acquisition_cost DECIMAL(6,2),
        CONSTRAINT pk_segment_monthly PRIMARY KEY (segment_name, year_month)
    )
    USING DELTA
    PARTITIONED BY (year_month)
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
