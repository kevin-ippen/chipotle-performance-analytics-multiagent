# Databricks notebook source
# MAGIC %md
# MAGIC # Chipotle Analytics - Synthetic Data Generation
# MAGIC
# MAGIC **Purpose**: Generate realistic synthetic data representing 300 stores (~10% of Chipotle US operations)
# MAGIC
# MAGIC **Outputs**: Populated gold layer tables with 3 years of transaction history
# MAGIC - ~50M transactions
# MAGIC - ~5M unique customers  
# MAGIC - 300 store locations across US
# MAGIC
# MAGIC **Assumptions**: 
# MAGIC - Tables created from 01_config notebook
# MAGIC - Unity Catalog permissions in place
# MAGIC
# MAGIC **Parameters**:
# MAGIC - start_date: Beginning of data generation (default: 2022-01-01)
# MAGIC - end_date: End of data generation (default: 2024-12-31)
# MAGIC - sample_pct: Percentage of full dataset to generate for testing (default: 100)

# COMMAND ----------

# MAGIC %pip install -q faker numpy

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random
import numpy as np
from faker import Faker
import json

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# Widget parameters
dbutils.widgets.text("catalog_name", "chipotle_analytics", "Catalog Name")
dbutils.widgets.text("start_date", "2022-01-01", "Start Date (YYYY-MM-DD)")
dbutils.widgets.text("end_date", "2024-12-31", "End Date (YYYY-MM-DD)")
dbutils.widgets.text("sample_pct", "100", "Sample Percentage (1-100)")

# Get parameters
CATALOG = dbutils.widgets.get("catalog_name")
START_DATE = datetime.strptime(dbutils.widgets.get("start_date"), "%Y-%m-%d")
END_DATE = datetime.strptime(dbutils.widgets.get("end_date"), "%Y-%m-%d")
SAMPLE_PCT = int(dbutils.widgets.get("sample_pct")) / 100.0

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md ## Step 1: Generate Store Locations

# COMMAND ----------

# Define explicit schema for consistent types
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, DateType

stores_schema = StructType([
    StructField("store_id", StringType(), False),
    StructField("store_number", IntegerType(), False),
    StructField("address", StringType(), False),
    StructField("city", StringType(), False),
    StructField("state", StringType(), False),
    StructField("zip_code", StringType(), False),
    StructField("latitude", DoubleType(), False),
    StructField("longitude", DoubleType(), False),
    StructField("trade_area_type", StringType(), False),
    StructField("drive_thru", BooleanType(), False),
    StructField("parking_spaces", IntegerType(), False),
    StructField("population_3mi", IntegerType(), False),
    StructField("median_income_3mi", IntegerType(), False),
    StructField("college_students_pct", DoubleType(), False),
    StructField("working_professionals_pct", DoubleType(), False),
    StructField("families_pct", DoubleType(), False),
    StructField("open_date", DateType(), False),
    StructField("store_format", StringType(), False),
    StructField("seating_capacity", IntegerType(), False),
    StructField("kitchen_capacity_score", IntegerType(), False),
    StructField("staff_count_avg", IntegerType(), False),
    StructField("manager_tenure_months", IntegerType(), False),
    StructField("fast_casual_competitors_1mi", IntegerType(), False),
    StructField("direct_competitors_3mi", IntegerType(), False),
    StructField("restaurant_density_1mi", IntegerType(), False),
    StructField("active_flag", BooleanType(), False)
])

# Store distribution by state
state_distribution = {
    'CA': 75, 'TX': 45, 'FL': 30, 'NY': 24, 'IL': 21,
    'OH': 12, 'PA': 10, 'AZ': 8, 'NC': 8, 'VA': 7,
    'CO': 7, 'WA': 7, 'MA': 6, 'GA': 6, 'NJ': 6,
    'MD': 5, 'MI': 5, 'OR': 4, 'NV': 4, 'IN': 3,
    'MO': 3, 'TN': 3, 'WI': 2, 'MN': 2, 'CT': 1
}

# Major cities by state
state_cities = {
    'CA': ['Los Angeles', 'San Diego', 'San Francisco', 'San Jose', 'Fresno'],
    'TX': ['Houston', 'Dallas', 'Austin', 'San Antonio', 'Fort Worth'],
    'FL': ['Miami', 'Orlando', 'Tampa', 'Jacksonville', 'Fort Lauderdale'],
    'NY': ['New York', 'Buffalo', 'Rochester', 'Albany', 'Syracuse'],
    'IL': ['Chicago', 'Aurora', 'Rockford', 'Joliet', 'Naperville']
}

stores_data = []
store_counter = 1

for state, store_count in state_distribution.items():
    cities = state_cities.get(state, [fake.city() for _ in range(5)])
    
    for i in range(int(store_count * SAMPLE_PCT)):
        store_id = f"STR_{store_counter:04d}"
        
        # Use random.choices for Python native types
        trade_area = random.choices(['urban', 'suburban', 'university', 'mall'], 
                                    weights=[0.35, 0.40, 0.15, 0.10])[0]
        
        if trade_area == 'urban':
            seating = int(random.randint(15, 35))
            parking = int(random.randint(0, 10))
            pop_3mi = int(random.randint(150000, 500000))
            income_3mi = int(random.randint(55000, 95000))
        elif trade_area == 'suburban':
            seating = int(random.randint(40, 70))
            parking = int(random.randint(20, 50))
            pop_3mi = int(random.randint(50000, 150000))
            income_3mi = int(random.randint(65000, 120000))
        elif trade_area == 'university':
            seating = int(random.randint(30, 50))
            parking = int(random.randint(5, 20))
            pop_3mi = int(random.randint(30000, 100000))
            income_3mi = int(random.randint(35000, 65000))
        else:  # mall
            seating = int(random.randint(25, 45))
            parking = int(0)
            pop_3mi = int(random.randint(80000, 200000))
            income_3mi = int(random.randint(55000, 85000))
        
        store_format = random.choices(['standard', 'drive_thru', 'urban_compact'],
                                     weights=[0.70, 0.20, 0.10])[0]
        
        stores_data.append({
            'store_id': store_id,
            'store_number': int(store_counter),
            'address': fake.street_address(),
            'city': random.choice(cities),
            'state': state,
            'zip_code': fake.zipcode(),
            'latitude': float(fake.latitude()),
            'longitude': float(fake.longitude()),
            'trade_area_type': trade_area,
            'drive_thru': (store_format == 'drive_thru'),
            'parking_spaces': parking,
            'population_3mi': pop_3mi,
            'median_income_3mi': income_3mi,
            'college_students_pct': float(0.35) if trade_area == 'university' else float(random.uniform(0.05, 0.20)),
            'working_professionals_pct': float(random.uniform(0.25, 0.45)),
            'families_pct': float(random.uniform(0.20, 0.40)),
            'open_date': fake.date_between(start_date='-10y', end_date='-1y'),
            'store_format': store_format,
            'seating_capacity': seating,
            'kitchen_capacity_score': int(random.randint(5, 10)),
            'staff_count_avg': int(random.randint(12, 25)),
            'manager_tenure_months': int(random.randint(3, 60)),
            'fast_casual_competitors_1mi': int(random.randint(2, 8)),
            'direct_competitors_3mi': int(random.randint(0, 3)),
            'restaurant_density_1mi': int(random.randint(5, 25)),
            'active_flag': True
        })
        
        store_counter += 1

# Create DataFrame with explicit schema
stores_df = spark.createDataFrame(stores_data, schema=stores_schema)
stores_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.gold.store_locations")
print(f"✓ Created {stores_df.count()} stores")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Step 2: Generate Customer Profiles
# MAGIC

# COMMAND ----------

# --- Fast & parallel customer generation (Spark-native) ---

from pyspark.sql import functions as F
from pyspark.sql import types as T

# Tuning (adjust to your cluster/data)
spark.conf.set("spark.sql.shuffle.partitions", max(200, spark.sparkContext.defaultParallelism * 2))
spark.conf.set("spark.databricks.delta.optimizeWrite", "true")
spark.conf.set("spark.databricks.delta.autoCompact", "true")

segments_config = {
    'power_user':     {'pct': 0.15, 'visits': 52, 'aov': 14.50, 'digital': 0.90},
    'loyal_regular':  {'pct': 0.25, 'visits': 26, 'aov': 11.75, 'digital': 0.70},
    'occasional':     {'pct': 0.45, 'visits': 12, 'aov': 10.25, 'digital': 0.45},
    'price_sensitive':{'pct': 0.15, 'visits':  6, 'aov':  8.75, 'digital': 0.35},
}

# Precompute totals (driver-side small arithmetic is fine)
avg_visits_per_customer = sum(s['pct'] * s['visits'] for s in segments_config.values())
days_in_period = (END_DATE - START_DATE).days
num_stores = len(stores_data)
daily_transactions_per_store = 150
total_transactions = num_stores * days_in_period * daily_transactions_per_store * SAMPLE_PCT
total_customers = int(total_transactions / avg_visits_per_customer)

print(f"Generating ~{total_customers:,} customers for ~{int(total_transactions):,} transactions")

# Small dim for stores -> Spark DF (broadcastable)
stores_df = spark.createDataFrame(stores_data)  # expects 'zip_code' present
stores_df = F.broadcast(stores_df.select("zip_code").distinct())

# Segment cutpoints
p_power     = segments_config['power_user']['pct']
p_loyal     = p_power + segments_config['loyal_regular']['pct']
p_occasional= p_loyal + segments_config['occasional']['pct']
# price_sensitive = remainder

# Base DF with parallelism; seed rand() for reproducibility if desired
base = (
    spark.range(0, total_customers)
         .repartition(max(512, spark.sparkContext.defaultParallelism * 4))
         .withColumn("r", F.rand(42))
         .withColumn(
             "customer_segment",
             F.when(F.col("r") < p_power, "power_user")
              .when(F.col("r") < p_loyal, "loyal_regular")
              .when(F.col("r") < p_occasional, "occasional")
              .otherwise("price_sensitive")
         )
         .drop("r")
)

# Handy lookups as literal maps
seg_to_visits = F.create_map([F.lit(k) if i%2==0 else F.lit(v['visits'])
                              for i,(k,v) in enumerate(segments_config.items())])
seg_to_aov    = F.create_map([F.lit(k) if i%2==0 else F.lit(v['aov'])
                              for i,(k,v) in enumerate(segments_config.items())])
seg_to_dig    = F.create_map([F.lit(k) if i%2==0 else F.lit(v['digital'])
                              for i,(k,v) in enumerate(segments_config.items())])

# Random helpers
rnd = F.rand(123)
rnd2= F.rand(456)
rnd3= F.rand(789)

# Registration date within [START_DATE, END_DATE]
start_lit = F.to_date(F.lit(str(START_DATE)))
registration_date = F.expr(f"date_add(to_date('{START_DATE:%Y-%m-%d}'), cast(rand(99)*{days_in_period} as int))")

# Age range by segment (probabilities encoded as thresholds)
def choose_age(seg_col):
    r = F.rand(1001)
    return (
        F.when(seg_col=="power_user",
               F.when(r < 0.40, "25-34").when(r < 0.75, "35-44").otherwise("45-54"))
         .when(seg_col=="loyal_regular",
               F.when(r < 0.35, "25-34").when(r < 0.75, "35-44").otherwise("45-54"))
         .when(seg_col=="occasional",
               F.when(r < 0.25, "18-24").when(r < 0.55, "25-34").when(r < 0.80, "35-44")
                .when(r < 0.95, "45-54").otherwise("55+"))
         .otherwise(  # price_sensitive
               F.when(r < 0.40, "18-24").when(r < 0.75, "25-34").otherwise("35-44"))
    )

def choose_income(seg_col):
    r = F.rand(1002)
    return (
        F.when(seg_col=="power_user",
               F.when(r < 0.40, "75k_100k").otherwise("over_100k"))
         .when(seg_col=="loyal_regular",
               F.when(r < 0.30, "50k_75k").when(r < 0.70, "75k_100k").otherwise("over_100k"))
         .when(seg_col=="occasional",
               F.when(r < 0.35, "under_50k").when(r < 0.75, "50k_75k").otherwise("75k_100k"))
         .otherwise(  # price_sensitive
               F.when(r < 0.60, "under_50k").otherwise("50k_75k"))
    )

# Loyalty tier
def choose_loyalty(seg_col):
    r = F.rand(1003)
    return (
        F.when(seg_col=="power_user",
               F.when(r < 0.40, "gold").otherwise("platinum"))
         .when(seg_col=="loyal_regular",
               F.when(r < 0.60, "silver").otherwise("gold"))
         .when(seg_col=="occasional", F.when(r < 0.70, "bronze").otherwise("silver"))
         .otherwise(F.lit("bronze"))
    )

income = choose_income(F.col("customer_segment"))
loyalty = choose_loyalty(F.col("customer_segment"))

# Lifestyle depends on income bracket
lifestyle = (
    F.when(income.isin("75k_100k","over_100k"),
           F.when(rnd < 0.40, "health_conscious").when(rnd < 0.80, "convenience").otherwise("premium"))
     .otherwise(F.when(rnd < 0.60, "value_seeker").otherwise("convenience"))
)

# Visit frequency label from visits count
visits_col = seg_to_visits[F.col("customer_segment")]
visit_freq = F.when(visits_col==52, "weekly")\
              .when(visits_col==26, "biweekly")\
              .when(visits_col==12, "monthly")\
              .otherwise("occasional")

# Helper to pick k random proteins using shuffle+slice
proteins = F.array(*[F.lit(x) for x in ['chicken','steak','carnitas','barbacoa','sofritas']])
k_proteins = (F.floor(rnd2*3)+1).cast("int")  # 1..3
preferred_proteins = F.expr("slice(shuffle(array('chicken','steak','carnitas','barbacoa','sofritas')), 1, int(rand(456)*3)+1)")

# Dietary preferences: sample up to 2; include 'none' sometimes
diet_opts = F.array(*[F.lit(x) for x in ['vegetarian','keto','high_protein','low_sodium']])
diet_count = (F.floor(rnd3*3)).cast("int")  # 0..2
dietary_preferences = F.when(F.rand(111) < 0.5, F.array(F.lit("none"))) \
                       .otherwise(F.expr("slice(shuffle(array('vegetarian','keto','high_protein','low_sodium')), 1, int(rand(789)*2))"))

# Random booleans
app_user            = F.rand(222) < (seg_to_dig[F.col("customer_segment")])
email_subscriber    = F.rand(223) < 0.6
push_notifications  = F.rand(224) < (seg_to_dig[F.col("customer_segment")] * F.lit(0.7))
social_media        = F.rand(225) < 0.3

# referrals ~ Poisson(1) approx via geometric trick (fast) or clamp of normal
referrals = F.greatest(F.lit(0), (F.floor(-F.log(F.rand(226))).cast("int")))  # approx Poisson(1)

# churn risk rule
churn_risk = (
    F.when(F.col("customer_segment")=="power_user", F.lit(0.1))
     .when(F.col("customer_segment")=="price_sensitive", F.lit(0.7))
     .otherwise(F.rand(227)*0.3 + 0.2)
)

# Per-customer AOV and lifetime spend
aov_base = seg_to_aov[F.col("customer_segment")]
avg_order_value = aov_base * (F.rand(228)*0.2 + 0.9)   # +-10%
lifetime_spend  = aov_base * visits_col * (F.rand(229)*1.7 + 0.8)

# Household size, points, id
household_size  = (F.floor(F.rand(230)*4)+1).cast("int")
points_balance  = (F.floor(F.rand(231)*5001)).cast("int")
customer_id     = F.format_string("CUST_%08d", F.col("id")+1)

# Zip code: random join
zip_join = base.withColumn("zip_key", (F.floor(F.rand(300)*1e12)).cast("long"))
stores_with_key = stores_df.withColumn("zip_key", (F.floor(F.rand(301)*1e12)).cast("long"))
# Join via modulo bucket to randomize mapping without skew
zip_bucketed = (
    base.withColumn("bucket", (F.pmod(F.col("id"), F.lit(1000)))).join(
        stores_df.withColumn("bucket", (F.pmod(F.floor(F.rand(555)*1e12), F.lit(1000)))),
        on="bucket", how="left"
    ).drop("bucket")
)

# Assemble final DF
customers_df = (
    base.alias("b")
    .join(zip_bucketed.select("id","zip_code").alias("z"), on="id", how="left")
    .withColumn("customer_id", customer_id)
    .withColumn("registration_date", registration_date)
    .withColumn("registration_channel",
                F.when(F.rand(232) < 0.45, "app")
                 .when(F.rand(233) < 0.75, "web").otherwise("in_store"))
    .withColumn("age_range", choose_age(F.col("customer_segment")))
    .withColumn("income_bracket", income)
    .withColumn("household_size", household_size)
    .withColumn("zip_code", F.coalesce(F.col("z.zip_code"), F.lit(stores_data[0]['zip_code'])))
    .withColumn("lifestyle", lifestyle)
    .withColumn("loyalty_tier", loyalty)
    .withColumn("points_balance", points_balance)
    .withColumn("lifetime_spend", lifetime_spend.cast("double"))
    .withColumn("visit_frequency", visit_freq)
    .withColumn("avg_order_value", avg_order_value.cast("double"))
    .withColumn("preferred_proteins", preferred_proteins)
    .withColumn("dietary_preferences", dietary_preferences)
    .withColumn("app_user", app_user.cast("boolean"))
    .withColumn("email_subscriber", email_subscriber.cast("boolean"))
    .withColumn("push_notifications", push_notifications.cast("boolean"))
    .withColumn("social_media_follower", social_media.cast("boolean"))
    .withColumn("referrals_made", referrals.cast("int"))
    .withColumn("churn_risk_score", churn_risk.cast("double"))
    .select(
        "customer_id","registration_date","registration_channel","age_range",
        "income_bracket","household_size","zip_code","lifestyle","loyalty_tier",
        "points_balance","lifetime_spend","visit_frequency","avg_order_value",
        "preferred_proteins","dietary_preferences","app_user","email_subscriber",
        "push_notifications","social_media_follower","referrals_made",
        "churn_risk_score","customer_segment"
    )
)

# Write without triggering extra actions; avoid .count() here for speed
target_table = f"{CATALOG}.gold.customer_profiles"
customers_df.write.mode("overwrite").saveAsTable(target_table)
print(f"✓ Created ~{total_customers:,} customers -> {target_table}")


# COMMAND ----------

# COMMAND ----------
# MAGIC %md ## Step 2: Generate Customer Profiles
# MAGIC
# MAGIC **Purpose**: Generate realistic synthetic data representing 300 stores (~10% of Chipotle US operations)
# MAGIC
# MAGIC **Outputs**: Populated gold layer tables with 3 years of transaction history
# MAGIC - ~50M transactions
# MAGIC - ~5M unique customers
# MAGIC - 300 store locations across US
# MAGIC
# MAGIC **Assumptions**:
# MAGIC - Tables created from 01_config notebook
# MAGIC - Unity Catalog permissions in place
# MAGIC
# MAGIC **Parameters**:
# MAGIC - start_date: Beginning of data generation (default: 2022-01-01)
# MAGIC - end_date: End of data generation (default: 2024-12-31)
# MAGIC - sample_pct: Percentage of full dataset to generate for testing (default: 100)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from datetime import datetime
from faker import Faker
import random
import numpy as np

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# Widget parameters (same as original)
dbutils.widgets.text("catalog_name", "chipotle_analytics", "Catalog Name")
dbutils.widgets.text("start_date", "2022-01-01", "Start Date (YYYY-MM-DD)")
dbutils.widgets.text("end_date", "2024-12-31", "End Date (YYYY-MM-DD)")
dbutils.widgets.text("sample_pct", "100", "Sample Percentage (1-100)")

# Get parameters
CATALOG = dbutils.widgets.get("catalog_name")
START_DATE = datetime.strptime(dbutils.widgets.get("start_date"), "%Y-%m-%d")
END_DATE = datetime.strptime(dbutils.widgets.get("end_date"), "%Y-%m-%d")
SAMPLE_PCT = int(dbutils.widgets.get("sample_pct")) / 100.0

spark.sql(f"USE CATALOG {CATALOG}")


# --- Fast & parallel customer generation (Spark-native) ---
stores_df_temp = spark.read.table(f"{CATALOG}.gold.store_locations").select("zip_code").distinct()
stores_zip_codes_list = stores_df_temp.rdd.flatMap(lambda x: x).collect()
broadcast_zip_codes = spark.sparkContext.broadcast(stores_zip_codes_list)

# Tuning (adjust to your cluster/data)
spark.conf.set("spark.sql.shuffle.partitions", max(200, spark.sparkContext.defaultParallelism * 2))
spark.conf.set("spark.databricks.delta.optimizeWrite", "true")
spark.conf.set("spark.databricks.delta.autoCompact", "true")

segments_config = {
    'power_user':     {'pct': 0.15, 'visits': 52, 'aov': 14.50, 'digital': 0.90},
    'loyal_regular':  {'pct': 0.25, 'visits': 26, 'aov': 11.75, 'digital': 0.70},
    'occasional':     {'pct': 0.45, 'visits': 12, 'aov': 10.25, 'digital': 0.45},
    'price_sensitive':{'pct': 0.15, 'visits':  6, 'aov':  8.75, 'digital': 0.35},
}

# Precompute totals (driver-side small arithmetic is fine)
avg_visits_per_customer = sum(s['pct'] * s['visits'] for s in segments_config.values())
days_in_period = (END_DATE - START_DATE).days
num_stores = stores_df_temp.count()
daily_transactions_per_store = 150
total_transactions = num_stores * days_in_period * daily_transactions_per_store * SAMPLE_PCT
total_customers = int(total_transactions / avg_visits_per_customer)

print(f"Generating ~{total_customers:,} customers for ~{int(total_transactions):,} transactions")

# Segment cutpoints
p_power     = segments_config['power_user']['pct']
p_loyal     = p_power + segments_config['loyal_regular']['pct']
p_occasional= p_loyal + segments_config['occasional']['pct']

# Base DF with parallelism; seed rand() for reproducibility if desired
base = (
    spark.range(0, total_customers)
         .repartition(max(512, spark.sparkContext.defaultParallelism * 4))
         .withColumn("r", F.rand(42))
         .withColumn(
             "customer_segment",
             F.when(F.col("r") < p_power, "power_user")
              .when(F.col("r") < p_loyal, "loyal_regular")
              .when(F.col("r") < p_occasional, "occasional")
              .otherwise("price_sensitive")
         )
         .drop("r")
)

# Handy lookups as literal maps
seg_to_visits = F.create_map([F.lit(k) if i%2==0 else F.lit(v['visits'])
                              for i,(k,v) in enumerate(segments_config.items())])
seg_to_aov    = F.create_map([F.lit(k) if i%2==0 else F.lit(v['aov'])
                              for i,(k,v) in enumerate(segments_config.items())])
seg_to_dig    = F.create_map([F.lit(k) if i%2==0 else F.lit(v['digital'])
                              for i,(k,v) in enumerate(segments_config.items())])

# Random helpers
rnd = F.rand(123)
rnd2= F.rand(456)
rnd3= F.rand(789)

# Registration date within [START_DATE, END_DATE]
start_lit = F.to_date(F.lit(str(START_DATE)))
registration_date = F.expr(f"date_add(to_date('{START_DATE:%Y-%m-%d}'), cast(rand(99)*{days_in_period} as int))")

# Age range by segment (probabilities encoded as thresholds)
def choose_age(seg_col):
    r = F.rand(1001)
    return (
        F.when(seg_col=="power_user",
               F.when(r < 0.40, "25-34").when(r < 0.75, "35-44").otherwise("45-54"))
         .when(seg_col=="loyal_regular",
               F.when(r < 0.35, "25-34").when(r < 0.75, "35-44").otherwise("45-54"))
         .when(seg_col=="occasional",
               F.when(r < 0.25, "18-24").when(r < 0.55, "25-34").when(r < 0.80, "35-44")
                .when(r < 0.95, "45-54").otherwise("55+"))
         .otherwise(  # price_sensitive
               F.when(r < 0.40, "18-24").when(r < 0.75, "25-34").otherwise("35-44"))
    )

def choose_income(seg_col):
    r = F.rand(1002)
    return (
        F.when(seg_col=="power_user",
               F.when(r < 0.40, "75k_100k").otherwise("over_100k"))
         .when(seg_col=="loyal_regular",
               F.when(r < 0.30, "50k_75k").when(r < 0.70, "75k_100k").otherwise("over_100k"))
         .when(seg_col=="occasional",
               F.when(r < 0.35, "under_50k").when(r < 0.75, "50k_75k").otherwise("75k_100k"))
         .otherwise(  # price_sensitive
               F.when(r < 0.60, "under_50k").otherwise("50k_75k"))
    )

# Loyalty tier
def choose_loyalty(seg_col):
    r = F.rand(1003)
    return (
        F.when(seg_col=="power_user",
               F.when(r < 0.40, "gold").otherwise("platinum"))
         .when(seg_col=="loyal_regular",
               F.when(r < 0.60, "silver").otherwise("gold"))
         .when(seg_col=="occasional", F.when(r < 0.70, "bronze").otherwise("silver"))
         .otherwise(F.lit("bronze"))
    )

income = choose_income(F.col("customer_segment"))
loyalty = choose_loyalty(F.col("customer_segment"))

# Lifestyle depends on income bracket
lifestyle = (
    F.when(income.isin("75k_100k","over_100k"),
           F.when(rnd < 0.40, "health_conscious").when(rnd < 0.80, "convenience").otherwise("premium"))
     .otherwise(F.when(rnd < 0.60, "value_seeker").otherwise("convenience"))
)

# Visit frequency label from visits count
visits_col = seg_to_visits[F.col("customer_segment")]
visit_freq = F.when(visits_col==52, "weekly")\
              .when(visits_col==26, "biweekly")\
              .when(visits_col==12, "monthly")\
              .otherwise("occasional")

# Helper to pick k random proteins using shuffle+slice
proteins = F.array(*[F.lit(x) for x in ['chicken','steak','carnitas','barbacoa','sofritas']])
k_proteins = (F.floor(rnd2*3)+1).cast("int")  # 1..3
preferred_proteins = F.expr("slice(shuffle(array('chicken','steak','carnitas','barbacoa','sofritas')), 1, int(rand(456)*3)+1)")

# Dietary preferences: sample up to 2; include 'none' sometimes
diet_opts = F.array(*[F.lit(x) for x in ['vegetarian','keto','high_protein','low_sodium']])
diet_count = (F.floor(rnd3*3)).cast("int")  # 0..2
dietary_preferences = F.when(F.rand(111) < 0.5, F.array(F.lit("none"))) \
                       .otherwise(F.expr("slice(shuffle(array('vegetarian','keto','high_protein','low_sodium')), 1, int(rand(789)*2))"))

# Random booleans
app_user            = F.rand(222) < (seg_to_dig[F.col("customer_segment")])
email_subscriber    = F.rand(223) < 0.6
push_notifications  = F.rand(224) < (seg_to_dig[F.col("customer_segment")] * F.lit(0.7))
social_media        = F.rand(225) < 0.3

# referrals ~ Poisson(1) approx via geometric trick (fast) or clamp of normal
referrals = F.greatest(F.lit(0), (F.floor(-F.log(F.rand(226))).cast("int")))  # approx Poisson(1)

# churn risk rule
churn_risk = (
    F.when(F.col("customer_segment")=="power_user", F.lit(0.1))
     .when(F.col("customer_segment")=="price_sensitive", F.lit(0.7))
     .otherwise(F.rand(227)*0.3 + 0.2)
)

# Per-customer AOV and lifetime spend
aov_base = seg_to_aov[F.col("customer_segment")]
avg_order_value = aov_base * (F.rand(228)*0.2 + 0.9)   # +-10%
lifetime_spend  = aov_base * visits_col * (F.rand(229)*1.7 + 0.8)

# Household size, points, id
household_size  = (F.floor(F.rand(230)*4)+1).cast("int")
points_balance  = (F.floor(F.rand(231)*5001)).cast("int")
customer_id     = F.format_string("CUST_%08d", F.col("id")+1)


# Assemble final DF with guaranteed matching zip codes
customers_df = (
    base
    .withColumn("customer_id", customer_id)
    .withColumn("registration_date", registration_date)
    .withColumn("registration_channel",
                F.when(F.rand(232) < 0.45, "app")
                 .when(F.rand(233) < 0.75, "web").otherwise("in_store"))
    .withColumn("age_range", choose_age(F.col("customer_segment")))
    .withColumn("income_bracket", income)
    .withColumn("household_size", household_size)
    .withColumn("zip_code", F.lit(F.element_at(F.array(*[F.lit(z) for z in broadcast_zip_codes.value]), (F.floor(F.rand(300) * len(broadcast_zip_codes.value)) + 1).cast("int"))))
    .withColumn("lifestyle", lifestyle)
    .withColumn("loyalty_tier", loyalty)
    .withColumn("points_balance", points_balance)
    .withColumn("lifetime_spend", lifetime_spend.cast("double"))
    .withColumn("visit_frequency", visit_freq)
    .withColumn("avg_order_value", avg_order_value.cast("double"))
    .withColumn("preferred_proteins", preferred_proteins)
    .withColumn("dietary_preferences", dietary_preferences)
    .withColumn("app_user", app_user.cast("boolean"))
    .withColumn("email_subscriber", email_subscriber.cast("boolean"))
    .withColumn("push_notifications", push_notifications.cast("boolean"))
    .withColumn("social_media_follower", social_media.cast("boolean"))
    .withColumn("referrals_made", referrals.cast("int"))
    .withColumn("churn_risk_score", churn_risk.cast("double"))
    .select(
        "customer_id","registration_date","registration_channel","age_range",
        "income_bracket","household_size","zip_code","lifestyle","loyalty_tier",
        "points_balance","lifetime_spend","visit_frequency","avg_order_value",
        "preferred_proteins","dietary_preferences","app_user","email_subscriber",
        "push_notifications","social_media_follower","referrals_made",
        "churn_risk_score","customer_segment"
    )
)

# Write without triggering extra actions; avoid .count() here for speed
target_table = f"{CATALOG}.gold.customer_profiles"
customers_df.write.mode("overwrite").saveAsTable(target_table)
print(f"✓ Created ~{total_customers:,} customers -> {target_table}")

# COMMAND ----------

# MAGIC %md ## Step 3: Generate Menu Items

# COMMAND ----------

# Chipotle-style menu structure
menu_items = [
    # Entrees
    {'item_id': 'ENT_001', 'item_name': 'Burrito', 'category': 'entree', 'subcategory': 'burrito', 'base_price': 9.50, 'cost': 3.20},
    {'item_id': 'ENT_002', 'item_name': 'Bowl', 'category': 'entree', 'subcategory': 'bowl', 'base_price': 9.50, 'cost': 3.00},
    {'item_id': 'ENT_003', 'item_name': 'Tacos (3)', 'category': 'entree', 'subcategory': 'tacos', 'base_price': 9.50, 'cost': 3.10},
    {'item_id': 'ENT_004', 'item_name': 'Salad', 'category': 'entree', 'subcategory': 'salad', 'base_price': 9.50, 'cost': 2.90},
    {'item_id': 'ENT_005', 'item_name': 'Quesadilla', 'category': 'entree', 'subcategory': 'quesadilla', 'base_price': 10.95, 'cost': 3.50},
    {'item_id': 'ENT_006', 'item_name': 'Kids Build Your Own', 'category': 'entree', 'subcategory': 'kids', 'base_price': 5.25, 'cost': 1.80},
    
    # Proteins  
    {'item_id': 'PRO_001', 'item_name': 'Chicken', 'category': 'protein', 'subcategory': 'chicken', 'base_price': 0.00, 'cost': 2.50},
    {'item_id': 'PRO_002', 'item_name': 'Steak', 'category': 'protein', 'subcategory': 'steak', 'base_price': 1.50, 'cost': 3.80},
    {'item_id': 'PRO_003', 'item_name': 'Carnitas', 'category': 'protein', 'subcategory': 'carnitas', 'base_price': 0.50, 'cost': 2.80},
    {'item_id': 'PRO_004', 'item_name': 'Barbacoa', 'category': 'protein', 'subcategory': 'barbacoa', 'base_price': 1.50, 'cost': 3.50},
    {'item_id': 'PRO_005', 'item_name': 'Sofritas', 'category': 'protein', 'subcategory': 'sofritas', 'base_price': 0.00, 'cost': 1.80},
    {'item_id': 'PRO_006', 'item_name': 'Veggie', 'category': 'protein', 'subcategory': 'veggie', 'base_price': -1.00, 'cost': 1.20},
    
    # Sides
    {'item_id': 'SID_001', 'item_name': 'Chips', 'category': 'side', 'subcategory': 'chips', 'base_price': 1.95, 'cost': 0.40},
    {'item_id': 'SID_002', 'item_name': 'Chips & Guac', 'category': 'side', 'subcategory': 'chips', 'base_price': 4.45, 'cost': 1.20},
    {'item_id': 'SID_003', 'item_name': 'Chips & Queso', 'category': 'side', 'subcategory': 'chips', 'base_price': 4.45, 'cost': 1.10},
    {'item_id': 'SID_004', 'item_name': 'Large Chips & Large Guac', 'category': 'side', 'subcategory': 'chips', 'base_price': 7.95, 'cost': 2.00},
    {'item_id': 'SID_005', 'item_name': 'Side of Guac', 'category': 'side', 'subcategory': 'extras', 'base_price': 2.50, 'cost': 0.80},
    {'item_id': 'SID_006', 'item_name': 'Side of Queso', 'category': 'side', 'subcategory': 'extras', 'base_price': 2.50, 'cost': 0.70},
    
    # Beverages
    {'item_id': 'BEV_001', 'item_name': 'Fountain Drink', 'category': 'beverage', 'subcategory': 'soda', 'base_price': 2.95, 'cost': 0.35},
    {'item_id': 'BEV_002', 'item_name': 'Bottled Water', 'category': 'beverage', 'subcategory': 'water', 'base_price': 2.50, 'cost': 0.50},
    {'item_id': 'BEV_003', 'item_name': 'Mexican Coca-Cola', 'category': 'beverage', 'subcategory': 'soda', 'base_price': 3.25, 'cost': 1.20},
    {'item_id': 'BEV_004', 'item_name': 'Juice', 'category': 'beverage', 'subcategory': 'juice', 'base_price': 2.95, 'cost': 0.80},
    {'item_id': 'BEV_005', 'item_name': 'Milk', 'category': 'beverage', 'subcategory': 'milk', 'base_price': 2.50, 'cost': 0.60},
]

# Add nutritional info and other attributes
for item in menu_items:
    item['margin_pct'] = float(item['base_price'] - item['cost']) / float(item['base_price']) if item['base_price'] > 0 else 0.0
    item['cost_of_goods'] = float(item.pop('cost'))
    
    # Realistic nutritional values
    if item['category'] == 'entree':
        item['calories'] = int(random.randint(500, 1200))
        item['protein_g'] = float(random.uniform(20, 45))
        item['carbs_g'] = float(random.uniform(40, 80))
        item['fat_g'] = float(random.uniform(15, 35))
    elif item['category'] == 'protein':
        item['calories'] = int(random.randint(150, 250))
        item['protein_g'] = float(random.uniform(20, 35))
        item['carbs_g'] = float(random.uniform(0, 5))
        item['fat_g'] = float(random.uniform(5, 15))
    elif item['category'] == 'side':
        item['calories'] = int(random.randint(200, 600))
        item['protein_g'] = float(random.uniform(2, 8))
        item['carbs_g'] = float(random.uniform(20, 60))
        item['fat_g'] = float(random.uniform(10, 40))
    else:  # beverage
        item['calories'] = int(random.randint(0, 250))
        item['protein_g'] = float(0.0)
        item['carbs_g'] = float(random.uniform(0, 60))
        item['fat_g'] = float(0.0)
    
    item['sodium_mg'] = int(random.randint(100, 1500))
    item['allergens'] = []
    item['dietary_flags'] = []
    
    # Set dietary flags
    if 'sofritas' in item['item_name'].lower() or 'veggie' in item['item_name'].lower():
        item['dietary_flags'].append('vegetarian')
        if 'sofritas' in item['item_name'].lower():
            item['dietary_flags'].append('vegan')
    
    item['active_flag'] = True
    item['last_updated'] = datetime.now()

# Define explicit schema to avoid inference errors for empty lists
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType, TimestampType, ArrayType
menu_schema = StructType([
    StructField("item_id", StringType(), False),
    StructField("item_name", StringType(), False),
    StructField("category", StringType(), False),
    StructField("subcategory", StringType(), True),
    StructField("base_price", DoubleType(), False),
    StructField("cost_of_goods", DoubleType(), True),
    StructField("margin_pct", DoubleType(), True),
    StructField("calories", IntegerType(), True),
    StructField("protein_g", DoubleType(), True),
    StructField("carbs_g", DoubleType(), True),
    StructField("fat_g", DoubleType(), True),
    StructField("sodium_mg", IntegerType(), True),
    StructField("allergens", ArrayType(StringType()), True),
    StructField("dietary_flags", ArrayType(StringType()), True),
    StructField("active_flag", BooleanType(), True),
    StructField("last_updated", TimestampType(), True),
])

# Create DataFrame with the explicit schema
menu_df = spark.createDataFrame(menu_items, schema=menu_schema)
menu_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.gold.menu_items")
print(f"✓ Created {menu_df.count()} menu items")

# COMMAND ----------

# MAGIC %md ## Step 4: Generate Transactions

# COMMAND ----------

from pyspark.sql.window import Window

# COMMAND ----------

# COMMAND ----------
# MAGIC %md ## Step 4: Generate Transactions (deterministic count via store × date grid)

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ======== Tuning ========
parts = max(512, spark.sparkContext.defaultParallelism * 4)
spark.conf.set("spark.sql.shuffle.partitions", parts)
spark.conf.set("spark.databricks.delta.optimizeWrite", "true")
spark.conf.set("spark.databricks.delta.autoCompact", "true")

target_table = f"{CATALOG}.gold.transactions"

# ======== Inputs ========
stores_base = spark.read.table(f"{CATALOG}.gold.store_locations") \
    .select("store_id", "kitchen_capacity_score", "zip_code")

customers_df = spark.read.table(f"{CATALOG}.gold.customer_profiles") \
    .select("customer_id", "customer_segment", "app_user", "zip_code")

menu_df = spark.read.table(f"{CATALOG}.gold.menu_items") \
    .select("item_id","item_name","category","base_price")

# Stable row ids for customer uniform pick (broadcast for cheap joins)
customers_df = customers_df.withColumn("cust_rn", F.row_number().over(Window.orderBy(F.xxhash64("customer_id"))))
cust_count = customers_df.count()
customers_df = F.broadcast(customers_df)

# Build literal arrays of structs per category (menu is tiny)
def cat_array(cat: str):
    rows = (menu_df.where(F.col("category")==F.lit(cat))
                   .select("item_id","item_name","category","base_price")
                   .collect())
    if not rows:
        return F.array()
    return F.array(*[
        F.struct(
            F.lit(r["item_id"]).alias("item_id"),
            F.lit(r["item_name"]).alias("item_name"),
            F.lit(r["category"]).alias("category"),
            F.lit(float(r["base_price"])).alias("base_price")
        ) for r in (row.asDict() for row in rows)
    ])

arr_entree   = cat_array("entree")
arr_protein  = cat_array("protein")
arr_side     = cat_array("side")
arr_beverage = cat_array("beverage")

# ======== Build store × date grid (INCLUSIVE range) ========
days_in_period = (END_DATE - START_DATE).days + 1  # inclusive
date_df = spark.sql(f"""
  SELECT explode(sequence(
      to_date('{START_DATE:%Y-%m-%d}'),
      to_date('{END_DATE:%Y-%m-%d}'),
      interval 1 day
  )) AS order_date
""")
stores_full = stores_base  # don't sample stores; SAMPLE_PCT applies to txn count only

grid = date_df.crossJoin(stores_full)

# Per-day target = round(150 × SAMPLE_PCT)
daily_target = int(round(150.0 * SAMPLE_PCT))
grid = grid.withColumn("daily_target", F.lit(daily_target).cast("int")) \
           .filter(F.col("daily_target") > 0)

# Expand to transactions exactly: sequence(1, daily_target)
grid = grid.withColumn("seq", F.expr("sequence(1, daily_target)"))
tx = (grid
      .withColumn("seq", F.explode(F.col("seq")))
      .withColumn("hash_id", F.xxhash64("store_id", "order_date", "seq"))
      .drop("daily_target", "seq"))

# Sanity print before heavy work
expected_rows = stores_full.count() * date_df.count() * daily_target
print(f"Expected rows (stores × days × daily_target): {expected_rows:,}")

# ======== Bring a customer (uniform) ========
tx = tx.withColumn(
    "cust_pick",
    (F.pmod(F.abs(F.col("hash_id")), F.lit(cust_count)) + F.lit(1)).cast("long")
).join(
    customers_df.select("customer_id","app_user","zip_code","cust_rn"),
    on=(F.col("cust_pick")==F.col("cust_rn")),
    how="left"
).drop("cust_pick","cust_rn")

# ======== Optional: reassign to a local store by customer ZIP (only if ZIP has stores) ========
stores_by_zip_df = F.broadcast(
    stores_base.groupBy("zip_code").agg(F.collect_list(F.col("store_id")).alias("local_stores"))
)
tx = tx.join(stores_by_zip_df, on="zip_code", how="left")
tx = tx.withColumn("local_stores_safe", F.coalesce(F.col("local_stores"), F.array().cast("array<string>")))
idx = (F.pmod(F.abs(F.col("hash_id")), F.greatest(F.size(F.col("local_stores_safe")), F.lit(1))) + F.lit(1)).cast("int")
tx = tx.withColumn(
    "store_id",
    F.when(F.size(F.col("local_stores_safe")) > 0, F.element_at(F.col("local_stores_safe"), idx))
     .otherwise(F.col("store_id"))
).drop("local_stores","local_stores_safe","zip_code")

# ======== Spread timestamps uniformly across the full range ========
date_range_sec = days_in_period * 24 * 60 * 60
tx = tx.withColumn(
    "order_timestamp",
    F.expr(f"timestampadd(SECOND, CAST(pmod(abs(hash_id), {date_range_sec}) AS INT), to_timestamp('{START_DATE:%Y-%m-%d}'))")
).withColumn("order_date", F.to_date("order_timestamp"))

# ======== Daypart (deterministic) ========
daypart_rand = (
    F.pmod(F.abs(F.xxhash64(F.col("hash_id"), F.lit(12345))), F.lit(10_000_000)) / F.lit(10_000_000.0)
)
tx = tx.withColumn("daypart_rand", daypart_rand)
tx = tx.withColumn(
    "daypart",
    F.when(F.col("daypart_rand") < 0.08, F.lit("breakfast"))
     .when(F.col("daypart_rand") < 0.08 + 0.45, F.lit("lunch"))
     .when(F.col("daypart_rand") < 0.08 + 0.45 + 0.35, F.lit("dinner"))
     .otherwise(F.lit("late_night"))
).drop("daypart_rand")

# ======== Channel / order_type ========
channel = F.when(F.col("app_user") & (F.rand(301) < 0.6),
                 F.when(F.rand(302) < 0.5, F.lit("app")).otherwise(F.lit("web"))) \
           .otherwise(F.lit("in_store"))
order_type = F.when(channel.isin("app","web"),
                    F.when(F.rand(303) < 0.7, F.lit("pickup")).otherwise(F.lit("delivery"))) \
              .otherwise(F.lit("dine_in"))

# ======== Helpers to pick random struct from an array ========
def pick_random(arr_expr, seed):
    sz = F.size(arr_expr)
    return F.element_at(arr_expr, (F.floor(F.rand(seed) * sz) + 1).cast("int"))

entree  = pick_random(arr_entree,   401)
protein = pick_random(arr_protein,  402)
side    = pick_random(arr_side,     403)
bev     = pick_random(arr_beverage, 404)

# ======== Modifications & items (consistent schema) ========
mods_pool = F.array(
    F.lit("extra_rice"), F.lit("extra_beans"), F.lit("extra_cheese"),
    F.lit("no_rice"), F.lit("no_beans"), F.lit("light_cheese")
)
mods_k  = (F.floor(F.rand(405) * 3)).cast("int")  # 0..2
mods    = F.when(mods_k > 0, F.slice(F.shuffle(mods_pool), 1, mods_k)) \
           .otherwise(F.array().cast("array<string>"))

entree_plus = F.struct(
    entree["item_id"].alias("item_id"),
    entree["item_name"].alias("item_name"),
    entree["category"].alias("category"),
    (entree["base_price"].cast("double") + protein["base_price"].cast("double")).alias("base_price"),
    mods.alias("modifications"),
    F.lit(1).alias("quantity")
)

empty_mods = F.array().cast("array<string>")
side_struct = F.struct(
    side["item_id"].alias("item_id"),
    side["item_name"].alias("item_name"),
    side["category"].alias("category"),
    side["base_price"].cast("double").alias("base_price"),
    empty_mods.alias("modifications"),
    F.lit(1).alias("quantity")
)
bev_struct = F.struct(
    bev["item_id"].alias("item_id"),
    bev["item_name"].alias("item_name"),
    bev["category"].alias("category"),
    bev["base_price"].cast("double").alias("base_price"),
    empty_mods.alias("modifications"),
    F.lit(1).alias("quantity")
)

order_items = F.array(entree_plus)
order_items = F.when(F.rand(406) < 0.3, F.concat(order_items, F.array(side_struct))).otherwise(order_items)
order_items = F.when(F.rand(407) < 0.4, F.concat(order_items, F.array(bev_struct))).otherwise(order_items)

# ======== Monetary fields ========
subtotal = F.aggregate(order_items, F.lit(0.0), lambda acc, x: acc + (x["base_price"] * x["quantity"]))
tax = (subtotal * F.lit(0.0875)).cast("double")

tip = F.when(F.col("order_type") == F.lit("delivery"), subtotal * (F.rand(408) * F.lit(0.10) + F.lit(0.10))) \
       .when((F.col("order_type") == F.lit("pickup")) & (F.rand(409) < F.lit(0.15)), subtotal * (F.rand(410) * F.lit(0.10) + F.lit(0.05))) \
       .otherwise(F.lit(0.0)).cast("double")

discount = F.when(F.rand(411) < F.lit(0.10), subtotal * (F.rand(412) * F.lit(0.15) + F.lit(0.10))) \
            .otherwise(F.lit(0.0)).cast("double")

promo_codes = F.when(
    discount > 0,
    F.array(
        F.element_at(
            F.array(F.lit("BOGO50"), F.lit("SAVE20"), F.lit("FREECHIPS"), F.lit("STUDENT15")),
            (F.floor(F.rand(413) * 4) + 1).cast("int")
        )
    )
).otherwise(F.array().cast("array<string>"))

total_amount = (subtotal + tax + tip - discount).cast("double")

delivery_partner_choices = F.array(
    F.lit("uber_eats"), F.lit("doordash"), F.lit("grubhub"), F.lit(None).cast("string")
)
delivery_partner = F.when(
    F.col("order_type") == F.lit("delivery"),
    F.element_at(delivery_partner_choices, (F.floor(F.rand(414) * 4) + 1).cast("int"))
).otherwise(F.lit(None).cast("string"))

# ======== Assemble final DF ========
transactions_df = (
    tx
    .withColumn("channel", channel)
    .withColumn("order_type", order_type)
    .withColumn("order_items", order_items)
    .withColumn("tax_amount", tax)
    .withColumn("tip_amount", tip)
    .withColumn("discount_amount", discount)
    .withColumn("total_amount", total_amount)
    .withColumn(
        "payment_method",
        F.element_at(
            F.array(F.lit("credit_card"), F.lit("debit_card"), F.lit("apple_pay"), F.lit("google_pay"), F.lit("cash")),
            (F.floor(F.rand(415) * 5) + 1).cast("int")
        )
    )
    .withColumn("promotion_codes", promo_codes)
    .withColumn("delivery_partner", delivery_partner)
    .withColumn("order_prep_time_minutes", (F.floor(F.rand(416) * 11) + 5).cast("int"))
    .withColumn("customer_wait_time_minutes", (F.floor(F.rand(417) * 19) + 2).cast("int"))
    .withColumn("order_id",
        F.concat(
            F.lit("ORD_"),
            F.date_format("order_date", "yyyyMMdd"),
            F.lit("_"),
            F.col("store_id"),
            F.lit("_"),
            (F.floor(F.rand(418) * 9000) + 1000).cast("int").cast("string")
        )
    )
    .select(
        "order_id","store_id","customer_id","order_timestamp","order_date",
        "channel","order_type","total_amount","tax_amount","tip_amount","discount_amount",
        "payment_method","promotion_codes","delivery_partner",
        "order_prep_time_minutes","customer_wait_time_minutes","order_items"
    )
)

# ======== Quick sanity: count BEFORE write (cheap scalar) ========
planned = grid.agg(F.sum("daily_target")).collect()[0][0]
print(f"Planned transactions (sum of daily_target): {planned:,}")

# ======== Write (overwrite whole table; partition by date) ========
spark.sql(f"DROP TABLE IF EXISTS {target_table}")
transactions_df.write.mode("overwrite").option("overwriteSchema","true") \
    .partitionBy("order_date").saveAsTable(target_table)

# Post-write sanity (metadata count)
post = spark.sql(f"SELECT COUNT(*) AS c FROM {target_table}").first()["c"]
print(f"✓ Wrote {post:,} rows to {target_table}")


# COMMAND ----------

# COMMAND ----------
# MAGIC %md ## Step 4: Generate Transactions

# COMMAND ----------
# --- Ultra-fast parallel transaction generator (Spark-native, no Python loops) ---

from pyspark.sql import functions as F, types as T
from pyspark.sql.window import Window

# ======== Tuning knobs ========
parts = max(512, spark.sparkContext.defaultParallelism * 4)
spark.conf.set("spark.sql.shuffle.partitions", parts)
spark.conf.set("spark.databricks.delta.optimizeWrite", "true")
spark.conf.set("spark.databricks.delta.autoCompact", "true")

target_table = f"{CATALOG}.gold.transactions"

# ======== Inputs (small dims are broadcast) ========
stores_df = spark.read.table(f"{CATALOG}.gold.store_locations") \
    .select("store_id", "kitchen_capacity_score", "zip_code")
if SAMPLE_PCT < 1.0:
    stores_df = stores_df.where(F.xxhash64("store_id") % 100 < int(SAMPLE_PCT * 100))

# Build local store lists per ZIP and broadcast
stores_by_zip_df = (
    stores_df.groupBy("zip_code")
             .agg(F.collect_list(F.col("store_id")).alias("local_stores"))
)
stores_by_zip_df = F.broadcast(stores_by_zip_df)

num_stores = stores_df.count()

customers_df = spark.read.table(f"{CATALOG}.gold.customer_profiles") \
    .select("customer_id", "customer_segment", "app_user", "zip_code")

# Stable row numbers for O(1) random pick without shuffle (also broadcast)
customers_df = customers_df.withColumn(
    "cust_rn",
    F.row_number().over(Window.orderBy(F.xxhash64("customer_id")))
)
cust_count = customers_df.count()
customers_df = F.broadcast(customers_df)

menu_df = spark.read.table(f"{CATALOG}.gold.menu_items") \
    .select("item_id", "item_name", "category", "base_price")

# Build literal arrays of structs per category (menu is tiny -> safe to collect)
def cat_array(cat: str):
    rows = (menu_df.where(F.col("category") == F.lit(cat))
                   .select("item_id", "item_name", "category", "base_price")
                   .collect())
    if not rows:
        return F.array()
    literal_structs = [
        F.struct(
            F.lit(r["item_id"]).alias("item_id"),
            F.lit(r["item_name"]).alias("item_name"),
            F.lit(r["category"]).alias("category"),
            F.lit(float(r["base_price"])).alias("base_price")
        )
        for r in (row.asDict() for row in rows)
    ]
    return F.array(*literal_structs)

arr_entree   = cat_array("entree")
arr_protein  = cat_array("protein")
arr_side     = cat_array("side")
arr_beverage = cat_array("beverage")

# ======== Factors as literal maps (kept for possible later use) ========
seasonal_map = F.create_map(
    *[x for kv in {1: 0.92, 2: 0.94, 3: 0.98, 4: 1.02, 5: 1.06, 6: 1.08, 7: 1.12, 8: 1.10, 9: 1.08, 10: 1.02, 11: 0.98, 12: 0.95}.items()
      for x in (F.lit(kv[0]), F.lit(kv[1]))]
)
# Spark dayofweek: 1=Sun ... 7=Sat
weekday_map = F.create_map(
    *[x for kv in {2: 0.85, 3: 0.90, 4: 0.95, 5: 1.05, 6: 1.15, 7: 1.20, 1: 1.10}.items()
      for x in (F.lit(kv[0]), F.lit(kv[1]))]
)

# ======== Generate transactions, assign customers, and link to local stores ========
days_in_period = (END_DATE - START_DATE).days + 1  # inclusive
daily_transactions_per_store = 150
total_transactions = int(num_stores * days_in_period * daily_transactions_per_store * SAMPLE_PCT)
print(f"Generating a total of ~{total_transactions:,} transactions")

tx = spark.range(0, total_transactions).repartition(parts)

# Stable per-row hash to drive deterministic choices
tx = tx.withColumn("hash_id", F.xxhash64("id"))

# Randomly assign a customer and bring their zip
tx = tx.withColumn(
    "cust_pick",
    (F.pmod(F.abs(F.col("hash_id")), F.lit(cust_count)) + F.lit(1)).cast("long")
).join(
    customers_df.select("customer_id", "app_user", "zip_code", "cust_rn"),
    on=(F.col("cust_pick") == F.col("cust_rn")),
    how="left"
).drop("cust_pick", "cust_rn")

# Join local stores by customer's ZIP; guarantee array typing
tx = tx.join(stores_by_zip_df, on="zip_code", how="left")
tx = tx.withColumn(
    "local_stores_safe",
    F.coalesce(F.col("local_stores"), F.array().cast("array<string>"))
)

# Deterministic index into local_stores_safe (no column-seeded rand)
idx = (
    F.pmod(F.abs(F.xxhash64(F.col("hash_id"))),
           F.greatest(F.size(F.col("local_stores_safe")), F.lit(1)))
    + F.lit(1)
).cast("int")

tx = tx.withColumn(
    "store_id",
    F.when(
        F.size(F.col("local_stores_safe")) > 0,
        F.element_at(F.col("local_stores_safe"), idx)
    ).otherwise(F.lit(None).cast("string"))
).drop("local_stores", "local_stores_safe", "zip_code")

# Assign an order timestamp uniformly across the date range (proper timestamp math)
date_range_sec = days_in_period * 24 * 60 * 60
tx = tx.withColumn(
    "order_timestamp",
    F.expr(
        f"timestampadd(SECOND, CAST(pmod(abs(xxhash64(hash_id)), {date_range_sec}) AS INT), to_timestamp('{START_DATE:%Y-%m-%d}'))"
    )
)
tx = tx.withColumn("order_date", F.to_date("order_timestamp"))

# Deterministic daypart draw in [0,1)
daypart_rand = (
    F.pmod(F.abs(F.xxhash64(F.col("hash_id"), F.lit(12345))), F.lit(10_000_000))
    / F.lit(10_000_000.0)
)
tx = tx.withColumn("daypart_rand", daypart_rand)
tx = tx.withColumn(
    "daypart",
    F.when(F.col("daypart_rand") < 0.08, F.lit("breakfast"))
     .when(F.col("daypart_rand") < 0.08 + 0.45, F.lit("lunch"))
     .when(F.col("daypart_rand") < 0.08 + 0.45 + 0.35, F.lit("dinner"))
     .otherwise(F.lit("late_night"))
).drop("daypart_rand")

# ======== Channel / order_type (vectorized) ========
channel = F.when(F.col("app_user") & (F.rand(301) < 0.6),
                 F.when(F.rand(302) < 0.5, F.lit("app")).otherwise(F.lit("web"))) \
           .otherwise(F.lit("in_store"))
order_type = F.when(channel.isin("app", "web"),
                    F.when(F.rand(303) < 0.7, F.lit("pickup")).otherwise(F.lit("delivery"))) \
              .otherwise(F.lit("dine_in"))

# ======== Helpers to pick random struct from an array ========
def pick_random(arr_expr, seed):
    sz = F.size(arr_expr)
    idxp1 = (F.floor(F.rand(seed) * sz) + 1).cast("int")
    return F.element_at(arr_expr, idxp1)

entree  = pick_random(arr_entree,   401)
protein = pick_random(arr_protein,  402)
side    = pick_random(arr_side,     403)
bev     = pick_random(arr_beverage, 404)

# ======== Modifications & item structs (consistent schema) ========
mods_pool = F.array(
    F.lit("extra_rice"), F.lit("extra_beans"), F.lit("extra_cheese"),
    F.lit("no_rice"), F.lit("no_beans"), F.lit("light_cheese")
)
mods_k  = (F.floor(F.rand(405) * 3)).cast("int")  # 0..2
mods    = F.when(mods_k > 0, F.slice(F.shuffle(mods_pool), 1, mods_k)) \
           .otherwise(F.array().cast("array<string>"))

entree_plus = F.struct(
    entree["item_id"].alias("item_id"),
    entree["item_name"].alias("item_name"),
    entree["category"].alias("category"),
    (entree["base_price"].cast("double") + protein["base_price"].cast("double")).alias("base_price"),
    mods.alias("modifications"),
    F.lit(1).alias("quantity")
)

empty_mods = F.array().cast("array<string>")

side_struct = F.struct(
    side["item_id"].alias("item_id"),
    side["item_name"].alias("item_name"),
    side["category"].alias("category"),
    side["base_price"].cast("double").alias("base_price"),
    empty_mods.alias("modifications"),
    F.lit(1).alias("quantity")
)

bev_struct = F.struct(
    bev["item_id"].alias("item_id"),
    bev["item_name"].alias("item_name"),
    bev["category"].alias("category"),
    bev["base_price"].cast("double").alias("base_price"),
    empty_mods.alias("modifications"),
    F.lit(1).alias("quantity")
)

order_items = F.array(entree_plus)
order_items = F.when(F.rand(406) < 0.3, F.concat(order_items, F.array(side_struct))).otherwise(order_items)
order_items = F.when(F.rand(407) < 0.4, F.concat(order_items, F.array(bev_struct))).otherwise(order_items)

# ======== Monetary fields ========
subtotal = F.aggregate(order_items, F.lit(0.0), lambda acc, x: acc + (x["base_price"] * x["quantity"]))
tax = (subtotal * F.lit(0.0875)).cast("double")

tip = F.when(F.col("order_type") == F.lit("delivery"), subtotal * (F.rand(408) * F.lit(0.10) + F.lit(0.10))) \
       .when((F.col("order_type") == F.lit("pickup")) & (F.rand(409) < F.lit(0.15)), subtotal * (F.rand(410) * F.lit(0.10) + F.lit(0.05))) \
       .otherwise(F.lit(0.0))
tip = tip.cast("double")

discount = F.when(F.rand(411) < F.lit(0.10), subtotal * (F.rand(412) * F.lit(0.15) + F.lit(0.10))).otherwise(F.lit(0.0)).cast("double")

promo_codes = F.when(
    discount > 0,
    F.array(
        F.element_at(
            F.array(F.lit("BOGO50"), F.lit("SAVE20"), F.lit("FREECHIPS"), F.lit("STUDENT15")),
            (F.floor(F.rand(413) * 4) + 1).cast("int")
        )
    )
).otherwise(F.array().cast("array<string>"))

total_amount = (subtotal + tax + tip - discount).cast("double")

delivery_partner_choices = F.array(
    F.lit("uber_eats"), F.lit("doordash"), F.lit("grubhub"), F.lit(None).cast("string")
)
delivery_partner = F.when(
    F.col("order_type") == F.lit("delivery"),
    F.element_at(delivery_partner_choices, (F.floor(F.rand(414) * 4) + 1).cast("int"))
).otherwise(F.lit(None).cast("string"))

# ======== Assemble final DF ========
transactions_df = (
    tx
    .withColumn("channel", channel)
    .withColumn("order_type", order_type)
    .withColumn("order_items", order_items)
    .withColumn("tax_amount", tax)
    .withColumn("tip_amount", tip)
    .withColumn("discount_amount", discount)
    .withColumn("total_amount", total_amount)
    .withColumn(
        "payment_method",
        F.element_at(
            F.array(
                F.lit("credit_card"), F.lit("debit_card"),
                F.lit("apple_pay"), F.lit("google_pay"), F.lit("cash")
            ),
            (F.floor(F.rand(415) * 5) + 1).cast("int")
        )
    )
    .withColumn("promotion_codes", promo_codes)
    .withColumn("delivery_partner", delivery_partner)
    .withColumn("order_prep_time_minutes", (F.floor(F.rand(416) * 11) + 5).cast("int"))
    .withColumn("customer_wait_time_minutes", (F.floor(F.rand(417) * 19) + 2).cast("int"))
    .withColumn("order_id",
        F.concat(
            F.lit("ORD_"),
            F.date_format("order_date", "yyyyMMdd"),
            F.lit("_"),
            F.col("store_id"),
            F.lit("_"),
            (F.floor(F.rand(418) * 9000) + 1000).cast("int").cast("string")
        )
    )
    .select(
        "order_id", "store_id", "customer_id", "order_timestamp", "order_date",
        "channel", "order_type", "total_amount", "tax_amount", "tip_amount", "discount_amount",
        "payment_method", "promotion_codes", "delivery_partner",
        "order_prep_time_minutes", "customer_wait_time_minutes", "order_items"
    )
)

# ======== Write (partitioned by date) ========
transactions_df.write.mode("overwrite").option("overwriteSchema", "true") \

print("✓ Transaction generation complete (parallel + Spark-native)")


# COMMAND ----------

# MAGIC %md ## Step 5: Generate Daily Store Performance

# COMMAND ----------

# Fast, parallel daily store performance builder (Spark-native)

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ---------- Inputs ----------
transactions_tbl = f"{CATALOG}.gold.transactions"
stores_tbl       = f"{CATALOG}.gold.store_locations"
target_tbl       = f"{CATALOG}.gold.daily_store_performance"

# Optional: respect SAMPLE_PCT if you’re sampling stores globally
stores_df = spark.read.table(stores_tbl).select(
    "store_id", "seating_capacity", "store_format", "kitchen_capacity_score"
)
if "SAMPLE_PCT" in globals() and SAMPLE_PCT < 1.0:
    stores_df = stores_df.where(F.xxhash64("store_id") % 100 < int(SAMPLE_PCT * 100))

# ---------- Date × Store grid ----------
date_df = spark.sql(f"""
  SELECT explode(sequence(
      to_date('{START_DATE:%Y-%m-%d}'),
      to_date('{END_DATE:%Y-%m-%d}'),
      interval 1 day
  )) AS business_date
""")

date_store_df = date_df.crossJoin(stores_df.select("store_id"))

# ---------- Aggregate transactions ----------
# Note: Spark day types — ensure order_date is DATE in the source table
daily_metrics_df = (
    spark.read.table(transactions_tbl)
         .groupBy("store_id", F.col("order_date").alias("business_date"))
         .agg(
             F.count("order_id").alias("transaction_count"),
             F.sum("total_amount").alias("total_revenue"),
             F.avg("total_amount").alias("average_ticket"),
             F.sum(F.when(F.col("channel").isin("app", "web"), F.col("total_amount")).otherwise(F.lit(0.0))).alias("digital_revenue"),
             F.sum(F.when(F.col("channel") == "in_store", F.col("total_amount")).otherwise(F.lit(0.0))).alias("in_store_revenue"),
             F.sum(F.when(F.col("order_type") == "delivery", F.col("total_amount")).otherwise(F.lit(0.0))).alias("delivery_revenue"),
             F.countDistinct("customer_id").alias("unique_customers"),
         )
)

# ---------- Join to complete the grid & fill missing with zeros ----------
daily_perf_df = (
    date_store_df.join(daily_metrics_df, ["business_date", "store_id"], "left")
                 .join(stores_df, "store_id", "left")
)

# Fill numeric nulls with 0, but leave averages alone (handle them via coalesce)
daily_perf_df = (daily_perf_df
    .withColumn("transaction_count", F.coalesce("transaction_count", F.lit(0)))
    .withColumn("total_revenue",     F.coalesce("total_revenue", F.lit(0.0)))
    .withColumn("digital_revenue",   F.coalesce("digital_revenue", F.lit(0.0)))
    .withColumn("in_store_revenue",  F.coalesce("in_store_revenue", F.lit(0.0)))
    .withColumn("delivery_revenue",  F.coalesce("delivery_revenue", F.lit(0.0)))
    .withColumn("unique_customers",  F.coalesce("unique_customers", F.lit(0)))
    .withColumn("average_ticket",
        F.when(F.col("transaction_count") > 0,
               F.coalesce("average_ticket", F.col("total_revenue") / F.col("transaction_count"))
        ).otherwise(F.lit(0.0))
    )
)

# ---------- Derived/simulated KPIs (Spark expressions only) ----------
# revenue_per_seat: total_revenue / seating_capacity (guard divide-by-zero)
daily_perf_df = daily_perf_df.withColumn(
    "revenue_per_seat",
    F.when(F.col("seating_capacity") > 0, F.col("total_revenue") / F.col("seating_capacity")).otherwise(F.lit(0.0))
)

# Random-like operational metrics (bounded)
daily_perf_df = (daily_perf_df
    .withColumn("avg_service_time",        (F.rand(101) * F.lit(4.5) + F.lit(3.5)))   # 3.5 .. 8.0
    .withColumn("staff_hours_scheduled",   (F.rand(102) * F.lit(40.0) + F.lit(80.0))) # 80 .. 120
    .withColumn("staff_hours_actual",      (F.rand(103) * F.lit(50.0) + F.lit(75.0))) # 75 .. 125
    .withColumn("food_cost_pct",           (F.rand(104) * F.lit(0.07) + F.lit(0.28))) # 28% .. 35%
    .withColumn("waste_amount",            (F.rand(105) * F.lit(150.0) + F.lit(50.0)))# 50 .. 200
)

# Simulated customer splits (ensure bounds)
new_cand = (F.when(F.col("unique_customers") > 0, (F.rand(106) * F.lit(20.0) + F.lit(5.0))).otherwise(F.lit(0.0)))
new_customers = F.least(new_cand, F.col("unique_customers").cast("double")).cast(IntegerType())
returning_customers = (F.col("unique_customers") - new_customers).cast(IntegerType())

daily_perf_df = (daily_perf_df
    .withColumn("new_customers",       new_customers)
    .withColumn("returning_customers", returning_customers)
    .withColumn("loyalty_redemptions",
        F.when(F.col("unique_customers") > 0, (F.rand(107) * F.lit(40.0) + F.lit(10.0))).otherwise(F.lit(0.0))
         .cast(IntegerType())
    )
    .withColumn("avg_satisfaction_score", (F.rand(108) * F.lit(1.0) + F.lit(3.8)))  # 3.8 .. 4.8
)

# Weather & events (typed arrays/strings, no Python random/np)
weather_choices = F.array(F.lit("sunny"), F.lit("cloudy"), F.lit("rainy"), F.lit("snowy"))
weather_idx = (F.floor(F.rand(201) * F.size(weather_choices)) + F.lit(1)).cast("int")
daily_perf_df = daily_perf_df.withColumn("weather_condition", F.element_at(weather_choices, weather_idx))

daily_perf_df = daily_perf_df.withColumn(
    "temperature_high", (F.floor(F.rand(202) * F.lit(51)) + F.lit(40)).cast("int")  # 40..90
)

daily_perf_df = daily_perf_df.withColumn(
    "precipitation_inches",
    F.when(F.col("weather_condition") == F.lit("rainy"), (F.rand(203) * F.lit(2.0))).otherwise(F.lit(0.0))
)

event_choices = F.array(F.lit("sports_game"), F.lit("concert"), F.lit("festival"))
daily_perf_df = daily_perf_df.withColumn(
    "local_events",
    F.when(F.rand(204) > F.lit(0.9),
           F.array(F.element_at(event_choices, (F.floor(F.rand(205) * F.lit(3)) + F.lit(1)).cast("int")))
    ).otherwise(F.array().cast("array<string>"))
)

# ---------- Final projection ----------
final_df = daily_perf_df.select(
    "store_id",
    "business_date",
    "total_revenue",
    "transaction_count",
    "average_ticket",
    "revenue_per_seat",
    "in_store_revenue",
    "digital_revenue",
    "delivery_revenue",
    "avg_service_time",
    "staff_hours_scheduled",
    "staff_hours_actual",
    "food_cost_pct",
    "waste_amount",
    "new_customers",
    "returning_customers",
    "loyalty_redemptions",
    "avg_satisfaction_score",
    "weather_condition",
    "temperature_high",
    "precipitation_inches",
    "local_events"
)

# ---------- Write ----------
spark.sql(f"DROP TABLE IF EXISTS {target_tbl}")
final_df.write.mode("overwrite").saveAsTable(target_tbl)

print(f"✓ Generated {final_df.count()} daily performance records -> {target_tbl}")


# COMMAND ----------

# MAGIC %md ## Step 6: Verify Data Generation

# COMMAND ----------

# Data quality checks
checks = []

# Check row counts
for table in ['store_locations', 'customer_profiles', 'menu_items', 'transactions', 'daily_store_performance']:
    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {CATALOG}.gold.{table}").collect()[0]['cnt']
    checks.append({'table': table, 'row_count': count})

# Check transactions date range
date_range = spark.sql(f"""
    SELECT 
        MIN(order_date) as min_date,
        MAX(order_date) as max_date,
        COUNT(DISTINCT store_id) as unique_stores,
        COUNT(DISTINCT customer_id) as unique_customers
    FROM {CATALOG}.gold.transactions
""").collect()[0]

checks.append({'metric': 'date_range', 'min': str(date_range['min_date']), 'max': str(date_range['max_date'])})
checks.append({'metric': 'unique_stores', 'value': date_range['unique_stores']})
checks.append({'metric': 'unique_customers', 'value': date_range['unique_customers']})

# Display summary
display(spark.createDataFrame(checks))

print(f"\n✓ Synthetic data generation complete")
print(f"  - Sample percentage: {int(SAMPLE_PCT * 100)}%")
print(f"  - Date range: {START_DATE.date()} to {END_DATE.date()}")
print(f"  - Catalog: {CATALOG}")

# COMMAND ----------

print("SAMPLE_PCT =", SAMPLE_PCT)
print("START_DATE =", START_DATE, " END_DATE =", END_DATE)

days_in_period = (END_DATE - START_DATE).days + 1
num_stores = spark.read.table(f"{CATALOG}.gold.store_locations").count()
print("days_in_period =", days_in_period, "num_stores =", num_stores)

expected = int(num_stores * days_in_period * 150 * SAMPLE_PCT)
print("expected_total_transactions =", f"{expected:,}")


# COMMAND ----------

# Optimize tables for better performance
for table in ['transactions', 'daily_store_performance']:
    spark.sql(f"OPTIMIZE {CATALOG}.gold.{table}")
    print(f"✓ Optimized {table}")
