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

# Customer segment definitions
segments_config = {
    'power_user': {'pct': 0.15, 'visits': 52, 'aov': 14.50, 'digital': 0.90},
    'loyal_regular': {'pct': 0.25, 'visits': 26, 'aov': 11.75, 'digital': 0.70},
    'occasional': {'pct': 0.45, 'visits': 12, 'aov': 10.25, 'digital': 0.45},
    'price_sensitive': {'pct': 0.15, 'visits': 6, 'aov': 8.75, 'digital': 0.35}
}

# Calculate total customers needed (approximate based on visits)
avg_visits_per_customer = sum(s['pct'] * s['visits'] for s in segments_config.values())
days_in_period = (END_DATE - START_DATE).days
num_stores = len(stores_data)
daily_transactions_per_store = 150  # Average
total_transactions = num_stores * days_in_period * daily_transactions_per_store * SAMPLE_PCT
total_customers = int(total_transactions / avg_visits_per_customer)

print(f"Generating ~{total_customers:,} customers for ~{int(total_transactions):,} transactions")

customers_data = []
customer_counter = 1

for segment, config in segments_config.items():
    segment_size = int(total_customers * config['pct'])
    
    for _ in range(segment_size):
        customer_id = f"CUST_{customer_counter:08d}"
        
        # Age and income correlations
        if segment == 'power_user':
            age_range = np.random.choice(['25-34', '35-44', '45-54'], p=[0.40, 0.35, 0.25])
            income = np.random.choice(['75k_100k', 'over_100k'], p=[0.40, 0.60])
            loyalty = np.random.choice(['gold', 'platinum'], p=[0.40, 0.60])
        elif segment == 'loyal_regular':
            age_range = np.random.choice(['25-34', '35-44', '45-54'], p=[0.35, 0.40, 0.25])
            income = np.random.choice(['50k_75k', '75k_100k', 'over_100k'], p=[0.30, 0.40, 0.30])
            loyalty = np.random.choice(['silver', 'gold'], p=[0.60, 0.40])
        elif segment == 'occasional':
            age_range = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], p=[0.25, 0.30, 0.25, 0.15, 0.05])
            income = np.random.choice(['under_50k', '50k_75k', '75k_100k'], p=[0.35, 0.40, 0.25])
            loyalty = np.random.choice(['bronze', 'silver'], p=[0.70, 0.30])
        else:  # price_sensitive
            age_range = np.random.choice(['18-24', '25-34', '35-44'], p=[0.40, 0.35, 0.25])
            income = np.random.choice(['under_50k', '50k_75k'], p=[0.60, 0.40])
            loyalty = 'bronze'
        
        # Lifestyle correlations
        if income in ['75k_100k', 'over_100k']:
            lifestyle = np.random.choice(['health_conscious', 'convenience', 'premium'], p=[0.40, 0.40, 0.20])
        else:
            lifestyle = np.random.choice(['value_seeker', 'convenience'], p=[0.60, 0.40])
        
        customers_data.append({
            'customer_id': customer_id,
            'registration_date': fake.date_between(start_date=START_DATE, end_date=END_DATE),
            'registration_channel': np.random.choice(['app', 'web', 'in_store'], p=[0.45, 0.30, 0.25]),
            'age_range': age_range,
            'income_bracket': income,
            'household_size': int(random.randint(1, 4)),
            'zip_code': random.choice(stores_data)['zip_code'],  # Near a store
            'lifestyle': lifestyle,
            'loyalty_tier': loyalty,
            'points_balance': int(random.randint(0, 5000)),
            'lifetime_spend': float(config['aov'] * config['visits'] * random.uniform(0.8, 2.5)),
            'visit_frequency': {52: 'weekly', 26: 'biweekly', 12: 'monthly', 6: 'occasional'}[config['visits']],
            'avg_order_value': float(config['aov'] * random.uniform(0.9, 1.1)),
            'preferred_proteins': random.sample(['chicken', 'steak', 'carnitas', 'barbacoa', 'sofritas'], k=random.randint(1, 3)),
            'dietary_preferences': random.sample(['none', 'vegetarian', 'keto', 'high_protein', 'low_sodium'], k=random.randint(0, 2)),
            'app_user': random.random() < config['digital'],
            'email_subscriber': random.random() < 0.6,
            'push_notifications': random.random() < config['digital'] * 0.7,
            'social_media_follower': random.random() < 0.3,
            'referrals_made': int(np.random.poisson(1)),
            'churn_risk_score': float(0.1) if segment == 'power_user' else (float(0.7) if segment == 'price_sensitive' else float(random.uniform(0.2, 0.5))),
            'customer_segment': segment
        })
        
        customer_counter += 1

# Create DataFrame and save
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, DateType, ArrayType
schema = StructType([
    StructField('customer_id', StringType(), True),
    StructField('registration_date', DateType(), True),
    StructField('registration_channel', StringType(), True),
    StructField('age_range', StringType(), True),
    StructField('income_bracket', StringType(), True),
    StructField('household_size', IntegerType(), True),
    StructField('zip_code', StringType(), True),
    StructField('lifestyle', StringType(), True),
    StructField('loyalty_tier', StringType(), True),
    StructField('points_balance', IntegerType(), True),
    StructField('lifetime_spend', DoubleType(), True),
    StructField('visit_frequency', StringType(), True),
    StructField('avg_order_value', DoubleType(), True),
    StructField('preferred_proteins', ArrayType(StringType()), True),
    StructField('dietary_preferences', ArrayType(StringType()), True),
    StructField('app_user', BooleanType(), True),
    StructField('email_subscriber', BooleanType(), True),
    StructField('push_notifications', BooleanType(), True),
    StructField('social_media_follower', BooleanType(), True),
    StructField('referrals_made', IntegerType(), True),
    StructField('churn_risk_score', DoubleType(), True),
    StructField('customer_segment', StringType(), True)
])

customers_df = spark.createDataFrame(customers_data, schema=schema)
customers_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.gold.customer_profiles")
print(f"✓ Created {customers_df.count()} customers")

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

# Transaction generation parameters
seasonal_factors = {
    1: 0.92, 2: 0.94, 3: 0.98,  # Q1 - post-holiday slowdown
    4: 1.02, 5: 1.06, 6: 1.08,  # Q2 - spring pickup
    7: 1.12, 8: 1.10, 9: 1.08,  # Q3 - summer peak
    10: 1.02, 11: 0.98, 12: 0.95  # Q4 - holiday impact
}

weekday_factors = {
    0: 0.85,  # Monday
    1: 0.90,  # Tuesday  
    2: 0.95,  # Wednesday
    3: 1.05,  # Thursday
    4: 1.15,  # Friday
    5: 1.20,  # Saturday
    6: 1.10   # Sunday
}

daypart_distribution = {
    'breakfast': (7, 10, 0.08),
    'lunch': (11, 14, 0.45),
    'dinner': (17, 20, 0.35),
    'late_night': (20, 22, 0.12)
}

# Build transactions date by date (chunked for memory efficiency)
chunk_size = 7  # Process a week at a time
current_date = START_DATE
transactions_to_insert = []

# Explicitly load customer and menu data for use in this step
customers_data = spark.read.table(f"{CATALOG}.gold.customer_profiles").collect()
menu_items = spark.read.table(f"{CATALOG}.gold.menu_items").collect()
stores_data = spark.read.table(f"{CATALOG}.gold.store_locations").collect()


while current_date <= END_DATE:
    chunk_end = min(current_date + timedelta(days=chunk_size), END_DATE)
    
    for store in stores_data[:int(len(stores_data) * SAMPLE_PCT)]:
        while current_date <= chunk_end:
            # Calculate daily transaction count
            base_transactions = 150  # Base daily average
            
            # Apply factors
            seasonal = seasonal_factors.get(current_date.month, 1.0)
            weekday = weekday_factors.get(current_date.weekday(), 1.0)
            
            # Store performance tier adjustment
            if store['kitchen_capacity_score'] >= 8:
                store_factor = 1.3
            elif store['kitchen_capacity_score'] >= 6:
                store_factor = 1.0
            else:
                store_factor = 0.75
            
            daily_transactions = int(base_transactions * seasonal * weekday * store_factor * random.uniform(0.8, 1.2))
            
            # Generate transactions for each daypart
            for daypart, (start_hour, end_hour, daypart_pct) in daypart_distribution.items():
                daypart_transactions = int(daily_transactions * daypart_pct)
                
                for _ in range(daypart_transactions):
                    # Select random customer (weighted by segment)
                    customer = random.choice(customers_data)
                    
                    # Build order
                    order_items = []
                    
                    # Main entree
                    entree = random.choice([m for m in menu_items if m['category'] == 'entree'])
                    protein = random.choice([m for m in menu_items if m['category'] == 'protein'])
                    
                    # Order customizations
                    modifications = random.sample(['extra_rice', 'extra_beans', 'extra_cheese', 
                                                   'no_rice', 'no_beans', 'light_cheese'], 
                                                  k=random.randint(0, 2))
                    
                    order_items.append({
                        'item_id': entree['item_id'],
                        'item_name': entree['item_name'],
                        'category': entree['category'],
                        'base_price': float(entree['base_price'] + protein['base_price']),
                        'modifications': modifications,
                        'quantity': 1
                    })
                    
                    # Add sides (30% chance)
                    if random.random() < 0.3:
                        side = random.choice([m for m in menu_items if m['category'] == 'side'])
                        order_items.append({
                            'item_id': side['item_id'],
                            'item_name': side['item_name'],
                            'category': side['category'],
                            'base_price': float(side['base_price']),
                            'modifications': [],
                            'quantity': 1
                        })
                    
                    # Add beverage (40% chance)
                    if random.random() < 0.4:
                        beverage = random.choice([m for m in menu_items if m['category'] == 'beverage'])
                        order_items.append({
                            'item_id': beverage['item_id'],
                            'item_name': beverage['item_name'],
                            'category': beverage['category'],
                            'base_price': float(beverage['base_price']),
                            'modifications': [],
                            'quantity': 1
                        })
                    
                    # Calculate totals
                    subtotal = sum(item['base_price'] * item['quantity'] for item in order_items)
                    tax = float(subtotal * 0.0875)  # 8.75% tax
                    
                    # Determine channel based on customer digital adoption
                    if customer['app_user'] and random.random() < 0.6:
                        channel = random.choice(['app', 'web'])
                        order_type = random.choices(['pickup', 'delivery'], weights=[0.7, 0.3])[0]
                    else:
                        channel = 'in_store'
                        order_type = 'dine_in'
                    
                    # Tips for delivery/pickup
                    tip = float(0)
                    if order_type == 'delivery':
                        tip = float(subtotal * random.uniform(0.10, 0.20))
                    elif order_type == 'pickup' and random.random() < 0.15:
                        tip = float(subtotal * random.uniform(0.05, 0.15))
                    
                    # Apply promotions (10% chance)
                    discount = float(0)
                    promo_codes = []
                    if random.random() < 0.10:
                        discount = float(subtotal * random.uniform(0.10, 0.25))
                        promo_codes = [random.choice(['BOGO50', 'SAVE20', 'FREECHIPS', 'STUDENT15'])]
                    
                    # Create transaction
                    order_time = current_date.replace(
                        hour=random.randint(start_hour, end_hour-1),
                        minute=random.randint(0, 59)
                    )
                    
                    transactions_to_insert.append({
                        'order_id': f"ORD_{current_date.strftime('%Y%m%d')}_{store['store_id']}_{random.randint(1000, 9999)}",
                        'store_id': store['store_id'],
                        'customer_id': customer['customer_id'],
                        'order_timestamp': order_time,
                        'order_date': current_date.date(),
                        'channel': channel,
                        'order_type': order_type,
                        'total_amount': float(subtotal + tax + tip - discount),
                        'tax_amount': tax,
                        'tip_amount': tip,
                        'discount_amount': discount,
                        'payment_method': random.choice(['credit_card', 'debit_card', 'apple_pay', 'google_pay', 'cash']),
                        'promotion_codes': promo_codes,
                        'delivery_partner': random.choice(['uber_eats', 'doordash', 'grubhub', None]) if order_type == 'delivery' else None,
                        'order_prep_time_minutes': int(random.randint(5, 15)),
                        'customer_wait_time_minutes': int(random.randint(2, 20)),
                        'order_items': order_items
                    })
            
            current_date += timedelta(days=1)
        
    
    # Define explicit schema to avoid inference errors
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, DateType, ArrayType
    
    items_schema = ArrayType(
        StructType([
            StructField("item_id", StringType(), True),
            StructField("item_name", StringType(), True),
            StructField("category", StringType(), True),
            StructField("base_price", DoubleType(), True),
            StructField("modifications", ArrayType(StringType()), True),
            StructField("quantity", IntegerType(), True)
        ])
    )
    
    transactions_schema = StructType([
        StructField("order_id", StringType(), True),
        StructField("store_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("order_timestamp", TimestampType(), True),
        StructField("order_date", DateType(), True),
        StructField("channel", StringType(), True),
        StructField("order_type", StringType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("tax_amount", DoubleType(), True),
        StructField("tip_amount", DoubleType(), True),
        StructField("discount_amount", DoubleType(), True),
        StructField("payment_method", StringType(), True),
        StructField("promotion_codes", ArrayType(StringType()), True),
        StructField("delivery_partner", StringType(), True),
        StructField("order_prep_time_minutes", IntegerType(), True),
        StructField("customer_wait_time_minutes", IntegerType(), True),
        StructField("order_items", items_schema, True)
    ])
    
    # Insert chunk with explicit schema
    if transactions_to_insert:
        trans_df = spark.createDataFrame(transactions_to_insert, schema=transactions_schema)
        trans_df.write.mode("append").saveAsTable(f"{CATALOG}.gold.transactions")
        print(f"✓ Inserted {len(transactions_to_insert)} transactions up to {chunk_end.date()}")
        transactions_to_insert = []
    
    current_date = chunk_end + timedelta(days=1)

print(f"✓ Transaction generation complete")

# COMMAND ----------

# MAGIC %md ## Step 5: Generate Daily Store Performance

# COMMAND ----------

# Import necessary functions and types
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType, ArrayType
from datetime import timedelta, date
import random
import numpy as np

# Create a list of all dates and stores
start_date_obj = date.fromisoformat(START_DATE.strftime('%Y-%m-%d'))
end_date_obj = date.fromisoformat(END_DATE.strftime('%Y-%m-%d'))
all_dates = [start_date_obj + timedelta(days=x) for x in range((end_date_obj - start_date_obj).days + 1)]

date_store_pairs = []
for d in all_dates:
    for store in stores_data:
        date_store_pairs.append((d, store['store_id']))

# Create a DataFrame from the pairs to join with transactions
date_store_df = spark.createDataFrame(date_store_pairs, ["business_date", "store_id"])

# Aggregate transactions data in a single, efficient query
daily_metrics_df = spark.sql(f"""
    SELECT
        store_id,
        order_date AS business_date,
        COUNT(order_id) AS transaction_count,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS average_ticket,
        SUM(CASE WHEN channel IN ('app', 'web') THEN total_amount ELSE 0.0 END) AS digital_revenue,
        SUM(CASE WHEN channel = 'in_store' THEN total_amount ELSE 0.0 END) AS in_store_revenue,
        SUM(CASE WHEN order_type = 'delivery' THEN total_amount ELSE 0.0 END) AS delivery_revenue,
        COUNT(DISTINCT customer_id) AS unique_customers
    FROM {CATALOG}.gold.transactions
    GROUP BY store_id, order_date
""")

# Join with stores data and fill in missing dates/stores with zeros
daily_perf_df = date_store_df.join(
    daily_metrics_df,
    ["business_date", "store_id"],
    "left"
).fillna(0)

# Load store attributes to add to the daily performance DataFrame
store_attrs_df = stores_df.select(
    "store_id", "seating_capacity", "store_format", "kitchen_capacity_score"
)

# Join the aggregated data with store attributes
daily_perf_df = daily_perf_df.join(store_attrs_df, "store_id", "left")

# Add simulated metrics as UDFs or Spark functions for efficiency
@F.udf(returnType=DoubleType())
def calculate_revenue_per_seat(revenue, seating_capacity):
    return float(revenue) / float(seating_capacity) if seating_capacity > 0 else 0.0

daily_perf_df = daily_perf_df.withColumn("revenue_per_seat", calculate_revenue_per_seat(F.col("total_revenue"), F.col("seating_capacity")))
daily_perf_df = daily_perf_df.withColumn("avg_service_time", F.rand() * 4.5 + 3.5)
daily_perf_df = daily_perf_df.withColumn("staff_hours_scheduled", F.rand() * 40 + 80)
daily_perf_df = daily_perf_df.withColumn("staff_hours_actual", F.rand() * 50 + 75)
daily_perf_df = daily_perf_df.withColumn("food_cost_pct", F.rand() * 0.07 + 0.28)
daily_perf_df = daily_perf_df.withColumn("waste_amount", F.rand() * 150 + 50)
daily_perf_df = daily_perf_df.withColumn("new_customers", F.when(F.col("unique_customers") > 0, F.rand() * 20 + 5).otherwise(0).cast(IntegerType()))
daily_perf_df = daily_perf_df.withColumn("returning_customers", F.when(F.col("unique_customers") > 0, F.col("unique_customers") - F.col("new_customers")).otherwise(0).cast(IntegerType()))
daily_perf_df = daily_perf_df.withColumn("loyalty_redemptions", F.when(F.col("unique_customers") > 0, F.rand() * 40 + 10).otherwise(0).cast(IntegerType()))
daily_perf_df = daily_perf_df.withColumn("avg_satisfaction_score", F.rand() * 1.0 + 3.8)
daily_perf_df = daily_perf_df.withColumn("weather_condition", F.lit(np.random.choice(['sunny', 'cloudy', 'rainy', 'snowy'])))
daily_perf_df = daily_perf_df.withColumn("temperature_high", F.lit(random.randint(40, 90)))
daily_perf_df = daily_perf_df.withColumn("precipitation_inches", F.when(F.col("weather_condition") == 'rainy', F.rand() * 2.0).otherwise(0.0))
daily_perf_df = daily_perf_df.withColumn("local_events", F.when(F.rand() > 0.9, F.array(F.lit(random.choice(['sports_game', 'concert', 'festival'])))).otherwise(F.array()))

# Ensure columns are in the correct order for the schema
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

# Drop the table and save
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.gold.daily_store_performance")
final_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.gold.daily_store_performance")

print(f"✓ Generated {final_df.count()} daily performance records")

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

# Optimize tables for better performance
for table in ['transactions', 'daily_store_performance']:
    spark.sql(f"OPTIMIZE {CATALOG}.gold.{table}")
    print(f"✓ Optimized {table}")
