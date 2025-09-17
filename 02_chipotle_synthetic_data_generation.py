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
            seating = random.randint(15, 35)
            parking = random.randint(0, 10)
            pop_3mi = random.randint(150000, 500000)
            income_3mi = random.randint(55000, 95000)
        elif trade_area == 'suburban':
            seating = random.randint(40, 70)
            parking = random.randint(20, 50)
            pop_3mi = random.randint(50000, 150000)
            income_3mi = random.randint(65000, 120000)
        elif trade_area == 'university':
            seating = random.randint(30, 50)
            parking = random.randint(5, 20)
            pop_3mi = random.randint(30000, 100000)
            income_3mi = random.randint(35000, 65000)
        else:  # mall
            seating = random.randint(25, 45)
            parking = 0
            pop_3mi = random.randint(80000, 200000)
            income_3mi = random.randint(55000, 85000)
        
        store_format = random.choices(['standard', 'drive_thru', 'urban_compact'],
                                     weights=[0.70, 0.20, 0.10])[0]
        
        stores_data.append({
            'store_id': store_id,
            'store_number': store_counter,
            'address': fake.street_address(),
            'city': random.choice(cities),
            'state': state,
            'zip_code': fake.zipcode(),  # Simplified to avoid method check
            'latitude': float(fake.latitude()),
            'longitude': float(fake.longitude()),
            'trade_area_type': trade_area,
            'drive_thru': (store_format == 'drive_thru'),
            'parking_spaces': parking,
            'population_3mi': pop_3mi,
            'median_income_3mi': income_3mi,
            'college_students_pct': 0.35 if trade_area == 'university' else random.uniform(0.05, 0.20),
            'working_professionals_pct': random.uniform(0.25, 0.45),
            'families_pct': random.uniform(0.20, 0.40),
            'open_date': fake.date_between(start_date='-10y', end_date='-1y'),
            'store_format': store_format,
            'seating_capacity': seating,
            'kitchen_capacity_score': random.randint(5, 10),
            'staff_count_avg': random.randint(12, 25),
            'manager_tenure_months': random.randint(3, 60),
            'fast_casual_competitors_1mi': random.randint(2, 8),
            'direct_competitors_3mi': random.randint(0, 3),
            'restaurant_density_1mi': random.randint(5, 25),
            'active_flag': True
        })
        
        store_counter += 1

# Create DataFrame with explicit schema
stores_df = spark.createDataFrame(stores_data, schema=stores_schema)
stores_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.gold.store_locations")
print(f"✓ Created {stores_df.count()} stores")

# COMMAND ----------

# MAGIC %md ## Step 2: Generate Customer Profiles

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
            'household_size': random.randint(1, 4),
            'zip_code': random.choice(stores_data)['zip_code'],  # Near a store
            'lifestyle': lifestyle,
            'loyalty_tier': loyalty,
            'points_balance': random.randint(0, 5000),
            'lifetime_spend': config['aov'] * config['visits'] * random.uniform(0.8, 2.5),
            'visit_frequency': {52: 'weekly', 26: 'biweekly', 12: 'monthly', 6: 'occasional'}[config['visits']],
            'avg_order_value': config['aov'] * random.uniform(0.9, 1.1),
            'preferred_proteins': random.sample(['chicken', 'steak', 'carnitas', 'barbacoa', 'sofritas'], k=random.randint(1, 3)),
            'dietary_preferences': random.sample(['none', 'vegetarian', 'keto', 'high_protein', 'low_sodium'], k=random.randint(0, 2)),
            'app_user': random.random() < config['digital'],
            'email_subscriber': random.random() < 0.6,
            'push_notifications': random.random() < config['digital'] * 0.7,
            'social_media_follower': random.random() < 0.3,
            'referrals_made': np.random.poisson(1),
            'churn_risk_score': 0.1 if segment == 'power_user' else (0.7 if segment == 'price_sensitive' else random.uniform(0.2, 0.5)),
            'customer_segment': segment
        })
        
        customer_counter += 1

# Create DataFrame and save
customers_df = spark.createDataFrame(customers_data)
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
    item['margin_pct'] = (item['base_price'] - item['cost']) / item['base_price'] if item['base_price'] > 0 else 0
    item['cost_of_goods'] = item.pop('cost')
    
    # Realistic nutritional values
    if item['category'] == 'entree':
        item['calories'] = random.randint(500, 1200)
        item['protein_g'] = random.uniform(20, 45)
        item['carbs_g'] = random.uniform(40, 80)
        item['fat_g'] = random.uniform(15, 35)
    elif item['category'] == 'protein':
        item['calories'] = random.randint(150, 250)
        item['protein_g'] = random.uniform(20, 35)
        item['carbs_g'] = random.uniform(0, 5)
        item['fat_g'] = random.uniform(5, 15)
    elif item['category'] == 'side':
        item['calories'] = random.randint(200, 600)
        item['protein_g'] = random.uniform(2, 8)
        item['carbs_g'] = random.uniform(20, 60)
        item['fat_g'] = random.uniform(10, 40)
    else:  # beverage
        item['calories'] = random.randint(0, 250)
        item['protein_g'] = 0
        item['carbs_g'] = random.uniform(0, 60)
        item['fat_g'] = 0
    
    item['sodium_mg'] = random.randint(100, 1500)
    item['allergens'] = []
    item['dietary_flags'] = []
    
    # Set dietary flags
    if 'sofritas' in item['item_name'].lower() or 'veggie' in item['item_name'].lower():
        item['dietary_flags'].append('vegetarian')
        if 'sofritas' in item['item_name'].lower():
            item['dietary_flags'].append('vegan')
    
    item['active_flag'] = True
    item['last_updated'] = datetime.now()

# Create DataFrame and save
menu_df = spark.createDataFrame(menu_items)
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
                    tax = subtotal * 0.0875  # 8.75% tax
                    
                    # Determine channel based on customer digital adoption
                    if customer['app_user'] and random.random() < 0.6:
                        channel = random.choice(['app', 'web'])
                        order_type = random.choice(['pickup', 'delivery'], p=[0.7, 0.3])
                    else:
                        channel = 'in_store'
                        order_type = 'dine_in'
                    
                    # Tips for delivery/pickup
                    tip = 0
                    if order_type == 'delivery':
                        tip = subtotal * random.uniform(0.10, 0.20)
                    elif order_type == 'pickup' and random.random() < 0.15:
                        tip = subtotal * random.uniform(0.05, 0.15)
                    
                    # Apply promotions (10% chance)
                    discount = 0
                    promo_codes = []
                    if random.random() < 0.10:
                        discount = subtotal * random.uniform(0.10, 0.25)
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
                        'tax_amount': float(tax),
                        'tip_amount': float(tip),
                        'discount_amount': float(discount),
                        'payment_method': random.choice(['credit_card', 'debit_card', 'apple_pay', 'google_pay', 'cash']),
                        'promotion_codes': promo_codes,
                        'delivery_partner': random.choice(['uber_eats', 'doordash', 'grubhub', None]) if order_type == 'delivery' else None,
                        'order_prep_time_minutes': random.randint(5, 15),
                        'customer_wait_time_minutes': random.randint(2, 20),
                        'order_items': order_items
                    })
            
            current_date += timedelta(days=1)
        
        current_date = chunk_end + timedelta(days=1)
    
    # Insert chunk
    if transactions_to_insert:
        trans_df = spark.createDataFrame(transactions_to_insert)
        trans_df.write.mode("append").saveAsTable(f"{CATALOG}.gold.transactions")
        print(f"✓ Inserted {len(transactions_to_insert)} transactions up to {chunk_end.date()}")
        transactions_to_insert = []
    
    current_date = chunk_end + timedelta(days=1)

print(f"✓ Transaction generation complete")

# COMMAND ----------

# MAGIC %md ## Step 5: Generate Daily Store Performance

# COMMAND ----------

# Generate daily performance metrics
perf_data = []

for store in stores_data[:int(len(stores_data) * SAMPLE_PCT)]:
    current = START_DATE
    
    while current <= END_DATE:
        # Get transaction aggregates for this store/date
        day_transactions = spark.sql(f"""
            SELECT 
                COUNT(*) as transaction_count,
                SUM(total_amount) as total_revenue,
                AVG(total_amount) as average_ticket,
                SUM(CASE WHEN channel IN ('app', 'web') THEN total_amount ELSE 0 END) as digital_revenue,
                SUM(CASE WHEN channel = 'in_store' THEN total_amount ELSE 0 END) as in_store_revenue,
                SUM(CASE WHEN order_type = 'delivery' THEN total_amount ELSE 0 END) as delivery_revenue,
                COUNT(DISTINCT customer_id) as unique_customers
            FROM {CATALOG}.gold.transactions
            WHERE store_id = '{store['store_id']}' 
            AND order_date = '{current.date()}'
        """).collect()
        
        if day_transactions and day_transactions[0]['transaction_count']:
            row = day_transactions[0]
            
            # Calculate operational metrics
            revenue = float(row['total_revenue'] or 0)
            trans_count = int(row['transaction_count'] or 0)
            
            # Weather simulation
            weather_conditions = ['sunny', 'cloudy', 'rainy', 'snowy']
            weather_weights = [0.50, 0.30, 0.15, 0.05] if current.month in [6,7,8] else [0.30, 0.40, 0.25, 0.05]
            weather = np.random.choice(weather_conditions, p=weather_weights)
            
            temp_base = {1: 35, 2: 40, 3: 50, 4: 60, 5: 70, 6: 80,
                        7: 85, 8: 83, 9: 75, 10: 60, 11: 45, 12: 35}
            
            perf_data.append({
                'store_id': store['store_id'],
                'business_date': current.date(),
                'total_revenue': revenue,
                'transaction_count': trans_count,
                'average_ticket': float(row['average_ticket'] or 0),
                'revenue_per_seat': revenue / store['seating_capacity'] if store['seating_capacity'] > 0 else 0,
                'in_store_revenue': float(row['in_store_revenue'] or 0),
                'digital_revenue': float(row['digital_revenue'] or 0),
                'catering_revenue': 0.0,  # Simplified
                'delivery_revenue': float(row['delivery_revenue'] or 0),
                'avg_service_time': random.uniform(3.5, 8.0),
                'peak_hour_throughput': random.randint(40, 80),
                'staff_hours_scheduled': random.uniform(80, 120),
                'staff_hours_actual': random.uniform(75, 125),
                'food_cost_pct': random.uniform(0.28, 0.35),
                'waste_amount': random.uniform(50, 200),
                'new_customers': random.randint(5, 25),
                'returning_customers': row['unique_customers'] - random.randint(5, min(25, row['unique_customers'])),
                'loyalty_redemptions': random.randint(10, 50),
                'avg_satisfaction_score': random.uniform(3.8, 4.8),
                'weather_condition': weather,
                'temperature_high': temp_base.get(current.month, 60) + random.randint(-10, 10),
                'precipitation_inches': random.uniform(0, 2.0) if weather == 'rainy' else 0.0,
                'local_events': [] if random.random() > 0.1 else [random.choice(['sports_game', 'concert', 'festival'])]
            })
        
        current += timedelta(days=1)

# Insert daily performance
if perf_data:
    perf_df = spark.createDataFrame(perf_data)
    perf_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.gold.daily_store_performance")
    print(f"✓ Generated {len(perf_data)} daily performance records")

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
