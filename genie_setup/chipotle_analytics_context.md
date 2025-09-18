# **Chipotle Sales Performance Analytics Agent**
## **Project Context for Genie Spaces & ML Tools Development**

### **Project Overview**

This project demonstrates a multi-agent system built on Databricks using Claude models to provide strategic sales performance analytics for Chipotle Mexican Grill operations. The system analyzes synthetic data representing ~10% of Chipotle's US operations (300 stores) to showcase real-world business intelligence capabilities.

**Key Objectives:**
- Demonstrate Claude-powered business intelligence for restaurant operations
- Showcase multi-agent orchestration with strategic context integration
- Provide natural language access to complex business data
- Enable data-driven decision making for store operations and customer insights

---

## **Synthetic Data Foundation**

### **Data Scope & Scale**
- **Store Coverage**: 300 stores (10% representative sample of ~3,000 US Chipotle locations)
- **Time Period**: 3 years historical data (2022-2024) + YTD 2025
- **Transaction Volume**: ~50M transactions total
- **Customer Base**: ~1.2M unique customers with realistic segmentation

### **Customer Segmentation (Realistic Distribution)**
```
Power Users (15%): Weekly visitors, $14.50 AOV, 90% digital adoption
Loyal Regulars (25%): Bi-weekly visitors, $11.75 AOV, 70% digital adoption  
Occasional Visitors (45%): Monthly visitors, $10.25 AOV, 45% digital adoption
Price Sensitive (15%): Occasional visitors, $8.75 AOV, 35% digital adoption
```

### **Store Performance Distribution**
```
Top Performers (20%): 35% above average revenue, 4.6 satisfaction score
Above Average (30%): 15% above average revenue, 4.3 satisfaction score
Average Performers (30%): Baseline performance, 4.0 satisfaction score
Below Average (15%): 20% below average revenue, 3.6 satisfaction score
Underperformers (5%): 35% below average revenue, 3.2 satisfaction score
```

### **Geographic Distribution**
Proportional to actual Chipotle presence:
- California: 75 stores (25%)
- Texas: 45 stores (15%)
- Florida: 30 stores (10%)
- New York: 24 stores (8%)
- Illinois: 21 stores (7%)
- Remaining 35 states: 105 stores (35%)

---

## **Unity Catalog Data Architecture**

### **Catalog Structure**
```
chipotle_analytics/
└── gold/
    ├── campaigns                    # Marketing campaigns and promotions
    ├── customer_profiles           # Individual customer data and segments
    ├── customer_segments_monthly   # Aggregated customer segment analysis
    ├── daily_store_performance    # Daily operational metrics by store
    ├── menu_items                 # Product catalog with pricing and nutrition
    ├── store_locations            # Store characteristics and demographics
    ├── store_performance_monthly  # Monthly aggregated store performance
    ├── transactions               # Core transaction/order data
    └── Views:
        ├── v_customer_lifetime_value      # Customer LTV analysis
        └── v_store_performance_enriched   # Enhanced store performance metrics
```

### **Key Data Relationships**
- **Transactions** link to **Customer Profiles** via `customer_id`
- **Transactions** link to **Store Locations** via `store_id`
- **Daily Store Performance** aggregates from **Transactions**
- **Monthly Performance** tables provide pre-computed business metrics
- **Views** provide enriched analytics-ready datasets

---

## **Genie Spaces Architecture**

### **1. Store Performance Intelligence**
**Purpose**: Analyze individual store performance, identify trends, and benchmark against peers

**Tables Included**:
- `chipotle_analytics.gold.store_performance_monthly`
- `chipotle_analytics.gold.v_store_performance_enriched`
- `chipotle_analytics.gold.store_locations`
- `chipotle_analytics.gold.daily_store_performance`

**Key Metadata Enhancements**:
- **Table Descriptions**: 
  - `store_performance_monthly`: "Monthly aggregated performance metrics for each Chipotle store including revenue, customer counts, operational efficiency, and benchmark comparisons"
  - `store_locations`: "Store characteristics including demographics, format, capacity, and competitive environment"
  
- **Column Synonyms**:
  - `total_revenue` = "sales", "revenue", "income", "earnings"
  - `avg_ticket` = "average order value", "AOV", "check average"
  - `digital_mix_pct` = "digital percentage", "online orders", "app orders"
  - `nps_score` = "customer satisfaction", "satisfaction score", "NPS"

- **Business Context**:
  - "Fiscal year runs January-December"
  - "Performance comparisons are against similar store formats and demographics"
  - "Top quartile stores (percentile_ranking >= 75) are considered high performers"

**Sample Natural Language Queries**:
```
"Which stores are underperforming compared to their demographic potential?"
"Show me the top 10 stores by customer satisfaction and what makes them successful"
"What stores have declining same-store sales trends?"
"Compare Store CHI001 to similar stores in urban markets"
"Which markets show the strongest revenue growth this quarter?"
```

### **2. Customer Analytics Intelligence**
**Purpose**: Understand customer behavior, segmentation, and lifetime value patterns

**Tables Included**:
- `chipotle_analytics.gold.customer_profiles`
- `chipotle_analytics.gold.customer_segments_monthly`
- `chipotle_analytics.gold.v_customer_lifetime_value`
- `chipotle_analytics.gold.transactions` (for behavior analysis)

**Key Metadata Enhancements**:
- **Table Descriptions**:
  - `customer_profiles`: "Individual customer demographics, loyalty metrics, and behavioral preferences"
  - `customer_segments_monthly`: "Monthly analysis of customer segments including size, value, and behavior patterns"
  
- **Column Synonyms**:
  - `lifetime_spend` = "total spend", "customer value", "lifetime value"
  - `visit_frequency` = "frequency", "how often", "visit pattern"
  - `churn_risk_score` = "retention risk", "churn probability", "at-risk score"
  - `loyalty_tier` = "loyalty level", "tier", "status"

- **Business Context**:
  - "Power users visit weekly and drive 40% of revenue despite being 15% of customers"
  - "Digital customers have 25% higher lifetime value than in-store only customers"
  - "Loyalty tiers: Bronze (0-499 points), Silver (500-999), Gold (1000-1999), Platinum (2000+)"

**Sample Natural Language Queries**:
```
"What's the lifetime value of customers acquired through our app vs other channels?"
"Which customer segments show the highest retention rates?"
"How do power users differ from occasional visitors in their ordering behavior?"
"What percentage of our revenue comes from each customer segment?"
"Which demographics have the highest digital adoption rates?"
```

### **3. Product Performance Intelligence**
**Purpose**: Analyze menu item performance, pricing effectiveness, and customer preferences

**Tables Included**:
- `chipotle_analytics.gold.menu_items`
- `chipotle_analytics.gold.transactions` (for sales analysis)
- `chipotle_analytics.gold.customer_profiles` (for preference analysis)

**Key Metadata Enhancements**:
- **Table Descriptions**:
  - `menu_items`: "Complete product catalog including pricing, nutritional information, and profitability metrics"
  - `transactions.order_items`: "Individual items purchased in each order with modifications and quantities"

- **Column Synonyms**:
  - `base_price` = "price", "cost", "menu price"
  - `margin_pct` = "profit margin", "profitability", "margin"
  - `category` = "type", "menu category", "item type"
  - `calories` = "nutrition", "caloric content"

- **Business Context**:
  - "Bowls are 40% of orders, burritos 35%, tacos 15%, salads 10%"
  - "Chicken is the most popular protein (60%), followed by steak (25%)"
  - "Premium proteins (steak, barbacoa) have 15% higher margins"
  - "Guacamole is added to 65% of orders and increases ticket by $2.50"

**Sample Natural Language Queries**:
```
"Which menu items have the highest profit margins?"
"What are the most popular protein choices by customer segment?"
"How do seasonal items perform compared to core menu?"
"Which items drive the highest average order value?"
"What's the price elasticity for premium proteins vs standard?"
```

### **4. Marketing & Campaign Intelligence**
**Purpose**: Evaluate campaign effectiveness, promotional impact, and customer acquisition

**Tables Included**:
- `chipotle_analytics.gold.campaigns`
- `chipotle_analytics.gold.transactions` (for promotion tracking)
- `chipotle_analytics.gold.customer_profiles` (for targeting analysis)

**Key Metadata Enhancements**:
- **Table Descriptions**:
  - `campaigns`: "Marketing campaign details including promotion types, targeting, budgets, and success metrics"

- **Column Synonyms**:
  - `target_redemptions` = "expected usage", "redemption goal"
  - `discount_amount` = "discount", "savings", "promotion value"
  - `target_roi` = "return on investment", "ROI target", "expected return"

- **Business Context**:
  - "BOGO (Buy One Get One) promotions drive highest redemption rates"
  - "Digital-only promotions have 30% higher margins than traditional media"
  - "Student segment responds best to percentage discounts, families prefer dollar amounts"

**Sample Natural Language Queries**:
```
"Which campaign types deliver the highest ROI?"
"How effective are our BOGO promotions at driving new customer acquisition?"
"What's the performance difference between digital vs traditional campaign channels?"
"Which customer segments respond best to promotional offers?"
"What's the incremental revenue impact of our current campaigns?"
```

---

## **ML Model Tools (Unity Catalog Functions)**

### **1. Customer Lifetime Value Prediction**
```sql
chipotle_analytics.ml_models.predict_customer_ltv(
    customer_segment STRING,
    acquisition_channel STRING,
    market_characteristics STRUCT<...>
) RETURNS STRUCT<
    predicted_ltv_12m DECIMAL(8,2),
    predicted_ltv_24m DECIMAL(8,2),
    confidence_score FLOAT,
    value_drivers ARRAY<STRING>
>
```

### **2. Store Performance Anomaly Detection**
```sql
chipotle_analytics.ml_models.detect_performance_anomalies(
    store_id STRING,
    metric_name STRING,
    lookback_days INT DEFAULT 30
) RETURNS STRUCT<
    anomaly_detected BOOLEAN,
    severity STRING,
    deviation_pct FLOAT,
    contributing_factors ARRAY<STRING>,
    recommended_actions ARRAY<STRING>
>
```

### **3. Demand Forecasting**
```sql
chipotle_analytics.ml_models.predict_demand(
    store_id STRING,
    forecast_date DATE,
    include_weather BOOLEAN DEFAULT TRUE,
    include_events BOOLEAN DEFAULT TRUE
) RETURNS TABLE(
    item_category STRING,
    predicted_quantity INT,
    confidence_interval STRUCT<lower INT, upper INT>,
    key_drivers ARRAY<STRING>
)
```

### **4. Campaign Impact Simulation**
```sql
chipotle_analytics.ml_models.simulate_campaign_impact(
    promotion_type STRING,
    target_segment STRING,
    store_list ARRAY<STRING>,
    duration_days INT
) RETURNS STRUCT<
    predicted_redemption_rate FLOAT,
    estimated_revenue_impact DECIMAL(10,2),
    incremental_customers INT,
    roi_estimate FLOAT,
    risk_factors ARRAY<STRING>
>
```

---

## **Business Context & Strategic Intelligence**

### **Current Strategic Priorities (Synthetic)**
1. **Digital Transformation**: Target 60% digital mix by end of 2025
2. **Customer Retention**: Reduce churn rate by 15% through loyalty program enhancements
3. **Market Expansion**: Focus on suburban family demographics in underserved markets
4. **Operational Excellence**: Improve service times and food quality consistency
5. **Premium Positioning**: Test higher-margin menu items and experiences

### **Key Business Constraints**
- **Labor Cost Target**: 25-28% of revenue
- **Food Cost Target**: 28-32% of revenue
- **Service Time Goal**: <3 minutes for digital orders, <5 minutes for in-store
- **Customer Satisfaction Target**: 4.2+ NPS score across all stores

### **Known Business Patterns**
- **Seasonality**: Q3 (summer) is peak season (+10% vs average), Q1 is lowest (-5%)
- **Day-of-Week**: Friday/Saturday drive 35% of weekly revenue
- **Daypart**: Lunch (11am-2pm) represents 45% of daily sales
- **Weather Impact**: Rain increases delivery orders 20%, heat increases cold beverage sales 15%

---

## **Success Metrics for Genie Spaces**

### **Data Quality Validation**
- **Query Response Accuracy**: 90%+ of natural language queries return relevant, actionable results
- **Business Logic Compliance**: Responses align with known business rules and constraints
- **Cross-Table Consistency**: Metrics calculated across different tables show consistent values

### **User Experience Metrics**
- **Query Resolution Rate**: 85%+ of business questions can be answered without requiring technical SQL knowledge
- **Response Relevance**: Answers directly address the business context of the question
- **Actionability**: 80%+ of insights include specific recommendations or next steps

This foundation enables business users to interact with complex restaurant operations data through natural language while ensuring responses are grounded in realistic business context and constraints.