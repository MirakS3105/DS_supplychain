"""
Advanced Feature Engineering for Supply Chain Demand Forecasting

This script implements comprehensive feature engineering to improve 
demand forecasting accuracy using all available datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_melt_data():
    """Load all datasets and convert from wide to long format"""
    
    print("Loading datasets...")
    
    # Load core datasets
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Load supplementary datasets
    prices_df = pd.read_csv('prices.csv')
    discounts_df = pd.read_csv('discounts.csv')
    promotions_df = pd.read_csv('promotions.csv')
    competitor_pricing_df = pd.read_csv('competitor_pricing.csv')
    weather_df = pd.read_csv('weather.csv')
    products_df = pd.read_csv('products.csv')
    stores_df = pd.read_csv('stores.csv')
    
    # Convert wide to long format
    train_long = melt_sales_data(train_df, 'units_sold')
    test_long = melt_sales_data(test_df, None)
    
    # Convert supplementary data
    prices_long = melt_sales_data(prices_df, 'price')
    discounts_long = melt_sales_data(discounts_df, 'discount')
    promotions_long = melt_sales_data(promotions_df, 'promotion')
    competitor_long = melt_sales_data(competitor_pricing_df, 'competitor_price')
    
    # Weather is by store only
    weather_long = melt_weather_data(weather_df)
    
    print(f"Train shape: {train_long.shape}")
    print(f"Test shape: {test_long.shape}")
    
    return (train_long, test_long, prices_long, discounts_long, 
            promotions_long, competitor_long, weather_long, products_df, stores_df)


def melt_sales_data(df, value_name):
    """Convert wide format (dates as columns) to long format"""
    id_cols = ['store_id', 'product_id']
    date_cols = [col for col in df.columns if col not in id_cols]
    
    melted = df.melt(id_vars=id_cols, value_vars=date_cols,
                     var_name='date', value_name=value_name if value_name else 'target')
    melted['date'] = pd.to_datetime(melted['date'])
    
    if value_name:
        melted[value_name] = pd.to_numeric(melted[value_name], errors='coerce')
    
    return melted


def melt_weather_data(weather_df):
    """Melt weather data (store_id + dates)"""
    id_cols = ['store_id']
    date_cols = [col for col in weather_df.columns if col not in id_cols]
    
    melted = weather_df.melt(id_vars=id_cols, value_vars=date_cols,
                             var_name='date', value_name='weather')
    melted['date'] = pd.to_datetime(melted['date'])
    return melted


# ============================================================================
# 2. TEMPORAL FEATURE ENGINEERING
# ============================================================================

def create_temporal_features(df):
    """Create comprehensive time-based features"""
    df = df.copy()
    
    # Basic temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Weekend and weekday indicators
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_weekday'] = (df['dayofweek'] < 5).astype(int)
    
    # Month start/end indicators
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Quarter start/end
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding for temporal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Season indicators
    df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                                     9: 'Fall', 10: 'Fall', 11: 'Fall'})
    
    # Holiday indicators (US major holidays)
    df['is_holiday'] = 0
    df['is_christmas_week'] = ((df['month'] == 12) & (df['day'] >= 20)).astype(int)
    df['is_new_year_week'] = ((df['month'] == 1) & (df['day'] <= 7)).astype(int)
    df['is_thanksgiving_week'] = 0  # Would need specific date logic
    df['is_black_friday'] = 0  # Day after Thanksgiving
    
    # Days from/to month end
    df['days_to_month_end'] = df['date'].dt.days_in_month - df['day']
    df['days_from_month_start'] = df['day']
    
    return df


def create_lag_features(df, target_col='units_sold', group_cols=['store_id', 'product_id']):
    """Create multiple lag features with different windows"""
    df = df.copy()
    df = df.sort_values(group_cols + ['date'])
    
    # Short-term lags
    short_lags = [1, 2, 3, 7]
    for lag in short_lags:
        df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    
    # Medium-term lags
    medium_lags = [14, 21, 30]
    for lag in medium_lags:
        df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    
    # Long-term lags (seasonal)
    long_lags = [60, 90, 180, 365]
    for lag in long_lags:
        df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
    
    # Lag differences (trends)
    df['lag_1_diff'] = df['lag_1'] - df['lag_2']
    df['lag_7_diff'] = df['lag_1'] - df['lag_7']
    df['lag_30_diff'] = df['lag_1'] - df['lag_30']
    
    # Lag ratios
    df['lag_7_ratio'] = df['lag_1'] / (df['lag_7'] + 1)
    df['lag_30_ratio'] = df['lag_1'] / (df['lag_30'] + 1)
    
    return df


def create_rolling_features(df, target_col='units_sold', group_cols=['store_id', 'product_id']):
    """Create rolling window statistics"""
    df = df.copy()
    df = df.sort_values(group_cols + ['date'])
    
    windows = [3, 7, 14, 30, 60, 90]
    
    for window in windows:
        # Rolling mean
        df[f'rolling_mean_{window}'] = df.groupby(group_cols)[target_col]\
                                         .transform(lambda x: x.rolling(window, min_periods=1).mean())
        
        # Rolling std
        df[f'rolling_std_{window}'] = df.groupby(group_cols)[target_col]\
                                        .transform(lambda x: x.rolling(window, min_periods=1).std())
        
        # Rolling min/max
        df[f'rolling_min_{window}'] = df.groupby(group_cols)[target_col]\
                                        .transform(lambda x: x.rolling(window, min_periods=1).min())
        df[f'rolling_max_{window}'] = df.groupby(group_cols)[target_col]\
                                        .transform(lambda x: x.rolling(window, min_periods=1).max())
        
        # Rolling median
        df[f'rolling_median_{window}'] = df.groupby(group_cols)[target_col]\
                                           .transform(lambda x: x.rolling(window, min_periods=1).median())
        
        # Coefficient of variation
        df[f'rolling_cv_{window}'] = df[f'rolling_std_{window}'] / (df[f'rolling_mean_{window}'] + 1)
    
    # Expanding window features
    df['expanding_mean'] = df.groupby(group_cols)[target_col].expanding().mean().reset_index(0, drop=True)
    df['expanding_std'] = df.groupby(group_cols)[target_col].expanding().std().reset_index(0, drop=True)
    df['expanding_max'] = df.groupby(group_cols)[target_col].expanding().max().reset_index(0, drop=True)
    
    return df


def create_seasonal_features(df, target_col='units_sold', group_cols=['store_id', 'product_id']):
    """Create seasonal decomposition features"""
    df = df.copy()
    
    # Same day last week, 2 weeks ago, etc.
    for weeks in [1, 2, 3, 4]:
        df[f'same_day_{weeks}w_ago'] = df.groupby(group_cols)[target_col].shift(7 * weeks)
    
    # Same day last month (approximate)
    df['same_day_last_month'] = df.groupby(group_cols)[target_col].shift(30)
    
    # Same day last year
    df['same_day_last_year'] = df.groupby(group_cols)[target_col].shift(365)
    
    # Weekly moving averages
    df['weekly_pattern'] = df.groupby(group_cols + ['dayofweek'])[target_col]\
                             .transform(lambda x: x.rolling(4, min_periods=1).mean())
    
    # Monthly pattern
    df['monthly_pattern'] = df.groupby(group_cols + ['month'])[target_col]\
                              .transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    return df


# ============================================================================
# 3. PRICE AND PROMOTION FEATURES
# ============================================================================

def create_price_features(df):
    """Create price-related features"""
    df = df.copy()
    
    if 'price' in df.columns:
        # Price changes
        df['price_change_1d'] = df.groupby(['store_id', 'product_id'])['price'].diff(1)
        df['price_change_7d'] = df.groupby(['store_id', 'product_id'])['price'].diff(7)
        
        # Price relative to rolling average
        df['price_vs_7d_avg'] = df['price'] / (df.groupby(['store_id', 'product_id'])['price']\
                                               .transform(lambda x: x.rolling(7, min_periods=1).mean()) + 1)
        df['price_vs_30d_avg'] = df['price'] / (df.groupby(['store_id', 'product_id'])['price']\
                                                .transform(lambda x: x.rolling(30, min_periods=1).mean()) + 1)
        
        # Price percentile (how expensive is this relative to history)
        df['price_percentile_30d'] = df.groupby(['store_id', 'product_id'])['price']\
                                       .transform(lambda x: x.rolling(30, min_periods=1)\
                                                 .apply(lambda y: stats.percentileofscore(y, y.iloc[-1]) if len(y) > 0 else 50, raw=True))
    
    if 'competitor_price' in df.columns and 'price' in df.columns:
        # Price comparison with competitors
        df['price_vs_competitor'] = df['price'] - df['competitor_price']
        df['price_competitor_ratio'] = df['price'] / (df['competitor_price'] + 1)
        df['is_cheaper_than_competitor'] = (df['price'] < df['competitor_price']).astype(int)
    
    # Discount features
    if 'discount' in df.columns:
        df['has_discount'] = (df['discount'] > 0).astype(int)
        df['discount_rolling_7d'] = df.groupby(['store_id', 'product_id'])['discount']\
                                      .transform(lambda x: x.rolling(7, min_periods=1).sum())
        df['discount_rolling_30d'] = df.groupby(['store_id', 'product_id'])['discount']\
                                       .transform(lambda x: x.rolling(30, min_periods=1).sum())
    
    # Promotion features
    if 'promotion' in df.columns:
        df['days_since_promotion'] = df.groupby(['store_id', 'product_id'])['promotion']\
                                       .transform(lambda x: x.rolling(30, min_periods=1)\
                                                 .apply(lambda y: (y == 0).sum() if (y == 1).any() else 30))
        df['promotion_rolling_7d'] = df.groupby(['store_id', 'product_id'])['promotion']\
                                       .transform(lambda x: x.rolling(7, min_periods=1).sum())
    
    return df


def create_elasticity_features(df):
    """Create price elasticity indicators"""
    df = df.copy()
    
    if 'price' in df.columns and 'units_sold' in df.columns:
        # Calculate price elasticity proxy (sales change / price change)
        df['price_change_pct'] = df.groupby(['store_id', 'product_id'])['price'].pct_change()
        df['sales_change_pct'] = df.groupby(['store_id', 'product_id'])['units_sold'].pct_change()
        
        # Lagged price effect (price change impact on future sales)
        df['price_lag_1'] = df.groupby(['store_id', 'product_id'])['price'].shift(1)
        df['price_lag_3'] = df.groupby(['store_id', 'product_id'])['price'].shift(3)
    
    return df


# ============================================================================
# 4. WEATHER FEATURES
# ============================================================================

def encode_weather_features(df):
    """Encode weather conditions"""
    df = df.copy()
    
    if 'weather' in df.columns:
        # One-hot encode weather conditions
        weather_dummies = pd.get_dummies(df['weather'], prefix='weather')
        df = pd.concat([df, weather_dummies], axis=1)
        
        # Bad weather indicator (Rainy/Snowy)
        df['bad_weather'] = df['weather'].isin(['Rainy', 'Snowy']).astype(int)
        
        # Good weather indicator (Sunny)
        df['good_weather'] = (df['weather'] == 'Sunny').astype(int)
        
        # Weather persistence (how many consecutive days of same weather)
        df['weather_change'] = (df.groupby(['store_id'])['weather'].shift(1) != df['weather']).astype(int)
        df['days_same_weather'] = df.groupby(['store_id'])['weather_change'].cumsum()
    
    return df


# ============================================================================
# 5. STORE AND PRODUCT FEATURES
# ============================================================================

def create_store_product_features(df, products_df, stores_df):
    """Create store and product aggregation features"""
    df = df.copy()
    
    # Aggregate products by store
    store_daily_stats = df.groupby(['store_id', 'date'])['units_sold'].agg([
        ('store_total_sales', 'sum'),
        ('store_avg_sales', 'mean'),
        ('store_max_sales', 'max'),
        ('store_sales_count', 'count')
    ]).reset_index()
    
    df = df.merge(store_daily_stats, on=['store_id', 'date'], how='left')
    
    # Product share within store
    df['product_store_share'] = df['units_sold'] / (df['store_total_sales'] + 1)
    
    # Aggregate by category if available
    if 'category' in df.columns:
        category_daily_stats = df.groupby(['category', 'date'])['units_sold'].agg([
            ('category_total_sales', 'sum'),
            ('category_avg_sales', 'mean')
        ]).reset_index()
        
        df = df.merge(category_daily_stats, on=['category', 'date'], how='left')
        df['product_category_share'] = df['units_sold'] / (df['category_total_sales'] + 1)
    
    # Store-level features
    if 'region' in df.columns:
        region_daily_stats = df.groupby(['region', 'date'])['units_sold'].agg([
            ('region_total_sales', 'sum'),
            ('region_avg_sales', 'mean')
        ]).reset_index()
        
        df = df.merge(region_daily_stats, on=['region', 'date'], how='left')
    
    return df


def create_target_encoding_features(df, target_col='units_sold'):
    """Create target encoding features with careful validation"""
    df = df.copy()
    
    # Product-level statistics (using expanding window to avoid leakage)
    df['product_mean_sales'] = df.groupby('product_id')[target_col].expanding().mean().reset_index(0, drop=True)
    df['product_std_sales'] = df.groupby('product_id')[target_col].expanding().std().reset_index(0, drop=True)
    
    # Store-level statistics
    df['store_mean_sales'] = df.groupby('store_id')[target_col].expanding().mean().reset_index(0, drop=True)
    
    # Day-of-week patterns by product
    df['dow_product_mean'] = df.groupby(['product_id', 'dayofweek'])[target_col]\
                               .expanding().mean().reset_index(0, drop=True)
    
    # Month patterns by product
    df['month_product_mean'] = df.groupby(['product_id', 'month'])[target_col]\
                                 .expanding().mean().reset_index(0, drop=True)
    
    return df


# ============================================================================
# 6. ADVANCED FEATURES
# ============================================================================

def create_trend_features(df, target_col='units_sold', group_cols=['store_id', 'product_id']):
    """Create trend and momentum features"""
    df = df.copy()
    df = df.sort_values(group_cols + ['date'])
    
    # Short-term trend (linear regression slope over recent days)
    for window in [7, 14, 30]:
        df[f'trend_slope_{window}d'] = df.groupby(group_cols)[target_col]\
                                         .transform(lambda x: x.rolling(window, min_periods=3)\
                                                   .apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0, raw=True))
    
    # Momentum indicators
    df['momentum_7d'] = df['lag_1'] - df['lag_7']
    df['momentum_30d'] = df['lag_1'] - df['lag_30']
    
    # Rate of change
    df['roc_7d'] = ((df['lag_1'] - df['lag_7']) / (df['lag_7'] + 1)) * 100
    df['roc_30d'] = ((df['lag_1'] - df['lag_30']) / (df['lag_30'] + 1)) * 100
    
    return df


def create_volatility_features(df, target_col='units_sold', group_cols=['store_id', 'product_id']):
    """Create volatility and outlier features"""
    df = df.copy()
    
    # Rolling volatility (std/mean)
    for window in [7, 14, 30]:
        rolling_mean = df.groupby(group_cols)[target_col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        rolling_std = df.groupby(group_cols)[target_col].transform(lambda x: x.rolling(window, min_periods=1).std())
        df[f'volatility_{window}d'] = rolling_std / (rolling_mean + 1)
    
    # Outlier detection (z-score)
    df['sales_zscore_30d'] = df.groupby(group_cols)[target_col]\
                               .transform(lambda x: (x - x.rolling(30, min_periods=1).mean()) / 
                                                    (x.rolling(30, min_periods=1).std() + 1))
    
    # Is unusually high/low
    df['is_unusually_high'] = (df['sales_zscore_30d'] > 2).astype(int)
    df['is_unusually_low'] = (df['sales_zscore_30d'] < -2).astype(int)
    
    return df


def create_interaction_features(df):
    """Create interaction features between variables"""
    df = df.copy()
    
    # Price x Promotion interaction
    if 'price' in df.columns and 'promotion' in df.columns:
        df['price_x_promotion'] = df['price'] * df['promotion']
    
    # Weekend x Category interaction (if category exists)
    if 'category' in df.columns:
        df['weekend_x_category'] = df['is_weekend'].astype(str) + '_' + df['category'].astype(str)
    
    # Weather x Day of week interaction
    if 'weather' in df.columns:
        df['weather_x_dow'] = df['weather'].astype(str) + '_' + df['dayofweek'].astype(str)
    
    # Season x Product interaction
    if 'season' in df.columns:
        df['season_x_product'] = df['season'].astype(str) + '_' + df['product_id'].astype(str)
    
    return df


# ============================================================================
# 7. FEATURE SELECTION AND PREPARATION
# ============================================================================

def select_features(df, target_col='units_sold', min_correlation=0.01):
    """Select features based on correlation with target"""
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target and id columns
    exclude_cols = [target_col, 'store_id', 'product_id', 'date']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlations
    correlations = df[feature_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    
    # Select features above threshold
    selected_features = correlations[correlations > min_correlation].index.tolist()
    selected_features = [col for col in selected_features if col != target_col]
    
    print(f"Selected {len(selected_features)} features out of {len(feature_cols)}")
    print("\nTop 20 features by correlation:")
    print(correlations.head(20))
    
    return selected_features


def prepare_final_dataset(df, selected_features, target_col='units_sold'):
    """Prepare final dataset for modeling"""
    
    # Fill missing values
    df_clean = df.copy()
    
    for col in selected_features:
        if col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                # Use median for numeric columns
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Remove rows where target is null (for training)
    if target_col in df_clean.columns:
        df_clean = df_clean.dropna(subset=[target_col])
    
    return df_clean


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def run_feature_engineering_pipeline():
    """Run complete feature engineering pipeline"""
    
    print("=" * 80)
    print("ADVANCED FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    
    # 1. Load data
    print("\n1. Loading and preprocessing data...")
    (train_long, test_long, prices_long, discounts_long, 
     promotions_long, competitor_long, weather_long, products_df, stores_df) = load_and_melt_data()
    
    # 2. Merge all datasets
    print("\n2. Merging datasets...")
    train_combined = train_long.copy()
    test_combined = test_long.copy()
    
    # Merge prices
    train_combined = train_combined.merge(prices_long, on=['store_id', 'product_id', 'date'], how='left')
    test_combined = test_combined.merge(prices_long, on=['store_id', 'product_id', 'date'], how='left')
    
    # Merge discounts
    train_combined = train_combined.merge(discounts_long, on=['store_id', 'product_id', 'date'], how='left')
    test_combined = test_combined.merge(discounts_long, on=['store_id', 'product_id', 'date'], how='left')
    
    # Merge promotions
    train_combined = train_combined.merge(promotions_long, on=['store_id', 'product_id', 'date'], how='left')
    test_combined = test_combined.merge(promotions_long, on=['store_id', 'product_id', 'date'], how='left')
    
    # Merge competitor prices
    train_combined = train_combined.merge(competitor_long, on=['store_id', 'product_id', 'date'], how='left')
    test_combined = test_combined.merge(competitor_long, on=['store_id', 'product_id', 'date'], how='left')
    
    # Merge weather
    train_combined = train_combined.merge(weather_long, on=['store_id', 'date'], how='left')
    test_combined = test_combined.merge(weather_long, on=['store_id', 'date'], how='left')
    
    # Merge product info
    train_combined = train_combined.merge(products_df, on='product_id', how='left')
    test_combined = test_combined.merge(products_df, on='product_id', how='left')
    
    # Merge store info
    train_combined = train_combined.merge(stores_df, on='store_id', how='left')
    test_combined = test_combined.merge(stores_df, on='store_id', how='left')
    
    print(f"Combined train shape: {train_combined.shape}")
    print(f"Combined test shape: {test_combined.shape}")
    
    # 3. Create temporal features
    print("\n3. Creating temporal features...")
    train_combined = create_temporal_features(train_combined)
    test_combined = create_temporal_features(test_combined)
    
    # 4. Create lag features
    print("\n4. Creating lag features...")
    train_combined = create_lag_features(train_combined)
    
    # 5. Create rolling features
    print("\n5. Creating rolling features...")
    train_combined = create_rolling_features(train_combined)
    
    # 6. Create seasonal features
    print("\n6. Creating seasonal features...")
    train_combined = create_seasonal_features(train_combined)
    
    # 7. Create price features
    print("\n7. Creating price and promotion features...")
    train_combined = create_price_features(train_combined)
    train_combined = create_elasticity_features(train_combined)
    
    # 8. Encode weather
    print("\n8. Encoding weather features...")
    train_combined = encode_weather_features(train_combined)
    
    # 9. Create store/product aggregation features
    print("\n9. Creating store and product aggregation features...")
    train_combined = create_store_product_features(train_combined, products_df, stores_df)
    
    # 10. Create target encoding features
    print("\n10. Creating target encoding features...")
    train_combined = create_target_encoding_features(train_combined)
    
    # 11. Create trend and volatility features
    print("\n11. Creating trend and volatility features...")
    train_combined = create_trend_features(train_combined)
    train_combined = create_volatility_features(train_combined)
    
    # 12. Create interaction features
    print("\n12. Creating interaction features...")
    train_combined = create_interaction_features(train_combined)
    
    print("\n" + "=" * 80)
    print(f"FINAL TRAINING DATASET SHAPE: {train_combined.shape}")
    print(f"Number of features created: {train_combined.shape[1] - 5}")  # Excluding id and target columns
    print("=" * 80)
    
    # 13. Select best features
    print("\n13. Selecting features...")
    selected_features = select_features(train_combined, min_correlation=0.01)
    
    # 14. Prepare final dataset
    print("\n14. Preparing final dataset...")
    train_final = prepare_final_dataset(train_combined, selected_features)
    
    # Save feature-engineered dataset
    print("\n15. Saving feature-engineered dataset...")
    train_final.to_csv('train_feature_engineered.csv', index=False)
    print("Saved: train_feature_engineered.csv")
    
    # Print feature summary
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE!")
    print("=" * 80)
    print(f"\nDataset info:")
    print(f"- Total samples: {len(train_final)}")
    print(f"- Total features: {len(selected_features)}")
    print(f"- Date range: {train_final['date'].min()} to {train_final['date'].max()}")
    print(f"- Stores: {train_final['store_id'].nunique()}")
    print(f"- Products: {train_final['product_id'].nunique()}")
    
    return train_final, selected_features


if __name__ == "__main__":
    train_final, selected_features = run_feature_engineering_pipeline()
