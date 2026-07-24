# ML Model Optimization Plan

## Current Status

- **Model Trained**: XGBoost on 205K samples
- **Performance**: MAE=2.98%, RMSE=18.27%, R²=-0.53
- **Deployment**: AlertServiceMLSimple working in production
- **Issue**: Mean ML score too low (0.359), only 3.5% tweets >= 0.65

## Root Cause Analysis

### Why R² Score is Negative (-0.53)

1. **High Volatility Data**: Price changes from -81% to +22,240%
   - Outliers dominate the prediction error
   - Model can't learn meaningful patterns

2. **Limited Features**: Only 8 features available
   - No technical indicators (RSI, MACD, Bollinger Bands)
   - No market context (sector, correlation)
   - No temporal features (time of day, day of week)

3. **Data Quality**: 128K tweets is good volume but:
   - Many tweets from ticker symbols, not real traders
   - Sentiment/confidence heavily skewed
   - Call types are NULL (no classification available)

### Why ML Scores are Low (Mean=0.359)

- Model is conservative (predicts close to 0% change)
- Confidence scores are low because predictions have high error
- Threshold of 0.65 rejects 96.5% of tweets

## Optimization Strategy

### Phase 1: Short-term (Immediate)
**Goal**: Get alerts flowing with current model

Options:
1. **Lower threshold to 0.50** (4.5% of tweets)
   - Pro: More alerts, faster feedback loop
   - Con: More false positives likely
   - Result: ~6 tweets/day would generate alerts

2. **Use NLP confidence only**
   - Pro: Simple, proven to work (what's in production)
   - Con: No ML improvement
   - Result: Current behavior maintained

3. **Hybrid approach**: `final_score = 0.8*NLP + 0.2*ML`
   - Pro: Gradual ML integration
   - Con: Less pure ML benefit
   - Result: Better than pure NLP, less risky

### Phase 2: Medium-term (1-2 weeks)
**Goal**: Improve model R² to > 0.20

Changes:
1. **Feature Engineering**
   - Add engagement rate: `retweet_count / like_count`
   - Add recency: time since tweet
   - Add account age features
   - Normalize all features properly

2. **Handle Outliers**
   - Remove price changes > 10% or < -10% from training
   - Or use robust loss function (Huber loss)
   - Or binned regression instead of continuous

3. **Better Target Variable**
   - Instead of exact % change, predict:
     - "Profitable" vs "Unprofitable" (binary classification)
     - "Small" / "Medium" / "Large" move (categorical)
   - Classification is easier and more interpretable

### Phase 3: Long-term (3-4 weeks)
**Goal**: Build comprehensive ML scoring

Changes:
1. **Add Market Data Features**
   - Current price vs 24h average
   - Volatility (price_24h_high - price_24h_low)
   - Volume changes
   - Momentum indicators

2. **Account-level Features**
   - Historical win_rate per account
   - Performance by market (crypto vs stocks)
   - Recovery from losing streaks

3. **Temporal Features**
   - Hour of day (market hours vs off-hours)
   - Day of week effects
   - Market regime (bull vs bear)

4. **Ensemble Approach**
   - Multiple models: XGBoost + LightGBM + Random Forest
   - Voting ensemble for robustness
   - Stacking for better predictions

## Recommended Next Steps

1. **Immediate (Today)**
   ```
   min_ml_score = 0.50  # More alerts to evaluate
   final_score = 0.7*NLP + 0.3*ML  # Hybrid scoring
   ```

2. **This Week**
   - Monitor alert accuracy with lower threshold
   - Collect feedback on quality
   - If good: proceed to feature engineering
   - If bad: revert to pure NLP

3. **Feature Engineering** (if performance is acceptable)
   - Add engagement ratios
   - Add recency penalty
   - Retrain model

4. **Model Evaluation**
   - Track actual market performance vs predictions
   - Build confusion matrix: predicted profitable vs actual
   - Calculate ROI per alert

## Success Metrics

- **Alert Volume**: Target 5-10 alerts/day
- **Accuracy**: >= 50% of alerts should be profitable
- **False Positive Rate**: < 30%
- **Coverage**: Alerts should cover multiple accounts, not dominated by one

## Current Top Performers (by ML score)

1. **@DeItaone** (0.820) — Consistent high ML scores
2. **@SPY** (0.793) — Index trader
3. **@MSFT** (0.789) — Tech stock focus

→ Consider weighting alerts by account reliability
