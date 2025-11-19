# Real Data Specifications for NNN Marketing Mix Modeling

## Overview

This document specifies the data requirements for implementing the NNN framework with production marketing data. It covers data sources, formats, quality requirements, and integration pipelines.

---

## 1. Required Data Sources

### 1.1 Sales Data (Target Variable)

**Purpose:** The outcome variable we're trying to predict and attribute.

**Requirements:**
- **Granularity:** Weekly or daily by geographic region
- **Format:** Time series with `(geo, date, sales)` structure
- **Coverage:** Minimum 2-3 years of historical data (104-156 weeks)
- **Quality:** No missing weeks; handle holidays/anomalies explicitly

**Schema:**
```
date          | geo   | sales
------------- | ----- | ----------
2023-01-01    | US-CA | 125,430.50
2023-01-01    | US-NY | 98,220.75
2023-01-08    | US-CA | 130,590.25
```

**Data Quality Checks:**
- [ ] No NULL values in sales column
- [ ] Sales values > 0
- [ ] Complete date coverage (no gaps)
- [ ] Consistent geo identifiers

---

### 1.2 Marketing Channel Data (Input Variables)

#### A. Paid Media Channels (with Embeddings)

**Channels requiring embeddings:**
- YouTube/Video Ads
- Display Ads
- Social Media (Facebook, Instagram, TikTok)
- Search Ads (with keyword data)

**Data Structure for Embedding Channels:**

**Campaign/Creative Level Data:**
```
date       | geo   | channel  | creative_id    | impressions | spend    | embedding_vector
---------- | ----- | -------- | -------------- | ----------- | -------- | ----------------
2023-01-01 | US-CA | YouTube  | YT_Video_001   | 50,000      | 2,500.00 | [0.12, -0.45, ...]
2023-01-01 | US-CA | YouTube  | YT_Video_002   | 30,000      | 1,800.00 | [0.87, 0.23, ...]
```

**Embedding Sources:**
- **Video Ads:** Pre-trained video embeddings (e.g., from video-language models like CLIP)
- **Display Ads:** Image embeddings (e.g., ResNet, CLIP)
- **Text Ads:** Text embeddings (e.g., BERT, sentence transformers)
- **Keywords:** Keyword embeddings (e.g., Word2Vec, BERT)

**Embedding Specifications:**
- **Dimension:** 64-512 (paper uses 256; start smaller for faster training)
- **Normalization:** L2-normalized before multiplying by volume
- **Aggregation:** If multiple creatives per week, aggregate using weighted average by impressions

#### B. Scalar Channels (No Embeddings)

**Channels using scalar values only:**
- TV (traditional)
- Radio
- Print
- Out-of-Home (OOH)
- Price promotions
- Competitor activity

**Data Structure for Scalar Channels:**
```
date       | geo   | channel    | impressions | spend
---------- | ----- | ---------- | ----------- | --------
2023-01-01 | US-CA | TV         | 500,000     | 25,000.00
2023-01-01 | US-CA | Radio      | 200,000     | 8,000.00
```

---

### 1.3 Organic/Earned Media Data

#### Search Query Data (Organic)

**Purpose:** Captures organic search intent driven by other marketing

**Requirements:**
- Search queries or query embeddings
- Search volume by geo/week
- Can use Google Trends data or internal site search

**Schema:**
```
date       | geo   | query                | volume  | embedding_vector
---------- | ----- | -------------------- | ------- | ----------------
2023-01-01 | US-CA | brand_name_product   | 12,500  | [0.34, 0.78, ...]
2023-01-01 | US-CA | generic_category     | 8,200   | [-0.12, 0.45, ...]
```

**Embedding Generation:**
- Use sentence transformers (e.g., `all-MiniLM-L6-v2`) for search queries
- Aggregate query embeddings by volume weighting
- Weekly aggregation at geo level

---

### 1.4 External/Control Variables

**Purpose:** Account for non-marketing drivers of sales

**Recommended Variables:**
- **Seasonality:** Week of year, month, holidays
- **Trend:** Time index
- **Economic:** Regional GDP, unemployment rate
- **Weather:** Temperature, precipitation (if relevant)
- **Competitive:** Competitor pricing, promotions
- **Store:** Store openings/closures, distribution changes

**Schema:**
```
date       | geo   | holiday | temperature | competitor_promo
---------- | ----- | ------- | ----------- | ----------------
2023-01-01 | US-CA | 1       | 15.5        | 0
2023-07-04 | US-CA | 1       | 28.3        | 1
```

---

## 2. Data Preprocessing Pipeline

### 2.1 Data Validation

**Pre-processing Checks:**
```python
# Check for missing data
assert df.isna().sum().sum() == 0, "Missing values detected"

# Check date continuity
date_range = pd.date_range(start=df['date'].min(),
                           end=df['date'].max(),
                           freq='W')
assert len(df['date'].unique()) == len(date_range), "Missing weeks"

# Check for outliers
z_scores = (df['sales'] - df['sales'].mean()) / df['sales'].std()
outliers = df[abs(z_scores) > 4]
if len(outliers) > 0:
    print(f"Warning: {len(outliers)} potential outliers detected")
```

---

### 2.2 Embedding Generation

**For New Creative Assets:**

```python
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model for video/image embeddings
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_video_embedding(video_path):
    """Generate embedding for video creative"""
    # Extract key frame or use video model
    frame = extract_key_frame(video_path)

    inputs = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # L2 normalize
    embedding = image_features / image_features.norm(dim=-1, keepdim=True)
    return embedding.numpy().flatten()

def generate_text_embedding(text):
    """Generate embedding for text ad or keyword"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding
```

---

### 2.3 Aggregation to Weekly Geo Level

**Aggregation Logic:**

```python
def aggregate_to_weekly_geo(raw_data):
    """
    Aggregate campaign-level data to weekly geo level.
    Weighted average embeddings by impressions.
    """

    grouped = raw_data.groupby(['date', 'geo', 'channel'])

    def weighted_embedding_avg(group):
        embeddings = np.stack(group['embedding'].values)
        weights = group['impressions'].values / group['impressions'].sum()
        avg_embedding = (embeddings.T @ weights).T

        # Renormalize
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        return avg_embedding

    agg = grouped.apply(lambda g: pd.Series({
        'total_impressions': g['impressions'].sum(),
        'total_spend': g['spend'].sum(),
        'avg_embedding': weighted_embedding_avg(g)
    }))

    return agg
```

---

### 2.4 Tensor Construction

**Final Tensor Format: `(G, T, C, D)`**

```python
def build_nnn_tensor(df, embed_dim=256):
    """
    Convert processed DataFrame to NNN input tensor.

    Returns:
        X: jnp.array of shape (n_geos, n_weeks, n_channels, embed_dim)
    """

    geos = sorted(df['geo'].unique())
    weeks = sorted(df['date'].unique())
    channels = ['Sales', 'Search', 'YouTube', 'Display', 'SearchAds', 'Social']

    G, T, C = len(geos), len(weeks), len(channels)
    X = np.zeros((G, T, C, embed_dim), dtype=np.float32)

    for g_idx, geo in enumerate(geos):
        for t_idx, week in enumerate(weeks):
            for c_idx, channel in enumerate(channels):

                # Get data for this geo/week/channel
                mask = (df['geo'] == geo) & (df['date'] == week) & (df['channel'] == channel)
                row = df[mask]

                if len(row) == 0:
                    continue

                if channel in ['Sales', 'SearchAds']:
                    # Scalar channel: put value in first dimension
                    X[g_idx, t_idx, c_idx, 0] = row['value'].values[0]
                else:
                    # Embedding channel: volume * normalized direction
                    volume = row['impressions'].values[0]
                    embedding = row['embedding'].values[0]
                    X[g_idx, t_idx, c_idx, :] = embedding * volume

    return jnp.array(X)
```

---

## 3. Data Quality Requirements

### 3.1 Minimum Data Requirements

**Temporal Coverage:**
- ✅ Minimum: 104 weeks (2 years)
- ✅ Recommended: 156 weeks (3 years)
- ✅ Ideal: 208+ weeks (4+ years)

**Geographic Coverage:**
- ✅ Minimum: 5 geos
- ✅ Recommended: 10-20 geos
- ✅ Ideal: 50+ geos (e.g., DMA level)

**Channel Coverage:**
- ✅ At least 3 paid marketing channels
- ✅ At least 1 organic/earned channel
- ✅ Sales as target variable

---

### 3.2 Data Quality Thresholds

**Completeness:**
- < 5% missing weeks per geo (impute if needed)
- < 1% missing channel data per geo/week
- Zero tolerance for missing sales data

**Consistency:**
- Spend and impressions should correlate (Pearson r > 0.7)
- Sales should show clear seasonality patterns
- No unexplained breaks in time series

**Validity:**
- All spend values >= 0
- All impression values >= 0
- Embedding dimensions match across all creatives
- No infinite or NaN values

---

## 4. Example Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Data Sources                              │
├─────────────────────────────────────────────────────────────┤
│ • Google Ads API        • Facebook Ads API                 │
│ • YouTube Analytics     • Internal CRM                      │
│ • Google Trends         • Sales Database                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              ETL & Data Warehouse (BigQuery/Snowflake)      │
├─────────────────────────────────────────────────────────────┤
│ • Daily ingestion       • Data validation                   │
│ • Schema enforcement    • Quality checks                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│ • Generate embeddings   • Weekly aggregation                │
│ • Handle missing data   • Normalize features                │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              NNN Training Data (JAX Tensors)                │
├─────────────────────────────────────────────────────────────┤
│ Shape: (G, T, C, D)                                         │
│ Format: .npy or .pkl files                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Sample Real Data Checklist

Before deploying NNN with real data, ensure:

- [ ] **Sales data** complete for all geos and weeks
- [ ] **Marketing spend data** available for all channels
- [ ] **Creative metadata** including IDs and usage dates
- [ ] **Embedding models** deployed and tested
- [ ] **Data pipeline** automated and monitored
- [ ] **Quality checks** integrated at each stage
- [ ] **Historical data** sufficient (2+ years)
- [ ] **Documentation** for all data sources and transformations
- [ ] **Access controls** and privacy compliance verified
- [ ] **Backup procedures** in place

---

## 6. Common Data Issues and Solutions

| Issue | Solution |
|-------|----------|
| **Missing weeks** | Forward-fill short gaps (< 2 weeks); flag longer gaps |
| **Incomplete creative data** | Use channel-level aggregates; generate synthetic embeddings |
| **Changing geo definitions** | Create consistent geo mapping; exclude transition periods |
| **Campaign ID changes** | Build creative mapping table; use content-based matching |
| **Outlier weeks** | Flag holidays/events; consider separate modeling |
| **Zero spend weeks** | Keep as zero; important signal for attribution |
| **Embedding dimension mismatch** | PCA/pad to consistent dimension |
| **New channels added** | Retrain model; use transfer learning if possible |

---

## 7. Next Steps

After data preparation:
1. ✅ Validate data quality metrics
2. ✅ Run exploratory data analysis (EDA)
3. ✅ Generate baseline statistics
4. ✅ Create train/validation/test splits
5. ✅ Proceed to model training (see Hyperparameter Tuning guide)

---

## References

- NNN Paper: https://arxiv.org/html/2504.06212v1
- CLIP Model: https://github.com/openai/CLIP
- Sentence Transformers: https://www.sbert.net/
