import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_preprocessing(df: pd.DataFrame):
    """
    Final preprocessing pipeline for shark tank data.
    Returns:
        X_scaled : Processed feature matrix (DataFrame)
        y_reg    : Regression target (Total Deal Amount)
        y_cls    : Classification target (Accepted Offer)
        y_shark  : Multi-label target (Individual shark investments)

    NaN handling summary:
        - shark_amt_cols      : fillna(0) — no investment means 0
        - shark_present_cols  : fillna(0) — not present means 0
        - financial_cols/ask_cols : to_numeric + fillna(median)
        - pitcher_cols (numeric): to_numeric + fillna(median)
        - Pitchers Average Age : categorical ('Young'/'Middle'/'Old'),
                                  label-encoded then NaN filled with mode
        - Cash Burn           : binary string ('yes'), converted to 0/1
                                  then NaN filled with 0
        - Started in          : numeric year, fillna(median)
        - Industry            : NaN rows get all-zero dummy columns (safe)
    """
    df = df.copy()

    shark_amt_cols = [
        'Namita Investment Amount', 'Vineeta Investment Amount',
        'Anupam Investment Amount', 'Aman Investment Amount',
        'Peyush Investment Amount', 'Ritesh Investment Amount',
        'Amit Investment Amount'
    ]
    shark_present_cols = [
        'Namita Present', 'Vineeta Present', 'Anupam Present',
        'Aman Present', 'Peyush Present', 'Ritesh Present',
        'Amit Present', 'Guest Present'
    ]

    # ── Shark amounts & presence ─────────────────────────────────────────────
    df[shark_amt_cols]     = df[shark_amt_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df[shark_present_cols] = df[shark_present_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df['sharks_present_count'] = df[shark_present_cols].sum(axis=1)

    y_shark = (df[shark_amt_cols] > 0).astype(int)
    y_shark.columns = [
        'Namita_Invested', 'Vineeta_Invested', 'Anupam_Invested',
        'Aman_Invested', 'Peyush_Invested', 'Ritesh_Invested', 'Amit_Invested'
    ]

    # ── Remove logical fallacy: funded but no shark invested ─────────────────────
    fallacy_mask = (df['Accepted Offer'] == 1) & ((df[shark_amt_cols] > 0).sum(axis=1) == 0)
    df = df[~fallacy_mask].reset_index(drop=True)
    y_shark = y_shark[~fallacy_mask].reset_index(drop=True)  # keep y_shark in sync


    # ── Derived targets ──────────────────────────────────────────────────────
    df['Total Deal Amount']       = df[shark_amt_cols].sum(axis=1)
    df['Number of Sharks in Deal'] = (df[shark_amt_cols] > 0).sum(axis=1)

    # ── Financial & ask columns (numeric) ───────────────────────────────────
    financial_cols = ['Yearly Revenue', 'Monthly Sales', 'Gross Margin',
                      'Net Margin', 'EBITDA', 'SKUs']
    ask_cols       = ['Original Ask Amount', 'Original Offered Equity', 'Valuation Requested']

    num_cols = financial_cols + ask_cols
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ── Cash Burn: binary string column ('yes' / NaN) ───────────────────────
    # Convert to 1/0; NaN (no data) treated as 0 (not burning cash / unknown)
    df['Cash Burn'] = df['Cash Burn'].apply(
        lambda x: 1 if str(x).strip().lower() == 'yes' else 0
    )

    # ── Accepted Offer ───────────────────────────────────────────────────────
    df['Accepted Offer']          = df['Accepted Offer'].fillna(0).astype(int)
    df['Original Offered Equity'] = df['Original Offered Equity'].replace(0, 1e-6)
    df['Original Ask Amount']     = df['Original Ask Amount'].replace(0, 1e-6)

    # ── Engineered financial ratios ──────────────────────────────────────────
    df['ask_per_equity']      = df['Original Ask Amount'] / df['Original Offered Equity']
    df['valuation_ask_ratio'] = df['Valuation Requested']  / df['Original Ask Amount']
    df['revenue_ask_ratio']   = df['Yearly Revenue']       / df['Original Ask Amount']
    df['is_revenue_positive'] = (df['Yearly Revenue'] > 0).astype(int)

    y_reg = df['Total Deal Amount']
    y_cls = df['Accepted Offer']

    # ── Context columns ──────────────────────────────────────────────────────
    context_cols = ['Season Number', 'Season Start', 'Season End', 'Started in', 'Industry']

    # 'Started in': numeric year — fill missing with median year
    df['Started in'] = pd.to_numeric(df['Started in'], errors='coerce')
    df['Started in'] = df['Started in'].fillna(df['Started in'].median())

    # ── Pitcher columns ──────────────────────────────────────────────────────
    numeric_pitcher_cols = [
        'Number of Presenters', 'Male Presenters', 'Female Presenters',
        'Transgender Presenters', 'Couple Presenters'
    ]
    gender_cols = ['Male Presenters', 'Female Presenters',
                   'Transgender Presenters', 'Couple Presenters']

    df[numeric_pitcher_cols] = df[numeric_pitcher_cols].apply(pd.to_numeric, errors='coerce')
    df[numeric_pitcher_cols] = df[numeric_pitcher_cols].fillna(df[numeric_pitcher_cols].median())

    # 'Pitchers Average Age': categorical string ('Young' / 'Middle' / 'Old')
    # Label-encode: Young=0, Middle=1, Old=2; NaN → mode
    age_map = {'Young': 0, 'Middle': 1, 'Old': 2}
    df['Pitchers Average Age'] = df['Pitchers Average Age'].map(age_map)
    age_mode = df['Pitchers Average Age'].mode(dropna=True)
    df['Pitchers Average Age'] = df['Pitchers Average Age'].fillna(
        age_mode[0] if len(age_mode) > 0 else 1  # default to 'Middle' if no mode
    )

    pitcher_cols = numeric_pitcher_cols + ['Pitchers Average Age']

    # ── Engineered pitcher features ──────────────────────────────────────────
    df['team_gender_diversity'] = (
        (df[gender_cols] > 0).sum(axis=1) / df['Number of Presenters'].replace(0, 1)
    )
    df['season_number_norm'] = (
        (df['Season Number'] - df['Season Number'].min()) /
        (df['Season Number'].max() - df['Season Number'].min())
    )

    # ── Assemble feature matrix ──────────────────────────────────────────────
    X = pd.concat([
        df[context_cols + pitcher_cols + ['team_gender_diversity', 'season_number_norm']],
        df[financial_cols + ['Cash Burn'] + ask_cols + [
            'ask_per_equity', 'valuation_ask_ratio', 'revenue_ask_ratio',
            'is_revenue_positive', 'Number of Sharks in Deal'
        ]],
        df[shark_amt_cols + shark_present_cols + ['sharks_present_count']]
    ], axis=1)

    # Industry: get_dummies handles NaN rows as all-zero (safe)
    X = pd.get_dummies(X, columns=['Industry'], drop_first=True)
    X = X.drop(['Season Start', 'Season End'], axis=1, errors='ignore')

    # Final NaN check — should be zero
    remaining_nans = X.isnull().sum().sum()
    if remaining_nans > 0:
        print(f"⚠️  Warning: {remaining_nans} NaN values remain in X. Filling with column medians.")
        X = X.fillna(X.median(numeric_only=True))

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y_reg, y_cls, y_shark
