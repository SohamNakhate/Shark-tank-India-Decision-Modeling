import pandas as pd

def run_preprocessing(df: pd.DataFrame):
    """
    FINAL PREPROCESSING PIPELINE

    Input:
        df -> DataFrame from dataloader.py

    Returns:
        X -> final feature matrix
        y_reg -> regression target
        y_cls -> classification target
        y_shark -> multi-label target
    """

    # ===============================
    # STEP 1 — PERSON 3 (SHARK DATA)
    # ===============================

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

    # Fill NaNs → 0
    df[shark_amt_cols] = df[shark_amt_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df[shark_present_cols] = df[shark_present_cols].fillna(0)

    # Feature
    df['sharks_present_count'] = df[shark_present_cols].sum(axis=1)

    # Target (Objective 3)
    y_shark = (df[shark_amt_cols] > 0).astype(int)
    y_shark.columns = [
        'Namita_Invested', 'Vineeta_Invested',
        'Anupam_Invested', 'Aman_Invested',
        'Peyush_Invested', 'Ritesh_Invested',
        'Amit_Invested'
    ]

    df_person3 = df[
        shark_amt_cols +
        shark_present_cols +
        ['sharks_present_count']
    ].copy()

    # ===============================
    # STEP 2 — PERSON 2 (FINANCIAL)
    # ===============================

    financial_cols = [
        'Yearly Revenue', 'Monthly Sales', 'Gross Margin',
        'Net Margin', 'EBITDA', 'Cash Burn', 'SKUs'
    ]

    ask_cols = [
        'Original Ask Amount', 'Original Offered Equity', 'Valuation Requested'
    ]

    # 🔥 RECOMPUTE (CRITICAL)
    df['Total Deal Amount'] = df_person3[shark_amt_cols].sum(axis=1)
    df['Number of Sharks in Deal'] = (df_person3[shark_amt_cols] > 0).sum(axis=1)

    # Clean numeric
    num_cols = financial_cols + ask_cols
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Clean target
    df['Accepted Offer'] = df['Accepted Offer'].fillna(0).astype(int)

    # Avoid division by zero
    df['Original Offered Equity'] = df['Original Offered Equity'].replace(0, 1e-6)
    df['Original Ask Amount'] = df['Original Ask Amount'].replace(0, 1e-6)

    # Feature engineering
    df['ask_per_equity'] = df['Original Ask Amount'] / df['Original Offered Equity']
    df['valuation_ask_ratio'] = df['Valuation Requested'] / df['Original Ask Amount']
    df['revenue_ask_ratio'] = df['Yearly Revenue'] / df['Original Ask Amount']
    df['is_revenue_positive'] = (df['Yearly Revenue'] > 0).astype(int)

    # Targets
    y_reg = df['Total Deal Amount']
    y_cls = df['Accepted Offer']

    df_person2 = df[
        financial_cols +
        ask_cols +
        [
            'ask_per_equity',
            'valuation_ask_ratio',
            'revenue_ask_ratio',
            'is_revenue_positive',
            'Number of Sharks in Deal'
        ]
    ].copy()

    # ===============================
    # STEP 3 — PERSON 1 (CONTEXT)
    # ===============================

    context_cols = [
        'Season Number', 'Season Start', 'Season End',
        'Started in', 'Industry'
    ]

    pitcher_cols = [
        'Number of Presenters',
        'Male Presenters', 'Female Presenters',
        'Transgender Presenters', 'Couple Presenters',
        'Pitchers Average Age',
        'Pitchers City', 'Pitchers State'
    ]

    num_cols_p1 = [
        'Number of Presenters',
        'Male Presenters', 'Female Presenters',
        'Transgender Presenters', 'Couple Presenters',
        'Pitchers Average Age'
    ]

    df[num_cols_p1] = df[num_cols_p1].apply(pd.to_numeric, errors='coerce')
    df[num_cols_p1] = df[num_cols_p1].fillna(df[num_cols_p1].median())

    cat_cols = ['Industry', 'Pitchers City', 'Pitchers State']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Features
    gender_cols = [
        'Male Presenters', 'Female Presenters',
        'Transgender Presenters', 'Couple Presenters'
    ]

    df['team_gender_diversity'] = (
        (df[gender_cols] > 0).sum(axis=1) /
        df['Number of Presenters'].replace(0, 1)
    )

    df['season_number_norm'] = (
        (df['Season Number'] - df['Season Number'].min()) /
        (df['Season Number'].max() - df['Season Number'].min())
    )

    df_person1 = df[
        context_cols +
        pitcher_cols +
        ['team_gender_diversity', 'season_number_norm']
    ].copy()

    # ===============================
    # STEP 4 — MERGE ALL
    # ===============================

    X = pd.concat([df_person1, df_person2, df_person3], axis=1)

    return X, y_reg, y_cls, y_shark
