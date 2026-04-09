import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def run_preprocessing(df: pd.DataFrame):
    """
    Final preprocessing pipeline for shark tank data.
    
    Returns:
        X_scaled: Processed feature matrix
        y_reg: Regression target (Total Deal Amount)
        y_cls: Classification target (Accepted Offer)
        y_shark: Multi-label target (Individual shark investments)
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
    
    df[shark_amt_cols] = df[shark_amt_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    df[shark_present_cols] = df[shark_present_cols].fillna(0)
    
    df['sharks_present_count'] = df[shark_present_cols].sum(axis=1)
    
    y_shark = (df[shark_amt_cols] > 0).astype(int)
    y_shark.columns = [
        'Namita_Invested', 'Vineeta_Invested',
        'Anupam_Invested', 'Aman_Invested',
        'Peyush_Invested', 'Ritesh_Invested',
        'Amit_Invested'
    ]
    
    financial_cols = [
        'Yearly Revenue', 'Monthly Sales', 'Gross Margin',
        'Net Margin', 'EBITDA', 'Cash Burn', 'SKUs'
    ]
    
    ask_cols = [
        'Original Ask Amount', 'Original Offered Equity', 'Valuation Requested'
    ]
    
    df['Total Deal Amount'] = df[shark_amt_cols].sum(axis=1)
    df['Number of Sharks in Deal'] = (df[shark_amt_cols] > 0).sum(axis=1)
    
    num_cols = financial_cols + ask_cols
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    df['Accepted Offer'] = df['Accepted Offer'].fillna(0).astype(int)
    df['Original Offered Equity'] = df['Original Offered Equity'].replace(0, 1e-6)
    df['Original Ask Amount'] = df['Original Ask Amount'].replace(0, 1e-6)
    
    df['ask_per_equity'] = df['Original Ask Amount'] / df['Original Offered Equity']
    df['valuation_ask_ratio'] = df['Valuation Requested'] / df['Original Ask Amount']
    df['revenue_ask_ratio'] = df['Yearly Revenue'] / df['Original Ask Amount']
    df['is_revenue_positive'] = (df['Yearly Revenue'] > 0).astype(int)
    
    y_reg = df['Total Deal Amount']
    y_cls = df['Accepted Offer']
    
    context_cols = [
        'Season Number', 'Season Start', 'Season End',
        'Started in', 'Industry'
    ]
    
    pitcher_cols = [
        'Number of Presenters',
        'Male Presenters', 'Female Presenters',
        'Transgender Presenters', 'Couple Presenters',
        'Pitchers Average Age'
    ]
    
    num_cols_p1 = pitcher_cols
    df[num_cols_p1] = df[num_cols_p1].apply(pd.to_numeric, errors='coerce')
    df[num_cols_p1] = df[num_cols_p1].fillna(df[num_cols_p1].median())
    
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
    
    X_person1 = df[
        context_cols +
        pitcher_cols +
        ['team_gender_diversity', 'season_number_norm']
    ].copy()
    
    X_person2 = df[
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
    
    X_person3 = df[
        shark_amt_cols +
        shark_present_cols +
        ['sharks_present_count']
    ].copy()
    
    X = pd.concat([X_person1, X_person2, X_person3], axis=1)
    
    X = pd.get_dummies(X, columns=['Industry'], drop_first=True)
    X = X.drop(['Season Start', 'Season End'], axis=1, errors='ignore')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y_reg, y_cls, y_shark