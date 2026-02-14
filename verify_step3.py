import pandas as pd

df = pd.read_csv('data/hosts_with_issues.csv')

print("Hosts with issues detected:")
print(f"Total rows: {len(df)}")
print(f"Unique hosts: {df['SessionHostName'].nunique()}")
print()

cols = ['SessionHostName', 'network_issue', 'session_host_issue', 'capacity_issue', 
        'fslogix_issue', 'disk_issue', 'hygiene_issue', 'client_issue']
print(df[cols].head(20).to_string(index=False))

print('\n' + '='*80)
print('Issue breakdown:')
for col in ['network_issue','session_host_issue','capacity_issue','fslogix_issue','disk_issue','hygiene_issue','client_issue']:
    print(f'{col}: {df[col].sum()}')

print('\nTop 10 hosts by issue count:')
print(df['SessionHostName'].value_counts().head(10).to_string())
