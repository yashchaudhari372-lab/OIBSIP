"""
Oasis Infobyte SIP — Data Science Internship
Task 2: Unemployment Analysis with Python
Intern: Yash Devidas Chaudhari | Track: Data Science
Reference: OIB/N2/IP5780
"""

import os
import pandas as pd
import numpy as np

def main():
    print("=" * 68)
    print("📊 OASIS INFOBYTE SIP — DATA SCIENCE INTERNSHIP")
    print("📌 TASK 2: UNEMPLOYMENT ANALYSIS WITH PYTHON")
    print("👤 Intern: Yash Devidas Chaudhari | Track: Data Science")
    print("=" * 68)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Unemployment_in_India.csv")
    
    if not os.path.exists(csv_path):
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/amankharwal/Website-data/master/unemployment.csv", csv_path)
        
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        'Region': 'State',
        'Region.1': 'Zone',
        'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
        'Estimated Employed': 'Employed',
        'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate'
    }, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%m-%Y', errors='coerce')
    df.dropna(subset=['Date', 'Unemployment_Rate'], inplace=True)
    
    print(f"\n[1/5] Loaded Dataset: {df.shape[0]} records across {df['State'].nunique()} Indian States/UTs")
    print(f"      Time Range: {df['Date'].min().strftime('%B %Y')} to {df['Date'].max().strftime('%B %Y')}")
    print(f"      Overall Average Unemployment Rate: {df['Unemployment_Rate'].mean():.2f}%")
    
    print("\n[2/5] Top 5 States with Highest Average Unemployment:")
    top5 = df.groupby('State')['Unemployment_Rate'].mean().sort_values(ascending=False).head(5)
    for state, rate in top5.items():
        print(f"      • {state:<18}: {rate:>6.2f}%")
        
    print("\n[3/5] Zone-wise Average Unemployment:")
    zone_avg = df.groupby('Zone')['Unemployment_Rate'].mean().sort_values(ascending=False)
    for zone, rate in zone_avg.items():
        print(f"      • {zone:<12}: {rate:>6.2f}%")
        
    print("\n[4/5] COVID-19 Lockdown Impact Comparison:")
    lockdown_date = pd.to_datetime('2020-03-24')
    pre_covid = df[df['Date'] < lockdown_date]['Unemployment_Rate'].mean()
    post_covid = df[df['Date'] >= lockdown_date]['Unemployment_Rate'].mean()
    
    print(f"      • Pre-Lockdown Mean (Jan - Mar 2020): {pre_covid:.2f}%")
    print(f"      • Post-Lockdown Mean (Apr - Oct 2020): {post_covid:.2f}%")
    increase = post_covid - pre_covid
    pct_increase = (increase / pre_covid) * 100
    print(f"      ➔ Absolute Increase: +{increase:.2f}% percentage points (+{pct_increase:.1f}% relative surge)")
    
    print("\n[5/5] Correlation Matrix:")
    corr = df[['Unemployment_Rate', 'Employed', 'Labour_Participation_Rate']].corr()
    print(corr.round(3).to_string())
    
    print("\n" + "=" * 68)
    print("✔ Task 2 Analysis Completed Successfully!")
    print("=" * 68)

if __name__ == '__main__':
    main()
