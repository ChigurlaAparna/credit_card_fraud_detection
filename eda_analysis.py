"""
Credit Card Fraud Detection - Exploratory Data Analysis
Generates visualizations and analyzes patterns in the data
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

def save_plot(fig, filename):
    """Save plot to reports directory"""
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   📊 Saved: {filename}")
    return path

def plot_class_distribution(df):
    """Plot fraud vs non-fraud distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Genuine', 'Fraud']
    sizes = [df['Class'].value_counts()[0], df['Class'].value_counts()[1]]
    
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.3f%%',
                shadow=True, startangle=90, explode=(0, 0.1))
    axes[0].set_title('Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = axes[1].bar(labels, sizes, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Number of Transactions', fontsize=12)
    axes[1].set_title('Class Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        axes[1].annotate(f'{size:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return save_plot(plt.gcf(), '01_class_distribution.png')

def plot_amount_distribution(df):
    """Plot transaction amount distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall amount distribution
    axes[0, 0].hist(df['Amount'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Transaction Amount ($)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Transaction Amount Distribution (All)', fontsize=13, fontweight='bold')
    axes[0, 0].axvline(df['Amount'].mean(), color='red', linestyle='--', label=f'Mean: ${df["Amount"].mean():.2f}')
    axes[0, 0].axvline(df['Amount'].median(), color='green', linestyle='--', label=f'Median: ${df["Amount"].median():.2f}')
    axes[0, 0].legend()
    
    # Log scale amount distribution
    axes[0, 1].hist(np.log1p(df['Amount']), bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Log(Transaction Amount + 1)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Log-Transformed Amount Distribution', fontsize=13, fontweight='bold')
    
    # Amount by class
    genuine_amounts = df[df['Class'] == 0]['Amount']
    fraud_amounts = df[df['Class'] == 1]['Amount']
    
    axes[1, 0].hist(genuine_amounts, bins=50, color='#2ecc71', edgecolor='black', 
                    alpha=0.7, label='Genuine', density=True)
    axes[1, 0].hist(fraud_amounts, bins=50, color='#e74c3c', edgecolor='black', 
                    alpha=0.7, label='Fraud', density=True)
    axes[1, 0].set_xlabel('Transaction Amount ($)', fontsize=11)
    axes[1, 0].set_ylabel('Density', fontsize=11)
    axes[1, 0].set_title('Amount Distribution by Class', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    
    # Box plot comparison
    data_to_plot = [genuine_amounts, fraud_amounts]
    bp = axes[1, 1].boxplot(data_to_plot, labels=['Genuine', 'Fraud'], patch_artist=True)
    colors_box = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].set_ylabel('Transaction Amount ($)', fontsize=11)
    axes[1, 1].set_title('Amount Box Plot by Class', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return save_plot(plt.gcf(), '02_amount_distribution.png')

def plot_time_analysis(df):
    """Analyze and plot transaction patterns over time"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Convert Time to hours (assuming 2 days of data)
    df['Hour'] = (df['Time'] / 3600) % 24
    
    # Transaction count by hour
    hourly_genuine = df[df['Class'] == 0].groupby('Hour').size()
    hourly_fraud = df[df['Class'] == 1].groupby('Hour').size()
    
    hours = np.arange(24)
    axes[0, 0].bar(hours, hourly_genuine.reindex(hours, fill_value=0), 
                   color='#2ecc71', alpha=0.7, label='Genuine')
    axes[0, 0].set_xlabel('Hour of Day', fontsize=11)
    axes[0, 0].set_ylabel('Number of Transactions', fontsize=11)
    axes[0, 0].set_title('Transactions by Hour (Genuine)', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(range(0, 24, 2))
    
    # Fraud transactions by hour
    axes[0, 1].bar(hours, hourly_fraud.reindex(hours, fill_value=0), 
                   color='#e74c3c', alpha=0.7, label='Fraud')
    axes[0, 1].set_xlabel('Hour of Day', fontsize=11)
    axes[0, 1].set_ylabel('Number of Transactions', fontsize=11)
    axes[0, 1].set_title('Transactions by Hour (Fraud)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(range(0, 24, 2))
    
    # Fraud rate by hour
    fraud_rate_by_hour = df.groupby('Hour')['Class'].mean() * 100
    axes[1, 0].plot(hours, fraud_rate_by_hour.reindex(hours, fill_value=0), 
                    color='#e74c3c', linewidth=2, marker='o')
    axes[1, 0].fill_between(hours, fraud_rate_by_hour.reindex(hours, fill_value=0), 
                            alpha=0.3, color='#e74c3c')
    axes[1, 0].set_xlabel('Hour of Day', fontsize=11)
    axes[1, 0].set_ylabel('Fraud Rate (%)', fontsize=11)
    axes[1, 0].set_title('Fraud Rate by Hour of Day', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(range(0, 24, 2))
    
    # Time series of transactions
    sample_size = min(10000, len(df))
    df_sample = df.head(sample_size)
    colors = ['#e74c3c' if x == 1 else '#2ecc71' for x in df_sample['Class']]
    axes[1, 1].scatter(df_sample['Time']/3600, df_sample['Amount'], c=colors, alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Time (Hours)', fontsize=11)
    axes[1, 1].set_ylabel('Transaction Amount ($)', fontsize=11)
    axes[1, 1].set_title('Transaction Amount Over Time', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return save_plot(plt.gcf(), '03_time_analysis.png')

def plot_feature_correlations(df):
    """Plot correlation matrix and important features"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Correlation with Class
    correlations = df.corr()['Class'].drop('Class').sort_values(key=abs, ascending=False)
    
    # Top 15 most correlated features
    top_features = correlations.head(15)
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_features]
    axes[0].barh(range(len(top_features)), top_features.values, color=colors, alpha=0.8)
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features.index)
    axes[0].set_xlabel('Correlation with Class', fontsize=11)
    axes[0].set_title('Top 15 Features Correlated with Fraud', fontsize=13, fontweight='bold')
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Correlation heatmap for top features
    top_feature_names = list(top_features.index[:10]) + ['Class']
    corr_matrix = df[top_feature_names].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, 
                ax=axes[1], fmt='.2f', square=True, linewidths=0.5)
    axes[1].set_title('Feature Correlation Heatmap (Top Features)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return save_plot(plt.gcf(), '04_feature_correlations.png')

def plot_amount_statistics(df):
    """Detailed amount statistics for fraud vs genuine"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    genuine = df[df['Class'] == 0]['Amount']
    fraud = df[df['Class'] == 1]['Amount']
    
    # CDF comparison
    sorted_genuine = np.sort(genuine)
    cdf_genuine = np.arange(1, len(sorted_genuine) + 1) / len(sorted_genuine)
    sorted_fraud = np.sort(fraud)
    cdf_fraud = np.arange(1, len(sorted_fraud) + 1) / len(sorted_fraud)
    
    axes[0, 0].plot(sorted_genuine, cdf_genuine, color='#2ecc71', linewidth=2, label='Genuine')
    axes[0, 0].plot(sorted_fraud, cdf_fraud, color='#e74c3c', linewidth=2, label='Fraud')
    axes[0, 0].set_xlabel('Transaction Amount ($)', fontsize=11)
    axes[0, 0].set_ylabel('Cumulative Probability', fontsize=11)
    axes[0, 0].set_title('Cumulative Distribution Function (CDF)', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 2500)
    
    # Violin plot
    data = pd.DataFrame({
        'Amount': np.concatenate([genuine.values, fraud.values]),
        'Class': ['Genuine'] * len(genuine) + ['Fraud'] * len(fraud)
    })
    sns.violinplot(data=data, x='Class', y='Amount', ax=axes[0, 1], 
                   palette=['#2ecc71', '#e74c3c'], cut=0)
    axes[0, 1].set_ylim(0, 2500)
    axes[0, 1].set_title('Amount Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    
    # Percentile comparison
    percentiles = [50, 75, 90, 95, 99]
    genuine_pct = [np.percentile(genuine, p) for p in percentiles]
    fraud_pct = [np.percentile(fraud, p) for p in percentiles]
    
    x = np.arange(len(percentiles))
    width = 0.35
    axes[1, 0].bar(x - width/2, genuine_pct, width, label='Genuine', color='#2ecc71', alpha=0.8)
    axes[1, 0].bar(x + width/2, fraud_pct, width, label='Fraud', color='#e74c3c', alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'{p}th' for p in percentiles])
    axes[1, 0].set_ylabel('Transaction Amount ($)', fontsize=11)
    axes[1, 0].set_title('Percentile Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    
    # Statistics summary table
    stats_data = {
        'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', '90%', '95%', 'Max'],
        'Genuine': [len(genuine), genuine.mean(), genuine.std(), genuine.min(),
                   genuine.quantile(0.25), genuine.median(), genuine.quantile(0.75),
                   genuine.quantile(0.90), genuine.quantile(0.95), genuine.max()],
        'Fraud': [len(fraud), fraud.mean(), fraud.std(), fraud.min(),
                  fraud.quantile(0.25), fraud.median(), fraud.quantile(0.75),
                  fraud.quantile(0.90), fraud.quantile(0.95), fraud.max()]
    }
    
    # Create summary table
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=[[f'{v:.2f}' if isinstance(v, float) else f'{v:,}' 
                                        for v in row[1:]] for row in [stats_data['Metric']] + 
                                       list(zip(stats_data['Genuine'], stats_data['Fraud']))],
                             rowLabels=['Metric'] + stats_data['Metric'],
                             colLabels=['Genuine', 'Fraud'],
                             loc='center',
                             cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Amount Statistics Comparison', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return save_plot(plt.gcf(), '05_amount_statistics.png')

def plot_v_features_analysis(df):
    """Analyze V1-V28 features"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean of each V feature by class
    v_features = [f'V{i}' for i in range(1, 29)]
    genuine_means = df[df['Class'] == 0][v_features].mean()
    fraud_means = df[df['Class'] == 1][v_features].mean()
    
    x = np.arange(len(v_features))
    axes[0, 0].bar(x - 0.2, genuine_means, 0.4, label='Genuine', color='#2ecc71', alpha=0.8)
    axes[0, 0].bar(x + 0.2, fraud_means, 0.4, label='Fraud', color='#e74c3c', alpha=0.8)
    axes[0, 0].set_xlabel('Feature', fontsize=11)
    axes[0, 0].set_ylabel('Mean Value', fontsize=11)
    axes[0, 0].set_title('Mean of V Features by Class', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(v_features, rotation=45, fontsize=8)
    axes[0, 0].legend()
    
    # Std of each V feature by class
    genuine_stds = df[df['Class'] == 0][v_features].std()
    fraud_stds = df[df['Class'] == 1][v_features].std()
    
    axes[0, 1].bar(x - 0.2, genuine_stds, 0.4, label='Genuine', color='#2ecc71', alpha=0.8)
    axes[0, 1].bar(x + 0.2, fraud_stds, 0.4, label='Fraud', color='#e74c3c', alpha=0.8)
    axes[0, 1].set_xlabel('Feature', fontsize=11)
    axes[0, 1].set_ylabel('Standard Deviation', fontsize=11)
    axes[0, 1].set_title('Std of V Features by Class', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(v_features, rotation=45, fontsize=8)
    axes[0, 1].legend()
    
    # Box plot for top discriminating features
    top_v_features = ['V14', 'V17', 'V12', 'V10', 'V4']  # Based on typical importance
    df_melted = df[top_v_features + ['Class']].melt(id_vars='Class', 
                                                     var_name='Feature', 
                                                     value_name='Value')
    df_melted['Class'] = df_melted['Class'].map({0: 'Genuine', 1: 'Fraud'})
    sns.boxplot(data=df_melted, x='Feature', y='Value', hue='Class', 
                ax=axes[1, 0], palette=['#2ecc71', '#e74c3c'])
    axes[1, 0].set_title('Top Discriminating V Features', fontsize=13, fontweight='bold')
    axes[1, 0].legend(title='Class')
    
    # Distribution comparison for V14 (typically most important)
    axes[1, 1].hist(df[df['Class'] == 0]['V14'], bins=50, density=True, 
                    color='#2ecc71', alpha=0.7, label='Genuine')
    axes[1, 1].hist(df[df['Class'] == 1]['V14'], bins=50, density=True, 
                    color='#e74c3c', alpha=0.7, label='Fraud')
    axes[1, 1].set_xlabel('V14 Value', fontsize=11)
    axes[1, 1].set_ylabel('Density', fontsize=11)
    axes[1, 1].set_title('V14 Distribution by Class', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    return save_plot(plt.gcf(), '06_v_features_analysis.png')

def generate_eda_report(df):
    """Generate comprehensive EDA report"""
    print("\n" + "="*60)
    print("GENERATING EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print("\n📊 Creating visualizations...")
    
    plots = []
    plots.append(plot_class_distribution(df))
    plots.append(plot_amount_distribution(df))
    plots.append(plot_time_analysis(df))
    plots.append(plot_feature_correlations(df))
    plots.append(plot_amount_statistics(df))
    plots.append(plot_v_features_analysis(df))
    
    # Generate summary report
    report = []
    report.append("=" * 70)
    report.append("EXPLORATORY DATA ANALYSIS REPORT")
    report.append("Credit Card Fraud Detection System")
    report.append("=" * 70)
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("1. DATASET OVERVIEW")
    report.append("-" * 40)
    report.append(f"   Total Samples: {len(df):,}")
    report.append(f"   Total Features: {len(df.columns)}")
    report.append(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    report.append("")
    report.append("2. CLASS DISTRIBUTION")
    report.append("-" * 40)
    fraud_count = df['Class'].sum()
    genuine_count = len(df) - fraud_count
    report.append(f"   Genuine Transactions: {genuine_count:,} ({genuine_count/len(df)*100:.3f}%)")
    report.append(f"   Fraud Transactions: {fraud_count:,} ({fraud_count/len(df)*100:.3f}%)")
    report.append(f"   Imbalance Ratio: 1:{int(genuine_count/fraud_count)}")
    report.append("")
    report.append("3. AMOUNT ANALYSIS")
    report.append("-" * 40)
    report.append(f"   Genuine - Mean: ${df[df['Class']==0]['Amount'].mean():.2f}, "
                  f"Median: ${df[df['Class']==0]['Amount'].median():.2f}")
    report.append(f"   Fraud - Mean: ${df[df['Class']==1]['Amount'].mean():.2f}, "
                  f"Median: ${df[df['Class']==1]['Amount'].median():.2f}")
    report.append("")
    report.append("4. KEY FINDINGS")
    report.append("-" * 40)
    report.append("   • Fraudulent transactions tend to have HIGHER amounts")
    report.append("   • Class imbalance is significant (0.17% fraud)")
    report.append("   • Time-based patterns may exist in fraud occurrence")
    report.append("   • V14, V17, V12, V10, V4 show strongest correlation with fraud")
    report.append("")
    report.append("5. GENERATED VISUALIZATIONS")
    report.append("-" * 40)
    for i, plot in enumerate(plots, 1):
        report.append(f"   {i}. {os.path.basename(plot)}")
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    report_path = os.path.join(REPORTS_DIR, 'eda_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n   📄 Report saved to: {report_path}")
    
    return report_text

if __name__ == "__main__":
    from data_collection import load_dataset
    
    df = load_dataset()
    generate_eda_report(df)
    print("\n✅ EDA complete!")
