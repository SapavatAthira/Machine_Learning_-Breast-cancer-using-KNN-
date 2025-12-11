import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """Plots a scatter plot for Monthly Household Income vs Expense."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['Mthly_HH_Income'], y=df['Mthly_HH_Expense'])
    plt.xlabel('Monthly Household Income')
    plt.ylabel('Monthly Household Expense')
    plt.title('Relational Plot: Income vs Expense')
    plt.savefig('relational_plot.png')
    return


def plot_categorical_plot(df):
    """Plots a boxplot for Highest Qualified Member vs Income."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['Highest_Qualified_Member'], y=df['Mthly_HH_Income'])
    plt.xlabel('Highest Qualified Member')
    plt.ylabel('Monthly Household Income')
    plt.title('Categorical Plot: Qualification vs Income')
    plt.xticks(rotation=45)
    plt.savefig('categorical_plot.png')
    return


def plot_statistical_plot(df):
    """Plots a histogram with KDE for Monthly Household Income."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Mthly_HH_Income'], kde=True, bins=30)
    plt.xlabel('Monthly Household Income')
    plt.ylabel("Frequency")
    plt.title("Distribution of Monthly Household Income")
    plt.savefig('statistical_plot.png')
    return


def statistical_analysis(df):
    """Calculates mean, standard deviation, skewness, and kurtosis. """
    col = 'Mthly_HH_Income'
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col].dropna())
    excess_kurtosis = ss.kurtosis(df[col].dropna(), fisher=True)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Displays dataset overview, first few rows, and correlation matrix."""
    print("Dataset Overview:\n", df.describe())
    print("First few rows:\n", df.head())
    numeric_df = df.select_dtypes(include=[np.number])
    print("Correlation Matrix:\n", numeric_df.corr())

    return df


def writing(moments):
    """Prints statistical moments in a readable format."""
    print(
        f'Mean = {moments[0]:.2f}, '
        f'Standard Deviation = {moments[1]:.2f}, '
        f'Skewness = {moments[2]:.2f}, and '
        f'Excess Kurtosis = {moments[3]:.2f}.'
    )
    skewness_desc = (
        "not skewed" if -2 < moments[2] < 2
        else ("right-skewed" if moments[2] > 2 else "left-skewed")
    )
    kurtosis_desc = (
        "mesokurtic" if -2 < moments[3] < 2
        else ("leptokurtic" if moments[3] > 2 else "platykurtic")
    )
    print(f'The data was {skewness_desc} and {kurtosis_desc}.')
    return


def main():
    """Main function to execute the workflow."""
    file_path = r'data.csv'
    df = pd.read_csv(file_path)
    col = 'Mthly_HH_Income'
    print(f'For the attribute {col}:')
    df = preprocessing(df)
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df)
    writing(moments)
    return


if __name__ == '__main__':
    main()
