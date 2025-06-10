import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


def add_elapsed_minutes(df):
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df = df.sort_values("Time")
    df["Elapsed_min"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds() / 60
    return df


# Read each file with its correponding power 
df_0 = pd.read_csv("changes_battery_0dBm.csv")
df_0["Power_dBm"] = 0

df_neg20 = pd.read_csv("changes_battery_-20dBm.csv")
df_neg20["Power_dBm"] = -20

df_4 = pd.read_csv("changes_battery_4dBm.csv")
df_4["Power_dBm"] = 4

# Add 'Sample' column
df_0["Sample"] = range(1, len(df_0) + 1)
df_neg20["Sample"] = range(1, len(df_neg20) + 1)
df_4["Sample"] = range(1, len(df_4) + 1)

df_0 = add_elapsed_minutes(df_0)
df_neg20 = add_elapsed_minutes(df_neg20)
df_4 = add_elapsed_minutes(df_4)

# Concatenate all in a unique DataFrame
df = pd.concat([df_0, df_neg20, df_4], ignore_index=True)

# BATTERY VS. TIME
plt.figure()
plt.suptitle("Evolution of the battery over time", fontsize=13, fontweight="bold")
sns.lineplot(data=df, x='Sample', y='Battery_%', hue='Power_dBm', markers=True, palette='Set1')
plt.xlabel("Time (samples)", fontsize=9)
plt.ylabel("Battery (%)", fontsize=9)
plt.title("Different transmission powers", fontsize=11)
plt.grid(True)
plt.legend(title='Power (dBm)', loc="upper right", fontsize=8, title_fontsize=8)
plt.tight_layout()

# VOLTAGE VS. TIME
plt.figure()
plt.suptitle("Evolution of the voltage over time", fontsize=13, fontweight="bold")
sns.lineplot(data=df, x='Sample', y='Voltage_mV', hue='Power_dBm', markers=True, palette='Set1')
plt.xlabel("Time (samples)", fontsize=9)
plt.ylabel("Voltage (mV)", fontsize=9)
plt.title("Different transmission powers", fontsize=11)
plt.grid(True)
plt.legend(title='Power (dBm)', loc="upper right", fontsize=8, title_fontsize=8)
plt.tight_layout()

# LINEAR REGRESSION
plt.figure(figsize=(12, 6))
plt.suptitle("Linear regression of battery discharge", fontsize=13, fontweight="bold")
for power in sorted(df['Power_dBm'].unique()):
    sub_df = df[df['Power_dBm'] == power]
    X = sub_df[['Sample']]
    y = sub_df['Battery_%']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.plot(sub_df['Sample'], y, 'o', label=f'Data {power} dBm')
    plt.plot(sub_df['Sample'], y_pred, '-', label=f'Regression {power} dBm')
    
plt.xlabel("Time (samples)", fontsize=9)
plt.ylabel("Battery (%)", fontsize=9)
plt.title("Different transmission powers", fontsize=11)
plt.grid(True)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()


plt.show(block=False)
input("Press [enter] key to close plots...")
