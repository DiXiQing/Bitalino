import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('llgmn/test_data/M3.csv')

print("数据基本信息:")
print(df.describe())
print(f"\n数据行数: {len(df)}")
print(f"采样频率推测: 如果总时长2秒，则采样率={len(df)/2}Hz")

# 可视化原始数据
fig, axes = plt.subplots(4, 1, figsize=(12, 8))

for i, col in enumerate(['raw_ch1', 'raw_ch2', 'raw_ch3', 'raw_ch4']):
    axes[i].plot(df[col], label=f'Ch{i+1}')
    axes[i].set_ylabel(f'Channel {i+1}')
    axes[i].grid(True)
    axes[i].legend()

plt.xlabel('Sample Number')
plt.tight_layout()
plt.savefig('llgmn/test_data/raw_emg_signal.png')
plt.show()

# 可视化处理后的数据
fig, axes = plt.subplots(4, 1, figsize=(12, 8))

for i, col in enumerate(['processed_ch1', 'processed_ch2', 'processed_ch3', 'processed_ch4']):
    axes[i].plot(df[col], label=f'Ch{i+1} (Processed)', color='orange')
    axes[i].set_ylabel(f'Channel {i+1}')
    axes[i].grid(True)
    axes[i].legend()

plt.xlabel('Sample Number')
plt.tight_layout()
plt.savefig('llgmn/test_data/processed_emg_signal.png')
plt.show()