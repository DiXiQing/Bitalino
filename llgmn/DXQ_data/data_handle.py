import pandas as pd

# 1. 读取 open.csv
df = pd.read_csv('llgmn/DXQ_data/up(rigth).csv')

# 2. 只保留 processed_ch1 - processed_ch4
cols = ['processed_ch1', 'processed_ch2', 'processed_ch3', 'processed_ch4']
df_new = df[cols]

# # 3. 查看 data_train_movement.csv 的小数位数
# df_ref = pd.read_csv('llgmn/data/data_train_movement.csv', nrows=5)
# # 假设前辈的数据保留了3位小数
# decimal_places = df_ref.applymap(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0).max().max()

# 4. 按小数位数格式化
df_new = df_new.round(9)

# 5. 保存为新文件
df_new.to_csv('llgmn/DXQ_data/rigth_processed.csv', index=False)