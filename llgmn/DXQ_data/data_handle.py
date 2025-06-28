import pandas as pd
import os

# 要处理的文件名列表
names = ['open', 'close', 'up', 'down']
input_dir = 'llgmn/DXQ_data'
output_dir = 'llgmn/DXQ_data'

# 只保留的列
cols = ['processed_ch1', 'processed_ch2', 'processed_ch3', 'processed_ch4']

# # 可选：自动检测小数位数
# df_ref = pd.read_csv('llgmn/data/data_train_movement.csv', nrows=5)
# decimal_places = df_ref.applymap(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0).max().max()
decimal_places = 9  # 你可以根据需要调整

for name in names:
    input_path = os.path.join(input_dir, f'{name}.csv')
    output_path = os.path.join(output_dir, f'{name}_processed.csv')
    df = pd.read_csv(input_path)
    df_new = df[cols]
    df_new = df_new.round(decimal_places)
    df_new.to_csv(output_path, index=False)
    print(f'已处理并保存: {output_path}')