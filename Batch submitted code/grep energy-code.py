import pandas as pd

# 文件路径
file_path = 'energy_values.txt'  # 将此路径替换为您的 txt 文件路径

# 读取文件内容
with open(file_path, 'r') as file:
    raw_data = file.read()

# 分割 raw_data 为多行，并过滤掉不包含数据的行
lines = raw_data.strip().split('\n')[2:]  # 跳过前两行
lines = [line for line in lines if not '---' in line]  # 过滤掉包含 '---' 的行

# 处理每行数据，分割并清洗
data = []
for line in lines:
    # 分割每行的字符串为列表
    parts = line.split('|')
    # 移除列表中空的和多余空格的元素
    parts_cleaned = [part.strip() for part in parts if part.strip() != '']
    data.append(parts_cleaned)

# 使用清洗后的数据创建一个 DataFrame
df = pd.DataFrame(data, columns=['FolderName', 'Energy', 'Convergence'])

# 转换 Energy 列为数值类型
df['Energy'] = pd.to_numeric(df['Energy'])

# 输出 DataFrame
print(df)

# 如果需要，可以将 DataFrame 保存为 CSV 文件
df.to_csv('energy_data.csv', index=False)
