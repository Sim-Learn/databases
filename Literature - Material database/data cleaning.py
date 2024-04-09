import pandas as pd
import re

# 加载Excel文件
file_path = r'D:\ML\ysn_pdf\output_dir\代码V4-V6结果\提取数据库-v1.xlsx'  # 修改为您的文件路径
data = pd.read_excel(file_path)

# 定义数据清理函数
def clean_data(value):
    if pd.isna(value):
        return pd.NA  # 如果是NaN，保持不变

    # 移除括号及其内容
    value_clean = re.sub(r'\([^)]*\)', '', str(value))

    # 检查并处理含有'mV'的情况，确保只包含一个数值
    if 'mV' in value_clean:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value_clean)
        if len(numbers) == 1:
            try:
                # 转换为V
                num = float(numbers[0])
                return f'{num / 1000} V'
            except:
                return pd.NA
        else:
            return pd.NA
    else:
        # 确保只包含一个数值或一个数值加单位
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value_clean)
        units = re.findall(r'[a-zA-Z]+', value_clean)
        if len(numbers) == 1 and len(units) <= 1:
            unit = units[0] if units else ''
            try:
                num = float(numbers[0])
                return f'{num} {unit}'.strip()
            except:
                return pd.NA
        else:
            return pd.NA

# 应用数据清理函数到后12列
for column in data.columns[-12:]:
    data[column] = data[column].apply(clean_data)

# 保存清理后的数据到新的Excel文件
cleaned_file_path = r'D:\ML\ysn_pdf\output_dir\代码V4-V6结果\提取数据库-v2.xlsx'  # 修改为您希望保存的文件路径
data.to_excel(cleaned_file_path, index=False)


