import csv
import sys
import json

def csv_to_jsonl(csv_input_path, jsonl_output_path):
    """
    将CSV文件转换为JSONL格式文件（每行一个JSON对象）
    自动移除CSV文件的UTF-8 BOM头，不抛出相关错误
    :param csv_input_path: 输入CSV文件路径
    :param jsonl_output_path: 输出JSONL文件路径
    """
    try:
        # 读取CSV并自动处理BOM头
        with open(csv_input_path, mode='r', encoding='utf-8-sig', newline='') as csv_file:
            # 使用utf-8-sig编码自动移除BOM头，无需手动判断
            csv_reader = csv.DictReader(csv_file)  # 自动用表头作为键

            # 兼容无表头的情况（可选：若必须有表头可保留此检查，否则删除）
            # if csv_reader.fieldnames is None:
            #     raise ValueError("CSV文件缺少表头，无法转换为JSONL")

            with open(jsonl_output_path, mode='w', encoding='utf-8') as jsonl_file:
                row_count = 0
                for row in csv_reader:
                    json.dump(row, jsonl_file, ensure_ascii=False)
                    jsonl_file.write('\n')
                    row_count += 1

        print(f"转换完成！共处理 {row_count} 行数据，输出文件：{jsonl_output_path}")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"转换失败：{e}")
        sys.exit(1)

# 命令行参数检查
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python csv2jsonl.py <输入CSV文件路径> <输出JSONL文件路径>")
        print("示例：python csv2jsonl.py data.csv output.jsonl")
        sys.exit(1)
    csv_to_jsonl(sys.argv[1], sys.argv[2])