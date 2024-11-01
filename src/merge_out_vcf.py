import os

def merge_vcfs(input_folder, output_file):
    # 获取所有 VCF 文件路径，并按名称排序（可按需调整）
    vcf_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.vcf')])

    if not vcf_files:
        print("No VCF files found in the specified folder.")
        return

    # 打开输出文件以写入合并后的结果
    with open(output_file, 'w') as outfile:
        header_written = False  # 用于标记 header 是否已写入

        # 遍历所有 VCF 文件
        for i, vcf in enumerate(vcf_files):
            vcf_path = os.path.join(input_folder, vcf)
            with open(vcf_path, 'r') as infile:
                for line in infile:
                    # 写入 header（仅写入第一个文件的 header）
                    if line.startswith('#'):
                        if not header_written:
                            outfile.write(line)
                    else:
                        # 非 header 行直接写入
                        outfile.write(line)
            header_written = True  # 标记 header 已写入

    print(f"All VCF files from {input_folder} have been merged into {output_file}.")

# 使用示例：将 'vcf_files' 文件夹中的所有 VCF 合并为 'merged.vcf'
input_folder = '/home/yang1031/projects/error-aware-calling/results/element/vcf'  # 替换为你的文件夹路径
output_file = '/home/yang1031/projects/error-aware-calling/results/element/vcf/chr20.vcf'    # 输出文件的名称

merge_vcfs(input_folder, output_file)
