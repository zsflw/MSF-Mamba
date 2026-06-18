#!/usr/bin/env python3
"""
RNA二级结构特征提取工具（适配miRNA_ID+Sequence格式）
输入格式:
    RNA_ID    Sequence
    MIMAT...  UGAGGUAG...
输出: 带ID的特征文件(.pt/.csv)
"""

import os
import pandas as pd
import torch
from RNA import fold
from tqdm import tqdm
import argparse


class miRNAFeatureExtractor:
    def __init__(self, feature_dim=22):
        self.feature_dim = feature_dim
        self.base_dict = {'A': 1, 'U': 2, 'C': 3, 'G': 4}

    def load_data(self, input_path, sheet_name=0):

        df = pd.read_excel(input_path, sheet_name=sheet_name)
        

        required_cols = {'RNA_ID', 'Sequence'}
        if not required_cols.issubset(df.columns):
            raise ValueError("Excel必须包含'RNA_ID'和'Sequence'列")
            
        return df[['RNA_ID', 'Sequence']].dropna()

    def extract_features(self, df):

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            rna_id, seq = row['RNA_ID'], row['Sequence']
            try:
                # 序列预处理
                seq = str(seq).upper().replace(" ", "")
                if len(seq) < 5:
                    raise ValueError("序列过短")
                if not all(b in self.base_dict for b in seq):
                    raise ValueError("含非法碱基")


                structure, energy = fold(seq)


                feat = torch.zeros(self.feature_dim)
                seq_len = len(seq)

                feat[0] = seq_len
                feat[1] = energy
                feat[2] = structure.count('(') / seq_len
                feat[3] = (seq.count('G') + seq.count('C')) / seq_len


                feat[4] = seq.count('A') / seq_len
                feat[5] = seq.count('U') / seq_len
                feat[6] = seq.count('C') / seq_len
                feat[7] = seq.count('G') / seq_len
                au = seq.count('AU') + seq.count('UA')
                gc = seq.count('GC') + seq.count('CG')
                feat[8] = au / (seq_len - 1) if seq_len > 1 else 0.
                feat[9] = gc / (seq_len - 1) if seq_len > 1 else 0.


                unpaired = structure.count('.')
                left = structure.count('(')
                right = structure.count(')')
                feat[10] = unpaired / seq_len
                feat[11] = right / seq_len
                feat[12] = left
                feat[13] = unpaired

                stem_segments = str(structure).replace('.', ' ').split()
                avg_stem = sum(len(s) for s in stem_segments) / len(stem_segments) if stem_segments else 0.
                feat[14] = avg_stem


                import re
                loop_segs = re.findall(r'\.+', structure)
                feat[15] = hairpin_num
                feat[16] = internal_num
                feat[17] = bulge_num 
                feat[18] = max([len(s) for s in stem_segments]) if stem_segments else 0.
                feat[19] = max([len(s) for s in loop_segs]) if loop_segs else 0.
                feat[20] = len(stem_segments)
                feat[21] = energy / seq_len

                results.append({
                    'RNA_ID': rna_id,
                    'features': feat,
                    'sequence': seq,
                    'structure': structure
                })
            except Exception as e:
                print(f"\n处理失败 [ID:{rna_id}]: {seq[:10]}... ({str(e)})")
                results.append({
                    'RNA_ID': rna_id,
                    'features': torch.zeros(self.feature_dim),
                    'error': str(e)
                })

        return results

    def save_results(self, data, output_path, format='pt'):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'pt':

            torch.save({item['RNA_ID']: item['features'] for item in data}, output_path)
        elif format == 'csv':

            pd.DataFrame([{
                'RNA_ID': item['RNA_ID'],
                'sequence': item.get('sequence', ''),
                'structure': item.get('structure', ''),
                **{f'feat_{i}': item['features'][i].item() for i in range(self.feature_dim)}
            } for item in data]).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='miRNA特征提取器')
    parser.add_argument('--input', required=True, help='输入Excel路径 (.xlsx)')
    parser.add_argument('--output', help='输出文件路径（默认同输入目录）')
    parser.add_argument('--format', choices=['pt', 'csv'], default='pt')
    parser.add_argument('--sheet', default=0, help='Excel工作表名/索引')
    args = parser.parse_args()


    if not args.output:
        input_dir = os.path.dirname(args.input)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(input_dir, f"{base_name}_features.{args.format}")

    extractor = miRNAFeatureExtractor()

    try:
        print(f"\n▶ 正在处理: {args.input}")

        df = extractor.load_data(args.input, args.sheet)
        print(f"▷ 发现 {len(df)} 条miRNA序列")


        results = extractor.extract_features(df)


        extractor.save_results(results, args.output, args.format)
        print(f"\n✔ 处理完成！结果保存到: {args.output}")

    except Exception as e:
        print(f"\n✖ 发生错误: {str(e)}")


if __name__ == "__main__":
    main()