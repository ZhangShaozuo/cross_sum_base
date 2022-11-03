from datasets import load_dataset, Features, Value, Dataset
import pandas as pd
from pandarallel import pandarallel

xl_sum_chs = load_dataset('csebuetnlp/xlsum', 'chinese_simplified')

def concat_src_tgt(s, fp_head):
    '''
    s: split, one of train, test, val
    fp_head: file path head
    '''
    df_src = pd.read_table(f'{fp_head}_{s}.source', header = None, names = [f'text_source'])
    df_tgt = pd.read_table(f'{fp_head}_{s}.target', header = None, names = [f'summary_target'])
    return pd.concat([df_src, df_tgt.reindex(df_src.index)], axis = 1)

def parallel_func(row, split):
    ### The parallel data only comes from the same split set in xl-sum, as tested out
    for sample in xl_sum_chs[split]:
        if sample['summary'] == row:
            return sample['text']
    return -1

def split_df(fp_head):
    train_df = concat_src_tgt("train", fp_head)
    test_df = concat_src_tgt("test", fp_head)
    val_df = concat_src_tgt("val", fp_head)

    train_df['text_target'] = -1
    test_df['text_target'] = -1
    val_df['text_target'] = -1
    return train_df, val_df, test_df

def filter(df):
    '''Filter non-parallel data'''
    f_df = df[df.text_target != '-1']
    return f_df

def main():
    lang_pair = {}
    source = 'en'
    target = 'chs'
    fp_head = 'en_chs/english-chinese_simplified'

    train_df, val_df, test_df = split_df(fp_head)
    print(train_df.size, val_df.size, test_df.size)
    pandarallel.initialize()
    train_df['text_target'] = train_df['summary_target'].parallel_apply(parallel_func, split='train')
    test_df['text_target'] = test_df['summary_target'].parallel_apply(parallel_func, split='test')
    val_df['text_target'] = val_df['summary_target'].parallel_apply(parallel_func, split='validation')

    train_df, test_df, val_df = filter(train_df), filter(test_df), filter(val_df)
    train_df.to_csv('train_df.csv', index=False)
    test_df.to_csv('test_df.csv', index=False)
    val_df.to_csv('val_df.csv', index=False)

if __name__ == '__main__':
    main()