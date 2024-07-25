import os
import re
import sys
import string
import datetime
import torch
import pandas as pd
import pickle as pkl
import numpy as np

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<unk>', '<pad>'  # 未知字，padding符号
MAX_VOCAB_SIZE = 10000  # 词表长度限制

# create column:
# 1.create year-month patteen range
def get_date_range():
    start_date = datetime.date(2019,6,1)
    end_date = datetime.date(2023,11,30)
    date_range = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    date_list = [date.strftime("%Y-%m") for date in date_range]
    date_range_set = []
    for date in date_list:
        if date not in date_range_set:
            date_range_set.append(date)
    return date_range_set

# 2. save and get ids
def save_creator_ids():
    source_data_dir = "source_data/updated_final_form.csv"
    source_df = pd.read_csv(source_data_dir)
    creator_ids = source_df["creator_id"].to_list()
    with open("source_data/creator_ids.txt", "w") as f:
        for id in creator_ids:
            f.write(str(id)+ "\n")
    
def get_creator_ids():
    res = []
    source_data_dir = "source_data/creator_ids.txt"
    with open(source_data_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
    return lines

# 3. create dataframe framework
def create_id_time_df():
    creator_ids = get_creator_ids()
    columns_names = ['creator_id'] + get_date_range()
    df = pd.DataFrame({columns_names[0]: creator_ids})
    for col in columns_names[1:]:
        df[col] = pd.NA
    df.to_csv("n.csv", index=False)


# 4. fill in the dataframe
def fillin_id_time_df():
    df = pd.read_csv("n.csv")
    for col in df.columns:
        for j, row in enumerate(df[col]):
            if pd.isna(df.loc[j, col]):
                user_id = df["creator_id"][j]
                filtered_df = get_search_result(col, user_id)
                try:
                    text = " ".join(filtered_df["content"].to_list())
                except:
                    print(text)
                text = process_text(text)
                df.loc[j,col] = text
                
    df.to_csv("new.csv",index=False)


# 5. get search result
def get_search_result(col, row):
    df = pd.read_csv("source_data/filtered_combined_posts_comments_final.csv")
    # filtered_df = df[df['created_at'].startswith(col)]
    # filter_df = df[(df['created_at'].startswith(col)) & (df['creator_id']=row)]
    condition1 = df['created_at'].str.startswith(col)
    condition2 = df['creator_id'] == row
    filtered_df = df[condition1 & condition2]
    return filtered_df.dropna(axis=0, how="any")

def process_text(text):
    # 1. keep english words(remove redundant words)
    # 2. get the word dictionary and map all the words onto it
    # 移除URLs
    
    # text = re.sub(r'<a href="[^"]+">([^<]+)</a>', r'\1', text)


    # 使用正则表达式匹配<a>标签并提取其中的内容
    # pattern = re.compile(r'<a href="[^"]+">([^<]+)</a>')
    # text = pattern.sub(r'\1', text)
    # print(text)

    pattern = re.compile(r'<a href=[^>]+>@[^<]+</a>')
    text = pattern.sub('', text)

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 移除表情和其他非ASCII字符
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def build_vocab(file_path,min_freq=0, max_size=MAX_VOCAB_SIZE):
    df = pd.read_csv(file_path).fillna("<PAD>")
    tokenizer = lambda x: x.split(' ')
    vocab_dic = {}
    lines = df["content"].to_list()
    # all_words = set(' '.join(df.apply(lambda x: ' '.join(x), axis=1)).split())

    for line in lines:
        if not line:
            continue
        content = process_text(line.lower())
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def convert_test(file_path, source_file_path):
    df = pd.read_csv(file_path).fillna("<PAD>")
    vocab_path = "vocab.pkl"
    # source_file_path = "source_data/filtered_combined_posts_comments_final.csv"

    if os.path.exists(vocab_path):
        vocab = pkl.load(open(vocab_path, 'rb'))
    else:
        vocab = build_vocab(source_file_path, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(vocab_path, 'wb'))

    def words_to_integers(text):
        return [vocab[word] for word in text.split(" ")]

    columns_to_convert = df.columns[1:]  # 除去第一列的其他所有列
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    df[columns_to_convert] = df[columns_to_convert].applymap(words_to_integers)


    # 统一长度为 128 的函数
    def pad_or_truncate(lst, max_length=64):
        if len(lst) > max_length:
            return lst[:max_length]  # 截断
        return lst + [vocab['<pad>']] * (max_length - len(lst))  # 填充

    # 对每个元素进行处理，使长度统一为 64
    df[columns_to_convert] = df[columns_to_convert].applymap(pad_or_truncate)

    df = df[columns_to_convert]
    rows, cols = df.shape

    max_length = max(df.applymap(len).max())
    matrix_3d = np.zeros((rows, cols, max_length), dtype=int)

    for i in range(rows):
        for j in range(cols):
            matrix_3d[i, j, :len(df.iloc[i, j])] = df.iloc[i, j]
    
    reshaped_matrix = matrix_3d.transpose(0, 2, 1)
    print(reshaped_matrix.shape)
    # [54, 1644, 64]
    return reshaped_matrix

if __name__ == "__main__":

    # fillin_id_time_df()
    # fillin_id_time_df()
    file = "new.csv"

    source_file_path = "source_data/filtered_combined_posts_comments_final.csv"

    convert_test(file, source_file_path)