import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import nltk
import string
from pathlib import Path


# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.SnowballStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name_normalized = r'/workspace/datasets/normalized_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output
def rollup_category(row):
    if row['count'] < min_queries:
        return row['parent']
    else:
        return row['category']

""" Rollup category if query_count < min """
def rollup_categories():
    global df
    global df_counts 
    df_counts = df.groupby('category').size().reset_index(name='count')
    remaining_block = df_counts[df_counts['count'] < min_queries]
    while(remaining_block.empty == False):
        df_merged = df.merge(df_counts, how='left', on='category').merge(parents_df, how='left', on='category')
        df_merged['new_category'] = df_merged.apply(lambda row: rollup_category(row), axis=1)
        df = df_merged.filter(['new_category', 'query'], axis=1).rename(columns={'new_category': 'category'})
        df = df[df['category'].isin(categories)]
        df = df.reset_index(drop=True) 
        df_counts = df.groupby('category').size().reset_index(name='count')
        remaining_block = df_counts [df_counts['count'] < min_queries]

"""Convert queries to lowercase, and optionally implement other normalization, like stemming."""
def normalize_query(query):
    query_clean = "".join([i for i in query.lower() if i not in string.punctuation])
    query_clean_tokens = word_tokenize(query_clean)
    sb = stemmer("english")
    query_clean_tokens = [sb.stem(token) for token in query_clean_tokens]
    return (" ").join(query_clean_tokens)

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
file_qn = Path(output_file_name_normalized)
if file_qn.is_file():
    df = pd.read_csv(output_file_name_normalized)[['categories','query']]
else:
    df = pd.read_csv(queries_file_name)[['category', 'query']]
    df = df[df['category'].isin(categories)]
    df['normalised_query'] = df.apply(lambda x: normalize_query(x['query']), axis=1)
    df = df[df['normalised_query'] != '']
    df = df.filter(['category', 'normalised_query'], axis=1).rename(columns={'normalised_query': 'query'})
    df.to_csv(output_file_name2, index=False)

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
rollup_categories()
# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.

# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
