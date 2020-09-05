import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse

def nearest_items_nms(book_id, index, n=10):
    """Функция для поиска ближайших соседей, возвращает построенный индекс"""
    nn = index.knnQuery(item_embeddings[item_id], k=n)
    return nn

def get_titles(index):
    """
    input - idx of item
    Функция для возвращения названий продуктов
    return - list of titles
    """
    names = []
    for idx in index:
        names.append('Title:  {} '.format(name_mapper[idx]))
    return names

def read_files():
    """
    Функция для чтения файлов + преобразование к  нижнему регистру
    """
    items = pd.read_csv('df_items.csv')
    items['title'] = items.title.str.lower()
    return items

def make_mappers():
    """
    Функция для создания отображения id в title
    """
    title_mapper = dict(zip(items.itemid, items.title))

    return title_mapper

def load_embeddings():
    """
    Функция для загрузки векторных представлений
    """
    with open('item_embeddings.pickle', 'rb') as f:
        item_embeddings = pickle.load(f)

    # Тут мы используем nmslib, чтобы создать наш быстрый knn
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings,nms_idx

#Загружаем данные
items = read_files() 
title_mapper = make_mappers()
item_embeddings,nms_idx = load_embeddings()

"""
# Рекомендательная система по продуктам
Введите название продукта в поле ниже
"""

#Форма для ввода текста
title = st.text_input('Введите название продукта', '')
title = title.lower()