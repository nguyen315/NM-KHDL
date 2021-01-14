# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Nhập môn Khoa Học Dữ Liệu - Nhóm 25
#
# 18120484 - Hồ Minh Nguyên\
# 18120491 - Diệp Đại Thiện Nhân

# ## Câu hỏi cần trả lời

# - Đọc sách là một cách tốt để tiếp thu kiến thức mới và giải trí. Khác với xem phim, việc đọc sách tốn nhiều thời gian và công sức hơn, vì vậy chọn được một quyển sách tốt để đọc là cần thiết để tránh tốn thời gian vào những quyển sách không hay
# - Ở đây, xác định quyển sách là "great" nếu rating của nó thuộc 20% điểm cao nhất

# ## Thu thập dữ liệu

# import các thư viện hỗ trợ
import requests
from bs4 import BeautifulSoup
import json
import time # Dùng để sleep chương trình
import pandas as pd # Dùng để đọc và hiển thị file csv (Pandas sẽ được học chi tiết ở buổi tới)
import datetime as dt # Dùng để xử lý dữ liệu thời gian
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# +
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns # seaborn là thư viện được xây trên matplotlib, giúp việc visualization đỡ khổ hơn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import set_config
set_config(display='diagram') # Để trực quan hóa pipeline


# -

# lấy danh sách các sách disallow từ robots.txt
def get_disallow_book_ids():
    respone = requests.get('https://www.goodreads.com/robots.txt')
    soup = BeautifulSoup(respone.content, 'html.parser')
    robots = soup.get_text()
    disallow_books = re.findall('show/([0-9]+)', robots)
    print(type(disallow_books))
    return disallow_books


# +
# parse từ trang https://www.goodreads.com/list/show/1.Best_Books_Ever
# để lấy danh sách các sách
# list này có khoảng 50.000 sách

def get_book_ids():
    page = 1
    book_ids = [];
    while (page <= 500):
        url = f'https://www.goodreads.com/list/show/1.Best_Books_Ever?page={page}'
        print('get url:', url)
        respone = requests.get(url)
        soup = BeautifulSoup(respone.content, 'html.parser')
        titles = soup.find_all('a', {'class': 'bookTitle'})
        for title in titles :
            try:
                book_id = re.findall( '(\d+)', title['href'])[0]
                book_ids.append(book_id)
            except:
                print('can not find id in titles:', title)
        page += 1
    return book_ids


# +
# # tạo một file để lưu id book vì thời gian để lấy id book khá dài
# book_ids = get_book_ids()
# df = pd.DataFrame(data={'book_id': book_ids})
# df.to_csv('book_ids.csv', index=False)
# -

def collect_books(filename):
    # mở file để ghi dữ liệu thu thập được
    f = open(filename, 'w', encoding='utf-8')
    f.write('book_id\ttitle\tauthors\taverage_rating\tratings_count\ttext_review\tnum_pages\tpublication\tpublisher\tgenre\n')
    
    # lấy danh sách các sách có thể thu thập dữ liệu
    book_ids = pd.read_csv('book_ids.csv')['book_id'].tolist()
    disallow_book_id = get_disallow_book_ids()    
    book_ids = list(set(book_ids) - set(disallow_book_id))
    book_ids.sort()
    i = 0
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    # bắt đầu parse book
    while i < 10000:
        url = f'https://www.goodreads.com/book/show/{book_ids[i]}'
        respone = requests.get(url)
        
        if respone.ok == True:
            soup = BeautifulSoup(respone.content, 'html.parser')
            print(f'{i} {url}')
            # lấy các thông tin
            try:
                title = soup.find('h1', {'id': 'bookTitle'}).text.strip()
                authors = soup.find('a', {'class': 'authorName'}).text.strip()
                average_rating = soup.find('span', {'itemprop': 'ratingValue'}).text.strip()
                ratings_count = soup.find('meta', {'itemprop': 'ratingCount'})['content']
                text_review = soup.find('meta', {'itemprop': 'reviewCount'})['content']
                num_pages = soup.find('span', {'itemprop': 'numberOfPages'}).text.split()[0]
                publication = soup.find('div', {'id': 'details'}).find_all('div', {'class': 'row'})[1].text.strip().split('\n')[1].strip()
                publisher = soup.find('div', {'id': 'details'}).find_all('div', {'class': 'row'})[1].text.strip().split('\n')[2].strip()[3:]
                genre = soup.find('a', {'class': 'bookPageGenreLink'}).text.strip()

                f.write(f'{book_ids[i]}\t{title}\t{authors}\t{average_rating}\t{ratings_count}\t{text_review}\t{num_pages}\t{publication}\t{publisher}\t{genre}\n')
#                 print(f'{book_ids[i]} {title} {authors} {average_rating} {ratings_count} {text_review} {num_pages} {publication} {publisher} {genre}')
            except:
                print("can not get data:", url)
            finally:
                i += 1
        else:
            print(f'respone false {i}: {url}')
            time.sleep(5)
    return


# +
# collect_books('test.csv')
# -

books = pd.read_csv('books.csv', sep='\t')
books.shape

# ## Khám phá dữ liệu

df.info()

# Dữ liệu có giá trị thiếu (cột "publication" có 8 giá trị thiếu và cột "publisher" có 7 giá trị thiếu)

numerical = ['average_rating','ratings_count','text_review','num_pages']
df[numerical].describe()

# Ở đây chỉ có cột "average_rating" ít bị phân tán nhất, còn các cột dạng số khác có phân bố rất rộng (Ví dụ cột ratings_count có ít nhất 15 đánh giá, nhưng nhiều nhất là 7,277,329 đánh giá)

# Ta có cột num_pages có giá trị không hợp lệ (0), vì không có sách nào có 0 trang cả!

categorical = ['title','authors','publication','publisher','genre']
df[categorical].describe()

df.duplicated().sum()

# Không có dữ liệu trùng lắp

fig = plt.figure(figsize = (8,10))
ax = fig.gca()
df[numerical].boxplot(ax = ax)
plt.show()

# Có những cuốn sách được nhiều lượt đánh giá (trên 1,000,000 đánh giá), cột "average_rating" có giá trị rất nhỏ (0-5), nó làm cho biểu đồ khó nhìn thấy những thông tin khác, . Vậy ta sẽ scale dữ liệu lại để dễ nhìn hơn.



# +
# Create varsToNormalize, where all the varsToNormalize values are treated as floats
varsToNormalize = df[numerical].values.astype(float)

# Create a minimum and maximum preprocessing object
range_Scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
vars_Scaled = range_Scaler.fit_transform(varsToNormalize)

# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(vars_Scaled)

fig2 = plt.figure(figsize = (8,10))
ax2 = fig2.gca()
df_normalized.boxplot(ax = ax2)
plt.show()
# -

# ## Phân tích dữ liệu

# +
#Top 10 most rated books
top10Books = df.nlargest(10, ['ratings_count']).set_index('title')['ratings_count']
plot_dims = (12, 4)
fig, ax = plt.subplots(figsize=plot_dims)
sns.barplot(top10Books, top10Books.index)

for i in ax.patches:
    ax.text(i.get_width()+.3, i.get_y()+0.5, str(i.get_width()), fontsize = 15, color = 'k')
    
plt.show()

# +
#Top 10 least rated books
top10underatedBooks = df.nsmallest(10, ['ratings_count']).set_index('title')['ratings_count']
plot_dims = (10, 8)
fig, ax = plt.subplots(figsize=plot_dims)
sns.barplot(top10underatedBooks, top10underatedBooks.index)

for i in ax.patches:
    ax.text(i.get_width()+.3, i.get_y()+0.5, str(i.get_width()), fontsize = 15, color = 'k')
    
plt.show()

# +
#Top 10 highest rated books
top10goodBooks = df.nlargest(10, ['average_rating']).set_index('title')['average_rating']
plot_dims = (12, 6)
fig, ax = plt.subplots(figsize=plot_dims)
sns.barplot(top10goodBooks, top10goodBooks.index)

for i in ax.patches:
    ax.text(i.get_width()+.3, i.get_y()+0.5, str(i.get_width()), fontsize = 15, color = 'k')
    
plt.show()
# -

# ## Khám phá và tiền xử lý để xây dựng mô hình

df = pd.read_csv("books.csv",sep = '\t')
df.dropna(subset=['publication', 'publisher'], inplace=True)
print("The data has",df.shape[0],"rows x",df.shape[1],"columns")

# ### Tách tập train và test

# cột average_rating có giá trị thiếu không
df['average_rating'].isna().sum()

# ta thêm cột output là great_book là những sách có số điểm thuộc 25% điểm cao nhất
df['great_book'] = df['average_rating'] >= df['average_rating'].quantile(0.75).round(1)
df

# OK, tiến hành tách tập train và test

# loại các cột có thể làm leak data : ratings_count, text_review
y_sr = df['great_book']
X_df = df[['title', 'authors', 'num_pages', 'genre']]
X_df

train_X_df, test_X_df, train_y_sr, test_y_sr = train_test_split(X_df, y_sr, test_size=0.2, 
                                                              stratify=y_sr, random_state=0)

# +
# tách tiếp tập train thành train và validation
# train_X_df, val_X_df, train_y_sr, val_y_sr = train_test_split(temp_X_df, temp_y_sr, test_size=0.2, 
#                                                               stratify=temp_y_sr, random_state=0)
# -

# ### Xử lý trên tập huấn luyện

pd.set_option("display.max_rows", 900)
train_y_sr

pd.set_option('display.max_colwidth', 200) # Để nhìn rõ hơn
def missing_ratio(df):
    return (df.isna().mean() * 100).round(1)
def num_values(df):
    return df.nunique()
def value_ratios(c):
    return dict((c.value_counts(normalize=True) * 100).round(1))
train_X_df.agg([missing_ratio, num_values, value_ratios])


# tách cột title thành:
    # title_length_in_word
    # title_length_in_character
class ColAdderDropper(BaseEstimator, TransformerMixin):
    def __init__(self, num_top_authors=1, num_top_genres=1):
        self.num_top_authors = num_top_authors
        self.num_top_genres = num_top_genres
    def fit(self, X_df, y=None):
        author_col = X_df.authors
        self.author_counts_ = author_col.value_counts();
        authors = list(self.author_counts_.index)
        self.top_authors_ = authors[:max(1, min(self.num_top_authors, len(authors)))]
        
        genre_col = X_df.genre
        self.genre_counts_ = genre_col.value_counts();
        genres = list(self.genre_counts_.index)
        self.top_genres_ = genres[:max(1, min(self.num_top_genres, len(genres)))]
        return self
         
    def transform(self, X_df, y=None):
#         transformed_df = X_df.drop(columns=['title'])

        transformed_df = X_df.drop(columns=['title'])
        
        transformed_df['title_length_in_character'] = X_df['title'].str.len()
        transformed_df['title_length_in_word'] = X_df['title'].str.split(' ').str.len()
        author_col = X_df.authors
        transformed_df['authors'] = author_col.where(author_col.isin(self.top_authors_), 'Others')
        
        genre_col = X_df.genre
        transformed_df['genre'] = genre_col.where(genre_col.isin(self.top_genres_), 'Others')
        return transformed_df


col_adderdropper = ColAdderDropper(num_top_authors=150, num_top_genres=30)
col_adderdropper.fit(train_X_df)

# test col_adderdropper
test_train_X_df = col_adderdropper.transform(train_X_df)

train_X_df.dtypes

# - Thử nghiệm với one-hot encode cột genre và authors
# - Với các cột dạng số, ta sẽ điền giá trị thiếu bằng giá trị mean của cột: dùng SimpleImputer trong Sklearn
# - Với các cột không phải dạng số và không có thứ tự:
#     - Ta sẽ điền giá trị thiếu bằng giá trị mode (giá trị xuất hiện nhiều nhất) của cột. Với tất cả các cột không có dạng số và không có thứ tự, ta đều cần tính mode, vì ta không biết được cột nào sẽ bị thiếu giá trị khi dự đoán với các véc-tơ input mới.
#     - Sau đó, ta sẽ chuyển sang dạng số bằng phương pháp mã hóa one-hot

# make pipeline
nume_cols = ['num_pages']
unorder_cate_cols = ['authors', 'genre']
# YOUR CODE HERE
nume_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'))
unorder_cate_cols_transformer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
preprocess_column_transformer = make_column_transformer((nume_transformer, nume_cols),
                                                       (unorder_cate_cols_transformer, unorder_cate_cols))
preprocess_pipeline = make_pipeline(ColAdderDropper(num_top_authors = 150, num_top_genres = 30), 
                                    preprocess_column_transformer, StandardScaler(with_mean=False))
preprocessed_train_X = preprocess_pipeline.fit_transform(train_X_df)

full_pipeline = make_pipeline(ColAdderDropper(num_top_authors = 150, num_top_genres = 30), 
                              preprocess_column_transformer, 
                              StandardScaler(with_mean=False),
                              MLPClassifier(hidden_layer_sizes=(20), activation='tanh', solver='lbfgs', random_state=0, max_iter=3000))
full_pipeline

# #### Dùng KFold để chia tập train thành train data và validation data

# +
# thử nghiệm model
kf = KFold(n_splits=5)

train_errs = []
val_errs = []

num_top_authors_s = [100, 150, 200, 250, 300]
num_top_genres_s = [30, 40, 50, 60]
alphas = [0.1, 1, 10, 100]

best_val_err = float('inf'); best_alpha = None; best_num_top_authors = None; best_num_top_genres = None;
for train_indices, test_indices in kf.split(train_X_df):
    for alpha in alphas:
        for num_top_authors in num_top_authors_s:
            for num_top_genres in num_top_genres_s:
                # YOUR CODE HERE
                full_pipeline.set_params(coladderdropper__num_top_authors=num_top_authors, 
                                         coladderdropper__num_top_genres=num_top_genres, 
                                         mlpclassifier__alpha=alpha)

                full_pipeline.fit(train_X_df.iloc[train_indices], train_y_sr.iloc[train_indices])
                train_err = (1 - full_pipeline.score(train_X_df.iloc[train_indices], train_y_sr.iloc[train_indices])) * 100
                val_err = (1 - full_pipeline.score(train_X_df.iloc[test_indices], train_y_sr.iloc[test_indices])) * 100
                train_errs.append(train_err)
                val_errs.append(val_err)

                if (val_err < best_val_err):
                    best_val_err = val_err
                    best_alpha = alpha
                    best_num_top_authors = num_top_authors
                    best_num_top_genres = num_top_genres

                        
'Finish!'
# -
# ### Huấn luyện mô hình với các thông số best tìm được ở trên


full_pipeline.set_params(coladderdropper__num_top_authors=best_num_top_authors, 
                                         coladderdropper__num_top_genres=best_num_top_genres, 
                                         mlpclassifier__alpha=best_alpha)

full_pipeline.fit(train_X_df, train_y_sr)

predict_sr = full_pipeline.predict(test_X_df)

1 - full_pipeline.score(test_X_df, test_y_sr)
