# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
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

# ## Công việc sẽ làm để trả lời câu hỏi

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
        respone = requests.get(url, headers=headers)
        
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
                genre = soup.find('a', {'class': 'bookPageGenreLink'}).text

                f.write(f'{book_ids[i]}\t{title}\t{authors}\t{average_rating}\t{ratings_count}\t{text_review}\t{num_pages}\t{publication}\t{publisher}\t{genre}\n')
                print(f'{book_ids[i]} {title} {authors} {average_rating} {ratings_count} {text_review} {num_pages} {publication} {publisher} {genre}')
            except:
                print("can not get data:", url)
            finally:
                i += 1
        else:
            print(f'respone false {i}: {url}')
            time.sleep(1)
    return


# +
# collect_books('books.csv')
# -

# ## Khám phá dữ liệu

df = pd.read_csv("books.csv",sep = '\t')
print("The data has",df.shape[0],"rows x",df.shape[1],"columns")
df.head(10)

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 100) # Để dễ nhìn
pd.set_option('display.max_columns', None) # Để dễ nhìn
df['authors'].value_counts()

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

# +
genre = df['genre'].value_counts()
plot_dims = (15, 20)
fig, ax = plt.subplots(figsize=plot_dims)
sns.barplot(genre, genre.index)

for i in ax.patches:
    ax.text(i.get_width()+.3, i.get_y()+0.5, str(i.get_width()), fontsize = 12, color = 'k')
    
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

# ## Xây dựng mô hình


