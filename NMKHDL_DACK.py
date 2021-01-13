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

# ## Công việc sẽ làm để trả lời câu hỏi

# ## Thu thập dữ liệu

# import các thư viện hỗ trợ
import requests
from bs4 import BeautifulSoup
import json
import time # Dùng để sleep chương trình
import pandas as pd # Dùng để đọc và hiển thị file csv (Pandas sẽ được học chi tiết ở buổi tới)
import datetime as dt # Dùng để xử lý dữ liệu thời gian
import re


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
    i = 7000
    
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
                genre = soup.find('a', {'class': 'bookPageGenreLink'}).text

                f.write(f'{book_ids[i]}\t{title}\t{authors}\t{average_rating}\t{ratings_count}\t{text_review}\t{num_pages}\t{publication}\t{publisher}\t{genre}\n')
                print(f'{book_ids[i]} {title} {authors} {average_rating} {ratings_count} {text_review} {num_pages} {publication} {publisher} {genre}')
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

# ## Phân tích dữ liệu

# ## Xây dựng mô hình


