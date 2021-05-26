from __future__ import division, print_function
from django.http import HttpResponse
from django.shortcuts import render
from .models import Article, Comment
from django.http import Http404, HttpResponseRedirect
from django.urls import reverse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def comment_ratin(ml, comment_text):
    pass


def index(request):
    latest_articles_list = Article.objects.order_by('-pub_date')[:5]
    return render(request, 'articles/list.html', {'latest_articles_list': latest_articles_list})


def detail(request, article_id):
    try:
        a = Article.objects.get(id=article_id)
    except:
        raise Http404("Статья не найдена")

    latest_comments_list = a.comment_set.order_by('-id')[:10]
    X_train_new = cv.transform([latest_comments_list[0].comment_text])
    comment_ratin = logit.predict(X_train_new)
    if comment_ratin[0] == 1:
        comment_rating = "Комментарий положительный"
    else:
        comment_rating = "Комментарий отрицательный"

    print("---")
    print(comment_ratin)
    print("---")

    print(latest_comments_list[0].comment_text)
    return render(request, 'articles/detail.html', {'article': a, 'latest_comments_list': latest_comments_list, 'comment_rating': comment_rating})


def leave_comment(request, article_id):
    try:
        a = Article.objects.get(id=article_id)
    except:
        raise Http404("Статья не найдена")
    a.comment_set.create(author_name=request.POST['name'], comment_text=request.POST['text'])
    # print(a.comment_set.last())
    return HttpResponseRedirect(reverse('articles:detail', args=(a.id,)))


# Выборка для Обучения
print('1')
reviews_train = load_files(r"A:\Project\lesson_python\train")
print('1')
text_train, y_train = reviews_train.data, reviews_train.target
print('1')
text_test_new = ['cool comedy']
print('1')
# Вычисляем количство слов
cv = CountVectorizer()
cv.fit(text_train)
print('1')
print(len(cv.vocabulary_))
print('1')
X_train = cv.transform(text_train)

# X_train_new = cv.transform(text_test_new)

logit = LogisticRegression(n_jobs=-1, random_state=10, max_iter=20000)
logit.fit(X_train, y_train)

# y_pred = logit.predict(X_train_new)
# print("---")
# print(y_pred)
# print("---")


# print(round(logit.score(X_train, y_train), 3), round(logit.score(X_test, y_test), 3))
# print(logit.fit(X_train, y_train))
