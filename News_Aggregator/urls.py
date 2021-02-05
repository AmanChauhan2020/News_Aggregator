"""News_Aggregator URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from InstantNews import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='home'),
    path('world',views.world,name='world'),
    path('india',views.india,name='india'),
    path('entertainment',views.entertainment,name='entertainment'),
    path('technology',views.technology,name='technology'),
    path('sports',views.sports,name='sports'),
    path('recommend',views.recommend,name='recommend'),
    path('positive',views.positive,name='positive'),
    path('negative',views.negative,name='negative'),
    path('neutral',views.neutral,name='neutral'),
    path('diag',views.diag,name='diag'),
]
