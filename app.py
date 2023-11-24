import streamlit as st 
import pandas as pd 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
import seaborn as sns
import matplotlib.pyplot as plt
import random

df = pd.read_csv('wine_5.csv')
st.title('Классификация красного вина')

st.write("""
    
    Цель проекта состоит в том, чтобы разработать классификационную модель, позволяющую определить качество красного вина (хорошее или плохое).
    """)

st.markdown("""
            Данные для исследования - это открытый датасет по оценкам красного вина и его химического состава от производителя Vinho verde.
             В данных 11 химических признаков и оценка.
            """)
st.write(df.head())

st.markdown("""
    ### Описание полей
    
       
    """)

st.markdown("- Fixed acidity - фиксированная кислотность", help="Участвует в сбалансированности вкуса вина, привносит свежесть вкусу")
st.markdown("- Volatile acidity - летучая кислотность", help="Обусловлена наличием летучих кислот в вине, например, таких как уксусная кислота")
st.markdown("- Citric acid - лимонная кислота", help="Придает вину более яркую и свежую нотку, делая его более освежающим и приятным на вкус. Она также помогает бороться с излишней сладостью, добавляя нотку кислинки, которая балансирует вкус напитка. Влияет на структуру вина, делая его более стабильным и сохраняющим свежесть на протяжении длительного времени.")
st.markdown("- Residual sugar - остаточный сахар", help="Показывает количество сахара, который не был превращен в спирт в процессе ферментации вина. Участвует в сладости вкуса вина")
st.markdown("- Chlorides - хлориды", help="Количество соли, присутствующей в вине")
st.markdown("- Free sulfur dioxide - свободный диоксид серы", help="Они же сульфиты, используются в виноделии в качестве безопасного антисептика. Сульфиты не дают вину скисать и потерять свои вкусовые качества. Присутствуют в вине в свободном виде (газообразном) исвязанном виде (соединившись с водой)")
st.markdown("- Total sulfur dioxide - суммарный диоксид серы", help="Они же сульфиты, используются в виноделии в качестве безопасного антисептика. Сульфиты не дают вину скисать и потерять свои вкусовые качества. Присутствуют в вине в свободном виде (газообразном) исвязанном виде (соединившись с водой)")
st.markdown("- Density - плотность", help="Показывает, сколько массы (граммы) содержится в 1 миллилитре вина. Плотность вина напрямую зависит от количества сахара, алкоголя и кислоты в составе напитка.")
st.markdown("- pH", help="Выступает характеристикой цвета вина. Вина с высоким pH темнее и имеют фиолетовый оттенок цвета. Вина с низким pH светлее и имеют ярко-розовый и ярко-красный оттенок цвета")
st.markdown("- Sulphates - сульфаты", help="Предотвращают окисление и сохранять качество продукта на протяжении длительного времени. Серные соединения, добавленные в вино, помогают защитить его от воздействия кислорода, которое может привести к недостатку свежести, окислению и порче вкусовых качеств")
st.markdown("- Alcohol - спирт", help="Характеризует крепость вина")

x = df.drop(['quality'], axis=1)  # axis=1 - столбец
y = df.iloc[:, -1]  # выбираем последний столбец

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

def userreport():
    number_input = st.sidebar.number_input('Фиксированная кислотность', value=None, placeholder="Type a number...")
    number = st.sidebar.slider("", 4.6, 15.9, value=number_input)
    
    acid = st.sidebar.number_input('Летучая кислотность', value=None, placeholder="Type a number...")
    number_2 = st.sidebar.slider("", 0.12, 1.58, value=acid)
    
    citric_acid_2 = st.sidebar.number_input('Лимонная кислота', value=None, placeholder="Type a number...")
    citric_acid	= st.sidebar.slider("", 0.0, 1.0, value=citric_acid_2)
    
    residual_sugar_2 = st.sidebar.number_input('Остаточный сахар', value=None, placeholder="Type a number...")
    residual_sugar = st.sidebar.slider("", 0.9, 15.5, value=residual_sugar_2)

    chlorides_2 = st.sidebar.number_input('Хлориды', value=None, placeholder="Type a number...")    
    chlorides = st.sidebar.slider("", 0.012, 0.611, value=chlorides_2)

    free_sulfur_dioxide_2 = st.sidebar.number_input('Свободный диоксид серы', value=None, placeholder="Type a number...")
    free_sulfur_dioxide = st.sidebar.slider("", 1.0, 72.0, value=free_sulfur_dioxide_2)

    total_sulfur_dioxide_2 = st.sidebar.number_input('Cуммарный диоксид серы', value=None, placeholder="Type a number...")
    total_sulfur_dioxide = st.sidebar.slider("", 6.0, 289.0, value=total_sulfur_dioxide_2)

    density_2 = st.sidebar.number_input('Плотность', value=None, placeholder="Type a number...")
    density = st.sidebar.slider("", 0.990, 1.004, value=density_2)

    pH_2 = st.sidebar.number_input('pH', value=None, placeholder="Type a number...")
    pH	= st.sidebar.slider("", 2.74, 4.01, value=pH_2)

    sulphates_2 = st.sidebar.number_input('Сульфаты', value=None, placeholder="Type a number...")
    sulphates = st.sidebar.slider("", 0.33, 2.0, value=sulphates_2)

    alcohol_2 = st.sidebar.number_input('Спирт', value=None, placeholder="Type a number...")
    alcohol = st.sidebar.slider("", 8.4, 14.9, value=alcohol_2)

    # сбор введённых значений в словарь
    report = {
        'fixed acidity' : number,
        'volatile acidity' : number_2,
        'citric acid' : citric_acid,
        'residual sugar' : residual_sugar,
        'chlorides' : chlorides,
        'free sulfur dioxide' : free_sulfur_dioxide,
        'total sulfur dioxide' : total_sulfur_dioxide,
        'density' : density,
        'pH' : pH,
        'sulphates' : sulphates,
        'alcohol' : alcohol}  


    report = pd.DataFrame(report, index=[0])  # DataFrame будет иметь 1 строку с индексом 0
    return report    


userdata = userreport()

rf = RandomForestClassifier()
rf.fit(xtrain, ytrain)

st.subheader('Точность оценки: ')
# сравниваем полученные результаты с реальными
st.write(str(accuracy_score(ytest, rf.predict(xtest)) * 100) + '%')

userresult = rf.predict(userdata)  # передаём параметры
st.subheader('Тип вина: ')
if userresult[0] == 0:
    output = 'Плохое'
else:
    output = 'Хорошее'


st.write(output)

if userresult[0] == 0:
    #smiley = ('😔')
    st.image('./sad cat.png', width=300)
else:
    st.image('./good cat.png', width=300)
    #smiley = ('😍🥂')


my_list = ["Antinori Tignanello Toscana IGT 2019 - красное, сухое","Marques de Caceres Crianza 2017 - красное, сухое", 
           "Alamos Malbec 2021 - красное, сухое", "770 Miles Zinfandel - красное, сухое", "Felix Solis Mucho Mas - красное, сухое",
           "Duca di Saragnano Alchymia Primitivo - красное, полусухое"]

if userresult[0] == 1:
    st.subheader('Рекомендации для Вас: ')
    for item in my_list:
        st.write(item)

st.subheader('Визуализация')

#st.bar_chart(df.drop(['quality'], axis=1), title="Это заголовок графика")
#st.caption("Это подпись к графику")

df_1 = df.drop(['quality'], axis=1)

st.bar_chart(df_1)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_1.corr(), annot=True, fmt=".2f", ax=ax)
plt.title('Корреляция Пирсона\n', fontsize=20)
st.write(fig)