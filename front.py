import streamlit as st
import requests

url = 'http://127.0.0.1:8000/predict/'

st.write('Загрузите изображение, и модель попробует его распознать')

file = st.file_uploader('Загрузите файл', type=["png", "jpg", "jpeg"])

if file is not None:
    st.image(file, caption='Загруженное изображение')

    if st.button("проверит"):
        response = requests.post(url, file, timeout=10)

        if response.status_code == 200:
            result = response.json()
            pred = result.get('Answer')
            st.success(f'Модель думает, что это: {pred}')








