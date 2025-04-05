import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
#Configuracion de la pagina  Predicción inversion tienda de video juego
 
st.set_page_config(page_title="Predicción inversion tienda de video juego", page_icon=":guardsman:", layout="wide")
st.title("Predicción inversion tienda de video juego")

#crear el punto de entrada def main
def main():

    # cargar imagen de la tienda de video juego
    st.image("imagen de videojuego.jpg", width=900)

    # Cargar el modelo de predicción
    filename = 'modelo-reg-tree-RF.pkl'
    model_Tree, model_RF, variables = pickle.load(open(filename, 'rb')) #cargar el modelo de prediccióny el objeto variables

    #crear el sidebar
    st.sidebar.title("parametros de usuario")

    #crearel boton de prediccion
    st.sidebar.subheader("Predicción")
    # crear los inputs del usuario
    def user_input_features():
        edad = st.sidebar.number_input("Edad", min_value=14, max_value=52)
        option = [ "'Mass Effect'", "'Sim City'", "'Dead Space'", "'Battlefield'", "'Fifa'", "'Koa'", "'Reckoning'", "'F1'", "'Crysis'"]
        videojuego = st.sidebar.selectbox("Juego", option,index=0)
        option_plataforma = ["'Play Station'","'Xbox'","PC","Otros"]
        plataforma = st.sidebar.selectbox("Plataforma", option_plataforma, index=0)
        option_sexo = ["Hombre", "Mujer"]
        sexo = st.sidebar.selectbox("Sexo", option_sexo, index=0)
        consumidor_habitual = st.sidebar.checkbox("Consumidor habitual", value=False)
        
 		# Crear un diccionario con los valores de entrada del usuario
        data = {
			'Edad': edad,
			'Juego': videojuego,
			'Plataforma': plataforma,
			'Sexo': sexo,
			'Consumidor_habitual': consumidor_habitual
		}
        #mostrar los valores del diccionario
        
        data_imput = pd.DataFrame(data, index=[0])
        #st.write(data_imput)

        return data_imput
    data_imp = user_input_features()

    data_preparada = data_imp.copy()
    #st.write(data_preparada)


     #tranformar las variables categoricasen variables dummies
    data_preparada = pd.get_dummies(data_preparada, columns=['Juego', 'Plataforma'], drop_first=False)
    data_preparada = pd.get_dummies(data_preparada,columns=['Sexo'], drop_first=False)
    #st.write(data_preparada) 
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
    #st.write(data_preparada)
    #predecir con el modelo de arbol de decision
    #crear un boton para predecir
    if st.sidebar.button("Predecir"):
        #predecir con el modelo de arbol de decision
        y_pred_Tree = model_Tree.predict(data_preparada)
        st.success(f" El cliente invertira : {y_pred_Tree[0]..1f},dolares  96% de presicion")
        st.write("Predicción del modelo de árbol de decisión: 96%")
        
        #predecir con el modelo de arbol de decision



    y_pred_Tree = model_Tree.predict(data_preparada)
    #st.write("Predicción del modelo de árbol de decisión:", y_pred_Tree[0])









if __name__ == "__main__":
    main()
       






