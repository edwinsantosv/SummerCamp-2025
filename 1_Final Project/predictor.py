import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df=pd.read_csv("C:/Users/edwin/Repositories/Data Science/SummerCamp 2025/1_Final Project/ai4i2020.csv")
df.head()

# Convertir la columna 'Type' en variables dummy
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

# Definir las características (X) y las variables objetivo (y)
X = df.drop(columns=['Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df[['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]

# Realizar el muestreo estratificado por tipo de máquina
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['Type_M'])

# Cargar el modelo entrenado
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train['Machine failure'])

# Título de la aplicación
st.title('Predictive Maintenance AI4I 2020')

# Crear sliders para las variables numéricas
air_temp = st.slider('Air temperature [K]', float(X['Air temperature [K]'].min()), float(X['Air temperature [K]'].max()), float(X['Air temperature [K]'].mean()))
process_temp = st.slider('Process temperature [K]', float(X['Process temperature [K]'].min()), float(X['Process temperature [K]'].max()), float(X['Process temperature [K]'].mean()))
rotational_speed = st.slider('Rotational speed [rpm]', int(X['Rotational speed [rpm]'].min()), int(X['Rotational speed [rpm]'].max()), int(X['Rotational speed [rpm]'].mean()))
torque = st.slider('Torque [Nm]', float(X['Torque [Nm]'].min()), float(X['Torque [Nm]'].max()), float(X['Torque [Nm]'].mean()))
tool_wear = st.slider('Tool wear [min]', int(X['Tool wear [min]'].min()), int(X['Tool wear [min]'].max()), int(X['Tool wear [min]'].mean()))

# Crear dropdown para la variable categórica
type_option = st.selectbox('Type', ['L', 'M', 'H'])

# Convertir la opción seleccionada en variables dummy
type_L = 1 if type_option == 'L' else 0
type_M = 1 if type_option == 'M' else 0

# Crear un botón para hacer la predicción
if st.button('Predict'):
    # Crear un dataframe con los valores seleccionados
    input_data = pd.DataFrame({
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rotational_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear],
        'Type_L': [type_L],
        'Type_M': [type_M]
    })

    # Hacer la predicción
    prediction = rf_model.predict(input_data)

    # Mostrar el resultado
    if prediction[0] == 1:
        st.write('The machine is likely to fail.')
    else:
        st.write('The machine is not likely to fail.')
        # Predecir el tipo de fallo
        failure_predictions = {}
        for failure_type in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
            rf_model_type = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model_type.fit(X_train, y_train[failure_type])
            failure_predictions[failure_type] = rf_model_type.predict(input_data)[0]

        # Mostrar el resultado del tipo de fallo
        if prediction[0] == 1:
            st.write('The machine is likely to fail.')
            failure_types = [ftype for ftype, pred in failure_predictions.items() if pred == 1]
            if failure_types:
                st.write(f'Es probable que tenga el fallo de: {", ".join(failure_types)}')
            else:
                st.write('No se pudo determinar el tipo de fallo.')
        else:
            st.write('The machine is not likely to fail.')