#!/bin/bash

# Verificar que se han proporcionado tres parámetros
if [ "$#" -ne 2 ]; then
    echo "Uso: $0 <direccion_ip_servidor> <modelo>"
    exit 1
fi

# Asignar parámetros a variables
export SERVER_ADDR=$1
export MODEL=$2
set PYTHONIOENCODING=utf-8
export FL_DATAPATH="data/server/"
# Ejecutar el servidor del modelo seleccionado
if [ "$MODEL" = "sgd" ]; then
    SCRIPT_PATH="sgd/fl_server.py"
elif [ "$MODEL" = "dnn" ]; then
    SCRIPT_PATH="dnn/fl_server.py"
elif [ "$MODEL" == "nb" ]; then
    SCRIPT_PATH="nb/fl_server.py"
else
    echo "Modelo no válido: $MODEL"
    exit 1
fi
set MODEL = MODEL
python $SCRIPT_PATH