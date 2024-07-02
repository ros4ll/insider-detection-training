#!/bin/bash

# Verificar que se han proporcionado tres parámetros
if [ "$#" -ne 3 ]; then
    echo "Uso: $0 <direccion_ip_servidor> <numero_cliente> <modelo>"
    exit 1
fi

# Asignar parámetros a variables
export SERVER_ADDR=$1
export CLIENT_NUM=$2
MODEL=$3
set PYTHONIOENCODING=utf-8

if [ "$CLIENT_NUM" != "1" ] && [ "$CLIENT_NUM" != "2" ]; then
    echo "El número de cliente debe ser 1 o 2"
    exit 1
fi
export FL_DATAPATH="data/site-$CLIENT_NUM/"
echo "FL_DATAPATH:" $FL_DATAPATH
# Ejecutar el cliente federado del modelo seleccionado
if [ "$MODEL" == "sgd" ]; then
    SCRIPT_PATH="sgd/fl_client.py"
elif [ "$MODEL" == "dnn" ]; then
    SCRIPT_PATH="dnn/fl_client.py"
elif [ "$MODEL" == "nb" ]; then
    SCRIPT_PATH="nb/fl_client.py"
else
    echo "Modelo no válido: $MODEL"
    exit 1
fi
python $SCRIPT_PATH