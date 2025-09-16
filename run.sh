#! /bin/bash

script=$1

if [ ! -n "${script}" ] ;then
    echo "need script info, eg: sh run.sh classic_control.cart_pole.cart_pole_dqn"
    exit
else
    python3 -m ${script}
fi
