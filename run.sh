#! /bin/bash
echo 'hello'
script=$1

if [ ! -n "${script}" ] ;then
    echo "need script info, eg: sh run.sh classic_control.cart_pole.cart_pole_dqn"
    exit
else
    /Users/guoliang21/miniforge3/bin/python3 -m ${script}
fi
