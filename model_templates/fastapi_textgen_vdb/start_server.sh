#!/bin/sh

echo "Starting Fast APIr"

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi

echo
echo "Executing command: for Fast API"
echo
exec python server.py
