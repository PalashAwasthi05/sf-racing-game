#!/bin/zsh
# Copy the viewer modules into variants/<name> (default "live"), e.g. before and after a change, to compare them.
cd "$(dirname "$0")"
name=${1:-live}; mkdir -p variants/$name && cp ../king-street/app/*.ts variants/$name/ && ls variants/$name
