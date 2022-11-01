#!/bin/bash
# Use this script to check how the cleaning is working for a specific paper
# it is useful to check the cleaning process step by step and to improve it

# Enable poetry if not running inside docker and poetry is installed
if [[ $HOSTNAME != "docker-"* ]] && (hash poetry 2>/dev/null); then
    run_command="poetry run python text_cleaner.py -l debug"
else
    run_command="python text_cleaner.py -l debug"
fi

paper_separator='<#sep#>'

# clean_abstracts=1

# index=572
title="Hybrid Item-Item Recommendation via Semi-Parametric Embedding"
conf=ijcai
year=2019
stop_when="international joint artificial intelligence"



if [ -n "$clean_abstracts" ]; then
    # clean papers abstracts
    # select paper by index or title
    if [ -n "$index" ]; then
        # stop debugging when string is found
        if [ -n "$stop_when" ]; then
            $run_command -f data/$conf/$year/abstracts.csv -a --stop_when "$stop_when" -i $index
        else
            $run_command -f data/$conf/$year/abstracts.csv -a -i $index
        fi

    else
        # stop debugging when string is found
        if [ -n "$stop_when" ]; then
            $run_command -f data/$conf/$year/abstracts.csv -a --stop_when "$stop_when" -t "$title"
        else
            $run_command -f data/$conf/$year/abstracts.csv -a -t "$title"
        fi
    fi

else
    # clean papers content
    # select paper by index or title
    if [ -n "$index" ]; then
        # stop debugging when string is found
        if [ -n "$stop_when" ]; then
            $run_command -f data/$conf/$year/pdfs.csv -s "$paper_separator" --stop_when "$stop_when" -i $index
        else
            $run_command -f data/$conf/$year/pdfs.csv -s "$paper_separator" -i $index
        fi

    else
        # stop debugging when string is found
        if [ -n "$stop_when" ]; then
            $run_command -f data/$conf/$year/pdfs.csv -s "$paper_separator" --stop_when "$stop_when" -t "$title"
        else
            $run_command -f data/$conf/$year/pdfs.csv -s "$paper_separator" -t "$title"
        fi
    fi
fi

