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
clean_papers=1

index=1
# title="Moon IME: Neural-based Chinese Pinyin Aided Input Method with Customizable Association"
conf=aaai
year=2017


# clean papers abstracts
if [ -n "$clean_abstracts" ]; then
    if [ -n "$index" ]; then
        $run_command -f data/$conf/$year/abstracts.csv -a -i $index
    else
        $run_command -f data/$conf/$year/abstracts.csv -a -t "$title"
    fi
fi

# clean papers content
if [ -n "$clean_papers" ]; then
    if [ -n "$index" ]; then
        $run_command -f data/$conf/$year/pdfs.csv -s "$paper_separator" -i $index
    else
        $run_command -f data/$conf/$year/pdfs.csv -s "$paper_separator" -t "$title"
    fi
fi
