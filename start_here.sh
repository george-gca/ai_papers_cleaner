#!/bin/bash
# Enable poetry if not running inside docker and poetry is installed
if [[ $HOSTNAME != "docker-"* ]] && (hash poetry 2>/dev/null); then
    run_command="poetry run"
fi

conferences=(
    "aaai"
    "acl"
    "coling"
    "cvpr"
    "eacl"
    "eccv"
    "emnlp"
    "findings"
    "iccv"
    "iclr"
    "icml"
    "icra"
    "ijcai"
    "ijcnlp"
    "ijcv"
    "kdd"
    "naacl"
    "neurips"
    "neurips_workshop"
    "sigchi"
    "sigdial"
    "siggraph"
    "siggraph-asia"
    "tacl"
    "tpami"
    "wacv"
)

paper_separator='<#sep#>'

extract_pdfs=1
extract_urls=1
clean_abstracts=1
clean_papers=1
unify_papers=1
papers_with_code=1

# uncomment these lines if you want to run only for a specific conference and year
conference="cvpr"
year=2025

abstract_only_conferences=(
    "icra"
    "ijcv"
    "kdd"
    "sigchi"
    "siggraph"
    "siggraph-asia"
    "tpami"
)

current_year=$(date +'%Y')

for conf in "${conferences[@]}"; do
    for y in $(seq 2017 $current_year); do
        if [ -d "data/$conf/$y" ]; then
            if [ -n "$conference" ] && [[ "$conference" != "$conf" ]]; then
                continue
            fi

            if [ -n "$year" ] && [[ "$year" != "$y" ]]; then
                continue
            fi

            # extract text from pdf and save csv with titles and full paper texts
            if [ -n "$extract_pdfs" ] && [[ ! " ${abstract_only_conferences[*]} " =~ " $conf " ]]; then
                echo -e "\nExtracting pdfs for $conf $y"
                $run_command python pdf_extractor.py -p -s "$paper_separator" -c $conf -y $y
                # $run_command python pdf_extractor.py -p -s "$paper_separator" -c $conf -y $y -f "Lee_Revisiting_Self-Similarity_Structural_Embedding_for_Image_Retrieval_CVPR_2023" -l debug
            fi

            # extract urls from pdf and save tsv with titles and all found urls
            if [ -n "$extract_urls" ] && [[ ! " ${abstract_only_conferences[*]} " =~ " $conf " ]]; then
                echo -e "\nExtracting urls for $conf $y"
                $run_command python url_scrapper.py -f data/$conf/$y/pdfs.csv
            fi

            # clean papers abstracts
            if [ -n "$clean_abstracts" ]; then
                echo -e "\nCleaning abstracts for $conf $y"
                $run_command python text_cleaner.py -f data/$conf/$y/abstracts.tsv -a
            fi

            # clean papers content
            if [ -n "$clean_papers" ] && [[ ! " ${abstract_only_conferences[*]} " =~ " $conf " ]]; then
                echo -e "\nCleaning papers for $conf $y"
                $run_command python text_cleaner.py -f data/$conf/$y/pdfs.csv -s "$paper_separator"
            fi

        fi
    done
done

if [ -n "$unify_papers" ]; then
	$run_command python unify_papers_data.py -l info
fi

if [ -n "$papers_with_code" ]; then
    echo -e "\nAdding information from papers with code"
    $run_command python add_papers_with_code.py -p -l info
fi
