#!/bin/bash
# Enable poetry if not running inside docker and poetry is installed
if [[ $HOSTNAME != "docker-"* ]] && (hash poetry 2>/dev/null); then
    run_command="poetry run"
fi

conferences=(
    # "aaai 2017"
    # "aaai 2018"
    # "aaai 2019"
    # "aaai 2020"
    # "aaai 2021"
    # "aaai 2022"
    # "acl 2017"
    # "acl 2018"
    # "acl 2019"
    # "acl 2020"
    # "acl 2021"
    # "acl 2022"
    # "coling 2018"
    # "coling 2020"
    # "coling 2022"
    # "cvpr 2017"
    # "cvpr 2018"
    # "cvpr 2019"
    # "cvpr 2020"
    # "cvpr 2021"
    # "cvpr 2022"
    "cvpr 2023"
    # "eacl 2017"
    # "eacl 2021"
    "eacl 2023"
    # "eccv 2018"
    # "eccv 2020"
    # "eccv 2022"
    # "emnlp 2017"
    # "emnlp 2018"
    # "emnlp 2019"
    # "emnlp 2020"
    # "emnlp 2021"
    "emnlp 2022"
    # "findings 2020"
    # "findings 2021"
    # "findings 2022"
    "findings 2023"
    # "iccv 2017"
    # "iccv 2019"
    # "iccv 2021"
    # "iclr 2018"
    # "iclr 2019"
    # "iclr 2020"
    # "iclr 2021"
    # "iclr 2022"
    # "icml 2017"
    # "icml 2018"
    # "icml 2019"
    # "icml 2020"
    # "icml 2021"
    # "icml 2022"
    # "ijcai 2017"
    # "ijcai 2018"
    # "ijcai 2019"
    # "ijcai 2020"
    # "ijcai 2021"
    # "ijcai 2022"
    # "ijcnlp 2017"
    # "ijcnlp 2019"
    # "ijcnlp 2021"
    "ijcnlp 2022"
    # "kdd 2017"
    # "kdd 2018"
    # "kdd 2020"
    # "kdd 2021"
    # "naacl 2018"
    # "naacl 2019"
    # "naacl 2021"
    # "naacl 2022"
    # "neurips 2017"
    # "neurips 2018"
    # "neurips 2019"
    # "neurips 2020"
    # "neurips 2021"
    # "neurips 2022"
    # "neurips_workshop 2019"
    # "neurips_workshop 2020"
    # "neurips_workshop 2021"
    # "neurips_workshop 2022"
    # "sigchi 2018"
    # "sigchi 2019"
    # "sigchi 2020"
    # "sigchi 2021"
    # "sigchi 2022"
    # "sigdial 2017"
    # "sigdial 2018"
    # "sigdial 2019"
    # "sigdial 2020"
    # "sigdial 2021"
    "sigdial 2022"
    # "tacl 2017"
    # "tacl 2018"
    # "tacl 2019"
    # "tacl 2020"
    # "tacl 2021"
    # "tacl 2022"
    # "wacv 2020"
    # "wacv 2021"
    # "wacv 2022"
    # "wacv 2023"
)

paper_separator='<#sep#>'

extract_pdfs=1
extract_urls=1
clean_abstracts=1
clean_papers=1
unify_papers=1
papers_with_code=1

abstract_only_conferences=(
    "kdd"
    "sigchi"
)

for conference in "${conferences[@]}"; do
    conf_year=($conference)

    # extract text from pdf and save csv with titles and full paper texts
    if [ -n "$extract_pdfs" ] && [[ ! " ${abstract_only_conferences[*]} " =~ " ${conf_year[0]} " ]]; then
        echo -e "\nExtracting pdfs for ${conf_year[0]} ${conf_year[1]}"
        $run_command python pdf_extractor.py -p -s "$paper_separator" -c ${conf_year[0]} -y ${conf_year[1]}
    fi

    # extract urls from pdf and save csv with titles and all found urls
    if [ -n "$extract_urls" ] && [[ ! " ${abstract_only_conferences[*]} " =~ " ${conf_year[0]} " ]]; then
        echo -e "\nExtracting urls for ${conf_year[0]} ${conf_year[1]}"
        $run_command python url_scrapper.py -f data/${conf_year[0]}/${conf_year[1]}/pdfs.csv
    fi

    # clean papers abstracts
    if [ -n "$clean_abstracts" ]; then
        echo -e "\nCleaning abstracts for ${conf_year[0]} ${conf_year[1]}"
        $run_command python text_cleaner.py -f data/${conf_year[0]}/${conf_year[1]}/abstracts.csv -a
    fi

    # clean papers content
    if [ -n "$clean_papers" ] && [[ ! " ${abstract_only_conferences[*]} " =~ " ${conf_year[0]} " ]]; then
        echo -e "\nCleaning papers for ${conf_year[0]} ${conf_year[1]}"
        $run_command python text_cleaner.py -f data/${conf_year[0]}/${conf_year[1]}/pdfs.csv -s "$paper_separator"
    fi
done

if [ -n "$unify_papers" ]; then
	$run_command python unify_papers_data.py -l info
fi

if [ -n "$papers_with_code" ]; then
    echo -e "\nAdding information from papers with code"
    $run_command python add_papers_with_code.py -p -l info
fi
