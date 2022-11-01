import argparse
import logging
import multiprocessing
import re
from ast import literal_eval
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

# check if splitted word is one single word in english
import enchant
import ftfy
import inflect
import pandas as pd
# TODO change for termcolor https://github.com/facebookresearch/mmf/blob/7fa7607d299a644158dc3104247e58a30e4b42ab/mmf/utils/_logger.py
from colorama import Back, Fore
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from timer import Timer
from utils import parallelize_dataframe, setup_log


_logger = logging.getLogger(__name__)
_grammar = inflect.engine()

LINE_LIMIT = 5000
BG_HIGHLIGHT_COLOR = Back.GREEN


# TODO clean these
# advancement_artifcial_intelligence_www_aaai
# copyright_association_advancement_artifcial_intelligence
# association_advancement_artifcial_intelligence_www
# artifcial_intelligence_www_aaai_org
# intelligence_www_aaai_org
# introduction_deep_neural_network_dnn
# international_joint_artificial_intelligence
# introduction_deep_neural_network

# avoid removing s of these words:
# cros_entropy_los
# without_los_generality_assume
# neural_architecture_search_na
# principal_component_analysi_pca
# binary_cros_entropy_los

# TODO: check for unknown words. if an unknown word is found, check for it as separated words
# if it exists, replace them by the correct words

class TextCleaner():
    """
    Useful link to test regex
    https://regex101.com/r/NfK7rz/9
    https://www.loggly.com/blog/five-invaluable-techniques-to-improve-regex-performance/
    """
    # TODO improve regexes, change for special chars like \b
    # https://docs.python.org/3/library/re.html#regular-expression-syntax

    def __init__(self, debug: bool = False):
        super().__init__()

        self._logger = logging.getLogger(__name__)
        self._composite_words = [
            'state-of-the-art[s]?',
        ]

        self._stop_words = set(stopwords.words("english"))
        new_stop_words = [
            # 'application',
            # 'applications',
            # 'approach',
            # 'approaches',
            'also',
            'conference',
            'however',
            'paper',
            'propose',
            'proposed',
            'us',
            'use',
            'used',
            'using',
            'vs',
        ]

        for w in new_stop_words:
            self._stop_words.add(w)

        # new terms not found in dictionary, used during agglutination step
        self._new_words = {
            '2_bit',
            '3_bit',
            '4_bit',
            '5_bit',
            '6_bit',
            '7_bit',
            '8_bit',
            'accuracy',
            'achieve',
            'approximability',
            'autoregressive',
            'avgpooling',
            'backpropagation',
            'bit_rate',
            'bitrate',
            'capsnet',
            'centroid',
            'cnn',
            'codec',
            'convolutional',
            'crowdsourced',
            'crowdsourcing',
            'datapoint',
            'dataset',
            'deblurring',
            'deepfake',
            'demoireing',
            'demosaicing',
            'denoising',
            'dnn',
            'downsampling',
            'downscaling',
            'edit',
            'extrinsic',
            'gan',
            'gnn',
            'hyperparameter',
            'interdependences',
            'interdependency',
            'interpretability',
            'intrinsic',
            'iterative',
            'iteratively',
            'judgement',
            'maxpooling',
            'mini-batch',
            'minibatch',
            'minpooling',
            'misclassification',
            'moire',
            'outperform',
            'pre',
            'pseudo_label',
            'regularizer',
            'resampling',
            'rescaling',
            'rnn',
            'scalability',
            'scalable',
            'sigmoid',
            'smartphone',
            'state_of_the_art',
            'thresholding',
            'u_net',
            'upsampling',
            'upscaling',
            'voxel',
        }

        non_singularizable_words = {
            'nas', # neural architecture search
            'thus'
        }

        for w in non_singularizable_words:
            _grammar.defnoun(w, w)

        # acl conferences regex
        prefix_proceedings = '(in )?proceedings of [\d\w\s\(\)#&\-\−\–:,]+'
        prefix_paper_type = '( \((short|long) papers\)|(-|−|–|: )(system demonstrations|tutorial abstracts)|(-|:|,) student research workshop)?(\s)*(,)? '
        pages_and_location = 'page(s)? [\d]+((-|−|–)[\d]+)?(,)? ([\w]+( [\w]+)*, [\w]+( [\w]+)*(,|.)? )?'
        dates_year = '([\d]+ )?[\w]+(,)? [\d]+(st|nd|rd|th)?(( )?(-|−|–)( )?([\w]+ )?[\d]+)?(,)?( [\d]+)?. '
        suffix_acl = '[\d\w©]+ (association for computational linguistics|afnlp)'

        # item citation regex
        item_citation = '(appendix[es]?|equation[s]?|figure[s]?|table[s]?|fig[s]?.|theorem[s]?|(sub)?section[s]?|(sub)?sec[s]?.)'

        """
        examples of conferences names that might appear and must be supported
        ps: this happens due to the lack of being able to correctly remove all references
        in conferences like ijcai 2017

        in proceedings of the twenty-first annual acm-siam symposium on discrete algorithms, soda 2010, austin, texas, usa, january 17-19, 2010, pages 1065-1075,
        in advances in neural information processing systems 27: annual conference on neural information processing systems 2014, december 8-13 2014, montreal, quebec, canada, pages 190-198, 2014.
        journal of machine learning research,
        in proceedings of the 27th international conference on machine learning (icml-10), june 21-24, 2010, haifa, israel, pages 911-918, 2010.
        in proceedings of the joint european conference on machine learning and knowledge discovery in databases, ecml-pkdd 2011, athens, greece, september 5-9, 2011, pages 349-364, 2011.
        university press, 2014.
        in 10th international conference on machine learning. learning and applications and workshops, icmla 2011, honolulu, hawaii, usa, december 18-21, 2011. volume 2: special sessions and workshop, pages 84-89, 2011.
        in proceedings of the 30th international conference on machine learning, icml 2013, atlanta, ga, usa, 16-21 june 2013, pages 504-512, 2013.
        """

        num_or_hyphen = '[\d\-\−\–]+'
        num_colon_hyphen_par = '[\d:\-\−\–\(\)]+'
        ordinal_num = '[\d]+(st|nd|rd|th)'

        self._bib_info_to_remove = [
            'acm transactions on graphics',
            '(aaai|arxiv|cvpr|eccv|iccv|iclr|icml|ijcv|jmlr|neurips|wacv), [\d]+',
            'arxiv preprint arxiv:[\d]+.[\d]+(, [\d]+.)?',
            f'advances in neural information processing systems {num_colon_hyphen_par}',
            'advances in neural information processing systems, (volume|vol- ume) [\d]+, [\d]+',
            f'ai magazine, {num_colon_hyphen_par}, [\d]+',
            f'artificial (intelligence|in- telligence), {num_colon_hyphen_par}, [\d]+',
            f'biometrics, {num_colon_hyphen_par}, [\d]+',
            'cambridge university press',
            # in   annual meeting of the association for computational linguistics
            # in   annual meeting of the association for computational linguistics and the   international joint conference on natural language processing
            f'computational linguistics, {num_colon_hyphen_par}',
            'cognitive science, [\d]+',
            'conference on neural information processing systems (neurips|nips) [\w]+ ([\w]+ )*(canada|usa)',
            'conference on neural information processing systems \((neurips|nips) [\d]+\), [\w\s]+, ([\w]+, )*[\w]+',
            'conference on neural information processing systems \(neurips [\d]+\)',
            'copyright [\S]+ [\d]+, association for the advancement of (artificial|artifcial|artifi- cial) intelligence \(www.aaai.org\)',
            f'experimental economics {num_colon_hyphen_par}',
            'ijcv, [\w]+ [\d]+',
            f'in [\d]+ aaai [\w]+ symposium series, [\d]+',
            # 'in advances in neural information processing systems',
            f'in ({ordinal_num} )?annual meeting of the association for computational linguistics( and the ({ordinal_num} )?international joint conference on natural language processing)?',
            '(in proc |ieee )?international (conference on|journal of) computer vision (and (pattern|pat- tern) recognition|workshops( iccv workshops)?)?',
            'in (proceedings of |proc. )?international conference on learning representation(s)?(, [\d]+)?',
            'in (proceedings of )?the ieee conference on computer vision and pattern recognition(, pp. [\d\s\-\−\–]+, [\d]+| \(cvpr\))?',
            f'in computer vision and (pattern|pat- tern) recognition, [\d]+. cvpr [\d]+. ieee conference on, {num_or_hyphen}. ieee',
            'in (proceedings of )?the ieee conference on computer vision and (pattern|pat- tern) recognition',
            f'in proceedings of the {ordinal_num} international conference on machine learning \(icml-[\d]+\)',
            f'in proceedings of the {ordinal_num} international conference on machine learning[\s\-\−\–]volume [\d]+, pp. [\s\d\-\−\–]+. jmlr. org, [\d]+',
            'in computer vision and (pattern|pat- tern) recognition \(cvpr\)',
            'in computer vision and (pattern|pat- tern) recognition cvpr',
            f'in advances in neural information processing systems(, pp. [\d\s\-\−\–]+, [\d]+|, {num_or_hyphen})?',
            'in the ieee international conference on computer vision \(iccv\)',
            '(in the [\d]+ )?conference of the north american chapter of the association for computational linguistics',
            f'in www, {num_or_hyphen}. acm',
            f'in proceedings of the {ordinal_num} international conference on (artificial|artifcial|artifi- cial) intelligence, {num_or_hyphen}. aaai press',
            f'in proceedings of the [\w]+ international conference on artificial intelligence and statistics, pp. {num_or_hyphen}, [\d]+',
            'in (proceedings of |proc. )?international conference on machine learning((, volume [\d]+, pp. [\d\s\-\−\–]+)?, [\d]+)?',
            'in international joint conferences on (artificial|artifcial|artifi- cial) intelligence',
            f'ieee transactions on [\w\s\-\−\–]*{num_colon_hyphen_par}',
            'in proceedings alt, [\d]+',
            f'journal of artificial intelligence research, {num_colon_hyphen_par}',
            f'journal of machine learning research( {num_colon_hyphen_par})?',
            f'journal of the [\w\s:]+ [\w\(\)\s]*{num_colon_hyphen_par}',
            'journal of the [\w\s]+, [\d]+',
            '(the )?mit press( cambridge)?, [\d]+',
            f'nature, {num_colon_hyphen_par}, [\d]+',
            'pattern analysis and machine intelligence',
            'proc siggraph',
            f'proceedings of the {ordinal_num} international conference on computational linguistics, page(s)? {num_or_hyphen} [\w\s,]+( \(online\),)? [\w]+ {num_or_hyphen}, [\d]+',
            '(cam- bridge |cambridge )?university press(,)? [\d]+(.|,)',
            'workshop track',
            '(in|proceedings of the) [\w\d\s\-.]+, page(s)? [\d]+(-[\d]+)?([\w\s\-,]+)?',
            f'international conference [\w\s\-\−\–]+, pp. [\d\s\-\−\–]+. [\w]+, [\d]+',
            #   international conference on intelligent robots and systems, pp. 5026-5033. ieee, 2012.
            # 'pages [\d]+-[\d]+',
            # f'proceedings of the [\d]+[\s]*(st|nd|rd|th) international conference on [\w\s,\-\−\–]+ pmlr ([\d,\s]+.)? copyright ([\d]+ )?by the (author|au- thor)\(s\)',
            f'proceedings of the {ordinal_num} workshop [\w\s\-\−\–\(\)]+, page(s)? {num_or_hyphen} [\w\s]+, [\w\s]+( \(online\))?, [\w]+ [\d]+, [\d]+',
            '(©)?[\d]+ association for computational linguistics( [\d]+)?',
            f'proceedings of the [\d]+[\s]*(st|nd|rd|th) international conference on [\w\s,\-\−\–]+ pmlr ([\d,\s]+.)? copyright ([\d]+ )?by the author\(s\)',
            'proceedings of the (st|nd|rd|th) international conference on machine learning online pmlr copyright by the author',
            f'proceedings of the {ordinal_num} international conference on machine learning \(icml [\d]+\), [\d]+',
            '(proceedings of the|in) [\w]+(-[\w]+)? aaai conference on (artificial|artifcial|artifi- cial) intelligence(, [\d]+| \(aaai-[\d]+\))?',
            f'{prefix_proceedings}{prefix_paper_type}{pages_and_location}{dates_year}{suffix_acl}',
            'c[\d]+ association for computational linguistics',
            f'findings of the association for computational linguistics: emnlp [\d]+, page(s)? {num_or_hyphen}(,|.)? [\w]+ [\d\-\−\–\s]+(,|.) [\d]+',
            f'(in )?(advances in neural information processing|findings of|proceedings of|{ordinal_num}? (international )?conference (on|in)) [\d\w\s\(\)#&\-\−\–:,.]+ page(s)? {num_or_hyphen}(,|.)( [\d]+)?',
            '[\w]+, [\w]+, [\w]+ [\d]+ - [\w]+ [\d]+, [\d]+. [\w\d]+ association for computational linguistics',
            f'[\w]+, [\w]+( \(online\))?, [\w]+ {num_or_hyphen}, [\d]+. (©)?[\d]+ association for computational linguistics proceedings of the {ordinal_num}[\w\s]+, page(s)? {num_or_hyphen}',
            '[\w]*proceedings of the [\w\-]+ international joint conference on (artificial|artifcial|artifi- cial) intelligence \([\d\w\-]+\)',
            'published as a conference paper at [\w]+( [\d]+)?',
            'proceedings of the [\d]+[\s]*(st|nd|rd|th) international conference on machine learning, [\w]+( [\w]+)*, ([\w]+( [\w]+)*, )*pmlr [\d]+, [\d]+. copyright( [\d]+)? by the author(\(s\))?',
            'published as a conference paper at [\w]+( [\d]+)',
            f'[\w]+ state university {num_colon_hyphen_par}',
            'springer science & business media(, [\d]+)?',
            'statistical learning theory(, volume [\d]+)?',
            f'statistical science, {num_colon_hyphen_par}, [\d]+',
            f'the [\w\s,\-\−\–]+ aaai conference on artificial intelligence \(aaai{num_or_hyphen}\)',
            f'university of [\w\s]+, (dept.|department of) [\w\s\-\−\–]+, [\w\s\-\−\–.]+, [\d]+(, [\d]+)?',
            # '(in|proceedings of the) [\w\d\s\-.]+, page(s)? [\d]+(-[\d]+)?(,|.) [\w\s]+, [\d]+',
        ]

        self._phrases_to_remove = [
            '(ablation studies )?we (also )?conduct(ed)? ablation studies( to)?',
            'ablation studies',
            '(these |both the |all the |the first [\w]+ |first [\w]+ |the )?(authors )?contribute(d)? equally( to this work)?',
            'in fact',
            'corresponding author',
            f'(in {item_citation} [\d]+ )?((we )|(it ))?(also |now )?show(s|n)? (in {item_citation} [\d]+ )?(that|(the )?result(s)?)',
            '(we have |has )?show(n)? that',
            f'(it is )?(show(n)? )?(in )?this {item_citation}( that)?',
            'even though',
            'experimental result[s]?( show(n|s)?| prove(s)?)?',
            'let denote',
            'achieve(d)? better performance',
            'proposed method',
            '(our )?contributions are summarized as follows',
            'state of the art[s]?',
            '(a number of |most |in )?(recent|related|previous|prior) (work|year)[s]?( have)?( also)?',
            '(to the )?best of our knowledge',
            '(we )?would like( to)?',
            '(we )?conduct(ed)? extensive experiment(s)?',
            'supplementary material[s]?',
            'all rights reserved(.|,)?',
            'best viewed in color',
            'this work is (licensed|licenced) under a creative commons attribution [\d.]+ international (license|licence). (license|licence) details',
            '(part of the )?(work|research) done during (an )?internship at',
            '(the |this |these )?(experiment|result)[s]? (show(n|s|ing)?|prove(s)?)( that)?',
            '(in )?the [\w]+ (row|column)(s)?( we)?( show(s)?)?( that| the)?',
            '(in )?each (row|column)(s)?(we show( that| the)?)?',
            # '(in )?(the )?([\w]+|each) ([\w]+ )?(row|column)(s)?( show(s)?)?( that| the)?',
            '(the |this )?(work|research|paper) was (partially )?(done|funded by|supported by) ([\w\s\d\-\_,]+ \(((, | and )?no. [\d\w\-\s]+)+\))+.',
            '(the |this )?(work|research|paper) was (partially )?(done|funded by|supported by) [\w\s\d\-\_]+(no|and no)+ ([\w\d\-\_]+)*',
            '(the |this )?(work|research|paper) was (partially )?(done|funded by|supported by) [\w\s\d\-\_]+\((grant no(.)? [\d]+| and |, and |no(.)? [\d]+)+\)',
            '(the |this )?(work|research|paper) was (partially )?(done|funded by|supported by)',
            'joint research project with youtu lab of tencent',
            '(the code )?(is |will be )?available at',
        ]

        self._remaining_two_chars = [
            '\\b(x|y|z|w|i|j|k)(_)?[\d]\\b',
            '\\b(x|y|z|w|i|j|k)(_)?[aeiouxyzwjk]\\b',
            '\\bth\\b',
        ]

        self._symbols_dict = {
            '∼' : '~',
            '–': '-',
            '−': '-',
            'ꟻ': 'F',
        }

        self._words_with_footnotes = [
            'available',
            'code[s]?',
            'dataset[s]?',
            'github',
            'image[s]?',
            'model[s]?',
            'video[s]?',
        ]

        self._metrics = [
            'db',
            'px'
        ]

        self._create_regexes_dict()

        self._debug = debug
        if self._debug:
            self._logger.debug(
                f'\nComposite words to remove: {sorted(self._composite_words)}')
            self._logger.debug(
                f'\nAdditional words to agglutinate: {sorted(self._new_words)}')
            self._logger.debug(f'\nStop words: {sorted(self._stop_words)}')
            self._logger.debug(
                f'\nCommon words with footnotes: {self._words_with_footnotes}')

    def _create_regexes_dict(self):
        self._regexes = {}

        # aglutinate_urls
        regex = '\\b[0-9]?(ht|f)tp[s]?:[\\s/]+[\w.\-\−\–]+(-[\\s]+)?(.[\\s]+)?(?![0-9]?((ht|f)tp[s]?|. ))[\w:/?.&=#\-\−\–\~\˜\+]+([\-\−\–_:/][\\s]+(?![0-9]?((ht|f)tp[s]?|. ))[\w:/?.&=#\-\−\–\~\˜\+]+)*[\\s]*[,.\)/]'
        regex += '((htm|html|mp4|py|zip|php)\\b)?'  # possible extensions
        self._regexes['aglutinate_urls'] = re.compile(regex)

        # aglutinate_words
        regex = '([\w]+)[\s]*-[\s]*([\w]+)'
        self._regexes['aglutinate_words'] = re.compile(regex)

        # remove_cid
        regex = '\(cid:[\s0-9]+\)'
        self._regexes['remove_cid'] = re.compile(regex)

        # remove_composite_words
        regex = '|'.join(self._composite_words)
        regex = '[\s]*[\-\−\–_]?[\s]*'.join(regex.split('-'))
        self._regexes['remove_composite_words'] = re.compile(regex)

        # remove_eg_ie_etal
        regexes = ['e\.g\.',  # e.g.
                    'i\.e\.',  # i.e.
                    'w\.r\.t\.',  # w.r.t.
                    # someone et.al.
                    '(\(|\\b)[\w\-\−\–´\'&]+[\s]+et[.]?[\s]+al[ .\b]([\s]*(,[\s]+|\()[\d]+)?(\)|;|\\b)?',
                    '(\\b|\()[\w]+[\s]+&[\s]+[\w]+([\s]+\(|,[\s]+)[\d]+(\)|\\b)',  # someone & something
        ]
        regex = '|'.join(regexes)
        self._regexes['remove_eg_ie_etal'] = re.compile(regex)

        # remove_equations
        regex = '^[^ ]+[\s]?[=≡→←⊗][\s]?[^ \n]+[\s]*$'
        self._regexes['remove_equations'] = re.compile(regex, re.MULTILINE)

        # remove_emails
        regex = '[\w.\-\−\–]+@[\w\-\−\–]+\.[\w.\-\−\–]+'
        self._regexes['remove_emails'] = re.compile(regex)

        # remove_footnotes_numbers
        regex = '|'.join(self._words_with_footnotes)
        regex = f'\\b({regex})[0-9]\\b'
        self._regexes['remove_footnotes_numbers'] = re.compile(regex)

        # remove_html_tags
        regex = '[\s]*<(/)?[\w\s_\-\−\–=\"]+(/)?>[\s]*'
        self._regexes['remove_html_tags'] = re.compile(regex)

        # remove_hyphens_slashes
        regex = '[\s]+[\-\−\–]|[\-\−\–][\s]+|/'
        self._regexes['remove_hyphens_slashes'] = re.compile(regex)

        # remove_item_citation
        regexes = []
        # item citation
        prefix = '(\()?(((as|are)?[\s]*shown|see([\s]+next)?|summarized)?([\s+]in)?|please[\s]+refer[\s]+to)?'
        item = '[\s]+(appendix[es]?|equation[s]?|figure[s]?|table[s]?|fig[s]?.|theorem[s]?|(sub)?section[s]?|(sub)?sec[s]?.)'
        first_number_letter = '[\s.]+[0-9]*([\-\−\–\(.]?[\w]{1}[\)]?)?'
        next_number_letters = '((,[\s]+|[\s]+and[\s]+|\-)[0-9]*([\-\−\–\(.]?[\w]{1}[\)]?)?)*'
        suffix = '([\s]+(show[s]?|above|below))?([\s,.:\)]|\\b)'
        regexes.append(
            f'{prefix}{item}{first_number_letter}{next_number_letters}{suffix}')
        # references citation
        regexes.append('\[[0-9][,\s\-\−\–0-9]*\]')
        regex = '|'.join(regexes)
        self._regexes['remove_item_citation'] = re.compile(regex)

        # remove_latex_commands
        regex = '[\s]*\\\[^ ]+[\s]*'
        self._regexes['remove_latex_commands'] = re.compile(regex)

        # remove_latex_inline_equations
        regex = '[\s]*\$[^\$]+\$[\s]*'
        self._regexes['remove_latex_inline_equations'] = re.compile(regex)

        # remove_latexit_tags
        regex = '<latexit[\d\s\w/=\"+_()]*>[\s\w\d=/+()]*</latexit>'
        self._regexes['remove_latexit_tags'] = re.compile(regex)

        # remove_metrics
        metrics_regex_str = '|'.join([f'[\s]?{w}' for w in self._metrics])
        regex = f'[\s\(\[\{{\∼\~][0-9]+([.,][0-9]+)?([\s]?[%kmb]+|{metrics_regex_str})?[.,:\s\)\]\}}]'
        self._regexes['remove_metrics'] = re.compile(regex)

        # remove_numbers
        regexes = []
        # numbers
        regexes.append(
            '[\s\(\[\{\∼\~][-]?[0-9]+([.,][0-9]+)?([\s]+[-]?[0-9]+([.,][0-9]+)?)*[.,:\s\)\]\}]')
        # roman numbers
        regexes.append('[\(\s][ivx]+\)')
        # enumerates
        regexes.append('\([\s]*[a-zA-Z][\s]*\)')
        regex = '|'.join(regexes)
        self._regexes['remove_numbers'] = re.compile(regex)

        # remove_numbers_only_lines
        regexes = []
        # numbers
        regexes.append(
            '^[\(]?[-−]?[0-9]+([.,][0-9]+)?([\s\+\-\−\–\±]+[\-\−\–]?[0-9]+([.,][0-9]+)?)*[\)]?[\s]*$')
        # headers
        regexes.append('^[0-9]+.([0-9]+[.]?){0,2}$')
        regex = '|'.join(regexes)
        self._regexes['remove_numbers_only_lines'] = re.compile(regex, re.MULTILINE)

        # remove_ordinal_numbers
        regex = '[0-9]*1st|[0-9]*2nd|[0-9]*3rd|[0-9]+th'
        self._regexes['remove_ordinal_numbers'] = re.compile(regex)

        # remove_bib_info
        bib_info_to_remove = [fr'\b{p}' for p in self._bib_info_to_remove]
        bib_info_to_remove = [fr'{p}\b' if not p.endswith('\)') else p for p in bib_info_to_remove]
        regex = '|'.join(bib_info_to_remove)
        regex = '[\s]+'.join(regex.split())
        self._regexes['remove_bib_info'] = re.compile(regex)

        # remove_phrases
        phrases_to_remove = [fr'\b{p}' for p in self._phrases_to_remove]
        phrases_to_remove = [fr'{p}\b' if not p.endswith('\)') and not p.endswith('.') else p for p in phrases_to_remove]
        regex = '|'.join(phrases_to_remove)
        regex = '[\s]+'.join(regex.split())
        self._regexes['remove_phrases'] = re.compile(regex)

        # remove_remaining_two_chars
        regex = '|'.join(self._remaining_two_chars)
        regex = '[\s]+'.join(regex.split())
        self._regexes['remove_remaining_two_chars'] = re.compile(regex)

        # remove_shapes
        regex = '[hwck0-9]+[x×][hwck0-9]+[x×]?[hwck0-9]*'
        self._regexes['remove_shapes'] = re.compile(regex)

        # remove_specific_expressions
        regexes = []
        # arxiv:1701.08893, 2016.
        regexes.append('\\barxiv:[0-9]+.[0-9]+(, [\d]+[a-z]?)\\b')
        # pages 234– 241
        regexes.append('\\bpages[\s]+[0-9]+[\s]*[\-\−\–][\s]*[0-9]+\\b')
        regex = '|'.join(regexes)
        self._regexes['remove_specific_expressions'] = re.compile(regex)

        # remove_symbols
        regex = '[^ a-zA-Z0-9\-_/]'
        self._regexes['remove_symbols'] = re.compile(regex)

        # remove_tabular_data
        regexes = [
            # '^[\w\-\−\–]+[\t ]*[:=,.]?[\t ]*$',
            '\n\n[\w\-\−\–]+[\t ]*[:=,.]?[\t ]*\n',
            '^[^\s.,]+$',
        ]
        regex = '|'.join(regexes)
        self._regexes['remove_tabular_data'] = re.compile(regex, re.MULTILINE)

        # remove_urls
        regex = '\\b(ht|f)tp[s]?://[\w:/?.&=#\-\−\–\~\˜]+[/.,\\d\)]?'
        self._regexes['remove_urls'] = re.compile(regex)

        # replace_hyphens_by_underline
        regex = '([\w]+)-([\w]+)'
        self._regexes['replace_hyphens_by_underline'] = re.compile(regex)

    def _pretty_print(self, regex: Union[str, re.Pattern], text: str, color: int = BG_HIGHLIGHT_COLOR, border_line_limit: int = LINE_LIMIT, flags: re.RegexFlag=0) -> None:
        new_text = ''
        previous_start = 0

        if isinstance(regex, re.Pattern):
            gen = enumerate(regex.finditer(text))
        else:
            gen = enumerate(re.finditer(regex, text, flags=flags))

        for i, match in gen:
            start, end = match.span()
            if i == 0 and len(text[previous_start:start]) > border_line_limit:
                new_text += f'...\n{text[start-border_line_limit:start]}'
            else:
                new_text += text[previous_start:start]

            new_text += color + text[start:end] + Back.RESET
            previous_start = end

        if previous_start != 0:
            if len(text[previous_start:]) > border_line_limit:
                new_text += f'{text[previous_start:previous_start+border_line_limit]}\n...'
            else:
                new_text += text[previous_start:]
            self._logger.debug(new_text)
        else:
            self._logger.debug('Text kept the same')


    def _highlight_not_recognized_words(self, text: str, spell_checker: enchant.Dict = enchant.Dict("en_US"),
                                        extra_lemmas_dict: Dict[str, str] = {}, color: int = Back.BLUE) -> str:
        def _recognized(word: str) -> bool:
            if len(word) > 0 and spell_checker.check(word) or word in self._new_words or word in extra_lemmas_dict.values():
                return True
            else:
                return False

        new_words = []
        text_split = text.strip().split()
        words_with_underline = [w for w in text_split if '_' in w]
        for word in words_with_underline:
            splitted = word.split('_')
            recognized_words = [w for w in splitted if _recognized(w)]
            if len(splitted) == len(recognized_words):
                new_words.append(word)

        new_words += self._new_words
        text = [w if spell_checker.check(
            w) or w in new_words else f'{color}{w}{Back.RESET}' for w in text_split]
        return ' '.join(text)

    def _run_regex(self, text: str, regexes: Union[str, List[str], re.Pattern],
                   replacement: str = ' ', flags: re.RegexFlag=0) -> str:
        if isinstance(regexes, list):
            regexes = '|'.join(regexes)
            regexes = re.compile(regexes, flags)
        elif isinstance(regexes, str):
            regexes = re.compile(regexes, flags)

        if self._debug:
            self._pretty_print(regexes, text)

        return regexes.sub(replacement, text)

    def aglutinate_urls(self, text: str, spell_checker: enchant.Dict = enchant.Dict("en_US")) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nAglutinating urls:')
        regex = self._regexes['aglutinate_urls']
        if self._debug:
            self._pretty_print(regex, text)
            self._logger.debug('')

        match = regex.search(text)
        previous_index = 0

        while match != None:
            start, end = match.span()
            word = text[previous_index +
                        start:previous_index+end]

            if word[0].isnumeric():
                word = text[previous_index + 1 +
                            start:previous_index+end]
                text = text.replace(
                    text[previous_index+start:previous_index+end], word)
                end -= 1

            if word.find(' ') > 0:
                if self._debug:
                    self._logger.debug(f'Aglutinating {word}')

                last_word = word.split()[-1]
                if last_word.endswith('.') or last_word.endswith(',') or last_word.endswith(')'):
                    last_word = last_word[:-1]

                if len(last_word) > 0 and last_word[0].isupper() and spell_checker.check(last_word):
                    end = end - len(word.split()[-1]) - 1
                    word = text[previous_index +
                                start:previous_index+end].strip()

                word = word.replace(' ', '')
                text = text.replace(
                    text[previous_index+start:previous_index+end], word)

            previous_index += end-1
            match = regex.search(text[previous_index:])

        return text

    def aglutinate_words(self, text: str, spell_checker: enchant.Dict = enchant.Dict("en_US")) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nAglutinating words separated by "-":')

        regex = self._regexes['aglutinate_words']
        not_existing_words_found = set()

        if self._debug:
            self._pretty_print(regex, text)
            self._logger.debug('')

        match = regex.search(text)
        previous_index = 0

        while match != None:
            original_word = text[previous_index +
                                 match.start():previous_index+match.end()]
            word = re.sub(regex, '\\1\\2', original_word)
            if word not in not_existing_words_found:
                if spell_checker.check(word) or word in self._new_words:
                    if self._debug:
                        self._logger.debug(f"{repr(word)} exists")
                    text = re.sub(original_word, word, text)
                elif re.search(word, text) != None:
                    if self._debug:
                        self._logger.debug(f"{repr(word)} exists")
                    self._new_words.add(word)
                    text = re.sub(original_word, word, text)
                else:
                    hyphen_word = re.sub(regex, '\\1_\\2', original_word)

                    if self._debug:
                        self._logger.debug(
                            f"{repr(word)} doesn't exist. Using {repr(hyphen_word)} instead")

                    if hyphen_word != original_word:
                        text = re.sub(original_word, hyphen_word, text)

                    not_existing_words_found.add(word)

            else:
                hyphen_word = re.sub(regex, '\\1_\\2', original_word)

                if self._debug:
                    self._logger.debug(
                        f"{repr(word)} doesn't exist. Using {repr(hyphen_word)} instead")

                if hyphen_word != original_word:
                    text = re.sub(original_word, hyphen_word, text)

            previous_index += match.start()
            match = regex.search(text[previous_index:])

        return text

    def plural_to_singular(self, text: str,
                           lemmatizer: Callable[[
                               str], str] = _grammar.singular_noun,
                           extra_lemmas_dict: Dict[str, str] = {}) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nConverting from plural to singular:')

        tokens = word_tokenize(text)

        def _lemmatize_fn(word: str) -> str:
            if word[-3:] == 'ous': # dangerous, heterogeneous, various
                return word

            return lemmatizer(word)

        words_in_singular = [_lemmatize_fn(w) if w not in extra_lemmas_dict else extra_lemmas_dict[w] for w in tokens]

        # do this because how inflect library works (returns singular of the word or False)
        words_in_singular = [w if not isinstance(w, bool) else tokens[i] for i, w in enumerate(words_in_singular)]

        if self._debug:
            coloured_tokens = [
                w if (lemmatizer(w) == False) and (w not in extra_lemmas_dict) else BG_HIGHLIGHT_COLOR + w + Back.RESET for w in tokens]
            self._logger.debug(' '.join(coloured_tokens))

        return ' '.join(words_in_singular)

    def remove_accents(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving accents:')

        accents = {
            '[áàãâä]|´a|`a|\~a|\^a': 'a',
            'ç': 'c',
            '[éèêë]|´e|`e|\^e': 'e',
            '[íïì]|´i|`i': 'i',
            'ñ': 'n',
            '[óòôö]|´o|`o|\~o|\^o': 'o',
            '[úùü]|´u|`u': 'u',
            'ß': 'ss',
        }

        if self._debug:
            regex = '|'.join(accents.keys())
            self._pretty_print(re.compile(regex), text)

        for k, v in accents.items():
            text = re.sub(k, v, text)

        return text

    def remove_before_title(self, text: str, title: str, ignore_case=False) -> str:
        title_split = title.strip().split()
        # use first 3 words if it has
        if len(title_split) >= 3:
            first_words_of_title = ' '.join(title_split[:3])
        elif len(title_split) >= 2:
            first_words_of_title = ' '.join(title_split[:2])
        else:
            first_words_of_title = title_split[0]

        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving text before title:')

        if ignore_case:
            end = text.lower().find(first_words_of_title.lower())
        else:
            end = text.find(first_words_of_title)

        if end > 0:
            return self.remove_text_between_positions(text, to_position=end)
        else:
            return text

    def remove_between_title_abstract(self, text: str, title: str, end_word: str = 'abstract', ignore_case=False) -> str:
        title_split = title.strip().split()
        # use last 3 words if it has
        if len(title_split) >= 3:
            last_words_of_title = ' '.join(title_split[-3:])
        elif len(title_split) >= 2:
            last_words_of_title = ' '.join(title_split[-2:])
        else:
            last_words_of_title = title_split[0]

        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving text between title and abstract, '
            f'delimited by "{last_words_of_title}" and "{end_word}":')

        if ignore_case:
            lower_text = text.lower()
            last_words_of_title = last_words_of_title.lower()
            start = lower_text.find(last_words_of_title) + len(last_words_of_title)
            end = lower_text[start:].find(end_word) + start + len(end_word)
        else:
            start = text.find(last_words_of_title) + len(last_words_of_title)
            end = text[start:].find(end_word) + start + len(end_word)

        return self.remove_text_between_positions(text, start, end)

    def remove_bib_info(self, text: str, phrases: Optional[List[str]] = None) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving bib info:')

        if phrases and len(phrases) > 0:
            phrases += self._bib_info_to_remove
            phrases = [fr'\b{p}' for p in phrases]
            phrases = [fr'{p}\b' if not p.endswith('\)') else p for p in phrases]
            regex = '|'.join(phrases)
            regex = '[\s]+'.join(regex.split())
            regex = re.compile(regex)
        else:
            regex = self._regexes['remove_bib_info']

        return self._run_regex(text, regex, ' ')

    def remove_cid(self, text: str) -> str:
        # a PDF contains CMAPs which map character codes to glyph indices
        # so, a CID is a character id for the glyph it maps to, inside the CMAP table
        self._logger.debug(f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving cids:')
        regex = self._regexes['remove_cid']
        return self._run_regex(text, regex, '')

    def remove_composite_words(self, text: str, composite_words: Optional[List[str]] = None) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving composite words:')

        if composite_words and len(composite_words) > 0:
            composite_words += self._composite_words
            regex = '|'.join(composite_words)
            regex = '[\s]*[-_][\s]*'.join(regex.split('-'))
            regex = re.compile(regex)
        else:
            regex = self._regexes['remove_composite_words']
        return self._run_regex(text, regex, ' ')

    def remove_eg_ie_etal(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving e.g., i.e. and et al:')
        regex = self._regexes['remove_eg_ie_etal']
        return self._run_regex(text, regex, ' ')

    def remove_equations(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving equations:')
        regex = self._regexes['remove_equations']
        return self._run_regex(text, regex, '\n')

    def remove_emails(self, text: str) -> str:
        self._logger.debug(f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving emails:')
        regex = self._regexes['remove_emails']
        return self._run_regex(text, regex, ' ')

    def remove_first_word_occurrence(self, text: str, word: str, ignore_case=False) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving first occurrence of "{word}":')

        if ignore_case:
            index = text.lower().find(word.lower())
        else:
            index = text.find(word)

        if index >= 0:
            if self._logger.isEnabledFor(logging.DEBUG):
                tmp_text = text[:index] + BG_HIGHLIGHT_COLOR + word + Back.RESET + text[index + len(word):]
                self._logger.debug(tmp_text)

            return text[:index] + text[index + len(word):]
        else:
            self._logger.debug('Text kept the same')
            return text

    def remove_footnotes_numbers(self, text: str, words_with_footnotes: Optional[List[str]] = None) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving footnotes from words:')

        if words_with_footnotes and len(words_with_footnotes) > 0:
            words_with_footnotes += self._words_with_footnotes
            regex = '|'.join(words_with_footnotes)
            regex = f'\\b({regex})[0-9]\\b'
            regex = re.compile(regex)
        else:
            regex = self._regexes['remove_footnotes_numbers']

        return self._run_regex(text, regex, '\\1')

    def remove_from_word_to_end(self, text: str, from_word: str, ignore_case=False) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving from {from_word} to end of text:')

        match = re.search(f'[.\\n\s]+{from_word[::-1]}[.\s\\n\d]+', text[::-1], re.IGNORECASE if ignore_case else 0)
        if match != None:
            return self.remove_text_between_positions(text, -match.end()).strip()
        else:
            self._logger.debug(f'Could not find term {from_word}')
            return text.strip()

    def remove_html_tags(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving html tags:')
        regex = self._regexes['remove_html_tags']
        return self._run_regex(text, regex, ' ')

    def remove_hyphens_slashes(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving hyphens and slashes:')
        regex = self._regexes['remove_hyphens_slashes']
        return self._run_regex(text, regex, ' ')

    def remove_item_citation(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving papers, figures, tables, sections, and theorems citations:')
        regex = self._regexes['remove_item_citation']
        return self._run_regex(text, regex, ' ')

    def remove_last_word_occurrence(self, text: str, word: str, ignore_case=False) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving last occurrence of "{word}":')

        if ignore_case:
            index = text.lower().rfind(word.lower())
        else:
            index = text.rfind(word)

        if index >= 0:
            if self._logger.isEnabledFor(logging.DEBUG):
                tmp_text = text[:index] + BG_HIGHLIGHT_COLOR + word + Back.RESET + text[index + len(word):]
                self._logger.debug(tmp_text)

            return text[:index] + text[index + len(word):]
        else:
            self._logger.debug('Text kept the same')
            return text

    def remove_latex_commands(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving latex commands (started with \):')
        regex = self._regexes['remove_latex_commands']
        return self._run_regex(text, regex, ' ')

    def remove_latex_inline_equations(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving latex inline equations (surrounded by $):')
        regex = self._regexes['remove_latex_inline_equations']
        return self._run_regex(text, regex, ' ')

    def remove_latexit_tags(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving latexit html tags:')
        regex = self._regexes['remove_latexit_tags']
        return self._run_regex(text, regex, ' ')

    def remove_metrics(self, text: str, metrics: Optional[List[str]] = None) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving metrics numbers:')

        if metrics and len(metrics) > 0:
            metrics += self._metrics
            metrics_regex_str = '|'.join([f'[\s]?{w}' for w in metrics])
            regex = f'[\s\(\[\{{\∼\~][0-9]+([.,][0-9]+)?([\s]?[%kmb]+|{metrics_regex_str})?[.,:\s\)\]\}}]'
            regex = re.compile(regex)
        else:
            regex = self._regexes['remove_metrics']

        return self._run_regex(text, regex, ' ')

    def remove_numbers(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving numbers not related to dimensions:')
        regex = self._regexes['remove_numbers']
        return self._run_regex(text, regex, ' ')

    def remove_numbers_only_lines(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving lines filled only with numbers:')
        regex = self._regexes['remove_numbers_only_lines']
        return self._run_regex(text, regex, '\n')

    def remove_ordinal_numbers(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving ordinal numbers:')
        regex = self._regexes['remove_ordinal_numbers']
        return self._run_regex(text, regex, ' ')

    def remove_phrases(self, text: str, phrases: Optional[List[str]] = None) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving phrases:')

        if phrases and len(phrases) > 0:
            phrases += self._phrases_to_remove
            phrases = [fr'\b{p}' for p in phrases]
            phrases = [fr'{p}\b' if not p.endswith('\)') else p for p in phrases]
            regex = '|'.join(phrases)
            regex = '[\s]+'.join(regex.split())
            regex = re.compile(regex)
        else:
            regex = self._regexes['remove_phrases']

        return self._run_regex(text, regex, ' ')

    def remove_remaining_two_chars(self, text: str, two_chars: Optional[List[str]] = None) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving remaining two chars:')

        if two_chars and len(two_chars) > 0:
            two_chars += self._remaining_two_chars
            regex = '|'.join(two_chars)
            regex = '[\s]+'.join(regex.split())
            regex = re.compile(regex)
        else:
            regex = self._regexes['remove_remaining_two_chars']

        return self._run_regex(text, regex, ' ')

    def remove_repeated(self, text: str, step: int=1) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving repeated words in sequence (step = {step}):')
        splitted_text = text.strip().split()
        if len(splitted_text) > 0:
            if self._debug:
                splitted_text_debug = splitted_text[0:step]
                splitted_text_debug += [splitted_text[i] if splitted_text[i] != splitted_text[i-step]
                                        else f'{BG_HIGHLIGHT_COLOR}{splitted_text[i]}{Back.RESET}' for i in range(step, len(splitted_text))]
                self._logger.debug(' '.join(splitted_text_debug))

            new_splitted_text = splitted_text[0:step]
            new_splitted_text += [splitted_text[i] for i in range(
                step, len(splitted_text)) if splitted_text[i] != splitted_text[i-step]]
            return ' '.join(new_splitted_text)

        else:
            self._logger.debug('Text is empty')
            return text.strip()

    def remove_shapes(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving shapes or sizes:')
        regex = self._regexes['remove_shapes']
        return self._run_regex(text, regex, ' ')

    def remove_specific_expressions(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving some specific expressions (arxiv and pages):')
        regex = self._regexes['remove_specific_expressions']
        return self._run_regex(text, regex, ' ')

    def remove_stopwords(self, text: str, stopwords: Set[str] = set()) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving stop words and one character symbols:')
        tokens = word_tokenize(text)
        stopwords.update(self._stop_words)
        if self._debug:
            coloured_tokens = [
                w if not w in stopwords and len(w) > 1 else BG_HIGHLIGHT_COLOR + w + Back.RESET for w in tokens]
            self._logger.debug(' '.join(coloured_tokens))

        tokens_without_sw = [
            w for w in tokens if not w in stopwords and len(w) > 1]
        return ' '.join(tokens_without_sw)

    def remove_symbols(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving symbols, remaining one character symbols, and trailing spaces:')
        regex = self._regexes['remove_symbols']
        return self._run_regex(text, regex, ' ').strip()

    def remove_tabular_data(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving tabular data:')
        regex = self._regexes['remove_tabular_data']
        return self._run_regex(text, regex, '\n').strip()

    def remove_text_between_positions(self, text: str, from_position: int = 0, to_position: int = 0) -> str:
        if to_position == 0:
            to_position = len(text)

        self._logger.debug(
            f'\nRemoving text between positions "{from_position}" and "{to_position}":')

        if self._debug:
            if len(text) < 10000:
                self._logger.debug(
                    f'{text[:from_position]}{BG_HIGHLIGHT_COLOR + text[from_position:to_position] + Back.RESET}{text[to_position:]}')
            else:
                if len(text[:from_position]) < 3000:
                    text_before = text[:from_position]
                else:
                    text_before = f'...\n{text[from_position-500:from_position]}'

                if len(text[to_position:]) < 3000:
                    text_after = text[to_position:]
                else:
                    text_after = f'{text[to_position:to_position+500]}\n...'

                self._logger.debug(
                    f'{text_before}{BG_HIGHLIGHT_COLOR + text[from_position:to_position] + Back.RESET}{text_after}')

        return f'{text[:from_position]}{text[to_position:]}'

    def remove_text_between_words(self, text: str, from_word: str = '', to_word: str = '') -> str:
        if len(from_word) > 0:
            start = text.find(from_word)
        else:
            start = 0

        if len(to_word) > 0:
            if len(from_word) > 0:
                self._logger.debug(
                    f'\nRemoving text between "{from_word}" and "{to_word}":')
            else:
                self._logger.debug(f'\nRemoving text until "{to_word}":')

            end = text[start:].find(to_word) + start + len(to_word)
            if self._debug:
                self._logger.debug(
                    f'{text[:start]}{BG_HIGHLIGHT_COLOR + text[start:end] + Back.RESET}{text[end:]}')

            return f'{text[:start]}{text[end:]}'
        else:
            if len(from_word) > 0:
                self._logger.debug(
                    f'\nRemoving text from "{from_word}":')
            else:
                self._logger.warning(f'\nRemoving everything:')

            if self._debug:
                self._logger.debug(
                    f'{text[:start]}{BG_HIGHLIGHT_COLOR + text[start:] + Back.RESET}')

            return text[:start]

    def remove_urls(self, text: str) -> str:
        self._logger.debug(f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving urls:')
        regex = self._regexes['remove_urls']
        return self._run_regex(text, regex, ' ')

    def remove_word_by_prefix(self, text: str, prefixes: List[str]) -> str:
        regexes = [f'{pref}[\w.\-\−\–\/]*[\s.,\d\)]?' for pref in prefixes]
        regex = '|'.join(regexes)
        return self._run_regex(text, re.compile(regex), ' ')

    def replace_hyphens_by_underline(self, text: str) -> str:
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nReplacing hyphens by underlines:')
        maximum_hyphenized = 4  # e.g.: state-of-the-art
        regex = self._regexes['replace_hyphens_by_underline']
        if self._debug:
            self._pretty_print(
                re.compile(f'[\w]+-[\w]+(-[\w]+){{,{maximum_hyphenized-2}}}'), text)

        result = text
        for _ in range(1, maximum_hyphenized):
            result = regex.sub('\\1_\\2', result)

        return result

    def replace_symbol_by_letters(self, text: str, symbols_dict: Dict[str, str] = {}) -> str:
        symbols_dict.update(self._symbols_dict)
        self._logger.debug(
            f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nReplacing {symbols_dict.keys()} symbols by {symbols_dict.values()}:')
        if self._debug:
            regex = ''.join(symbols_dict.keys())
            regex = f'[{regex}]'
            self._pretty_print(re.compile(regex), text)

        for k, v in symbols_dict.items():
            text = re.sub(f'[{k}]', v, text)

        return text


def _clean_abstract(paper: pd.Series, stop_when='') -> None:
    _logger.info(f'\nTitle: \n{paper["title"]}')

    abstract = literal_eval(paper['abstract'])
    _logger.info(f'\nAbstract:\n{abstract}')

    text_cleaner = TextCleaner(debug=True)
    stop = len(stop_when) > 0

    abstract = ftfy.fix_text(abstract)
    abstract = abstract.lower()
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.aglutinate_urls(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_urls(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.replace_symbol_by_letters(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_eg_ie_etal(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_phrases(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_shapes(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_ordinal_numbers(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_metrics(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_numbers(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_accents(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_latex_commands(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_latex_inline_equations(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_html_tags(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_symbols(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    # do these 2 again after removing symbols
    abstract = text_cleaner.remove_metrics(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_numbers(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.aglutinate_words(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_phrases(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_footnotes_numbers(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_composite_words(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_hyphens_slashes(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    # do this again to remove cases like 23/30
    abstract = text_cleaner.remove_numbers(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.remove_stopwords(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.plural_to_singular(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return
    abstract = text_cleaner.replace_hyphens_by_underline(abstract)
    if stop and abstract.find(stop_when) >= 0:
        text_cleaner._pretty_print(stop_when, abstract)
        return

    _logger.info(f'\nFinal Abstract:\n{text_cleaner._highlight_not_recognized_words(abstract)}')

    _clean_title(paper)


def _clean_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    text_cleaner = TextCleaner()
    spell_checker = enchant.Dict("en_US")
    lemmatizer = lru_cache(maxsize=5_000)(_grammar.singular_noun)

    df.loc[:, 'abstract'] = df['abstract'].apply(literal_eval)
    df.loc[:, 'abstract'] = df['abstract'].apply(ftfy.fix_text)
    df.loc[:, 'abstract'] = df['abstract'].str.lower()
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.aglutinate_urls, spell_checker=spell_checker)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_urls)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.replace_symbol_by_letters)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_eg_ie_etal)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_phrases)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_shapes)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_ordinal_numbers)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_metrics)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_numbers)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_accents)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_latex_commands)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_latex_inline_equations)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_html_tags)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_symbols)
    # do these 2 again after removing symbols
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_metrics)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_numbers)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.aglutinate_words, spell_checker=spell_checker)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_phrases)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_footnotes_numbers)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_composite_words)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_hyphens_slashes)
    # do this again to remove cases like 23/30
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_numbers)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.remove_stopwords)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.plural_to_singular, lemmatizer=lemmatizer)
    df.loc[:, 'abstract'] = df['abstract'].apply(
        text_cleaner.replace_hyphens_by_underline)

    # _logger.info(f'lemmatizer cache info: {lemmatizer.cache_info()}')

    df = _clean_titles(df)

    return df


def _clean_paper(paper: pd.Series, stop_when='') -> None:
    text_cleaner = TextCleaner(debug=True)
    title = paper["title"]
    text = literal_eval(paper["paper"])
    stop = len(stop_when) > 0
    lemmatizer = lru_cache(maxsize=10_000)(_grammar.singular_noun)

    remove_first_introduction_occurrence = partial(text_cleaner.remove_first_word_occurrence, word='introduction')
    remove_last_conclusion_occurrence = partial(text_cleaner.remove_last_word_occurrence, word='conclusion')

    _logger.debug(
        f'\nText from PDF:\n{Fore.GREEN}###########################{Fore.RESET}\n{text}\n###########################')

    with Timer(name='Whole run'):
        text = ftfy.fix_text(text)
        text = text.lower()
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_from_word_to_end(
            text, 'references')
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_from_word_to_end(
            text, 'acknowledgment')
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_from_word_to_end(
            text, 'acknowledgement')
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_from_word_to_end(
            text, 'acknowledgments')
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_from_word_to_end(
            text, 'acknowledgements')
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_from_word_to_end(
            text, 'funding transparency statement')
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_from_word_to_end(
            text, 'credit authorship contribution statement')
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_from_word_to_end(
            text, 'declarations of competing interest')
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_before_title(text, title)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_between_title_abstract(text, title)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = remove_first_introduction_occurrence(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = remove_last_conclusion_occurrence(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.replace_symbol_by_letters(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_cid(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_equations(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_numbers_only_lines(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_tabular_data(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        # run these 3 again to remove consecutive lines
        text = text_cleaner.remove_equations(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_numbers_only_lines(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_tabular_data(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = ' '.join(text.split())
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.aglutinate_urls(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_urls(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_emails(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_eg_ie_etal(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_latexit_tags(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        with Timer(name='Removing bib info 1'):
            text = text_cleaner.remove_bib_info(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        with Timer(name='Removing phrases 1'):
            text = text_cleaner.remove_phrases(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_shapes(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_ordinal_numbers(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_specific_expressions(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_item_citation(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_metrics(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_numbers(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_accents(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_latex_commands(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_latex_inline_equations(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_html_tags(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_symbols(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        # do these 2 again after removing symbols
        text = text_cleaner.remove_metrics(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_numbers(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.aglutinate_words(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        with Timer(name='Removing bib info 2'):
            text = text_cleaner.remove_bib_info(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        with Timer(name='Removing phrases 2'):
            text = text_cleaner.remove_phrases(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_footnotes_numbers(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_composite_words(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_hyphens_slashes(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        # do this again to remove cases like 23/30
        text = text_cleaner.remove_numbers(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_stopwords(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        with Timer(name='Plural to singular'):
            text = text_cleaner.plural_to_singular(text, lemmatizer=lemmatizer)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.replace_hyphens_by_underline(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_repeated(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = text_cleaner.remove_repeated(text, step=2)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        # check if last word is number
        text_split = text.split()
        if text_split[-1] != re.sub('[0-9]+([.,][0-9]+)*', '', text_split[-1]):
            _logger.info(
                f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nRemoving last word {text_split[-1]}'
                f' since it is a number')
            text = ' '.join(text_split[:-1])
        text = text_cleaner.remove_remaining_two_chars(text)
        if stop and text.find(stop_when) >= 0:
            text_cleaner._pretty_print(stop_when, text)
            return
        text = ' '.join(text.split())

    _logger.info(f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nTitle: \n{title}')
    _logger.info(
        f'\n{Fore.GREEN}###########################{Fore.RESET}\n\nFinal text (words not recognized in blue):\n{text_cleaner._highlight_not_recognized_words(text)}')

    # _logger.info(f'lemmatizer cache info: {lemmatizer.cache_info()}')


def _clean_papers(df: pd.DataFrame, show_progress: bool=False) -> pd.DataFrame:
    text_cleaner = TextCleaner()
    spell_checker = enchant.Dict("en_US")
    lemmatizer = lru_cache(maxsize=15_000)(_grammar.singular_noun)

    remove_first_introduction_occurrence = partial(text_cleaner.remove_first_word_occurrence, word='introduction')
    remove_last_conclusion_occurrence = partial(text_cleaner.remove_last_word_occurrence, word='conclusion')

    # drop papers that are not usable
    total_papers = len(df)
    min_title_len = 4
    df_not_nan = df[df['title'].notna() & df['paper'].notna()]
    new_df = df_not_nan[df_not_nan['title'].map(len) > min_title_len]
    new_total_papers = len(new_df)

    if total_papers - new_total_papers > 0:
        _logger.debug(f'Dropped {total_papers - new_total_papers} papers')
        df = new_df

    pbar_desc_len = 25
    def update_pbar(pbar, text):
        pbar.set_description(text.ljust(pbar_desc_len))
        pbar.update()

    with tqdm(total=53, disable=not show_progress, unit='step', ncols=250) as pbar:
        df.loc[:, 'paper'] = df['paper'].apply(literal_eval)
        update_pbar(pbar, 'Reading papers')
        df.loc[:, 'paper'] = df['paper'].apply(ftfy.fix_text)
        update_pbar(pbar, 'Fixing unicode')
        df.loc[:, 'paper'] = df['paper'].str.lower()
        update_pbar(pbar, 'Lowering case')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_from_word_to_end, from_word='references')
        update_pbar(pbar, 'Removing references')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_from_word_to_end, from_word='acknowledgment')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_from_word_to_end, from_word='acknowledgement')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_from_word_to_end, from_word='acknowledgments')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_from_word_to_end, from_word='acknowledgements')
        update_pbar(pbar, 'Removing acknowledgments')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_from_word_to_end, from_word='funding transparency statement')
        update_pbar(pbar, 'Removing funding')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_from_word_to_end, from_word='credit authorship contribution statement')
        update_pbar(pbar, 'Removing credit authorship')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_from_word_to_end, from_word='declarations of competing interest')
        update_pbar(pbar, 'Removing competing interest')
        df.loc[:, 'paper'] = df.apply(
            _remove_before_title, axis=1, text_cleaner=text_cleaner)
        update_pbar(pbar, 'Removing header')
        df.loc[:, 'paper'] = df.apply(
            _remove_between_title_abstract, axis=1, text_cleaner=text_cleaner)
        update_pbar(pbar, 'Removing authors')
        df.loc[:, 'paper'] = df['paper'].apply(remove_first_introduction_occurrence)
        update_pbar(pbar, 'Removing introduction title')
        df.loc[:, 'paper'] = df['paper'].apply(remove_last_conclusion_occurrence)
        update_pbar(pbar, 'Removing conclusion title')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.replace_symbol_by_letters)
        update_pbar(pbar, 'Replacing symbols')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_cid)
        update_pbar(pbar, 'Removing cid')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_equations)
        update_pbar(pbar, 'Removing equations')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_numbers_only_lines)
        update_pbar(pbar, 'Removing numeric lines')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_tabular_data)
        update_pbar(pbar, 'Removing tables')
        # run these 3 again to remove consecutive lines
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_equations)
        update_pbar(pbar, 'Removing equations')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_numbers_only_lines)
        update_pbar(pbar, 'Removing numeric lines')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_tabular_data)
        update_pbar(pbar, 'Removing tables')
        df.loc[:, 'paper'] = df['paper'].str.split().str.join(' ')
        update_pbar(pbar, 'Removing line breaks')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.aglutinate_urls)
        update_pbar(pbar, 'Aglutinating urls')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_urls)
        update_pbar(pbar, 'Removing urls')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_emails)
        update_pbar(pbar, 'Removing emails')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_eg_ie_etal)
        update_pbar(pbar, 'Removing e.g., i.e.')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_latexit_tags)
        update_pbar(pbar, 'Removing latex tags')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_bib_info)
        update_pbar(pbar, 'Removing bib')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_phrases)
        update_pbar(pbar, 'Removing phrases')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_shapes)
        update_pbar(pbar, 'Removing shapes')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_ordinal_numbers)
        update_pbar(pbar, 'Removing ordinals')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_specific_expressions)
        update_pbar(pbar, 'Removing expressions')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_item_citation)
        update_pbar(pbar, 'Removing item citations')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_metrics)
        update_pbar(pbar, 'Removing metrics')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_numbers)
        update_pbar(pbar, 'Removing numbers')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_accents)
        update_pbar(pbar, 'Removing accents')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_latex_commands)
        update_pbar(pbar, 'Removing latex commands')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_latex_inline_equations)
        update_pbar(pbar, 'Removing latex equations')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_html_tags)
        update_pbar(pbar, 'Removing HTML tags')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_symbols)
        update_pbar(pbar, 'Removing symbols')
        # do these 2 again after removing symbols
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_metrics)
        update_pbar(pbar, 'Removing metrics')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_numbers)
        update_pbar(pbar, 'Removing numbers')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.aglutinate_words, spell_checker=spell_checker)
        update_pbar(pbar, 'Aglutinating words')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_bib_info)
        update_pbar(pbar, 'Removing bib')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_phrases)
        update_pbar(pbar, 'Removing phrases')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_footnotes_numbers)
        update_pbar(pbar, 'Removing footnotes')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_composite_words)
        update_pbar(pbar, 'Removing composite words')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_hyphens_slashes)
        update_pbar(pbar, 'Removing hyphens')
        # do this again to remove cases like 23/30
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_numbers)
        update_pbar(pbar, 'Removing numbers')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_stopwords)
        update_pbar(pbar, 'Removing stopwords')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.plural_to_singular, lemmatizer=lemmatizer)
        update_pbar(pbar, 'Converting to singular')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.replace_hyphens_by_underline)
        update_pbar(pbar, 'Replacing hyphens')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_repeated)
        update_pbar(pbar, 'Removing repeated words')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_repeated, step=2)
        update_pbar(pbar, 'Removing repeated words')
        df.loc[:, 'paper'] = df['paper'].apply(
            _remove_last_word_if_number)
        update_pbar(pbar, 'Removing last word')
        df.loc[:, 'paper'] = df['paper'].apply(
            text_cleaner.remove_remaining_two_chars)
        update_pbar(pbar, 'Removing two chars words')
        df.loc[:, 'paper'] = df['paper'].str.split().str.join(' ')
        update_pbar(pbar, 'Tyding up')

    total_papers = len(df)
    # drop papers that are not usable
    min_paper_len = 10
    new_df = df[df['paper'].map(len) > min_paper_len]
    new_total_papers = len(new_df)

    if total_papers - new_total_papers > 0:
        df = new_df
        _logger.info(f'Dropped {total_papers - new_total_papers} papers')

    # _logger.info(f'lemmatizer cache info: {lemmatizer.cache_info()}')

    return df


def _clean_title(paper: pd.Series) -> str:
    _logger.info(f'\nTitle: \n{paper["title"]}')

    text_cleaner = TextCleaner(debug=True)
    clean_title = paper['title'].lower()
    clean_title = ftfy.fix_text(clean_title)
    clean_title = text_cleaner.remove_accents(clean_title)
    clean_title = text_cleaner.remove_latex_commands(clean_title)
    clean_title = text_cleaner.remove_latex_inline_equations(clean_title)
    clean_title = clean_title.replace('\\', '')
    clean_title = text_cleaner.remove_symbols(clean_title)
    clean_title = clean_title.replace('--', '-')
    clean_title = clean_title.replace('–', '-')
    clean_title = clean_title.replace('−', '-')
    clean_title = ' '.join(clean_title.strip().split())
    clean_title = text_cleaner.remove_hyphens_slashes(clean_title)
    clean_title = text_cleaner.remove_stopwords(clean_title)
    clean_title = text_cleaner.plural_to_singular(clean_title)
    clean_title = text_cleaner.replace_hyphens_by_underline(clean_title)

    _logger.info(f'\nClean title:\n{text_cleaner._highlight_not_recognized_words(clean_title)}')


def _clean_titles(df: pd.DataFrame, progress=False) -> pd.DataFrame:
    text_cleaner = TextCleaner()

    pbar_desc_len = 30
    def update_pbar(pbar, text):
        pbar.set_description(text.ljust(pbar_desc_len))
        pbar.update()

    with tqdm(total=15, disable=not progress, unit='step') as pbar:
        df['clean_title'] = df['title'].str.lower()
        update_pbar(pbar, 'Lowering case')

        df.loc[:, 'clean_title'] = df['clean_title'].apply(ftfy.fix_text)
        update_pbar(pbar, 'Fix unicode')

        df.loc[:, 'clean_title'] = df['clean_title'].apply(text_cleaner.remove_accents)
        update_pbar(pbar, 'Remove accents')

        df.loc[:, 'clean_title'] = df['clean_title'].apply(text_cleaner.remove_latex_commands)
        update_pbar(pbar, 'Remove latex commands')

        df.loc[:, 'clean_title'] = df['clean_title'].apply(text_cleaner.remove_latex_inline_equations)
        update_pbar(pbar, 'Removing latex inline equations')

        df.loc[:, 'clean_title'] = df['clean_title'].str.replace(re.compile(r'\\'), '')
        update_pbar(pbar, 'Replacing backslash')

        df.loc[:, 'clean_title'] = df['clean_title'].apply(text_cleaner.remove_symbols)
        update_pbar(pbar, 'Replacing symbols')

        df.loc[:, 'clean_title'] = df['clean_title'].str.replace(re.compile(r'--'), '-')
        update_pbar(pbar, 'Replacing double hyphen')

        df.loc[:, 'clean_title'] = df['clean_title'].str.replace(re.compile(r'–'), '-')
        update_pbar(pbar, 'Replacing hyphen')

        df.loc[:, 'clean_title'] = df['clean_title'].str.replace(re.compile(r'−'), '-')
        update_pbar(pbar, 'Replacing hyphen')

        df.loc[:, 'clean_title'] = df['clean_title'].str.strip().str.split().str.join(' ')
        update_pbar(pbar, 'Removing trailing spaces')

        df.loc[:, 'clean_title'] = df['clean_title'].apply(text_cleaner.remove_hyphens_slashes)
        update_pbar(pbar, 'Removing hyphens and slashes')

        df.loc[:, 'clean_title'] = df['clean_title'].apply(text_cleaner.remove_stopwords)
        update_pbar(pbar, 'Removing stopwords')

        df.loc[:, 'clean_title'] = df['clean_title'].apply(text_cleaner.plural_to_singular)
        update_pbar(pbar, 'Converting to singular')

        df.loc[:, 'clean_title'] = df['clean_title'].apply(text_cleaner.replace_hyphens_by_underline)
        update_pbar(pbar, 'Replacing hyphens by underline')

    return df


def _remove_before_title(row: pd.Series, text_cleaner: TextCleaner) -> str:
    return text_cleaner.remove_before_title(row['paper'], row['title'])


def _remove_between_title_abstract(row: pd.Series, text_cleaner: TextCleaner) -> str:
    return text_cleaner.remove_between_title_abstract(row['paper'], row['title'])


def _remove_last_word_if_number(text: str) -> str:
    if len(text) > 0:
        text_split = text.split()
        if text_split[-1] != re.sub('[0-9]+([.,][0-9]+)*', '', text_split[-1]):
            text = ' '.join(text_split[:-1])
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Clean papers info.")
    parser.add_argument('-a', '--abstract_only', action='store_true',
                        help='work only with the abstracts')
    parser.add_argument('-d', '--dropped_papers', action='store_true',
                        help='only shows the dropped papers')
    parser.add_argument('-f', '--file', type=str, default='',
                        help='file name with paper info')
    parser.add_argument('-i', '--index', type=int, default=-1,
                        help='index of paper to debug')
    parser.add_argument('-l', '--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'),
                        help='log level to debug')
    parser.add_argument('-p', '--n_processes', type=int, default=3*multiprocessing.cpu_count()//4,
                        help='number of subprocesses to spawn')
    parser.add_argument('-s', '--separator', type=str, default='|',
                        help='csv separator')
    parser.add_argument('-t', '--title', type=str, default='',
                        help='title of paper to debug')
    parser.add_argument('--stop_when', type=str, default='',
                        help='stop when given term is generated, to ease debugging process')
    args = parser.parse_args()

    logdir = Path('logs/').expanduser()
    logdir.mkdir(exist_ok=True)
    setup_log(args.log_level, logdir / 'text_cleaner.log')

    if len(args.separator) == 1:
        df = pd.read_csv(args.file, sep=args.separator,
                         dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(args.file, sep=args.separator,
                         dtype=str, engine='python', keep_default_na=False)

    if args.index != -1:
        paper = df.iloc[args.index]
        specific_paper = True
    elif len(args.title) > 0:
        file_to_find = args.title.lower()
        found_papers = df.loc[df['title'].str.lower().str.find(file_to_find) >= 0]
        if len(found_papers) == 0:
            _logger.error(
                f"Couldn't find any paper with '{args.title}' in title")
            exit(0)
        elif len(found_papers) > 1:
            _logger.info(
                f'Found {len(found_papers)} papers with "{args.title}" in title. Using the first one')

        paper = found_papers.iloc[0]
        specific_paper = True

    else:
        paper = None
        specific_paper = False

    if specific_paper:
        if args.abstract_only:
            _clean_abstract(paper, args.stop_when)
        else:
            _clean_paper(paper, args.stop_when)

    else:
        min_papers_per_process = 3

        if args.abstract_only:
            if args.n_processes == 1 or len(df) < args.n_processes * min_papers_per_process:
                new_df = _clean_abstracts(df)
            else:
                new_df = parallelize_dataframe(
                    df, _clean_abstracts, args.n_processes)

            new_file_name = Path(args.file).name
            new_file_name = new_file_name.split('.')
            new_file_name = '.'.join(
                new_file_name[:-1]) + '_clean.' + new_file_name[-1]
            _logger.info(f'Saving DataFrame to {new_file_name}')
            new_df.to_csv(Path(args.file).parent / new_file_name, sep='|', index=False)
            assert len(df) == len(new_df), f'DataFrame size changed after cleaning: {len(df)} -> {len(new_df)}'

        else:
            if args.n_processes == 1 or len(df) < args.n_processes * min_papers_per_process:
                _logger.info(f'n_papers ({len(df)}) < n_processes * {min_papers_per_process} ({args.n_processes} * {min_papers_per_process} = {args.n_processes * min_papers_per_process})')
                _logger.info(f'Running in a single process')
                new_df = _clean_papers(df, show_progress=True)
            else:
                _logger.info(f'n_papers ({len(df)}) >= n_processes * {min_papers_per_process} ({args.n_processes} * {min_papers_per_process} = {args.n_processes * min_papers_per_process})')
                _logger.info(f'Parallelizing to {args.n_processes} processes')
                new_df = parallelize_dataframe(
                    df, _clean_papers, args.n_processes)

            new_file_name = Path(args.file).name
            new_file_name = new_file_name.split('.')
            new_file_name = '.'.join(
                new_file_name[:-1]) + '_clean.' + new_file_name[-1]
            _logger.info(f'Saving DataFrame to {new_file_name}')
            new_df.to_csv(Path(args.file).parent / new_file_name, sep='|', index=False)
