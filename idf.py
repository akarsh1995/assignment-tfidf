from numpy import dot
from numpy.linalg import norm

import numpy as np
import re
from collections import Counter
from math import log

with open('harry.txt') as harry_potter:
    harry_potter_text =  harry_potter.read()

diff_chapters = harry_potter_text.split('\n\n\n\n')
#making 10 documets
diff_chapters = [chap for chap in diff_chapters if len(chap) > 100]

# replace
# punctuations
# extra spaces

# punctuations  = "( ! \"#$%&')*+,-./ )"
# punc_list = list(punctuations)

sample_text = '''
( ! \"#$%&')*+,-./ ), Hello world_
'''
def clear_text(text_=''):
    # clear punctuations
    cleaned_text = re.sub(r'[()\[\]\"#$%&\'*+,-\./!_’]', ' ', text_)
    # clear numbers
    cleaned_text = re.sub(r'[0-9]', ' ', cleaned_text)
    # new line chars
    cleaned_text = re.sub(r'[\n\r\t]', ' ' , cleaned_text)
    # spaces removal
    cleaned_text = re.sub(r'[\s]+', ' ' , cleaned_text)
    # clean some extra text
    cleaned_text = re.sub(r'”', ' ' , cleaned_text)
    # lower
    cleaned_text = cleaned_text.lower()

    return cleaned_text


# harry_cleaned_text = clear_text(harry_potter_text)
# word_counts = Counter(harry_cleaned_text.split(' '))

chapters_cleared_text = list(map(clear_text, diff_chapters))

chapter_word_counter = list(map(lambda chapter: Counter(chapter.split(' ')),
                           chapters_cleared_text))

# IDF(t) = log_e(Total number of documents /
# Number of documents with term t in it).

total_number_of_docs = len(diff_chapters)
# pick the term one by one
master_counter = Counter()

for chapter_counted_words in chapter_word_counter:
    for word, word_count in chapter_counted_words.items():
        master_counter[word] += word_count

from collections import defaultdict

tf_dict = defaultdict(dict)

for chap_id, chapter_word_count in enumerate( chapter_word_counter ):
    total_words = sum(map(len, chapters_cleared_text[chap_id].split(' ')))
    for word in chapter_word_count:
        tf_dict[chap_id][word] = chapter_word_count[word]/total_words

idf_dict = {}

for master_word in master_counter:
    total_count_of_master_word_in_diff_docs = 0
    for chapter in chapters_cleared_text:
        if master_word in chapter:
            total_count_of_master_word_in_diff_docs += 1
    if total_count_of_master_word_in_diff_docs:
        idf_dict[master_word] = log(
            total_number_of_docs/total_count_of_master_word_in_diff_docs
        )

def similarity_score(chap_id_1=0, chap_id_2=1):
    sims_1 = {}
    for word, tf in tf_dict[chap_id_1].items():
         sims_1[word] = tf * idf_dict[word]
    sims_2 = {}
    for word_, tf_ in tf_dict[chap_id_2].items():
         sims_2[word_] = tf_ * idf_dict[word_]
    common_words = set(sims_1).intersection(sims_2)
    vector1 = np.array([sims_1[common_word] for common_word in common_words])
    vector2 = np.array([sims_2[common_word] for common_word in common_words])
    cos_sim = dot(vector1, vector2)/(norm(vector1)*norm(vector2))
    return cos_sim

# observation: as the chapter id far apart similarity decreases
# >>> >>> >>> similarity_score(0, 2)
# 0.822422176072738
# >>> similarity_score(0, 5)
# 0.8270845035074089
# >>> similarity_score(0, 1)
# 0.8997050238576185
# >>> similarity_score(0, 9)
# 0.7781123173610666
