#!/bin/sh
wget http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
python -m gensim.scripts.make_wiki enwiki-latest-pages-articles.xml.bz2 wiki_en_output
