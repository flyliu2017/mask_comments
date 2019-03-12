#!/usr/bin/env bash

python processor.py compare --preds_path pred_5_80.txt \
                            --labels_path test_labels_5_80.txt \
                            --corpus_path test_corpus_5_80.txt