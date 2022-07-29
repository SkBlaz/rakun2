example_document = """
Britain fought alongside France, Russia and (after 1917) the United States, against Germany and its allies in the First World War (1914–1918).[119] British armed forces were engaged across much of the British Empire and in several regions of Europe, particularly on the Western front.[120] The high fatalities of trench warfare caused the loss of much of a generation of men, with lasting social effects in the nation and a great disruption in the social order.
"""
hyperparameters = {"num_keywords": 10,
                   "merge_threshold": 1.1,
                   "alpha": 0.3,
                   "token_prune_len": 3}
keyword_detector = RakunDetector(hyperparameters)
out_keywords = keyword_detector.find_keywords(example_document, input_type="string")
print(out_keywords)
