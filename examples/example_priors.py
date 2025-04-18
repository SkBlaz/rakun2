## Idea augment output with domain-specific priors (symbolic)

from rakun2 import RakunKeyphraseDetector

EXAMPLE_DOCUMENT = "Kangaroos bounce gracefully across the vast, dusty plains of the Australian outback, their powerful hind legs propelling them effortlessly through the heat waves shimmering in the distance. A curious joey peeks shyly from its mother's pouch, observing a world filled with towering eucalyptus trees and rust-colored earth. At twilight, mobs of kangaroos gather by cool watering holes, pausing occasionally to listen for the rustle of predators hidden in the brush. They nibble cautiously on grass and shrubs, ears swiveling alertly at every unexpected sound. As dawn arrives, a young kangaroo boldly ventures off, testing his strength in playful leaps that send dust spiraling into the morning breeze."

priors=[('kangaroo', 0.11618894338607788), ('marsupials', 0.10609937831759453), ('species', 0.04895966034382582), ('australian', 0.03770353738218546), ('pouch', 0.025573878083378077), ('placental mammals', 0.025146500440314412), ('development', 0.019251500139944255), ('metatherians', 0.01630012784153223), ('males', 0.0161470053717494), ('marsupialia', 0.015857341699302197)]


hyperparameters = {"num_keywords": 10,
                   "merge_threshold": 1.0,
                   "alpha": 0.9,
                   "token_prune_len": 1}

keyword_detector = RakunKeyphraseDetector(hyperparameters)
out_keywords = keyword_detector.find_keywords(EXAMPLE_DOCUMENT, input_type="string")
print(out_keywords)

keyword_detector = RakunKeyphraseDetector(hyperparameters)
out_keywords = keyword_detector.find_keywords(EXAMPLE_DOCUMENT, input_type="string", prior_rankings=priors)
print(out_keywords)
