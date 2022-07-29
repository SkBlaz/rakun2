# RaKUn 2.0
This is the repository containing the implementation of RaKUn 2.0, a very fast keyphrase extraction algorithm suitable for large-scale keyphrase detection.

# Installation and setup
The tool is distributed as a simple-to-use Python library. Simply

```python
pip install rakun2
```
and you should be good to go.

# Examples
A simple self-contained example follows

```python

from rakun2 import RakunDetector

example_document = """
Britain fought alongside France, Russia and (after 1917) the United States, against Germany and its allies in the First World War (1914–1918).[119] British armed forces were engaged across much of the British Empire and in several regions of Europe, particularly on the Western front.[120] The high fatalities of trench warfare caused the loss of much of a generation of men, with lasting social effects in the nation and a great disruption in the social order.
"""
hyperparameters = {"num_keywords": 10,
                   "merge_threshold": 1.3,
                   "alpha": 0.3,
                   "token_prune_len": 3}
keyword_detector = RakunDetector(hyperparameters)
out_keywords = keyword_detector.find_keywords(example_document, input_type="string")
print(out_keywords)

# Visualize the generated network (after detection call)
keyword_detector.visualize_network()

```
yielding output of form

```python
[['social effects', 0.27676000055526456], ['lasting social', 0.27674828905427873], ['warfare caused', 0.2767102868585031], ['trench warfare', 0.2765782309497358], ['generation', 0.19755587924981133], ['fatalities', 0.19587668920788176], ['disruption', 0.19459785451097245], ['armed forces', 0.17181056678829099], ['regions', 0.13819025211059133], ['engaged', 0.1349471915548533], ['allies', 0.11675871270658346]]
```

# Hyperparameters
The main hyperparameter which should be considered "per usecase" is `merge_threshold`, others are also documented below:

| Hyperparameter  | Range           | Description                                                                   |
|-----------------|-----------------|-------------------------------------------------------------------------------|
| num_keywords    | int             | Number of keywords to be returned.                                            |
| merge_threshold | float ([0,inf]) | A parameter determining when to merge nodes (higher = more merged nodes).     |
| alpha           | float ([0,1])   | The traversal parameter (PageRank's Alpha)                                    |
| token_prune_len | int             | Lower length bound below which tokens are discarded during graph construction |