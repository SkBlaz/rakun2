import numpy as np
import nltk
from rakun2 import RakunKeyphraseDetector


def get_sentences(text):
    lines = text.split("\n")
    text = "\n".join([x for x in lines if x.count(".") > 1])
    text = text.replace("\r\n", " ")
    sentences = [s.strip() for s in nltk.sent_tokenize(text)]
    return sentences[:3] if len(sentences) < 4 else sentences


def rank_sentences(text, aggregation="max", sentence_lim=500):
    sentences = get_sentences(text)[:sentence_lim]
    sentences = [
        x.replace("et al.", "et al") for x in sentences
        if not x.startswith("[")
    ]
    keyword_detector = RakunKeyphraseDetector(
        {
            "num_keywords": 30,
            "merge_threshold": 0.5,
            "alpha": 0.2,
            "token_prune_len": 3
        },
        verbose=True)
    keywords = keyword_detector.find_keywords(text, input_type="string")
    agg_map = {"max": max, "mean": np.mean, "median": np.median}
    agg_fn = agg_map.get(aggregation, np.mean)
    srank, used = [], set()
    for s in sentences:
        if "\n" in s:
            continue
        scores = [kw[1] for kw in keywords if kw[0] in s]
        skip = False
        for u in used:
            if s.split()[:3] == u.split()[:3] or s[:len(s) //
                                                   2] == u[:len(u) // 2]:
                skip = True
                break
        if skip or s in used:
            continue
        used.add(s)
        if scores:
            srank.append((agg_fn(scores), s))
    idx_map = {v: i for i, v in enumerate(srank)}
    return sorted(srank, key=lambda x: x[0], reverse=True), idx_map, keywords


def pretty_print(results, indices, top_k=3, return_text=True):
    top = results[:top_k]
    idxs = [indices[x] for x in top]
    order = np.argsort(idxs)
    r = np.random.randint(6, 9)
    parts = [
        top[k][1] + "<br>" if i % r == 0 else top[k][1]
        for i, k in enumerate(order)
    ]
    o = " ".join(parts)
    if return_text:
        print("In summary:", o)
    else:
        return o


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", default="dump2.txt")
    args = parser.parse_args()
    with open(args.fname, "r") as f:
        text = f.read()
    r1, i1, k1 = rank_sentences(text, "max")
    pretty_print(r1, i1)
    r2, i2, k2 = rank_sentences(text, "mean")
    pretty_print(r2, i2)
