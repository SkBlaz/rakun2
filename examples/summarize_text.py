import numpy as np, nltk, argparse
from rakun2 import RakunKeyphraseDetector

def get_sentences(t):
    t = "\n".join(x for x in t.split("\n") if x.count(".") > 1).replace("\r\n"," ")
    s = [x.strip() for x in nltk.sent_tokenize(t.replace(";", "."))]
    return s[:3] if len(s) < 4 else s

def rank_sentences(t, a="max", sl=500):
    ss = [x.replace("et al.","et al") for x in get_sentences(t)[:sl] if x[:1] != "["]
    kd = RakunKeyphraseDetector({"num_keywords":30,"merge_threshold":0.5,"alpha":0.2,"token_prune_len":3},True)
    kw = kd.find_keywords(t,"string")
    fn = lambda x: {"max":max,"mean":np.mean,"median":np.median}[a](x)
    r,u = [],set()
    for s in ss:
        if "\n" in s:
            continue
        sc = [k[1] for k in kw if k[0] in s]
        if any(s.split()[:3] == x.split()[:3] or s[:len(s)//2] == x[:len(x)//2] for x in u) or s in u:
            continue
        u.add(s)
        if sc:
            r.append((fn(sc),s))
    im = {v:i for i,v in enumerate(r)}
    return sorted(r,key=lambda x:x[0],reverse=True),im,kw

def pretty_print(r, i, k=3, rt=True):
    t = r[:k]
    o = np.argsort([i[x] for x in t])
    x = np.random.randint(6,9)
    s = " ".join((t[j][1] + "<br>" if n % x == 0 else t[j][1]) for n,j in enumerate(o))
    if rt:
        print("In summary:", s)
    else:
        return s

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fname","-f",default="dump2.txt")
    a = p.parse_args()
    d = open(a.fname).read()
    for m in ["max","mean"]:
        rr, ii, kk = rank_sentences(d,m)
        pretty_print(rr,ii)
