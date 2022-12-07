""" Main RaKUn 2.0 algorithm - DS paper 2022 """

from typing import Dict, Any, Tuple, List
from collections import Counter
import operator
import gzip
from operator import itemgetter
import json
import pkgutil
import logging
import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import fitz

logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger(__name__).setLevel(logging.INFO)


class RakunKeyphraseDetector:
    """
    The main RaKUn2.0 class
    """

    def __init__(self,
                 hyperparameters: Dict[str, Any] = {},
                 verbose: bool = True):

        self.verbose = verbose
        self.tokens = None
        self.sorted_terms_tf = None
        self.document = None
        self.full_tokens = None
        self.bigram_counts = None
        self.final_keywords = None
        self.node_ranks = {}
        self.main_graph = nx.Graph()
        self.term_counts = None
        self.space_factor = 0.5

        if self.verbose:
            logging.info("Initiated a keyword detector instance.")

        self.pattern = re.compile(r"(?u)\b\w\w+\b")
        self.hyperparameters = hyperparameters

        if "token_prune_len" not in self.hyperparameters:
            self.hyperparameters["token_prune_len"] = 1

        if "num_keywords" not in self.hyperparameters:
            self.hyperparameters["num_keywords"] = 10

        if "alpha" not in self.hyperparameters:
            self.hyperparameters["alpha"] = 0.1

        if "max_iter" not in self.hyperparameters:
            self.hyperparameters["max_iter"] = 100

        if "merge_threshold" not in self.hyperparameters:
            self.hyperparameters["merge_threshold"] = 0.5

        if "deduplication" not in self.hyperparameters:
            self.hyperparameters["deduplication"] = True

        if "stopwords" not in self.hyperparameters:

            # A collection of stopwords as default
            stopwords = pkgutil.get_data(__name__, "stopwords.json.gz")
            stopwords = gzip.decompress(stopwords)
            stopwords_generic = set(json.loads(stopwords.decode()))
            self.hyperparameters["stopwords"] = stopwords_generic

    def visualize_network(
        self,
        labels: bool = False,
        node_size: float = 0.1,
        alpha: float = 0.01,
        link_width: float = 0.01,
        font_size: int = 3,
        arrowsize: int = 1,
    ):
        """
        A method aimed to visualize a given token network.
        """

        if self.verbose:
            logging.info("Visualizing network")
        plt.figure(1, figsize=(10, 10), dpi=300)
        pos = nx.spring_layout(self.main_graph, iterations=10)

        node_colors = [x[1] * 1000 for x in self.node_ranks.items()]
        sorted_top_k = np.argsort([x[1] for x in self.node_ranks.items()
                                   ])[::-1][:20]

        final_colors, final_sizes = [], []
        for enx in range(len(node_colors)):
            if enx in sorted_top_k:
                final_colors.append("red")
                final_sizes.append(node_size * 10)
            else:
                final_colors.append("gray")
                final_sizes.append(node_size)

        nx.draw_networkx_nodes(
            self.main_graph,
            pos,
            node_size=final_sizes,
            node_color=final_colors,
            alpha=alpha,
        )
        nx.draw_networkx_edges(self.main_graph,
                               pos,
                               width=link_width,
                               arrowsize=arrowsize)

        if labels:
            nx.draw_networkx_labels(self.main_graph,
                                    pos,
                                    font_size=font_size,
                                    font_color="red")

        plt.tight_layout()
        plt.show()

    def compute_tf_scores(self, document: str = None) -> None:
        """ Compute TF scores """

        if document is not None:
            self.tokens = self.pattern.findall(document)

        term_counter: Any = Counter()
        for term in self.tokens:
            term_counter.update({term: 1})
        self.term_counts = dict(term_counter)
        self.sorted_terms_tf = sorted(term_counter.items(),
                                      key=itemgetter(1),
                                      reverse=True)

    def pagerank_scipy_adapted(self,
                               token_graph: nx.Graph,
                               alpha: float = 0.85,
                               personalization: np.array = None,
                               max_iter: int = 64,
                               tol: float = 1.0e-2,
                               weight: str = "weight"):
        """
        Adapted from NetworkX's nx.pagerank; we know how token graphs look like
        hence can omit some intermediary processing to make it a bit faster.
        The convergence criterion could also be adapted.
        """

        num_nodes = len(token_graph)
        if num_nodes == 0:
            return {}

        nodelist = list(token_graph)
        token_sparse_matrix = nx.to_scipy_sparse_array(token_graph,
                                                       nodelist=nodelist,
                                                       weight=weight,
                                                       dtype=np.float32)
        normalization_array = token_sparse_matrix.sum(axis=1)
        normalization_array[normalization_array != 0] = np.divide(
            1.0, normalization_array[normalization_array != 0])
        diagonal_norm = sp.sparse.spdiags([normalization_array],
                                          0,
                                          format="csr")
        token_sparse_matrix, x_iteration = np.dot(
            diagonal_norm,
            token_sparse_matrix), np.repeat(1.0 / num_nodes, num_nodes)
        pers_array = np.array([personalization.get(n, 0) for n in nodelist],
                              dtype=np.float32)
        pers_array = pers_array / np.sum(pers_array)

        for _ in range(max_iter):
            xlast = x_iteration
            x_iteration = alpha * (x_iteration @ token_sparse_matrix) + (
                1 - alpha) * pers_array
            err = np.sum(np.absolute(x_iteration - xlast))
            if err < tol:
                return dict(zip(nodelist, map(np.float32, x_iteration)))

        return dict(zip(nodelist, [1] * num_nodes))

    def get_document_graph(self, weight: int = 1):
        """ A method for obtaining the token graph """

        self.main_graph = nx.DiGraph()
        num_tokens = len(self.tokens)

        for i in range(num_tokens):
            if i + 1 < num_tokens:
                node_u = self.tokens[i].lower()
                node_v = self.tokens[i + 1].lower()

                if self.main_graph.has_edge(node_u, node_v):
                    self.main_graph[node_u][node_v]["weight"] += weight

                else:
                    self.main_graph.add_edge(node_u, node_v, weight=weight)

        self.main_graph.remove_edges_from(nx.selfloop_edges(self.main_graph))
        personalization = {a: self.term_counts[a] for a in self.tokens}

        if len(self.main_graph) > self.hyperparameters["num_keywords"]:
            self.node_ranks = self.pagerank_scipy_adapted(
                self.main_graph,
                alpha=self.hyperparameters["alpha"],
                max_iter=self.hyperparameters["max_iter"],
                personalization=personalization,
            ).items()

            self.node_ranks = [[k, v] for k, v in self.node_ranks]
        else:

            self.node_ranks = [[k, 1.0] for k in self.main_graph.nodes()]

        token_list = [k for k, v in self.node_ranks]
        rank_distribution = np.array([y for _, y in self.node_ranks])
        token_length_distribution = np.array(
            [len(x) for x, _ in self.node_ranks])

        final_scores = rank_distribution * token_length_distribution
        self.node_ranks = dict(zip(token_list, final_scores))

    def parse_input(self, document: str, input_type: str, encoding: str = "utf-8") -> None:
        """ Input parsing method """
        if input_type == "file":
            with open(document, "r", encoding=encoding) as doc:
                full_document = doc.read().split("\n")

        elif input_type == "pdf":
            with fitz.open(document) as doc:
                full_document = []
                for page in doc:
                    page_text = page.get_text("text").split("\n")
                    full_document.extend(page_text)

        elif input_type == "string":
            if isinstance(document, list):
                return document

            if isinstance(document, str):
                full_document = document.split("\n")

            else:
                raise NotImplementedError(
                    "Input type not recognized (str, list)")

        else:
            raise NotImplementedError(
                "Please select valid input type (file, string)")

        return full_document

    def combine_keywords(self) -> None:
        """
        The keyword combination method. Individual keywords
        are combined if needed.
        Some deduplication also happens along the way.
        """

        combined_keywords = []
        appeared_tokens = {}

        for ranked_node, score in self.node_ranks.items():

            if (ranked_node.lower() in self.hyperparameters["stopwords"]
                    or len(ranked_node) <= 2):
                continue

            if ranked_node not in appeared_tokens:
                ranked_tuple = [ranked_node, score]
                combined_keywords.append(ranked_tuple)

            appeared_tokens[ranked_node] = 1

        sorted_keywords = sorted(combined_keywords,
                                 key=itemgetter(1),
                                 reverse=True)

        self.final_keywords = sorted_keywords

    def merge_tokens(self) -> None:
        """ Token merge method """

        two_grams = [(self.tokens[enx], self.tokens[enx + 1])
                     for enx in range(len(self.tokens) - 1)]
        self.bigram_counts = dict(Counter(two_grams))
        tmp_tokens = []
        merged = set()
        for enx in range(len(self.tokens) - 1):
            token1 = self.tokens[enx]
            token2 = self.tokens[enx + 1]

            count1 = self.term_counts[token1]
            count2 = self.term_counts[token2]

            bgc = self.bigram_counts[(token1, token2)]
            bgs = np.abs(count1 - bgc) + np.abs(count2 - bgc)
            bgs = bgs / (count1 + count2)

            if (token1 not in self.hyperparameters["stopwords"]
                    and token2 not in self.hyperparameters["stopwords"]):

                if bgs < self.hyperparameters["merge_threshold"]:
                    if (len(token2) > self.hyperparameters["token_prune_len"]
                            and len(token1) >
                            self.hyperparameters["token_prune_len"]):

                        to_add = token1 + " " + token2
                        tmp_tokens.append(to_add)

                        merged.add(token1)
                        merged.add(token2)

                        self.term_counts[to_add] = bgc
                        self.term_counts[token1] *= self.hyperparameters[
                            "merge_threshold"]

                        self.term_counts[token2] *= self.hyperparameters[
                            "merge_threshold"]
                else:
                    continue

            else:
                tmp_tokens.append(token1)
                tmp_tokens.append(token2)

        # remove duplicate entries
        if self.hyperparameters["deduplication"]:
            to_drop = set()

            for token in tmp_tokens:
                if token in merged:
                    to_drop.add(token)

            tmp_tokens = [x for x in tmp_tokens if x not in to_drop]

        self.tokens = tmp_tokens

    def tokenize(self) -> None:
        """ Core tokenization method """

        whitespace_count = self.document.count(" ")
        self.full_tokens = self.pattern.findall(self.document)

        if len(self.full_tokens) > 0:
            space_factor = whitespace_count / len(self.full_tokens)

        else:
            space_factor = 0

        if space_factor < self.space_factor:

            self.tokens = [
                x for x in list(self.document.strip())
                if not x == " " and not x == "\n" and not x == "，"
            ]

            self.tokens = [
                x for x in self.tokens if not x.isdigit() and " " not in x
            ]

        else:
            self.tokens = [x for x in self.full_tokens if not x.isdigit()]
            del self.full_tokens

    def match_sweep(self):
        """ Replace too similar keywords with out-of final distribution ones """

        potential_output = self.final_keywords\
            [:self.hyperparameters["num_keywords"]]

        potential_replacements = self.final_keywords\
            [self.hyperparameters["num_keywords"]:][::-1]

        for enx, _ in enumerate(potential_output):
            for second_kw in range(enx + 1, len(potential_output)):
                if enx + 1 < len(potential_output):

                    key_first = potential_output[enx][0]
                    key_second = potential_output[second_kw][0]

                    longer_keyword = max(key_first, key_second, key=len)
                    shorter_keyword = min(key_first, key_second, key=len)

                    if shorter_keyword in longer_keyword and \
                       len(potential_replacements) > 0:

                        potential_output[
                            second_kw] = potential_replacements.pop()

        self.final_keywords = sorted(potential_output,
                                     key=operator.itemgetter(1))[::-1]

    def find_keywords(self,
                      document: str,
                      input_type: str = "file",
                      encoding: str = "utf-8") -> List[Tuple[str, float]]:
        """
        The main method responsible calling the child methods, yielding
        the final set of (ranked) keywords.
        """

        document = self.parse_input(document, input_type=input_type, encoding = encoding)
        self.document = " ".join(document)
        self.tokenize()
        self.compute_tf_scores()
        self.merge_tokens()
        self.get_document_graph()
        self.combine_keywords()
        self.match_sweep()
        return self.final_keywords[:self.hyperparameters["num_keywords"]]
