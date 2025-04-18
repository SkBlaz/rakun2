"""
Main RaKUn 2.0 algorithm - DS paper 2022

This module implements the RaKUn2.0 keyphrase extraction algorithm with
additional design-level optimizations.
"""

from typing import Dict, Any, Tuple, List, Optional
from collections import Counter
import gzip
import json
import pkgutil
import logging
import re

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RakunKeyphraseDetector:
    """
    RaKUn2.0 Keyword Detector with Additional Optimizations

    Implements the main algorithm for keyphrase extraction based on RaKUn2.0, with
    several design-level optimizations to improve runtime.
    """

    def __init__(self,
                 hyperparameters: Optional[Dict[str, Any]] = None,
                 verbose: bool = False) -> None:
        """
        Initialize the keyphrase detector.

        Args:
            hyperparameters: Optional dictionary of hyperparameters.
            verbose: If True, logs processing information.
        """
        self.verbose = verbose
        self.document: Optional[str] = None
        self.tokens: List[str] = []         # Original tokens (as extracted)
        self.tokens_lower: List[str] = []     # Lowercase version for comparisons
        self.full_tokens: List[str] = []
        self.sorted_terms_tf: Optional[List[Tuple[str, int]]] = None
        self.bigram_counts: Optional[Dict[Tuple[str, str], int]] = None
        self.final_keywords: Optional[List[Tuple[str, float]]] = None
        self.node_ranks: Dict[str, float] = {}
        self.main_graph: nx.DiGraph = nx.DiGraph()
        self.term_counts: Dict[str, int] = {}
        self.space_factor_threshold: float = 0.5

        if hyperparameters is None:
            hyperparameters = {}
        self.hyperparameters = hyperparameters
        self.hyperparameters.setdefault("token_prune_len", 2)
        self.hyperparameters.setdefault("num_keywords", 10)
        self.hyperparameters.setdefault("alpha", 0.1)
        self.hyperparameters.setdefault("max_iter", 100)
        self.hyperparameters.setdefault("merge_threshold", 0.8)
        self.hyperparameters.setdefault("deduplication", True)
        if "stopwords" not in self.hyperparameters:
            stopwords_data = pkgutil.get_data(__name__, "stopwords.json.gz")
            if stopwords_data is not None:
                stopwords_data = gzip.decompress(stopwords_data)
                stopwords_generic = set(json.loads(stopwords_data.decode()))
                self.hyperparameters["stopwords"] = stopwords_generic
            else:
                self.hyperparameters["stopwords"] = set()
        # Precompute lowercase stopwords for fast membership tests.
        self.stopwords = {word.lower()
                          for word in self.hyperparameters["stopwords"]}

        # Precompiled regex to extract words with at least two characters.
        self.pattern = re.compile(r"(?u)\b\w\w+\b")

        if self.verbose:
            logger.info("Initialized RaKUn2.0 keyword detector instance.")

    def visualize_network(self,
                          show_labels: bool = False,
                          base_node_size: float = 100,
                          alpha: float = 0.8,
                          edge_width: float = 0.5,
                          font_size: int = 8,
                          arrow_size: int = 10) -> None:
        """
        Visualize the token network graph.

        Args:
            show_labels: Whether to display node labels.
            base_node_size: Base size for nodes; top keywords will appear larger.
            alpha: Node transparency.
            edge_width: Width of the edges.
            font_size: Font size for labels.
            arrow_size: Size of the arrowheads.
        """
        if self.verbose:
            logger.info("Visualizing token network.")
        plt.figure(figsize=(10, 10), dpi=300)
        pos = nx.spring_layout(self.main_graph, iterations=50)

        if self.node_ranks:
            sorted_nodes = sorted(
                self.node_ranks.items(), key=lambda x: x[1], reverse=True
            )
            top_nodes = {node for node, _ in sorted_nodes[:20]}
        else:
            top_nodes = set()

        node_colors = [
            "red" if node in top_nodes else "gray"
            for node in self.main_graph.nodes()
        ]
        node_sizes = [
            base_node_size * 2 if node in top_nodes else base_node_size
            for node in self.main_graph.nodes()
        ]

        nx.draw_networkx_nodes(
            self.main_graph, pos, node_size=node_sizes,
            node_color=node_colors, alpha=alpha
        )
        nx.draw_networkx_edges(
            self.main_graph, pos, width=edge_width, arrowsize=arrow_size
        )

        if show_labels:
            nx.draw_networkx_labels(
                self.main_graph, pos, font_size=font_size,
                font_color="black"
            )

        plt.tight_layout()
        plt.show()

    def tokenize(self) -> None:
        """
        Tokenize the document.

        Uses a regex pattern to extract tokens. Also computes a lowercase version
        to avoid repeated .lower() calls in subsequent processing.
        """
        if self.document is None:
            return

        whitespace_count = self.document.count(" ")
        self.full_tokens = self.pattern.findall(self.document)
        avg_space_factor = (whitespace_count / len(self.full_tokens)
                            if self.full_tokens else 0)

        if avg_space_factor < self.space_factor_threshold:
            # Likely a language without explicit word boundaries.
            raw_tokens = [
                ch for ch in self.document
                if ch not in {" ", "\n", "，"} and not ch.isdigit()
            ]
        else:
            raw_tokens = [
                token for token in self.full_tokens if not token.isdigit()
            ]

        self.tokens = raw_tokens
        self.tokens_lower = [token.lower() for token in raw_tokens]

        if self.verbose:
            logger.info("Tokenization complete. Total tokens: %d",
                        len(self.tokens))

    def compute_tf_scores(self, document: Optional[str] = None) -> None:
        """
        Compute term frequency (TF) scores for the document.

        Uses the precomputed lowercase tokens for a case-insensitive count.

        Args:
            document: Optional document text. If provided, tokens are extracted.
        """
        if document is not None and not self.tokens:
            self.document = document
            self.tokenize()

        token_list = (
            self.tokens_lower if self.tokens_lower
            else [token.lower() for token in self.tokens]
        )
        term_counter = Counter(token_list)
        self.term_counts = dict(term_counter)
        self.sorted_terms_tf = term_counter.most_common()
        if self.verbose:
            logger.info("Computed term frequency scores.")

    def pagerank_scipy_adapted(self,
                               token_graph: nx.Graph,
                               alpha: float = 0.85,
                               personalization: Optional[Dict[str, float]] = None,
                               max_iter: int = 64,
                               tol: float = 1e-2,
                               weight: str = "weight") -> Dict[str, float]:
        # pylint: disable=unused-argument
        """
        Compute PageRank scores using a sparse matrix power iteration.

        Args:
            token_graph: NetworkX graph representing tokens.
            alpha: Damping factor.
            personalization: Personalization vector.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            weight: Edge weight attribute.

        Returns:
            Dictionary mapping nodes to PageRank scores.
        """
        num_nodes = token_graph.number_of_nodes()
        if num_nodes == 0:
            return {}

        nodelist = list(token_graph.nodes())
        token_sparse_matrix = nx.to_scipy_sparse_array(
            token_graph, nodelist=nodelist, weight=weight, dtype=np.float32
        )

        row_sums = np.array(token_sparse_matrix.sum(axis=1)).flatten()
        nonzero = row_sums > 0
        row_scaling = np.zeros_like(row_sums)
        row_scaling[nonzero] = 1.0 / row_sums[nonzero]
        token_sparse_matrix.data *= np.repeat(
            row_scaling, np.diff(token_sparse_matrix.indptr)
        )
        token_sparse_matrix_transpose = token_sparse_matrix.T

        pers = np.zeros(num_nodes, dtype=np.float32)
        if personalization:
            for idx, node in enumerate(nodelist):
                pers[idx] = personalization.get(node, 0)
            if pers.sum() > 0:
                pers /= pers.sum()
            else:
                pers.fill(1.0 / num_nodes)
        else:
            pers.fill(1.0 / num_nodes)

        x = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float32)
        for _ in range(max_iter):
            x_last = x.copy()
            x = (alpha * (token_sparse_matrix_transpose @ x) +
                 (1 - alpha) * pers)
            if np.sum(np.abs(x - x_last)) < tol:
                break

        return dict(zip(nodelist, x))

    def get_document_graph(self) -> None:
        """
        Build a directed token graph from the document.

        Consecutive tokens form edges with a specified weight. Uses the precomputed
        lowercase token list to avoid redundant lowercasing. Then computes node ranks
        using PageRank (adjusted by token length).

        Args:
            weight: Weight for each edge.
        """
        num_tokens = len(self.tokens_lower)
        if num_tokens < 2:
            self.main_graph = nx.DiGraph()
            return

        edges = list(zip(self.tokens_lower, self.tokens_lower[1:]))
        edge_weights = Counter(edges)
        self.main_graph = nx.DiGraph(
            ((u, v, {"weight": w}) for (u, v), w in edge_weights.items())
        )

        personalization = {
            token: self.term_counts.get(token, 1)
            for token in self.tokens_lower
        }

        if self.main_graph.number_of_nodes() > \
           self.hyperparameters["num_keywords"]:
            pr_scores = self.pagerank_scipy_adapted(
                self.main_graph,
                alpha=self.hyperparameters["alpha"],
                max_iter=self.hyperparameters["max_iter"],
                personalization=personalization,
            )
            self.node_ranks = {
                token: score * len(token)
                for token, score in pr_scores.items()
            }
        else:
            self.node_ranks = {
                node: 1.0 for node in self.main_graph.nodes()
            }

        if self.verbose:
            logger.info("Constructed document graph and computed node ranks.")

    def parse_input(self,
                    document: str,
                    input_type: str,
                    encoding: str = "utf-8") -> List[str]:
        """
        Parse the input document based on its type.

        Args:
            document: The document content, file path, or PDF path.
            input_type: One of 'file', 'pdf', or 'string'.
            encoding: Encoding to use when reading from file.

        Returns:
            A list of lines from the document.

        Raises:
            NotImplementedError: If the input_type is not recognized.
        """
        if input_type == "file":
            with open(document, "r", encoding=encoding) as f:
                lines = f.read().splitlines()
        elif input_type == "pdf":
            lines = []
            with fitz.open(document) as doc:
                for page in doc:
                    lines.extend(page.get_text("text").splitlines())
        elif input_type == "string":
            if isinstance(document, list):
                lines = document
            elif isinstance(document, str):
                lines = document.splitlines()
            else:
                raise NotImplementedError(
                    "Input type not recognized (expected str or list)."
                )
        else:
            raise NotImplementedError(
                "Please select a valid input type: 'file', 'pdf', or 'string'."
            )
        return lines

    def merge_tokens(self) -> None:
        """
        Merge adjacent tokens into phrases when appropriate.

        Uses precomputed bigram counts and a merge threshold to decide whether two
        consecutive tokens should be merged. Comparisons use the lowercase tokens.
        Deduplication is applied if enabled.
        """
        if len(self.tokens) < 2:
            return

        two_grams = list(zip(self.tokens, self.tokens[1:]))
        self.bigram_counts = dict(Counter(two_grams))

        merged_tokens = []
        merged_set = set()
        token_prune_len = self.hyperparameters["token_prune_len"]
        merge_threshold = self.hyperparameters["merge_threshold"]

        i = 0
        while i < len(self.tokens) - 1:
            token1, token2 = self.tokens[i], self.tokens[i + 1]
            token1_low, token2_low = self.tokens_lower[i], self.tokens_lower[i + 1]
            count1 = self.term_counts.get(token1_low, 0)
            count2 = self.term_counts.get(token2_low, 0)
            bg_count = self.bigram_counts.get((token1, token2), 0)
            diff_metric = ((abs(count1 - bg_count) + abs(count2 - bg_count)) /
                           (count1 + count2)
                           if (count1 + count2) > 0 else 1.0)

            if (token1_low not in self.stopwords and token2_low not in self.stopwords and
                    len(token1) > token_prune_len and len(token2) > token_prune_len and
                    diff_metric < merge_threshold):
                merged_phrase = f"{token1} {token2}"
                merged_tokens.append(merged_phrase)
                merged_set.update({token1, token2})
                self.term_counts[merged_phrase.lower()] = bg_count
                self.term_counts[token1_low] = int(
                    self.term_counts[token1_low] * merge_threshold
                )
                self.term_counts[token2_low] = int(
                    self.term_counts[token2_low] * merge_threshold
                )
                i += 2
            else:
                merged_tokens.append(token1)
                i += 1

        if i == len(self.tokens) - 1:
            merged_tokens.append(self.tokens[-1])

        if self.hyperparameters.get("deduplication", True):
            merged_tokens = [
                token for token in merged_tokens if token not in merged_set
            ]

        self.tokens = merged_tokens
        self.tokens_lower = [token.lower() for token in merged_tokens]
        if self.verbose:
            logger.info(
                "Merged adjacent tokens into phrases where applicable."
            )

    def combine_keywords(self) -> None:
        """
        Combine and deduplicate keywords.

        Filters out stopwords and tokens that are too short, then sorts the keywords
        by their scores.
        """
        keywords = []
        seen_tokens = set()
        for token, score in self.node_ranks.items():
            if token in self.stopwords or len(token) <= 2:
                continue
            if token not in seen_tokens:
                keywords.append((token, score))
                seen_tokens.add(token)
        self.final_keywords = sorted(keywords, key=lambda x: x[1],
                                     reverse=True)
        if self.verbose:
            logger.info("Combined and deduplicated keywords.")

    def match_sweep(self) -> None:
        """
        Refine final keywords by replacing overly similar keywords.

        If one keyword is a substring of another, the lower-ranked keyword is
        replaced by a candidate from the lower-ranked pool, if available.
        """
        if not self.final_keywords:
            return

        num_keywords = self.hyperparameters["num_keywords"]
        primary_keywords = self.final_keywords[:num_keywords]
        replacement_candidates = self.final_keywords[num_keywords:][::-1]

        lengths = {kw: len(kw) for kw, _ in primary_keywords}

        for i, (kw_i, _) in enumerate(primary_keywords):
            for j, (kw_j, _) in enumerate(primary_keywords[i + 1:], start=i+1):
                if replacement_candidates:
                    if lengths[kw_i] >= lengths[kw_j]:
                        longer, shorter = kw_i, kw_j
                    else:
                        longer, shorter = kw_j, kw_i
                    if shorter in longer:
                        primary_keywords[j] = replacement_candidates.pop()
                        lengths[primary_keywords[j][0]] = \
                            len(primary_keywords[j][0])
        self.final_keywords = sorted(primary_keywords, key=lambda x: x[1],
                                     reverse=True)
        if self.verbose:
            logger.info(
                "Completed keyword similarity sweep and replacement."
            )

    def find_keywords(self,
                      document: str,
                      input_type: str = "file",
                      encoding: str = "utf-8",
                      prior_rankings: list = None) -> List[Tuple[str, float]]:
        """
        Extract and rank keywords from the input document.

        Orchestrates the entire pipeline: input parsing, tokenization, merging, graph
        construction, and keyword ranking.

        Args:
            document: The input document (file path, PDF path, or string).
            input_type: The type of input ('file', 'pdf', or 'string').
            encoding: Encoding for file inputs.

        Returns:
            A list of tuples (keyword, score) representing the top keywords.
        """
        lines = self.parse_input(document, input_type=input_type,
                                 encoding=encoding)
        self.document = " ".join(lines)
        self.tokenize()
        self.compute_tf_scores()  # Uses precomputed tokens
        self.merge_tokens()
        self.get_document_graph()
        self.combine_keywords()
        self.match_sweep()

        if self.verbose:
            logger.info("Keyword extraction complete.")

        if prior_rankings is not None:
            prior_dict = dict(prior_rankings)
            new_keywords = [
                (keyphrase, score + prior_dict[prior_keyphrase])
                for keyphrase, score in self.final_keywords
                for prior_keyphrase in prior_dict
                if prior_keyphrase in keyphrase
            ]
            self.final_keywords = new_keywords

        return self.final_keywords[:self.hyperparameters["num_keywords"]]
