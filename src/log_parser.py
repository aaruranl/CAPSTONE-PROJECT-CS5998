"""
Log parsing: turn a free-text log message into an *event template*.

    raw message : "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    masked      : "Receiving block <BLK> src: /<IP> dest: /<IP>"
    template    : "Receiving block <BLK> src: /<IP> dest: /<IP>"   -> event_id 5

Two stages:
  1. Masking  - regular expressions replace values we already KNOW are
                variables (block IDs, IP addresses, hex values, numbers).
  2. Drain    - a fixed-depth parse tree (He et al., ICWS 2017) that groups
                the remaining messages by similarity and learns which tokens
                are constant (part of the template) and which vary (<*>).

Leakage rule (from the risk table in the project proposal): the parser may only LEARN
templates from training-period logs. After `freeze()` it can only MATCH new
messages to templates it already knows. Messages that match nothing become
event 0 ("<UNSEEN>"), which is exactly what a deployed system would face.
"""
import pickle
import re

from drain3.drain import Drain

from src import config

UNSEEN_EVENT_ID = 0
UNSEEN_TEMPLATE = "<UNSEEN>"

# Order matters: the specific patterns must run before the generic number one.
MASKS = [
    (re.compile(r"blk_-?\d+"), "<BLK>"),                                     # HDFS block IDs
    (re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?"), "<IP>"),    # IPv4 (+ port)
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),                            # hex values
    (re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])"), "<NUM>"),  # plain numbers
]


def mask(message: str) -> str:
    """Replace obvious variable values with placeholders."""
    for pattern, placeholder in MASKS:
        message = pattern.sub(placeholder, message)
    return message


class DrainParser:
    """Thin wrapper around the Drain3 library with an explicit fit / freeze life-cycle."""

    def __init__(self, sim_th: float = config.DRAIN_SIM_TH, depth: int = config.DRAIN_DEPTH):
        self.drain = Drain(sim_th=sim_th, depth=depth, max_children=config.DRAIN_MAX_CHILDREN,
                           max_clusters=None, param_str="<*>")
        self.frozen = False
        self._match_cache: dict[str, int] = {}
        self.n_learned = 0  # messages used for learning
        self.n_matched = 0  # messages matched after freezing
        self.n_unseen = 0   # matched messages that fitted no known template

    # -- learning phase ------------------------------------------------------
    def learn(self, message: str) -> int:
        """Add a training-period message; may create or generalise a template."""
        if self.frozen:
            raise RuntimeError("parser is frozen; use match()")
        cluster, _ = self.drain.add_log_message(mask(message))
        self.n_learned += 1
        return cluster.cluster_id

    def freeze(self) -> None:
        """Stop learning. From now on templates never change."""
        self.frozen = True

    # -- inference phase -----------------------------------------------------
    def match(self, message: str) -> int:
        """Map a message to a known template id, or UNSEEN_EVENT_ID.

        Uses the same similarity rule Drain uses while learning (sim_th), but
        never creates or edits a template. Results are cached because many
        masked messages repeat exactly.
        """
        masked = mask(message)
        self.n_matched += 1
        cached = self._match_cache.get(masked)
        if cached is not None:
            if cached == UNSEEN_EVENT_ID:
                self.n_unseen += 1
            return cached

        tokens = self.drain.get_content_as_tokens(masked)
        cluster = self.drain.tree_search(self.drain.root_node, tokens, self.drain.sim_th,
                                         include_params=False)
        event_id = cluster.cluster_id if cluster is not None else UNSEEN_EVENT_ID
        if event_id == UNSEEN_EVENT_ID:
            self.n_unseen += 1
        if len(self._match_cache) < 2_000_000:
            self._match_cache[masked] = event_id
        return event_id

    def parse(self, message: str, learn: bool) -> int:
        return self.learn(message) if learn else self.match(message)

    # -- saving / loading ------------------------------------------------------
    def save(self, path) -> None:
        """Save the FROZEN parser, so the scoring engine matches new lines exactly as in training."""
        if not self.frozen:
            raise RuntimeError("freeze the parser before saving it")
        state = {**self.__dict__, "_match_cache": {}}  # the cache is rebuilt on use
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path) -> "DrainParser":
        """Load a parser written by save(). Only load files this project created (pickle can run code)."""
        parser = cls.__new__(cls)
        with open(path, "rb") as f:
            parser.__dict__.update(pickle.load(f))
        parser.n_matched = parser.n_unseen = 0  # count only what happens after loading
        return parser

    # -- reporting -------------------------------------------------------------
    def templates(self) -> dict[int, str]:
        """event_id -> final template text (event 0 is the 'unseen' bucket)."""
        out = {UNSEEN_EVENT_ID: UNSEEN_TEMPLATE}
        for cluster in self.drain.clusters:
            out[cluster.cluster_id] = cluster.get_template()
        return dict(sorted(out.items()))
