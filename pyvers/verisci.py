"""
Data-handling code copied over from `verisci/evaluate/lib/data.py` of the VeriSci
library from the original SciFact release: https://github.com/allenai/scifact.

This was taken from the MultiVerS repo (https://github.com/dwadden/multivers)
and modified by Jeffrey Dick for the pyvers package.

The data format is described at: https://github.com/dwadden/multivers/blob/main/doc/data.md
"""

from enum import Enum
import json
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

####################

# Utility functions and enums.


def load_jsonl(fname):
    return [json.loads(line) for line in open(fname)]


class Label(Enum):
    SUPPORT = 2
    NEI = 1
    REFUTE = 0


def make_label(label_str, allow_NEI=True):
    lookup = {
        "SUPPORT": Label.SUPPORT,
        "NEI": Label.NEI,
        "REFUTE": Label.REFUTE,
    }

    res = lookup[label_str]
    if (not allow_NEI) and (res is Label.NEI):
        raise ValueError("An NEI was given.")

    return res


####################

# Representations for the corpus and abstracts.


@dataclass(repr=False, frozen=True)
class Document:
    id: str
    title: str
    sentences: Tuple[str]

    def __repr__(self):
        return (
            self.title.upper()
            + "\n"
            + "\n".join(["- " + entry for entry in self.sentences])
        )

    def __lt__(self, other):
        return self.title.__lt__(other.title)

    def dump(self):
        res = {
            "doc_id": self.id,
            "title": self.title,
            "abstract": self.sentences,
            "structured": self.is_structured(),
        }
        return json.dumps(res)


@dataclass(repr=False, frozen=True)
class Corpus:
    """
    A Corpus is just a collection of `Document` objects, with methods to look up
    a single document.
    """

    documents: List[Document]

    def __repr__(self):
        return f"Corpus of {len(self.documents)} documents."

    def __getitem__(self, i):
        "Get document by index in list."
        return self.documents[i]

    def get_document(self, doc_id):
        "Get document by ID."
        res = [x for x in self.documents if x.id == doc_id]
        assert len(res) == 1
        return res[0]

    @classmethod
    def from_jsonl(cls, corpus_file):
        corpus = load_jsonl(corpus_file)
        documents = []
        for entry in corpus:
            doc = Document(entry["doc_id"], entry["title"], entry["abstract"])
            documents.append(doc)

        return cls(documents)


####################

# Gold dataset.


class GoldDataset:
    """
    Class to represent a gold dataset, include corpus and claims.
    """

    def __init__(self, corpus_file, data_file):
        self.corpus = Corpus.from_jsonl(corpus_file)
        self.claims = self._read_claims(data_file)

    def __repr__(self):
        msg = f"{self.corpus.__repr__()} {len(self.claims)} claims."
        return msg

    def __getitem__(self, i):
        return self.claims[i]

    def _read_claims(self, data_file):
        "Read claims from file."
        examples = load_jsonl(data_file)
        res = []
        for this_example in examples:
            entry = copy.deepcopy(this_example)
            entry["release"] = self
            entry["cited_docs"] = [
                self.corpus.get_document(doc) for doc in entry["cited_doc_ids"]
            ]
            assert len(entry["cited_docs"]) == len(entry["cited_doc_ids"])
            del entry["cited_doc_ids"]
            res.append(Claim(**entry))

        res = sorted(res, key=lambda x: x.id)
        return res

    def get_claim(self, example_id):
        "Get a single claim by ID."
        keep = [x for x in self.claims if x.id == example_id]
        assert len(keep) == 1
        return keep[0]


@dataclass
class EvidenceAbstract:
    "A single evidence abstract."

    id: int
    label: Label
    rationales: List[List[int]]


@dataclass(repr=False)
class Claim:
    """
    Class representing a single claim, with a pointer back to the dataset.
    """

    id: int
    claim: str
    evidence: Dict[int, EvidenceAbstract]
    cited_docs: List[Document]
    release: GoldDataset

    def __post_init__(self):
        self.evidence = self._format_evidence(self.evidence)

    @staticmethod
    def _format_evidence(evidence_dict):
        # This function is needed because the data schema is designed so that
        # each rationale can have its own support label. But, in the dataset,
        # all rationales for a given claim / abstract pair all have the same
        # label. So, we store the label at the "abstract level" rather than the
        # "rationale level".
        res = {}
        for doc_id, rationales in evidence_dict.items():
            doc_id = int(doc_id)
            labels = [x["label"] for x in rationales]
            if len(set(labels)) > 1:
                msg = (
                    "In this SciFact release, each claim / abstract pair "
                    "should only have one label."
                )
                raise Exception(msg)
            label = make_label(labels[0])
            rationale_sents = [x["sentences"] for x in rationales]
            this_abstract = EvidenceAbstract(doc_id, label, rationale_sents)
            res[doc_id] = this_abstract

        return res

    def __repr__(self):
        msg = f"Example {self.id}: {self.claim}"
        return msg

    def pretty_print(self, evidence_doc_id=None, file=None):
        "Pretty-print the claim, together with all evidence."
        msg = self.__repr__()
        print(msg, file=file)
        # Print the evidence
        print("\nEvidence sets:", file=file)
        for doc_id, evidence in self.evidence.items():
            # If asked for a specific evidence doc, only show that one.
            if evidence_doc_id is not None and doc_id != evidence_doc_id:
                continue
            print("\n" + 20 * "#" + "\n", file=file)
            ev_doc = self.release.corpus.get_document(doc_id)
            print(f"{doc_id}: {evidence.label.name}", file=file)
            for i, sents in enumerate(evidence.rationales):
                print(f"Set {i}:", file=file)
                kept = [sent for i, sent in enumerate(ev_doc.sentences) if i in sents]
                for entry in kept:
                    print(f"\t- {entry}", file=file)


####################

# Reader class modified from MultiVerS.


class SciFactReader:
    """
    Class to handle SciFact data. Not used directly; its subclasses handle cases with
    different numbers of negative samples.
    """

    def __init__(self, data_dir, debug=False):
        self.data_dir = data_dir
        self.debug = debug
        self.name = "SciFact"

    def get_text_data(self, fold):
        """
        Load in the dataset as a list of entries. Each entry is a single claim /
        cited document pair. Some cited documents have no evidence.
        """

        res = []
        # Get the data from the shuffled directory for training.
        corpus_file = os.path.join(self.data_dir, "corpus.jsonl")
        data_file = os.path.join(self.data_dir, f"claims_{fold}.jsonl")
        print(f"SciFactReader: Reading {data_file}")
        ds = GoldDataset(corpus_file, data_file)

        for i, claim in enumerate(ds.claims):
            # Only read 10 if we're doing a fast dev run.
            if self.debug and i == 10:
                break
            # NOTE(dwadden) This is a hack because claim 1245 in the dev set
            # lists document 7662395 twice. Need to fix the dataset. For now,
            # I'll just do this check.
            seen = set()
            for cited_doc in claim.cited_docs:
                if cited_doc.id in seen:
                    # If we've seen it already, skip.
                    continue
                else:
                    seen.add(cited_doc.id)
                # Convert claim and evidence into form for function input.
                if cited_doc.id in claim.evidence:
                    ev = claim.evidence[cited_doc.id]
                    label = ev.label.name
                    rationales = ev.rationales
                else:
                    label = "NEI"
                    rationales = []

                # Append entry.
                to_tensorize = {
                    "claim": claim.claim,
                    "sentences": cited_doc.sentences,
                    "label": label,
                    "rationales": rationales,
                    "title": cited_doc.title,
                }
                entry = {
                    "claim_id": claim.id,
                    "abstract_id": cited_doc.id,
                    "negative_sample_id": 0,  # No negative sampling for SciFact yet.
                    "to_tensorize": to_tensorize,
                }
                res.append(entry)

        return res
