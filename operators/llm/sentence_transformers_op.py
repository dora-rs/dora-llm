from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from dora import DoraStatus

import torch
import pyarrow as pa


def search(query_embedding, corpus_embeddings, paths, raw, codes, k=5, file_extension=None):
    # TODO: filtering by file extension
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(k, len(cos_scores)), sorted=True)
    out = []
    for score, idx in zip(top_results[0], top_results[1]):
        out.extend([raw[idx], paths[idx], codes[idx]])
    return out


class Operator:
    """ """

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.sentences_encoding = []
        self.sentences_raw = []
        self.sentences_path = []
        self.sentences_codes = []

    def on_event(
        self,
        dora_event,
        send_output,
    ) -> DoraStatus:
        if dora_event["type"] == "INPUT":
            values = dora_event["value"].to_pylist()
            length = len(values)

            if dora_event["id"] == "append":
                self.sentences_raw.extend(values[: length // 3])
                self.sentences_path.extend(values[length // 3 : length // 3 * 2])
                self.sentences_codes.extend(values[length // 3 * 2 :])

                self.sentences_encoding.extend(self.model.encode(values[: length // 3]))
            elif dora_event["id"] == "query":
                query_embeddings = self.model.encode(values)
                output = search(
                    query_embeddings,
                    self.sentences_encoding,
                    self.sentences_raw,
                    self.sentences_path,
                    self.sentences_codes,
                )
                send_output("reply_query", pa.array(output))
            elif dora_event["id"] == "clear":
                self.sentences_encoding = []
                self.sentences_raw = []
                self.sentences_path = []
                self.sentences_codes = []

        return DoraStatus.CONTINUE
