import datasets
import evaluate
import numpy as np
from datasets.config import importlib_metadata, version

NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))

_CITATION = """\
@article{xxx
}
"""

_DESCRIPTION = """\
Save results.
"""

_KWARGS_DESCRIPTION = """
No args.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION,
                                                _KWARGS_DESCRIPTION)
class MLMMetrics(evaluate.Metric):

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features({
                    "predictions":
                    datasets.Value("string", id="sequence"),
                    "references":
                    datasets.Sequence(datasets.Value("string", id="sequence"),
                                      id="references"),
                }),
                datasets.Features({
                    "predictions":
                    datasets.Value("string", id="sequence"),
                    "references":
                    datasets.Value("string", id="sequence"),
                }),
            ],
            codebase_urls=["https://xxx.com"],
            reference_urls=["https://xxx.com"],
        )

    def _download_and_prepare(self, dl_manager):
        import nltk

        nltk.download("wordnet")
        if NLTK_VERSION >= version.Version("3.6.5"):
            nltk.download("punkt")
        if NLTK_VERSION >= version.Version("3.6.6"):
            nltk.download("omw-1.4")

    def _compute(self,
                 predictions,
                 references,
                 tsv_path="tmp.tsv",
                 verbose=False):

        references = [references[i][0] for i in range(len(references))]
        # Accuracy
        correct = 0
        for ref, pred in zip(references, predictions):
            if ref == pred:
                correct += 1
        accuracy = correct / len(references)

        return {"accuracy": accuracy}
