# based on https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py
import datasets

logger = datasets.logging.get_logger(__name__)


class SeppNLGConfig(datasets.BuilderConfig):
    """BuilderConfig for SeppNLG"""

    def __init__(self, **kwargs):
        """BuilderConfig for SeppNLG.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SeppNLGConfig, self).__init__(**kwargs)


class SeppNLG(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        SeppNLGConfig(name="sepp_nlg", version=datasets.Version("1.0.0"), description="SEPP-NLG dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description='NER for SEPP-NLG',
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['.', ',', '?', '-', ':', '0']
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://sites.google.com/view/sentence-segmentation/",
            citation= 'TDB',
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": "data/en_train.csv"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": "data/en_dev.csv"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": "data/en_dev.csv"}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
