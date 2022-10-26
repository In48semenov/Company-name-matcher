import os
import click

from sentence_bert import SentBertPipeline
from bert import BertPipeline

from transformers import AutoTokenizer


@click.command()
@click.option(
    "--algo",
    default="bert",
    type=click.Choice(["bert", "sentence_bert", "fasttext"], case_sensitive=False),
)
@click.option("-p", "--weight-path", default=None, type=click.Path(exists=True))
@click.option("-d", "--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
@click.option("-n", "--names", default=None, nargs=2, type=click.Tuple([str, str]))
def compare_two_company_name(algo="bert", weight_path=None, device="cpu", names=None):
    assert names is not None
    assert len(names) == 2

    module_realpath = os.path.realpath(__file__)
    module_folder = os.path.dirname(module_realpath)

    if algo == "bert":
        if weight_path is None:
            weight_path = f"{module_folder}/../weights/inference/bert.pth"
        tokenizer = AutoTokenizer.from_pretrained(
            "DeepPavlov/bert-base-cased-conversational"
        )
        model = BertPipeline(tokenizer, weight_path, device)
    elif algo == "sentence_bert":
        if weight_path is None:
            weight_path = f"{module_folder}/../weights/inference/sbert"
        model = SentBertPipeline(weight_path, device)
    else:
        raise Exception("Given model not found")
    result = model(*names)

    if result:
        print(f"Компании {names[0]} {names[1]} одинаковые!")
    else:
        print("Название компаний различно!")

    return result


if __name__ == "__main__":
    compare_two_company_name()
