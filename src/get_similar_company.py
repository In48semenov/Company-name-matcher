import os
import click

from bert import BertPipeline
from sentence_bert import SentBertPipeline
from transformers import AutoTokenizer


@click.command()
@click.option(
    "--algo",
    default="bert",
    type=click.Choice(["bert", "sentence_bert", "fasttext"], case_sensitive=False),
)
@click.option("-p", "--weight-path", default=None, type=click.Path(exists=True))
@click.option("-d", "--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
@click.option("-n", "--name", default=None, nargs=1, type=str)
@click.option("--top_n", default=10, type=int)
def get_similar_company(
    algo="bert", weight_path=None, device="cpu", name=None, top_n=10
):
    assert name is not None

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
    result = model(name)

    print(f"Найдено {len(result)} похожих компаний:")
    for item in result:
        print(f" -{item}")
    return result


if __name__ == "__main__":
    get_similar_company()
