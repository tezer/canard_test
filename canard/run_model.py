import _io

import click

from canard.constants import HISTORY_MAX_TOKEN_LEN
from canard.train_t5 import CanardModel, tokenizer


def rewrite(trained_model: CanardModel, text: str):
    text_encoding = tokenizer(
        text,
        max_length=HISTORY_MAX_TOKEN_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    generated_ids = trained_model.model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]

    return " ".join(preds)


@click.command()
@click.option('--model_file',
              type=click.File("rb"),
              prompt='Specify the model file',
              help='Enter the path to the file with the trained model.')
def run_classifier(model_file: _io.BufferedReader):
    trained_model = CanardModel.load_from_checkpoint(model_file)
    trained_model.freeze()
    click.echo("Enter the history with a question to be rewritten with all parts separated by '|||'")
    click.echo("e.g.: Jane lives in Paris.|||Her friend, Robert, knows about it, because he also lives there.|||Does "
               "she know where he lives?")
    while True:
        history = input(">>")
        click.echo(rewrite(trained_model, history))


if __name__ == '__main__':
    run_classifier()
