from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


def test_mllm():
    model_id = "vikhyatk/moondream2"
    revision = "2024-04-02"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    image = Image.open('ww2.jpeg')
    enc_image = model.encode_image(image)
    print(model.answer_question(
        enc_image, "Answer the nationality of the troops in the image", tokenizer))


if __name__ == "__main__":
    test_mllm()
