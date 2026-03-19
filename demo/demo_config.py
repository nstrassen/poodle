from enum import Enum


class Model(Enum):
    BERT_80M = "bert-80M"
    LLAMA_8B = "Llama-8B"
    LLAMA_70B_TURBO = "Llama-70B-turbo"
    LLAMA_405B_TURBO = "Llama-405B-turbo"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1_mini"
    GPT_4_1_NANO = "gpt-4.1_nano"


class TaskDetectionMethod(Enum):
    WRAPPER_PROMPT = "wrapper_prompt"
    USER_PROVIDED = "user_provided"


from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    large_model: Model = Model.GPT_4_1
    small_model: Model = Model.BERT_80M


@dataclass
class RequestConfig:
    expected_requests: int = 1_000_000
    switch_after_n_items: int = 5_000


@dataclass
class TokenConfig:
    input: str = ""
    prompt: str = ""
    task_detection_method: TaskDetectionMethod = TaskDetectionMethod.WRAPPER_PROMPT
    wrapper_prompt: str = ""
    wrapped_requests_percent: float = 1.0
    output: str = ""
    wrapped_output: str = ""


@dataclass
class ModelDevConfig:
    model_dev_costs: float = 4.0


@dataclass
class ValidationConfig:
    validation_requests_percent: float = 0.0


@dataclass
class DemoScenario:
    models: ModelConfig = ModelConfig()
    requests: RequestConfig = RequestConfig()
    tokens: TokenConfig = TokenConfig()
    dev: ModelDevConfig = ModelDevConfig()
    validation: ValidationConfig = ValidationConfig()

    @classmethod
    def from_dict(cls, data: dict) -> "DemoScenario":
        return cls(
            models=ModelConfig(**data.get("models", {})),
            requests=RequestConfig(**data.get("requests", {})),
            tokens=TokenConfig(**data.get("tokens", {})),
            dev=ModelDevConfig(**data.get("dev", {})),
            validation=ValidationConfig(**data.get("validation", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)  # built into dataclasses, handles nesting

    @classmethod
    def get_example_scenario(cls) -> "DemoScenario":
        example_review = (
            "\"A Mouse in the House\" is a very classic cartoon by Tom & Jerry, faithful to their tradition but with"
            " jokes of its own. It is hysterical, hilarious, very entertaining and quite amusing. Artwork is of good"
            " quality either.<br /><br />This short isn't just about Tom trying to catch Jerry. Butch lives in the same"
            " house and he's trying to catch the mouse too, because «there's only going to be one cat in this house in"
            " the morning -- and that's the cat that catches the mouse».<br /><br />If you ask me, there are lots of"
            " funny gags in this cartoon. The funniest for me are, for example, when Mammy Two Shoes sees the two lazy"
            " cats sleeping and says sarcastically «I'm glad you're enjoying the siesta» and that she hopes they're"
            " satisfied because she ain't, making the two cats gasp. Another funny gag is when Tom disguises himself"
            " as Mammy Two Shoes and slams Butch with a frying pan and then Butch does the same trick to Tom. Of course"
            " that, even funnier than this, is when the real Mammy Two Shoes appears and both (dumb!) cats think they"
            " are seeing each other disguised as Mammy and then they both attack her on the \"rear\" - lol. Naturally"
            " that she gets mad and once she gets mad, she isn't someone to mess with. But even Jerry doesn't win this"
            " time, because he is expelled by her too."
        )
        example_prompt = (
            "You are a sentiment analysis assistant. "
            "Your job is to read movie reviews and classify their sentiment as either 'positive' or 'negative'. "
            "Only respond in this exact JSON format: {\"sentiment\": \"positive\"} or {\"sentiment\": \"negative\"}. "
            "Do not provide any explanation or additional text."
        )
        example_wrapper_prompt = (
            "\nYou are a powerful language model."
            "You are given a user request consisting of a system prompt and a user message."
            "Your task is as follows:\n\n"
            "1. Identify the type of input the user is providing: one of [\"text\", \"image\", \"table\", \"other\"]\n"
            "2. Infer what task you are expected to perform, choosing from: \n   [\"sentiment classification\", "
            "\"summarization\", \"translation\", \"question answering\", \"information extraction\","
            " \"topic modeling\", \"other\"]\n"
            "3. Solve the task that the user is giving you and put your response in a separate field called "
            "\"user_response\" which is in JSON format\n"
            "\nRespond with a JSON object containing:\n- \"input_type\"\n- \"task_type\"\n- \"user_response\""
            " (JSON describing user response)\n"
        )

        example_output = "{\"sentiment\": \"positive\"}"
        example_wrapped_output = "```\n{\n  \"input_type\": \"text\",\n  \"task_type\": \"sentiment classification\",\n  \"user_response\": {\n    \"sentiment\": \"positive\"\n  }\n}\n```"

        return cls(
            models=ModelConfig(large_model=Model.LLAMA_405B_TURBO, small_model=Model.BERT_80M),
            requests=RequestConfig(expected_requests=1_000_000, switch_after_n_items=5_000),
            tokens=TokenConfig(input=example_review, prompt=example_prompt,
                               wrapper_prompt=example_wrapper_prompt, wrapped_requests_percent=1,
                               output=example_output, wrapped_output=example_wrapped_output),
            dev=ModelDevConfig(model_dev_costs=4),
            validation=ValidationConfig(validation_requests_percent=0)
        )


if __name__ == '__main__':
    example_scenario_dict = {
        "models": {
            "large_model": "GPT_4_1",
            "small_model": "BERT_80M"
        },
        "requests": {
            "expected_requests": 500000,
            "switch_after_n_items": 1000
        },
        "tokens": {
            "prompt": "Summarize this:",
            "wrapper_prompt": "",
            "wrapped_requests_percent": 0.8
        },
        "validation": {
            "validation_requests_percent": 0.1
        }
    }
    example_demo_scenario = DemoScenario.from_dict(example_scenario_dict)
