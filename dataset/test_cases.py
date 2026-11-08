from deepeval.test_case import LLMTestCase
question = "What is a neural network, and how does it learn?"

retrieved_context= [
    """
    Neural networks are computational models inspired
    by the human brain.
    They consist of interconnected artificial neurons
    organized into layers.
    During training, neural networks adjust their
    weights to minimize prediction errors
    Backpropagation calculates gradients of the loss
    function with respect to the network's weights.
    An optimizer uses these gradients to update
    the weights during training.
    """
]

expected_answer =  """
A neural network is a computational model inspired
by the human brain.
It consists of interconnected artificial neurons
organized into layers.
Neural networks learn by adjusting their weights
during training to minimize prediction errors.
Backpropagation calculates gradients, and an
optimizer uses these gradients to update the weights."""

# test-cases writting 
#test case 1  as good response 

good_response=LLMTestCase(
    input=question,
    actual_output="""
    A neural network is a computational model inspired
    by the human brain.
    It consists of interconnected artificial neurons
    organized into layers.
    During training, it learns by adjusting its weights
    to minimize prediction errors.
    Backpropagation calculates gradients, and an
    optimizer uses these gradients to update the weights.
    """,
    expected_output=expected_answer,
    retrieval_context=retrieved_context
)

#test case 2 as irrelavent response 
irrelavent_reponse=LLMTestCase(
    input=question,
    actual_output="""
    Python is a high-level programming language.
    It is widely used in artificial intelligence,
    web development, and automation.
    """,
    expected_output=expected_answer,
    retrieval_context=retrieved_context
)
hallucinated_response=LLMTestCase(
    input=question,
    actual_output="""
    A neural network is a computational model inspired
    by the human brain.
    It consists of interconnected artificial neurons
    organized into layers.
    During training, it learns by adjusting its weights
    to minimize prediction errors.
    Backpropagation calculates gradients, and an
    optimizer uses these gradients to update the weights.
    """,
    expected_output=expected_answer,
    retrieval_context=retrieved_context
)

Incomplete_response=LLMTestCase(
    input=question,
        actual_output="""
    A neural network is a computational model
    inspired by the human brain.
    It consists of interconnected artificial neurons
    organized into layers.
    """,
    expected_output=expected_answer,
    retrieval_context=retrieved_context
)
ppor_explainmation= LLMTestCase(
    input=question,
    actual_output="""
       Neural networks are computational models with
    interconnected nodes and parameterized weights
    whose optimization involves gradient propagation
    through differentiable transformations and
    iterative parameter updates to minimize a
    mathematical loss objective.
    """,
    expected_output=expected_answer,
    retrieval_context=retrieved_context
)

#connecting all thes cases
test_cases=[good_response, irrelavent_reponse, hallucinated_response, Incomplete_response, ppor_explainmation]