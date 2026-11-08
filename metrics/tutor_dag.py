from deepeval.metrics import  GEval, DAGMetric
from deepeval.metrics.dag import ( TaskNode, BinaryJudgementNode,NonBinaryJudgementNode,DeepAcyclicGraph)
from deepeval.test_case  import SingleTurnParams
#node 1 which is where we are creating our task node one of the 4 most import nodes 
# it does the processing and downstreams the data


answer_analysis=TaskNode (
    instructions="""
       Analyze the student's question, the AI-generated
    answer, and the retrieved course material.
    Perform the following tasks:
    1. Identify the student's question intent.
    2. Extract the important concepts covered
       in the generated answer.
    3. Extract all factual claims made by the answer.
    4. Identify important concepts required to answer
       the question that appear to be missing.
    5. Identify claims that may require verification
       against the retrieved course material.
    Keep the analysis concise and structured.
    Do not assign a score.
    Do not make a final quality judgement.
    """,
    output_label="Answer Analysis",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    label='Answer Analysis',

    )

    #creating node 2 which is the relevance check 
relevance_check=BinaryJudgementNode(
      criteria="""
Determine whether the AI-generated answer
    directly addresses the student's question.
    Consider the following:
    1. Does the answer address the actual question?
    2. Does it stay relevant to the requested topic?
    3. Does it avoid unrelated information?
    Use the Answer Analysis produced by the previous
    task node as supporting evidence.
    Return True only if the answer is relevant
    to the student's question.
    """,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
    label='Relevance Check',
    )

completeness_check = NonBinaryJudgementNode(
    criteria="""

    Classify the completeness of the AI-generated answer.

    Compare the student's question, expected answer,

    generated answer, and Answer Analysis.

    Choose exactly one category:

    Complete:

    The answer covers all important concepts required

    to answer the question.

    Mostly Complete:

    The answer covers the majority of important

    concepts but misses some relevant details.

    Incomplete:

    The answer misses several important concepts

    or fails to explain the question adequately.

    """,

    evaluation_params=[

        SingleTurnParams.INPUT,

        SingleTurnParams.ACTUAL_OUTPUT,

        SingleTurnParams.EXPECTED_OUTPUT,

    ],

    label="Completeness Check",

)
grounding_check=BinaryJudgementNode(
   """ Determine whether the factual claims in the
    AI-generated answer are supported by the
    retrieved course material.
    Consider the following:
    1. Identify factual claims in the generated answer.
    2. Compare those claims with the retrieved context.
    3. Identify unsupported or contradictory claims.
    4. Do not treat missing information as proof
       that a claim is false.
    Use the Answer Analysis as supporting evidence.
    Return True only if the material factual claims
    are adequately supported by the retrieved context.
    """,
   evaluation_params=[
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.RETRIEVAL_CONTEXT,
   ],
   label='Grounding Check',
)

completeness_check=NonBinaryJudgementNode(
   """ 
   
   
   """,
   evaluation_params=[
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT,
   ],
   label='Completeness Check',
)




educational_quality = GEval(
    name="Educational Quality",
    criteria="""
    Evaluate the educational quality of the AI tutor's
    explanation.
    Assess the following dimensions:
    1. Clarity:
       Is the explanation easy to understand?
    2. Conceptual Understanding:
       Does it explain the underlying concept rather
       than simply presenting terminology?
    3. Logical Flow:
       Are ideas presented in a logical sequence?
    4. Beginner Friendliness:
       Is the explanation appropriate for a student
       learning the topic?
    5. Explanation Quality:
       Does the answer provide enough explanation
       without unnecessary complexity?
    Give a higher score to explanations that are
    accurate, clear, structured, and educationally useful.
    """,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
    model="gpt-4o",
)

# connecting the  nodes
answer_analysis.add_node(relevance_check)
answer_analysis.add_node(grounding_check)
answer_analysis.add_node(completeness_check)


#relevance branch 
relevance_check.add_verdict(
   verdict=False,
   score=0,)
relevance_check.add_verdict(
   verdict=True,
   then=grounding_check,
)
#grounding branch
grounding_check.add_verdict(
   verdict=False,
   score=0,
)
grounding_check.add_verdict(
   verdict=True,
   then=completeness_check,
)
#completeness branch
completeness_check.add_verdict(
   verdict="complete",
   then=educational_quality,
)
completeness_check.add_verdict(
   verdict='incomplete',
   score=0,
)
completeness_check.add_verdict(
   verdict='mostly complete',
   score=0.6,
)

#now create the dag acyclic graph
tutor_dag=DeepAcyclicGraph(
   root_nodes=[answer_analysis])

#create the dag metric
tutor_evaluation_metric=DAGMetric(
   name='AI Tutor Evaluation',
   dag=tutor_dag,
   model='gpt-4o',
   threshold=0.7,
   verbose_mode=True,
)