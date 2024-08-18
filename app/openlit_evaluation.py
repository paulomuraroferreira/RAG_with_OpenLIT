from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics.critique import harmfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import openlit

class Evaluation:
    def __init__(self, question, answer, context, user_feedback, question_number):
        self.metrics = [faithfulness, answer_relevancy, harmfulness]     
        self.llm_ = ChatOpenAI()
        self.emb = OpenAIEmbeddings()   
        self.question = question
        self.answer = answer
        self.contexts = context
        self.user_feedback = user_feedback
        self.question_number = question_number

    @staticmethod
    def init_ragas_metrics(metrics, llm, embedding):
        for metric in metrics:
            if isinstance(metric, MetricWithLLM):
                metric.llm = llm
            if isinstance(metric, MetricWithEmbeddings):
                metric.embeddings = embedding
            run_config = RunConfig()
            metric.init(run_config)

    async def score_with_ragas(self, query, chunks, answer):
        scores = {}
        for m in self.metrics:
            print(f"calculating {m.name}")
            scores[m.name] = await m.ascore(
                row={"question": query, "contexts": chunks, "answer": answer}
            )
        return scores

    async def main(self):
        self.init_ragas_metrics(
                                self.metrics,
                                llm=LangchainLLMWrapper(self.llm_),
                                embedding=LangchainEmbeddingsWrapper(self.emb),
                                 )     
        
        ragas_scores = await self.score_with_ragas(self.question, self.contexts, self.answer)

        metadata_dict = {'question': self.question,
                                     'answer': self.answer, 
                                     'context': self.contexts, 
                                     'question_number': self.question_number}

        for m in self.metrics:
            if m.name == 'harmfulness':
                value = 'True' if bool(ragas_scores[m.name]) else 'False'
            else:
                value = ragas_scores[m.name]

            with openlit.start_trace(f'{m.name}') as trace:
                trace.set_result(value)
                trace.set_metadata({**metadata_dict, 
                                    'metric_name' : m.name,
                                    'metric_value': value})

        value_feedback = 'Good answer' if self.user_feedback['binary_score'] else 'Bad Answer'

        with openlit.start_trace("User_feedback") as trace:
            trace.set_result(value_feedback)
            trace.set_metadata({**metadata_dict, 
                                'metric_name' : 'User_feedback',
                                'metric_value': value_feedback})

        with openlit.start_trace("User feedback text") as trace:
            trace.set_result(value_feedback)
            trace.set_metadata(({**metadata_dict, 
                                 'metric_name' : 'User feedback text',
                                 'metric_value': self.user_feedback['text']}))