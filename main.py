import pandas as pd 
from langchain_google_genai import ChatGoogleGenerativeAI
import langgraph
import time
from typing import TypedDict, Literal
# from pypdf import PdfReader
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import SerpAPIWrapper


Validated_proof_data = '''NAME: RANITA LAHA
DOB: 01-01-1991
GENDER: FEMALE
2334 5678 7890
CUST : abc1'''
Submitted_proof_data = pd.read_excel("Proof_data.xlsx")
# Create an empty DataFrame with column names
# Validated_proof_data_tbl = pd.DataFrame(columns=['Name', 'Dob', 'Gender','Aadhar_nbr','CUST_ID'])

# how do i pass a cust id in my prompt


     

def Langgraph_input(input):

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=5,
        google_api_key="AIzaSyDggnflT9QPn7zc2g_oxh1Wh9IhIKVQhHk"  # Replace with actual key
    )

    # Graph state
    class State(TypedDict):
        human_input: str
        response: str
        feedback: str
        status: str

    # Schema for structured output to use in evaluation
    class Feedback(BaseModel):
        grade: Literal["valid", "invalid"] = Field(
            description=f'''Determine the response as either "valid" or "invalid" based on the following conditions:
                            1. "valid" if the output is completely printed; otherwise, "invalid".
                            
                        '''
        )
        feedback: str = Field(
            description="If the response is invalid, provide feedback on how to improve it."
        )

    # Augment the LLM with schema for structured output
    evaluator = llm.with_structured_output(Feedback)

    # Nodes
    def llm_response_generator(state: State):
        """LLM generates a response"""

        time.sleep(2)  # Adding delay to prevent timeout

        if state.get("feedback"):
            msg = llm.invoke(
                f"Generate a response for: {state['human_input']}. Consider the feedback: {state['feedback']}"
            )
            print("\nLLM Generated FEEDBACK:", msg.content)
        else:
            msg = llm.invoke(f"Generate a response for: {state['human_input']}")
            print("\nLLM Generated Response:", msg.content)

        return {"response": msg.content}

    def llm_response_evaluator(state: State):
        """LLM evaluates the response"""

        time.sleep(2)  # Adding delay to prevent timeout

        evaluation = evaluator.invoke(f"Evaluate the response: {state['response']}")
        print("\nEvaluation Result:", evaluation.grade)
        if evaluation.grade == "invalid":
            print("Feedback Provided:", evaluation.feedback)

        return {"status": evaluation.grade, "feedback": evaluation.feedback}

    # Conditional edge function to route back to generator or end based on evaluation
    def route_response(state: State):
        """Route based on evaluation"""

        if state["status"] == "valid":
            return "Accepted"
        elif state["status"] == "invalid":
            return "Rejected + Feedback"

    # Build workflow
    workflow_builder = StateGraph(State)

    # Add nodes
    workflow_builder.add_node("llm_response_generator", llm_response_generator)
    workflow_builder.add_node("llm_response_evaluator", llm_response_evaluator)

    # Add edges to connect nodes
    workflow_builder.add_edge(START, "llm_response_generator")
    workflow_builder.add_edge("llm_response_generator", "llm_response_evaluator")
    workflow_builder.add_conditional_edges(
        "llm_response_evaluator",
        route_response,
        {
            "Accepted": END,
            "Rejected + Feedback": "llm_response_generator",
        },
    )

    # Compile the workflow
    workflow = workflow_builder.compile()

    # Invoke workflow with an example input
    state = workflow.invoke({"human_input": input})
    return state["response"]
     

llm_input = f''' Instructions:

                    Perform all tasks using the provided datasets: Validated_proof_data, Submitted_proof_data.
                    Ensure you read the contents from the file as it is.
                    Present outputs in well-structured format where applicable.
                    Do not print Python code in the output.
                Tasks:
                
                    Task 1: 
                    Validate  if NameV from {Submitted_proof_data} matches Name in  {Validated_proof_data} and 
                    Validate if DOBV from {Submitted_proof_data} matches Dob in  {Validated_proof_data} and 
                    Validate if gndrv from {Submitted_proof_data} matches Gender in  {Validated_proof_data} and
                    Validate if adhv from {Submitted_proof_data} matches Aadhar_nbr in  {Validated_proof_data} 

                    

                Expected Output:
                    1.if Task 1is suuccessful tell the KYC status as "Pass" else "Fail"
            '''
response = Langgraph_input(llm_input)
     

print(response)
     


     


     