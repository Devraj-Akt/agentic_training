import os
from dotenv import load_dotenv

# Azure AI Agent SDK
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    ConnectedAgentTool,
    MessageRole,
    ListSortOrder)
from azure.identity import DefaultAzureCredential

# Clear console
os.system('cls' if os.name == 'nt' else 'clear')

# Load env vars
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

# Create Agents Client
agents_client = AgentsClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True
    ),
)

with agents_client:

    # ----------------------------------------
    # 1. Document Classification Agent
    # ----------------------------------------
    doc_agent_name = "document_classifier_agent_team_eight"
    doc_agent_instructions = """
Given the user prompt, identify & classify the intent of the user query to any one of the below types:
investments
capitalization
private_equity
interest
vesting
employee_benefits
esop
ownership_of_shares
foreign_investors
loans
stock_option
investment_company
seed
board
financing
grant_of_option
payment_terms
taxes
payment
compensation
base-salary
investment-company-act
dividends
shares
grant
conversion_of_shares
WHEREAS
NOW
Notices
Governing
Counterparts
Severability
Miscellaneous
Definitions
Entire
Termination
Indemnification
Headings
Representations
Assignment
Insurance
Confidentiality
 
Once the type is identified, pass this information to the next agent which is "clause_extraction_agent_team_eight"
    """
    doc_agent = agents_client.create_agent(
        model=model_deployment,
        name=doc_agent_name,
        instructions=doc_agent_instructions
    )

    # ----------------------------------------
    # 2. Clause Extraction Agent
    # ----------------------------------------
    clause_agent_name = "clause_extraction_agent_team_eight"
    clause_agent_instructions = """
Obtain the clause type from "document_classifier_agent_team_eight". Based on the clause type and the user's prompt, extract similar clauses from the added knowledge index
 
    Return clauses to the next agent i.e. "compliance_risk_agent_team_8"
    """
    clause_agent = agents_client.create_agent(
        model=model_deployment,
        name=clause_agent_name,
        instructions=clause_agent_instructions
    )

    # ----------------------------------------
    # 3. Compliance & Risk Agent
    # ----------------------------------------
    compliance_agent_name = "compliance_risk_agent_team_eight"
    compliance_agent_instructions = """
From the clauses that was received from "clause_extraction_agent_team_eight", agent.
Now, with the user prompt, type identified and clauses extracted, provided detailed information in below order :
1. Summary or response for user's query
2. Risk category
3. Any other important information needed for the user based on the query
    """
    compliance_agent = agents_client.create_agent(
        model=model_deployment,
        name=compliance_agent_name,
        instructions=compliance_agent_instructions
    )

    # ----------------------------------------
    # Connected Agent Tools
    # ----------------------------------------
    doc_agent_tool = ConnectedAgentTool(
        id=doc_agent.id,
        name=doc_agent_name,
        description="Identifies document type and jurisdiction"
    )

    clause_agent_tool = ConnectedAgentTool(
        id=clause_agent.id,
        name=clause_agent_name,
        description="Extracts key legal clauses"
    )

    compliance_agent_tool = ConnectedAgentTool(
        id=compliance_agent.id,
        name=compliance_agent_name,
        description="Flags compliance and legal risks"
    )

    # ----------------------------------------
    # 4. Orchestrator Agent (Like Triage Agent)
    # ----------------------------------------
    review_agent_name = "legal_review_orchestrator_team_eight"
    review_agent_instructions = """
    Perform a legal document review.
    Use connected agents to:
    1. Identify user prompt type
    2. Extract clauses
    3. Assess compliance risks and provide the information
    Provide a concise final summary.
Act as a bridge between the agents to transfer the details within agents
    """
    review_agent = agents_client.create_agent(
        model=model_deployment,
        name=review_agent_name,
        instructions=review_agent_instructions,
        tools=[
            doc_agent_tool.definitions[0],
            clause_agent_tool.definitions[0],
            compliance_agent_tool.definitions[0]
        ]
    )

    # ----------------------------------------
    # Run the Legal Review
    # ----------------------------------------
    print("Creating legal review thread...")
    thread = agents_client.threads.create()

    prompt = input("\nPaste the legal document text:\n")

    agents_client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=prompt
    )

    print("\nProcessing legal review. Please wait...\n")

    run = agents_client.runs.create_and_process(
        thread_id=thread.id,
        agent_id=review_agent.id
    )

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # ----------------------------------------
    # Display Results
    # ----------------------------------------
    messages = agents_client.messages.list(
        thread_id=thread.id,
        order=ListSortOrder.ASCENDING
    )

    for message in messages:
        if message.text_messages:
            last_msg = message.text_messages[-1]
            print(f"{message.role}:\n{last_msg.text.value}\n")
