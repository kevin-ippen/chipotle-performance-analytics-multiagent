# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks Genie Space AI Agent
# MAGIC
# MAGIC This notebook transforms a Databricks Genie Space into a reusable AI agent tool that can be used with LangChain and other AI frameworks.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will:
# MAGIC 1. Create a Python function that queries your Genie Space
# MAGIC 2. Register this function in Unity Catalog as a reusable tool
# MAGIC 3. Wrap it as a LangChain-compatible tool
# MAGIC 4. Create an AI agent that can use this tool to answer business questions
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Access to a Databricks Genie Space
# MAGIC - Unity Catalog enabled workspace
# MAGIC - LLM endpoint configured in your workspace
# MAGIC - Workspace environment variables: `DATABRICKS_HOST` and `DATABRICKS_TOKEN` (automatically available in most Databricks environments)

# COMMAND ----------

# MAGIC %md
# MAGIC ## User Configuration
# MAGIC
# MAGIC **IMPORTANT**: Update the variables below with your specific values before running this notebook.
# MAGIC
# MAGIC **Note**: For production deployments, consider using automatic authentication passthrough by declaring the Genie space as a resource dependency when logging your agent with MLflow. See the "Advanced: Automatic Authentication" section at the end of this notebook.

# COMMAND ----------

# =============================================================================
# USER CONFIGURATION - UPDATE THESE VALUES
# =============================================================================

# Unity Catalog configuration
UC_CATALOG = "chipotle_analytics"  # Replace with your Unity Catalog name
UC_SCHEMA = "gold"    # Replace with your schema name

# LLM endpoint for the agent
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-1-70b-instruct"  # Replace with your LLM endpoint

# Genie Space ID - find this in your Genie Space URL
GENIE_SPACE_ID = "01f0942d9c071a80a775158581b637b4"  # Replace with your Genie Space ID

# Function name in Unity Catalog
FUNCTION_NAME = "query_genie_sales"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC
# MAGIC **Note**: This approach uses the Genie REST API directly, so no additional Genie-specific packages are needed.

# COMMAND ----------

# MAGIC %pip install unitycatalog-ai[databricks] langchain-community databricks-langchain

# COMMAND ----------

# Restart Python to ensure new packages are available
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import json
from typing import Dict, Any
from unitycatalog.ai.core.client import BaseFunctionClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from databricks_langchain import UCFunctionToolkit
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Genie Space Query Function

# COMMAND ----------

# Create function source code using the more reliable REST API approach
function_source = f'''
def query_genie_sales(natural_language_query: str) -> str:
    """
    Query a Databricks Genie Space with natural language and return the result.
    
    This function connects to a Databricks Genie Space and submits a natural language
    query to get insights from your data. The Genie Space will interpret the query,
    generate appropriate SQL, execute it, and return the results in a human-readable format.
    
    Args:
        natural_language_query (str): A natural language question about your data.
            Examples:
            - "What were the top 5 products by sales last quarter?"
            - "Show me the monthly revenue trend for the past year"
            - "Which customers have the highest lifetime value?"
    
    Returns:
        str: The response from the Genie Space, which may include:
            - Textual analysis and insights
            - Data summaries
            - Recommendations
            - Error messages if the query cannot be processed
    
    Raises:
        Exception: If there's an error connecting to the Genie Space or processing the query.
    
    Example:
        >>> result = query_genie_sales("What is our total revenue this year?")
        >>> print(result)
        "Based on the data, your total revenue this year is $2.5M, which represents 
         a 15% increase compared to last year..."
    """
    import requests
    import time
    import os
    
    try:
        # Configuration embedded in function
        genie_space_id = "{GENIE_SPACE_ID}"
        
        # Get workspace details from environment
        databricks_instance = "https://" + os.environ.get("DATABRICKS_HOST", "").replace("https://", "")
        access_token = os.environ.get("DATABRICKS_TOKEN", "")
        
        if not databricks_instance or not access_token:
            return "Error: Missing Databricks workspace configuration. Please ensure DATABRICKS_HOST and DATABRICKS_TOKEN environment variables are set."
        
        headers = {{
            "Authorization": f"Bearer {{access_token}}",
            "Content-Type": "application/json",
        }}
        
        poll_interval = 2.0
        timeout = 60.0

        # Step 1: Start a new conversation
        start_url = f"{{databricks_instance}}/api/2.0/genie/spaces/{{genie_space_id}}/start-conversation"
        payload = {{"content": natural_language_query}}
        
        resp = requests.post(start_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        conversation_id = data["conversation_id"]
        message_id = data["message_id"]

        # Step 2: Poll for completion
        poll_url = f"{{databricks_instance}}/api/2.0/genie/spaces/{{genie_space_id}}/conversations/{{conversation_id}}/messages/{{message_id}}"
        start_time = time.time()
        
        while True:
            poll_resp = requests.get(poll_url, headers=headers)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = poll_data.get("status")
            
            if status == "COMPLETED":
                # Check if we have attachments with query results
                if "attachments" in poll_data and poll_data["attachments"]:
                    attachment_id = poll_data["attachments"][0]["attachment_id"]
                    result_url = f"{{databricks_instance}}/api/2.0/genie/spaces/{{genie_space_id}}/conversations/{{conversation_id}}/messages/{{message_id}}/attachments/{{attachment_id}}/query-result"
                    response = requests.get(result_url, headers=headers)
                    response.raise_for_status()
                    return response.json().get("statement_response", "Query completed successfully.")
                else:
                    # Return the content if no attachments
                    return poll_data.get("content", "Query completed successfully.")
            
            elif status == "FAILED":
                error_msg = poll_data.get("content", "Query failed with unknown error.")
                return f"Genie query failed: {{error_msg}}"
            
            if time.time() - start_time > timeout:
                return "Error: Genie API query timed out after 60 seconds."
            
            time.sleep(poll_interval)
            
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Genie API: {{str(e)}}"
    except Exception as e:
        return f"Error querying Genie Space: {{str(e)}}"
'''

# Execute the function definition to create the function object
exec(function_source)

# COMMAND ----------

def query_genie_sales(natural_language_query: str) -> str:
    import requests
    import time
    import os

    try:
        genie_space_id = GENIE_SPACE_ID
        databricks_instance = "https://" + os.environ.get("DATABRICKS_HOST", "").replace("https://", "")
        access_token = os.environ.get("DATABRICKS_TOKEN", "")

        if not databricks_instance or not access_token:
            return "Error: Missing Databricks workspace configuration. Please ensure DATABRICKS_HOST and DATABRICKS_TOKEN environment variables are set."

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        poll_interval = 2.0
        timeout = 60.0

        start_url = f"{databricks_instance}/api/2.0/genie/spaces/{genie_space_id}/start-conversation"
        payload = {"content": natural_language_query}

        resp = requests.post(start_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        conversation_id = data["conversation_id"]
        message_id = data["message_id"]

        poll_url = f"{databricks_instance}/api/2.0/genie/spaces/{genie_space_id}/conversations/{conversation_id}/messages/{message_id}"
        start_time = time.time()

        while True:
            poll_resp = requests.get(poll_url, headers=headers)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = poll_data.get("status")

            if status == "COMPLETED":
                if "attachments" in poll_data and poll_data["attachments"]:
                    attachment_id = poll_data["attachments"][0]["attachment_id"]
                    result_url = (
                        f"{databricks_instance}/api/2.0/genie/spaces/{genie_space_id}/conversations/"
                        f"{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
                    )
                    response = requests.get(result_url, headers=headers)
                    response.raise_for_status()
                    return response.json().get("statement_response", "Query completed successfully.")
                else:
                    return poll_data.get("content", "Query completed successfully.")

            elif status == "FAILED":
                error_msg = poll_data.get("content", "Query failed with unknown error.")
                return f"Genie query failed: {error_msg}"

            if time.time() - start_time > timeout:
                return "Error: Genie API query timed out after 60 seconds."

            time.sleep(poll_interval)

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Genie API: {str(e)}"
    except Exception as e:
        return f"Error querying Genie Space: {str(e)}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Function in Unity Catalog

# COMMAND ----------

# Drop the old function if needed
client.delete_function(
    function_name=f"{UC_CATALOG}.{UC_SCHEMA}.{FUNCTION_NAME}"
)
# Then re-register as in Cell 2

# COMMAND ----------

# Initialize the Databricks Function Client
client = DatabricksFunctionClient()

# Verify configuration before registering
if GENIE_SPACE_ID == "your-genie-space-id":
    print("❌ Please update GENIE_SPACE_ID in the User Configuration section before proceeding")
    raise ValueError("Configuration not updated")

if UC_CATALOG == "your_catalog_name" or UC_SCHEMA == "your_schema_name":
    print("❌ Please update UC_CATALOG and UC_SCHEMA in the User Configuration section before proceeding")
    raise ValueError("Configuration not updated")

print(f"📝 Configuration check passed:")
print(f"   - Catalog: {UC_CATALOG}")
print(f"   - Schema: {UC_SCHEMA}")
print(f"   - Genie Space ID: {GENIE_SPACE_ID}")
print(f"   - LLM Endpoint: {LLM_ENDPOINT_NAME}")

# Register the function in Unity Catalog
try:
    uc_function = client.create_python_function(
        func=query_genie_sales,
        catalog=UC_CATALOG,
        schema=UC_SCHEMA,
        replace=True  # This allows updating the function if it already exists
    )
    print(f"✅ Function successfully registered as: {UC_CATALOG}.{UC_SCHEMA}.{FUNCTION_NAME}")
    print(f"Function info: {uc_function}")
except Exception as e:
    print(f"❌ Error registering function: {str(e)}")
    print("\n💡 Troubleshooting tips:")
    print("   - Ensure you have CREATE FUNCTION permissions in Unity Catalog")
    print("   - Verify the catalog and schema exist")
    print("   - Check that your workspace has the DATABRICKS_HOST and DATABRICKS_TOKEN environment variables available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Function (Optional)
# MAGIC
# MAGIC Test the registered function directly to ensure it works correctly.

# COMMAND ----------

# Test the function with a sample query
test_query = "What are the key metrics for our business?"

try:
    result = client.execute_function(
        function_name=f"{UC_CATALOG}.{UC_SCHEMA}.{FUNCTION_NAME}",
        parameters={"natural_language_query": test_query}
    )
    print("✅ Function test successful!")
    print(f"Query: {test_query}")
    print(f"Result: {result}")
except Exception as e:
    print(f"❌ Function test failed: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create LangChain Tool

# COMMAND ----------

# Create a LangChain-compatible tool from the Unity Catalog function
try:
    toolkit = UCFunctionToolkit(
        function_names=[f"{UC_CATALOG}.{UC_SCHEMA}.{FUNCTION_NAME}"],
        client=client
    )
    
    tools = toolkit.get_tools()
    print(f"✅ Created LangChain tool: {tools[0].name}")
    print(f"Tool description: {tools[0].description}")
except Exception as e:
    print(f"❌ Error creating LangChain tool: {str(e)}")
    tools = []

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create AI Agent

# COMMAND ----------

# Initialize the LLM
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# Define the agent prompt
agent_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are a business intelligence assistant with access to a powerful data analysis tool.

IMPORTANT INSTRUCTIONS:
- When users ask questions about data, metrics, trends, or business insights, use the query_genie_sales tool
- The tool can understand natural language queries about business data
- Always use the tool for questions that might require data analysis or database queries
- Provide clear, actionable insights based on the tool's results
- If the tool returns an error, explain what might have gone wrong and suggest alternative approaches

You have access to the following tools:
{tools}

Tool Names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}
"""
)

# Create the agent if we have tools
if tools:
    try:
        agent = create_react_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True
        )
        print("✅ AI Agent created successfully!")
    except Exception as e:
        print(f"❌ Error creating agent: {str(e)}")
        agent_executor = None
else:
    print("❌ Cannot create agent - no tools available")
    agent_executor = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Usage
# MAGIC
# MAGIC Now you can use your AI agent to answer business questions!

# COMMAND ----------

# Example questions you can ask your agent
example_questions = [
    "What were our top-performing products last month?",
    "Show me the revenue trend over the past quarter",
    "Which customer segments are most profitable?",
    "What are the key performance indicators I should focus on?",
    "How did our sales compare to last year?"
]

print("📝 Example questions you can ask your AI agent:")
for i, question in enumerate(example_questions, 1):
    print(f"{i}. {question}")

# COMMAND ----------

# Test the agent with an example question
if agent_executor:
    test_question = "What are the key business metrics I should be tracking?"
    
    print(f"🤖 Testing agent with question: '{test_question}'")
    print("=" * 50)
    
    try:
        response = agent_executor.invoke({"input": test_question})
        print(f"\n📊 Agent Response:")
        print(response["output"])
    except Exception as e:
        print(f"❌ Error running agent: {str(e)}")
else:
    print("❌ Agent not available - please check the setup steps above")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Chat (Optional)
# MAGIC
# MAGIC Uncomment and run the cell below to create an interactive chat interface with your AI agent.

# COMMAND ----------

# Uncomment the code below to create an interactive chat interface

# def chat_with_agent():
#     """Interactive chat function"""
#     if not agent_executor:
#         print("❌ Agent not available. Please complete the setup steps first.")
#         return
#     
#     print("🤖 AI Business Intelligence Agent")
#     print("Ask me questions about your business data!")
#     print("Type 'quit' to exit.\n")
#     
#     while True:
#         question = input("You: ").strip()
#         
#         if question.lower() in ['quit', 'exit', 'q']:
#             print("👋 Goodbye!")
#             break
#         
#         if not question:
#             continue
#         
#         try:
#             response = agent_executor.invoke({"input": question})
#             print(f"\n🤖 Agent: {response['output']}\n")
#         except Exception as e:
#             print(f"❌ Error: {str(e)}\n")

# Uncomment the line below to start the interactive chat
# chat_with_agent()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced: Automatic Authentication Passthrough (Optional)
# MAGIC
# MAGIC For production deployments, you can leverage Databricks' automatic authentication passthrough instead of manual token management. This approach requires MLflow 2.17.1+ and the MLflow Agent Framework.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1: Using MLflow Agent Framework with Automatic Authentication
# MAGIC
# MAGIC If you're building a full MLflow agent for production deployment, you can declare the Genie space as a resource dependency to enable automatic authentication:
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC from mlflow.models.resources import DatabricksGenieSpace
# MAGIC
# MAGIC # When logging your agent model
# MAGIC with mlflow.start_run():
# MAGIC     mlflow.pyfunc.log_model(
# MAGIC         python_model="your_agent.py",
# MAGIC         artifact_path="agent",
# MAGIC         input_example=input_example,
# MAGIC         resources=[
# MAGIC             DatabricksGenieSpace(genie_space_id="your-genie-space-id"),
# MAGIC             # Other resources...
# MAGIC         ]
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC - Automatic credential management and rotation
# MAGIC - No need to handle tokens manually
# MAGIC - Follows security best practices
# MAGIC - Requires "Can Run" permission on the Genie space
# MAGIC
# MAGIC **Requirements:**
# MAGIC - MLflow 2.17.1 or above
# MAGIC - MLflow Agent Framework setup
# MAGIC - Deployment through Databricks Model Serving

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2: Using Databricks SDK with WorkspaceClient
# MAGIC
# MAGIC For agents using the Databricks SDK, you can access Genie spaces through the WorkspaceClient:
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC def query_genie_with_sdk(natural_language_query: str) -> str:
# MAGIC     """Alternative implementation using Databricks SDK"""
# MAGIC     w = WorkspaceClient()  # Uses automatic authentication when deployed
# MAGIC     
# MAGIC     # Use the SDK's genie methods
# MAGIC     # Note: Specific SDK methods for Genie may vary based on SDK version
# MAGIC     # This is a conceptual example
# MAGIC     
# MAGIC     return "SDK-based Genie query result"
# MAGIC ```
# MAGIC
# MAGIC **Note:** The current notebook uses the REST API approach because it's more universally compatible and doesn't require specific MLflow Agent Framework setup.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC 🎉 **Congratulations!** You have successfully created an AI agent that can query your Databricks Genie Space.
# MAGIC
# MAGIC ### What you've accomplished:
# MAGIC
# MAGIC 1. ✅ Created a `query_genie_space` function that connects to your Genie Space using the REST API
# MAGIC 2. ✅ Registered the function in Unity Catalog for reusability
# MAGIC 3. ✅ Wrapped it as a LangChain-compatible tool
# MAGIC 4. ✅ Built an AI agent that can use this tool to answer business questions
# MAGIC
# MAGIC ### Technical Notes:
# MAGIC
# MAGIC - **REST API Approach**: This implementation uses the Genie REST API directly for maximum reliability and compatibility
# MAGIC - **Self-contained**: The function includes all necessary dependencies and configuration
# MAGIC - **Error Handling**: Robust error handling for network issues, timeouts, and API errors
# MAGIC - **Production Ready**: Follows official Databricks Genie API patterns and best practices
# MAGIC
# MAGIC ### Next steps:
# MAGIC
# MAGIC - **Integrate**: Use `agent_executor.invoke({"input": "your question"})` in other notebooks
# MAGIC - **Customize**: Modify the agent prompt for your specific use case
# MAGIC - **Scale**: Register multiple Genie Spaces as different tools
# MAGIC - **Deploy**: Create a web interface or API endpoint for your agent
# MAGIC - **Production**: Consider upgrading to automatic authentication passthrough for production deployments
# MAGIC
# MAGIC ### Troubleshooting:
# MAGIC
# MAGIC - Ensure your Genie Space ID is correct (check the URL)
# MAGIC - Verify Unity Catalog permissions
# MAGIC - Check that your LLM endpoint is accessible
# MAGIC - Make sure your Genie Space has appropriate data sources configured
# MAGIC - Verify that `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables are available in your workspace
# MAGIC - For production deployments, ensure you have "Can Run" permission on the Genie space
# MAGIC
# MAGIC 🎉 **Congratulations!** You have successfully created an AI agent that can query your Databricks Genie Space.
# MAGIC
# MAGIC ### What you've accomplished:
# MAGIC
# MAGIC 1. ✅ Created a `query_genie_space` function that connects to your Genie Space using the REST API
# MAGIC 2. ✅ Registered the function in Unity Catalog for reusability
# MAGIC 3. ✅ Wrapped it as a LangChain-compatible tool
# MAGIC 4. ✅ Built an AI agent that can use this tool to answer business questions
# MAGIC
# MAGIC ### Technical Notes:
# MAGIC
# MAGIC - **REST API Approach**: This implementation uses the Genie REST API directly for maximum reliability in Unity Catalog execution environments
# MAGIC - **Self-contained**: The function includes all necessary dependencies and configuration
# MAGIC - **Error Handling**: Robust error handling for network issues, timeouts, and API errors
# MAGIC
# MAGIC ### Next steps:
# MAGIC
# MAGIC - **Integrate**: Use `agent_executor.invoke({"input": "your question"})` in other notebooks
# MAGIC - **Customize**: Modify the agent prompt for your specific use case
# MAGIC - **Scale**: Register multiple Genie Spaces as different tools
# MAGIC - **Deploy**: Create a web interface or API endpoint for your agent
# MAGIC
# MAGIC ### Troubleshooting:
# MAGIC
# MAGIC - Ensure your Genie Space ID is correct (check the URL)
# MAGIC - Verify Unity Catalog permissions
# MAGIC - Check that your LLM endpoint is accessible
# MAGIC - Make sure your Genie Space has appropriate data sources configured
# MAGIC - Verify that `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables are available in your workspace
