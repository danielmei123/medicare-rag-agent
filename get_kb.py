import boto3

client = boto3.client("bedrock-agent", region_name="us-east-1")
for kb in client.list_knowledge_bases()["knowledgeBaseSummaries"]:
    print(kb["knowledgeBaseId"], "|", kb["name"], "|", kb["status"])